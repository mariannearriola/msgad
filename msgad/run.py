import torch
import numpy as np
import argparse
import networkx as nx
from utils import *
from anom_detector import *
from dataloading import *
from loss_utils import *
import random 
import time
import gc
from model import *
import torch.nn.functional as F
import MADAN.Madan as md
from visualization import *
from torch.utils.tensorboard import SummaryWriter
from madan_analysis import madan_analysis
from label_analysis import LabelAnalysis
from itertools import combinations,product
import warnings ; warnings.filterwarnings("ignore")
import torch_scatter

def graph_anomaly_detection(exp_params):
    """Pipeline for autoencoder-based graph anomaly detection"""
    # load data
    exp_name = exp_params['EXP'] ; scales = exp_params['MODEL']['SCALES'] ; dataset = exp_params['DATASET']['NAME'] ; device = exp_params['DEVICE']
    load_data = (exp_params['DATASET']['DATADIR'] is not None and exp_params['DATASET']['DATALOAD'])
    seed_everything(82)
    dataloading = DataLoading(exp_params)
    transform = dgl.AddReverse(copy_edata=True)

    sp_adj, edge_idx, feats, truth = dataloading.load_anomaly_detection_dataset()
    anoms,norms=np.where(truth==1)[0],np.where(truth==0)[0]
    if sp_adj is not None:
        adj = sparse_matrix_to_tensor(sp_adj,feats)
    recons_label,pos_edges,neg_edges=None,None,None
    
    # initialize data loading
    if edge_idx is not None:
        adj = dgl.graph((edge_idx[0],edge_idx[1]),num_nodes=feats.shape[0])
        adj.ndata['feature'] = feats
        adj.edata['w'] = torch.ones(adj.number_of_edges()).to(adj.device)
    
    # intialize model (on given device)
    adj = adj.to(exp_params['DEVICE']) ; feats = feats.to(exp_params['DEVICE'])
    
    la = LabelAnalysis(exp_params['DATASET']['NAME'],anoms,norms,exp_name)

    g = adj.to('cpu')
    g_nx = dgl_to_nx(g)[0]
    
    
    nx.set_node_attributes(g_nx,feats,'feats')

    #sc_label_new,clusts=la.run_dend(g_nx,scales+1,return_all=True)
    dataset_scales, model_scales = exp_params['DATASET']['SCALES'], exp_params['MODEL']['SCALES']
    sc_label_new,clusts=la.run_dend(g_nx,scales,return_all=True)

    og_lbls = [adj for i in range(model_scales)]

    # sample an even number of intra-cluster & inter-cluster edges for each node
    if not os.path.exists(f'{dataset}_full_adj_{model_scales}_dataset{dataset_scales}.mat'):
        lbls,pos_edges_full,neg_edges_full = [],[],[]
        for clust_ind,clust in enumerate(clusts):
            print(clust_ind)
            clusters = {}
            for node, cluster_id in enumerate(clust,0):
                clusters.setdefault(cluster_id.item(), []).append(node)
            #pos_edges= np.array([(x, y) for nodes in clusters.values() for (x, y),z in zip(combinations(nodes, 2),range(10)) if z <= 10])
            
            pos_edges= torch.tensor([(x, y) for nodes in clusters.values() for x, y in combinations(nodes, 2)])
            # for each positive edge, replace connection with a negative edge. during dataloading, index both simultaneously
            #neg_edges = pos_edges
            
            pos_clusts = clust[pos_edges[:,1]]
            clust_offset=np.random.randint(1,(clust.max()),pos_clusts.shape[0])
            pos_clusts += clust_offset ; pos_clusts = pos_clusts % (clust.max()+1)
            #neg_edges = torch.stack([torch.tensor(np.random.choice(clusters[i.item()])) for i in pos_clusts])
            #neg_edges = torch.vstack((pos_edges[:,0],neg_edges)).T
            assert(torch.where(clust[pos_edges[:,0]]==clust[pos_edges[:,1]])[0].shape[0]==pos_edges.shape[0])
            #assert(torch.where(clust[neg_edges[:,0]]==clust[neg_edges[:,1]])[0].shape[0]==0)

            lbl_adj = dgl.graph((pos_edges[:,0],pos_edges[:,1]),num_nodes=adj.number_of_nodes()).to(exp_params['DEVICE'])
            lbl_adj.ndata['feature'] = feats.to(lbl_adj.device)
            lbl_adj.edata['w'] = torch.ones(pos_edges.shape[0]).to(lbl_adj.device)
            #lbls.append(lbl_adj)
            lbls.append(transform(lbl_adj))
            pos_edges_full.append(torch.vstack((pos_edges,pos_edges.flip(1))))
            #pos_edges_full.append(pos_edges)
            #neg_edges_full.append(neg_edges)
            
        save_mat = {'lbls':[i for i in lbls],'pos_edges':[i.to_sparse() for i in pos_edges_full]}#,'neg_edges':[i.to_sparse() for i in neg_edges_full]}
        with open(f'{dataset}_full_adj_{model_scales}_dataset{dataset_scales}.mat','wb') as fout:
            pkl.dump(save_mat,fout)
    else:
        with open(f'{dataset}_full_adj_{model_scales}_dataset{dataset_scales}.mat','rb') as fin:
            mat =pkl.load(fin)
        lbls,pos_edges_full = mat['lbls'],[i.to_dense() for i in mat['pos_edges']]#,[i.to_dense() for i in mat['neg_edges']]

    if 'tfinance' in exp_params['DATASET']['NAME'] and exp_params['VIS']['VIS_FILTERS'] == True:
        dataloader = None
    else:
        dataloader = [dataloading.fetch_dataloader(lbls[i],pos_edges_full[i],i) for i in range(len(clusts))]

    struct_model,feat_model=None,None
    struct_model,params,model_loaded = init_model(feats.size(1),exp_params,args)
    regularize = True if exp_params['MODEL']['CONTRASTIVE'] == True else False

    tb = SummaryWriter(log_dir=f'runs/{exp_name}')
        
    optimizer = torch.optim.Adam(struct_model.parameters(), lr = float(exp_params['MODEL']['LR']))

    # begin model training
    if dataloader is not None:
        print(dataloader.__len__(),'batches')
    else:
        dataloader = [None]
    
    if 'tfinance' in exp_params['EXP']:
        if exp_params['VIS']['VIS_CONCENTRATION'] == True:
            madan_analysis(adj,recons_label,anoms,exp_params)

    if exp_params['VIS']['VIS_FILTERS'] or exp_params['VIS']['VIS_LOSS']:
        visualizer = Visualizer(adj,feats,exp_params,sc_label_new,norms,anoms)
    else:
        visualizer = None

    attract_edges_sel,repel_edges_sel = None,None
    LossFunc = loss_func(adj,adj.ndata['feature'],exp_params,sample=True, recons='struct', alpha=None, clusts=None, regularize=True)
    seconds = time.time()
    for epoch in range(int(exp_params['MODEL']['EPOCH'])):
        epoch_l,iter = 0,0
        #edge_anom_mats,node_anom_mats,recons_a,res_a_all = init_recons_agg(adj.number_of_nodes(),adj.ndata['feature'].shape[1],exp_params)
        if model_loaded: break
        edge_ids=[]

        # unpack each of the dataloaders
        for batch,data_inds in enumerate(zip(*dataloader)):
            if load_data:
                try:
                    loaded_input,recons_label=dataloading.load_batch(batch,'train')
                except Exception as e:
                    print('err',batch)
            else:
                loaded_input = data_inds
            g_batch,pos_edges,neg_edges = zip(*loaded_input)
            #import ipdb ; ipdb.set_trace()
            if load_data: pos_edges = [i.to(device) for i in pos_edges] ; neg_edges = [i.to(device) for i in neg_edges] ; lbls = [i.to(device) for i in lbls]
            # for each dataloader...
            
            #in_nodes, pos_edges, neg_edges, g_batch, batch_sc_label = dataloading.get_edge_batch(loaded_input,sc_label_new)    
            recons_label = g_batch if not load_data else collect_recons_label(recons_label,exp_params['DEVICE'])

            edge_ids = [torch.cat((pos_edges[i],neg_edges[i]),axis=0) for i in range(len(pos_edges))]
            # UNIT TESTING
            try:
                for i in range(len(pos_edges)):
                    attract_e=torch.where(clusts[i][pos_edges[i]][:,0]==clusts[i][pos_edges[i][:,1]])[0]
                    assert(attract_e.shape[0]==pos_edges[i].shape[0])
                    repel_e=torch.where(clusts[i][pos_edges[i]][:,0]!=clusts[i][pos_edges[i][:,1]])[0]
                    attract_e_neg=torch.where(clusts[i][neg_edges[i]][:,0]==clusts[i][neg_edges[i][:,1]])[0]
                    assert(attract_e_neg.shape[0]==0)
                    assert(repel_e.shape[0]==0)
                    del attract_e,repel_e,attract_e_neg
            except Exception as e:
                print(e)
                import ipdb ; ipdb.set_trace()
            
            print('size of batch',g_batch[0].num_dst_nodes(),'nodes')
            if exp_params['DATASET']['DATASAVE']:
                dataloading.save_batch(loaded_input,recons_label,iter,'train')
                continue
            if struct_model:
                optimizer.zero_grad()
            for i in loaded_input: del i
            torch.cuda.empty_cache() ; gc.collect()

            check_gpu_usage('running model')
            if exp_params['MODEL']['IND'] == 'None':
                pos_edges_og = [edge_ids[ind][og_lbls[ind].has_edges_between(edge_ids[ind][:,0],edge_ids[ind][:,1]).nonzero().T[0]] for ind in range(len(og_lbls))]

                A_hat,res_a = struct_model(og_lbls,pos_edges_og,feats,edge_ids,vis=False,vis_name='epoch1',clusts=clusts)
            else:
                A_hat,res_a = struct_model(og_lbls,[pos_edges[exp_params['MODEL']['IND']]],feats,[edge_ids[exp_params['MODEL']['IND']]],vis=False,vis_name='epoch1',clusts=clusts)
            
            torch.cuda.empty_cache() ; gc.collect()
            check_gpu_usage('recons label collected, starting loss')
            #import ipdb ; ipdb.set_trace()
            loss,struct_loss,regloss,clustloss,nonclustloss = LossFunc.calc_loss(og_lbls, A_hat, edge_ids, clusts)
            
            # TODO: map clustloss scores to nodes in batch
            batch_scores_sc = torch.stack([gather_clust_info(clustloss[i].detach().cpu(),clusts[i],'std') for i in range(len(clustloss))])
            batch_scores = batch_scores_sc if batch == 0 else batch_scores + batch_scores_sc

            l = torch.sum(loss) if 'multi-scale' in exp_params['MODEL']['NAME'] else torch.mean(loss)
            epoch_l = loss.unsqueeze(0) if iter == 0 else torch.cat((epoch_l,l.unsqueeze(0)))
            if exp_params['MODEL']['DEBUG']:
                if iter % 100 == 0:
                    print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=', round(l.item(),3))
                
            iter += 1
            if 'cora' not in exp_params['DATASET']['NAME']: print('iter',iter)
            
            check_gpu_usage('about to backward')
            l.backward()
            optimizer.step()
            #del pos_edges, neg_edges, l, res_a,X_hat#,A_hat
            torch.cuda.empty_cache() ; gc.collect()
            
        print("Seconds since epoch =", (time.time()-seconds)/60)
        seconds = time.time()
        print("Epoch:", '%04d' % (epoch), "train_loss=", torch.round(torch.sum(loss),decimals=3).detach().cpu().item(), "losses=",torch.round(loss,decimals=4).detach().cpu())
    
        batch_scores /= len(dataloader)
        if exp_params['DATASET']['DATASAVE']: continue 
                
        print('epoch done',epoch,loss.detach().cpu())
        if True:
            if epoch == 0 and iter == 1:
                tb_writers = TBWriter(tb, truth, sc_label_new,clusts,anoms,norms,exp_params)
            labels = []
            mean_intras_tot,mean_inters_tot=[],[]
            fracs = [pos_edges[i].shape[0]/edge_ids[i].shape[0] for i in range(3)]
            anom_scores_all = []
            for sc,l in enumerate(loss):
                tb.add_scalar(f'Loss_{sc}', l, epoch)
                tb.add_scalar(f'RegLoss_{sc}', regloss[sc].sum(), epoch)
                tb.add_scalar(f'ClustLoss_{sc}', clustloss[sc].sum(), epoch)
                tb.add_scalar(f'NonClustLoss_{sc}', nonclustloss[sc].sum(), epoch)
                
                model_ind = sc if exp_params['MODEL']['IND']=='None' else exp_params['MODEL']['IND']
                #mean_intras,mean_inters,anom_scores=tb_writers.tb_write_anom(tb,g_batch,sc_label_new,edge_ids[model_ind],A_hat[sc], struct_loss[sc], sc,epoch, regloss[sc],clustloss,nonclustloss,clusts,sc_idx_inside,sc_idx_outside,entropies,anom_wise=True,fracs=fracs)
                mean_intras_cl,mean_inters_cl,anom_scores=tb_writers.tb_write_anom(adj,sc_label_new,edge_ids,A_hat[sc],res_a, struct_loss, sc,epoch, regloss[sc],clustloss,nonclustloss,clusts,anom_wise=False,fracs=fracs)
                anom_scores_all.append(anom_scores)
                mean_intras,mean_inters=mean_intras_cl,mean_inters_cl
                #labels.append(label)
                if sc == 0:
                    mean_intras_tot = mean_intras.unsqueeze(0) ; mean_inters_tot = mean_inters.unsqueeze(0)
                    mean_intras_tot_cl = mean_intras_cl.unsqueeze(0) ; mean_inters_tot_cl = mean_inters_cl.unsqueeze(0)
                else:
                    try:
                        mean_intras_tot = torch.cat((mean_intras_tot,mean_intras.unsqueeze(0)),axis=0)
                        mean_inters_tot = torch.cat((mean_inters_tot,mean_inters.unsqueeze(0)),axis=0)
                        mean_intras_tot_cl = torch.cat((mean_intras_tot_cl,mean_intras_cl.unsqueeze(0)),axis=0)
                        mean_inters_tot_cl = torch.cat((mean_inters_tot_cl,mean_inters_cl.unsqueeze(0)),axis=0)
                    except Exception as e:
                        print(e)
                        import ipdb ; ipdb.set_trace()
            sc_labels = np.unique(sc_label_new)
            for sc,l in enumerate(loss):
                if sc == 0: continue
                else:
                    for gr in range(len(mean_intras_tot[0])):
                        anom_sc = gr if gr != len(mean_intras_tot[0])-1 else 'norm'
                        mean_intra_grad = mean_intras_tot[sc,gr] -mean_intras_tot[sc-1,gr]
                        mean_inter_grad = mean_inters_tot[sc,gr] -mean_inters_tot[sc-1,gr]
                        tb.add_scalar(f'Intra_grad_{sc-1}:{sc}/Anom{anom_sc}', mean_intra_grad, epoch)
                        tb.add_scalar(f'Inter_grad_{sc-1}:{sc}/Anom{anom_sc}', mean_inter_grad, epoch)
                        #tb.add_scalar(f'Inter_grad_{(scales-(sc))}:{(scales-(sc+1))}Anom{anom_sc}',  mean_inter_grad, epoch)
            #import ipdb ; ipdb.set_trace()

            for ind,score in enumerate(anom_scores_all):
                #l_clust=torch.scatter_reduce(anom_scores_all[ind][ind].detach().cpu(), 0, clusts[ind], reduce="mean") ; l_clust = l_clust[clusts[ind]]#.to(edge_ids[0].device)#.mean()#/cl.shape[0]
                
                #anoms_found,percentages,scores=tb_writers.get_anom([l1_clust,l2_clust,l3_clust],anoms)
                #s1,s2,s3=l1_clust,l2_clust,l3_clust
                #gr_dict = self.update_dict(gr_dict,f'Percent_anom1',p1) ; gr_dict = self.update_dict(gr_dict,f'Percent_anom2',p2) ; gr_dict = self.update_dict(gr_dict,f'Percent_anom3',p3)
                # NOTE: want to minimize
                #gr_dict = self.update_dict(gr_dict,f'Num_anoms_detected1',len(anoms_found[0])) ; gr_dict = self.update_dict(gr_dict,f'Num_anoms_detected2',len(anoms_found[1])) ; gr_dict = self.update_dict(gr_dict,f'Num_anoms_detected3',len(anoms_found[2])) 
                _,prec1,ra1=tb_writers.a_clf.calc_prec(anom_scores_all[ind][ind].detach().cpu()[np.newaxis,...],truth,sc_label_new,verbose=False)
                #_,prec2,ra2=tb_writers.a_clf.calc_prec(s2[np.newaxis,...],truth,sc_label_new,clusts[1],input_scores=True,verbose=False)
                #_,prec3,ra3=tb_writers.a_clf.calc_prec(s3[np.newaxis,...],truth,sc_label_new,clusts[2],input_scores=True,verbose=False)
    
                for anom,prec in enumerate(prec1[0]):
                    tb.add_scalar(f'Precsc{ind+1}/anom{anom}', prec, epoch)
                    tb.add_scalar(f'ROC{ind+1}/anom{anom}', ra1[0][anom], epoch)
                '''
                for anom,prec in enumerate(prec1):
                    tb.add_scalar(f'Precsc2/anom{anom}', prec2[anom], epoch)
                    tb.add_scalar(f'ROC2/anom{anom}', ra2[anom], epoch)
                for anom,prec in enumerate(prec3):
                    tb.add_scalar(f'Precsc3/anom{anom}', prec3[anom], epoch)
                    tb.add_scalar(f'ROC3/anom{anom}', ra3[anom], epoch)
                '''
        #del loss,struct_loss,A_hat

        if epoch == 0:
            tot_loss = epoch_l.sum(0).unsqueeze(0)
        else:
            tot_loss = torch.cat((tot_loss,epoch_l.sum(0).unsqueeze(0)),dim=0)
        epoch_l = torch.sum(epoch_l)
        #epoch_l.backward()
        #optimizer.step()
        '''
        if struct_model:
            if struct_model.final_attn != None:
                # epoch x 3 x nodes x num filters
                train_attn_w[epoch,:,in_nodes]=F.softmax(struct_model.final_attn.detach().cpu(),1)#torch.unsqueeze(struct_model.attn,0).detach().cpu()
                #train_attn_w[epoch].scatter_(1,in_nodes.unsqueeze(0).unsqueeze(-1).to('cpu'),struct_model.final_attn.to(torch.float32).to('cpu'))
            train_losses[epoch]=torch.stack([torch.stack([torch.mean(j[j.nonzero()],axis=0) for j in i]) for i in torch.tensor(edge_anom_mats)]).squeeze(-1)
        '''
        for name, param in struct_model.named_parameters():
            tb.add_histogram(name, param.flatten(), epoch)

        # save/load trained model model
        if epoch > 1 and epoch == int(exp_params['MODEL']['EPOCH'])-1:
            torch.save(struct_model,f'{exp_name}.pt')

    print('done..')
    import ipdb ; ipdb.set_trace()
    a_clf = anom_classifier(exp_params)
    a_clf.title = 'loss'
    #a_clf.calc_prec(dgl_to_nx(adj)[0], edge_anom_mats, truth, sc_label_new, attns, clusts, cluster=False)
    

    a_clf.title = 'regloss' ; a_clf.stds = 5
    anoms_found = None
    losses = []
    #nonclustloss,clustloss=nonclustloss.detach().cpu(),clustloss.detach().cpu()
    #import ipdb ; ipdb.set_trace()
    # multiply by edges
    fracs = [pos_edges[i].shape[0]/edge_ids[i].shape[0] for i in range(3)]
    cl1,cl2,cl3=clustloss[0]*fracs[0],clustloss[1]*fracs[1],clustloss[2]*fracs[2]
    nc1,nc2,nc3=nonclustloss[0]*(1-fracs[0]),nonclustloss[1]*(1-fracs[1]),nonclustloss[2]*(1-fracs[2])
    l1 = (cl2-cl1)+(cl3-cl1)
    l2 = (cl3-cl2)+(nc1-nc2)
    l3 = (nc1-nc3)+(nc2-nc3)
    losses = [l1,l2,l3]
    for i in range(scales):
        scores,anoms_found = a_clf.calc_clust_prec(dgl_to_nx(adj)[0], clust_edge_anom_mats, nonclust_edge_anom_mats, i, anoms_found, anoms,norms, sc_label_new, attns, losses[i], cluster=False)
    #losses.append([clustloss[0]*fracs[0]+clustloss[],nonclustloss[0]*struct_loss[0],(clusts[0],clusts[0])])
    #losses.append([clustloss[1]*struct_loss[1],nonclustloss[1]*struct_loss[1],(clusts[1],clusts[1])])
    #losses.append([clustloss[2]*struct_loss[2],torch.zeros(struct_loss[0].shape),(clusts[2],clusts[2])])
    import ipdb ; ipdb.set_trace()


    #losses.append([clustloss[1]*struct_loss[1],torch.zeros(struct_loss[0].shape),(clusts[1],clusts[1])])
    #losses.append([clustloss[2]*struct_loss[2],nonclustloss[-2]*struct_loss[-2],(clusts[2],clusts[-2])])
    #losses.append([clustloss[3]*struct_loss[3],nonclustloss[-3]*struct_loss[-3],(clusts[3],clusts[-3])])
    stds = [5,4,4]
    #import ipdb ; ipdb.set_trace()
    a_clf.title = 'clustprec'

    for i in range(scales):
        a_clf.stds=stds[i]
        #edge_anom_mats = [np.zeros(edge_anom_mats[0].shape)]
        #node_anom_mats,edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,regloss[i].unsqueeze(0),feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats,recons_a,res_a_all,exp_params)
        #a_clf.calc_prec(dgl_to_nx(adj)[0], edge_anom_mats, truth, sc_label, attns, clusts, cluster=False)

        edge_anom_mats_ = [np.zeros(edge_anom_mats[0].shape)] ; node_anom_mats,clust_edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,losses[i][0].unsqueeze(0),node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats_,recons_a,res_a_all,exp_params)
        edge_anom_mats_ = [np.zeros(edge_anom_mats[0].shape)] ; node_anom_mats,nonclust_edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,losses[i][1].unsqueeze(0),node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats_,recons_a,res_a_all,exp_params)
        scores,anoms_found = a_clf.calc_clust_prec(dgl_to_nx(adj)[0], clust_edge_anom_mats, nonclust_edge_anom_mats, i, anoms_found, norms, sc_label, attns, losses[i][-1], cluster=False)
        

    LossFunc = loss_func(adj,adj.ndata['feature'],exp_params,sample=True, recons='struct', alpha=None, clusts=None, regularize=True)

    # accumulate node-wise anomaly scores via model evaluation
    struct_model.eval()

    edges=adj.edges('eid')
    #dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,device=exp_params['DEVICE'])
    if exp_params['DATASET']['DATASAVE'] == False:
        struct_scores, feat_scores = torch.zeros(scales,adj.number_of_nodes()).to(exp_params['DEVICE']),torch.zeros(exp_params['MODEL']['D'],adj.number_of_nodes()).to(exp_params['DEVICE'])
    iter = 0

    edge_anom_mats,node_anom_mats,recons_a,res_a_all = init_recons_agg(adj.number_of_nodes(),adj.ndata['feature'].shape[1],exp_params)

    all_samps = []
    if (exp_params['DATASET']['DATADIR'] is not None and exp_params['DATASET']['DATALOAD']):
        pass
        #random.shuffle(dataloader)
    else:
        dataloader = [dataloading.fetch_dataloader(x,y) for x,y in zip(lbls,pos_edges_full)]#,neg_edges_full)]
    recons_label,pos_edges,neg_edges=None,None,None
    # unpack each of the dataloaders
    for batch,data_inds in enumerate(zip(*dataloader)):
        if load_data:
            loaded_input,recons_label=dataloading.load_batch(batch,'train')
        else:
            loaded_input = [data_ind for data_ind in data_inds]
        # for each dataloader...
        in_nodes, pos_edges, neg_edges, g_batch, batch_sc_label = dataloading.get_edge_batch(loaded_input,sc_label_new)    
        recons_label = g_batch if not load_data else [collect_recons_label(i,exp_params['DEVICE']) for i in recons_label]
        
        # UNIT TESTING
        try:
            for i in range(len(pos_edges)):
                attract_e=torch.where(clusts[i][pos_edges[i]][:,0]==clusts[i][pos_edges[i][:,1]])[0]
                assert(attract_e.shape[0]==pos_edges[i].shape[0])
                repel_e=torch.where(clusts[i][pos_edges[i]][:,0]!=clusts[i][pos_edges[i][:,1]])[0]
                assert(repel_e.shape[0]==0)
                del attract_e,repel_e
        except Exception as e:
            print(e)
            import ipdb ; ipdb.set_trace()

        edge_ids = [torch.cat((pos_edges[i],neg_edges[i]),axis=0).T for i in range(len(pos_edges))]
        if exp_params['MODEL']['IND'] is not None:
            edge_ids = [edge_ids[exp_params['MODEL']['IND']]]

        if 'cora' not in exp_params['DATASET']['NAME']: print('size of batch',g_batch[0].num_dst_nodes(),'nodes')
        vis = True if (epoch == 0 and iter == 0 and exp_params['VIS']['VIS_FILTERS'] == True) else False
    
        #edges, feats = process_graph(lbls)
        # run evaluation
        # TODO: CHANGED FIRST ARG TO EDGE IDS
        A_hat,res_a = struct_model(lbls,edge_ids,feats,edge_ids,vis=vis,vis_name='test',clusts=clusts)

        # collect anomaly scores
        #edge_ids = torch.cat((pos_edges,neg_edges),axis=2).to(pos_edges.device)[:scales]
        
        edge_ids = torch.cat((pos_edges[0],neg_edges[0]),axis=1).unsqueeze(0).tile((scales,1,1))
        edge_ids_ = edge_ids
        node_ids_ = g_batch.nodes()

        loss, struct_loss, feat_cost,regloss,clustloss,nonclustloss = LossFunc.calc_loss(recons_label,A_hat, None, None, edge_ids, attract_edges_sel, repel_edges_sel,clusts)
        #loss, struct_loss, feat_cost = loss_func(recons_label, g_batch.ndata['feature'], A_hat, X_hat, res_a.detach().cpu(), edge_ids, sample=exp_params['MODEL']['SAMPLE_TEST'], recons=exp_params['MODEL']['RECONS'],alpha=exp_params['MODEL']['ALPHA'],clusts=clusts)
        if exp_params['MODEL']['NAME'] == 'gradate':
            loss = A_hat[0]

        if exp_params['MODEL']['NAME'] in ['gradate']:
            loss = A_hat[0]
            struct_loss = [A_hat[1]]
            if exp_params['MODEL']['SAMPLE_TEST']:
                l = torch.sum(loss)
            else:
                l = torch.mean(loss)
            
        node_anom_mats,edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,struct_loss,feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats,recons_a,res_a_all,exp_params)

        #if iter % 100 == 0 and exp_params['MODEL']['DEBUG']:
        #    print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=',round(l.item(),3))
        iter += 1
    

    # anomaly detection with final scores
    a_clf = anom_classifier(exp_params)
    print('structure scores')
    #import ipdb ; ipdb.set_trace()
    if exp_params['MODEL']['RECONS'] == 'both':
        print('struct scores')
        a_clf = anom_classifier(exp_params)
        a_clf.calc_prec(None, [np.mean(edge_anom_mats[0],axis=1)], truth, sc_label_new, train_attn_w)
        print('feat scores')
        a_clf.calc_prec(None, [np.mean(node_anom_mats[0],axis=1)], truth, sc_label_new, train_attn_w)
        print('combined scores')
        a_clf.calc_prec(None, loss.detach().cpu().numpy(), truth, sc_label_new, train_attn_w)
    else:
        if exp_params['DATASET']['BATCH_TYPE'] == 'node':
            if -1 in node_anom_mats[0]:
                raise('node not sampled')
            a_clf.calc_prec(None,node_anom_mats, truth, sc_label_new)
        else:
            if 'multi-scale' in exp_params['MODEL']['NAME']:
                #attns = F.softmax(train_attn_w[-1],1).numpy()
                attns = train_attn_w[-1].numpy()
            else:
                attns = None
            
            if not exp_params['MODEL']['SAMPLE_TEST']:
                a_clf.calc_prec(struct_loss.detach().cpu().numpy(), truth, sc_label_new)
            else:
                a_clf.title = 'loss'
                a_clf.calc_prec(edge_anom_mats, truth, sc_label_new, clusts)
                

                a_clf.title = 'regloss' ; a_clf.stds = 5
                anoms_found = None
                losses = []
                #nonclustloss,clustloss=nonclustloss.detach().cpu(),clustloss.detach().cpu()
                #import ipdb ; ipdb.set_trace()
                # multiply by edges
                losses.append([torch.zeros(struct_loss[0].shape),nonclustloss[0]*struct_loss[0],(clusts[0],clusts[0])])
                losses.append([clustloss[1]*struct_loss[1],nonclustloss[1]*struct_loss[1],(clusts[1],clusts[1])])
                losses.append([clustloss[2]*struct_loss[2],torch.zeros(struct_loss[0].shape),(clusts[2],clusts[2])])
                
                #losses.append([clustloss[1]*struct_loss[1],torch.zeros(struct_loss[0].shape),(clusts[1],clusts[1])])
                #losses.append([clustloss[2]*struct_loss[2],nonclustloss[-2]*struct_loss[-2],(clusts[2],clusts[-2])])
                #losses.append([clustloss[3]*struct_loss[3],nonclustloss[-3]*struct_loss[-3],(clusts[3],clusts[-3])])
                stds = [5,4,4]
                #import ipdb ; ipdb.set_trace()
                a_clf.title = 'clustprec'

                for i in range(scales):
                    a_clf.stds=stds[i]
                    #edge_anom_mats = [np.zeros(edge_anom_mats[0].shape)]
                    #node_anom_mats,edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,regloss[i].unsqueeze(0),feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats,recons_a,res_a_all,exp_params)
                    #a_clf.calc_prec(dgl_to_nx(adj)[0], edge_anom_mats, truth, sc_label, attns, clusts, cluster=False)

                    edge_anom_mats_ = [np.zeros(edge_anom_mats[0].shape)] ; node_anom_mats,clust_edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,losses[i][0].unsqueeze(0),feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats_,recons_a,res_a_all,exp_params)
                    edge_anom_mats_ = [np.zeros(edge_anom_mats[0].shape)] ; node_anom_mats,nonclust_edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,losses[i][1].unsqueeze(0),feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats_,recons_a,res_a_all,exp_params)
                    scores,anoms_found = a_clf.calc_clust_prec(dgl_to_nx(adj)[0], clust_edge_anom_mats, nonclust_edge_anom_mats, i, anoms_found, norms, sc_label, attns, losses[i][-1], cluster=False)
                    
                    '''
                    a_clf.title = 'nonclustloss'
                    edge_anom_mats = [np.zeros(edge_anom_mats[0].shape)]
                    node_anom_mats,edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,nonclustloss[i].unsqueeze(0),feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats,recons_a,res_a_all,exp_params)
                    a_clf.calc_clust_prec(dgl_to_nx(adj)[0], edge_anom_mats, truth, sc_label, attns, clusts, cluster=False)
                    '''
            print('done..')
            import ipdb ; ipdb.set_trace()

    if visualizer is not None and args.datasave is False:
        if train_attn_w is not None:
            train_attn_w = train_attn_w.detach().to('cpu').numpy()
            #train_attn_w=train_attn_w[-1].sum(axis=1).numpy()
        #if exp_params['VIS']['VIS_FILTERS'] == True:
        #visualizer.plot_final_filters([np.array([i.weight.detach().cpu().numpy() for i in j.filters]) for j in struct_model.module_list],[F.softmax(i.lam).detach().cpu().numpy() for i in struct_model.module_list])
        #visualizer.plot_final_filters([np.array([i.weight.detach().cpu().numpy() for i in j.filters]) for j in struct_model.module_list])#,[(i.lam).detach().cpu().numpy() for i in struct_model.module_list])
        # plot losses
        visualizer.plot_loss_curve((tot_loss.T).detach().cpu().numpy())
        #visualizer.plot_recons(recons_a,recons_label)
        #visualizer.plot_filters(res_a_all)
    tb.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasave', default=False, type=bool, help='whether to save data')
    parser.add_argument('--dataload', default=False, type=bool, help='whether to load data')
    parser.add_argument('--epoch', default=None, help='Training epoch')
    parser.add_argument('--config', default='cora', type=str, help='path to config file')
    args = parser.parse_args()
    exp_params = prep_args(args)

    graph_anomaly_detection(exp_params)