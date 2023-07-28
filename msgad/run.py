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
from models.gcad import *
from model import *
import torch.nn.functional as F
import MADAN.Madan as md
from visualization import *
from torch.utils.tensorboard import SummaryWriter
from madan_analysis import madan_analysis
from label_analysis import LabelAnalysis
from itertools import combinations,product
import warnings ; warnings.filterwarnings("ignore")

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
    all_nodes = np.array(list(g_nx.nodes()))

    #sc_label_new,clusts=la.run_dend(g_nx,scales+1,return_all=True)
    sc_label_new,clusts=la.run_dend(g_nx,exp_params['DATASET']['SCALES']+1,return_all=True)

    #lbls.append(lbl_adj)
    #del lbl_adj, elist, clusters
    #import ipdb ; ipdb.set_trace()
    if not os.path.exists(f'{dataset}_full_adj.mat'):
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
            neg_edges = torch.stack([torch.tensor(np.random.choice(clusters[i.item()])) for i in pos_clusts])
            neg_edges = torch.vstack((pos_edges[:,0],neg_edges)).T
            assert(torch.where(clust[pos_edges[:,0]]==clust[pos_edges[:,1]].shape[0]==pos_edges.shape[0]))
            assert(torch.where(clust[neg_edges[:,0]]==clust[neg_edges[:,1]].shape[0]==0))

            #    neg_edges = torch.stack([torch.stack((node1, torch.tensor(random.choice(all_nodes[(clust[all_nodes]!=clust[node1]).nonzero()])[0]))) for node1, _ in pos_edges])
            
            lbl_adj = dgl.graph((pos_edges[:,0],pos_edges[:,1])).to(exp_params['DEVICE'])
            lbl_adj.ndata['feature'] = feats.to(lbl_adj.device)
            lbl_adj.edata['w'] = torch.ones(pos_edges.shape[0]).to(lbl_adj.device)
            
            lbls.append(transform(lbl_adj))
            pos_edges_full.append(pos_edges)
            neg_edges_full.append(neg_edges)
            
        save_mat = {'lbls':[i for i in lbls],'pos_edges':[i.to_sparse() for i in pos_edges_full],'neg_edges':[i.to_sparse() for i in neg_edges_full]}
        with open(f'{dataset}_full_adj.mat','wb') as fout:
            pkl.dump(save_mat,fout)
    else:
        with open(f'{dataset}_full_adj.mat','rb') as fin:
            mat =pkl.load(fin)
        lbls,pos_edges_full,neg_edges_full = mat['lbls'],[i.to_dense() for i in mat['pos_edges']],[i.to_dense() for i in mat['neg_edges']]

    if 'tfinance' in exp_params['DATASET']['NAME'] and exp_params['VIS']['VIS_FILTERS'] == True:
        dataloader = None
    else:
        dataloader = [dataloading.fetch_dataloader(lbls[i],pos_edges_full[i],neg_edges_full[i]) for i in range(len(clusts))]

    struct_model,feat_model=None,None
    struct_model,params,model_loaded = init_model(feats.size(1),exp_params,args)
    regularize = True if exp_params['MODEL']['CONTRASTIVE'] == True else False

    if 'weibo' in exp_params['DATASET']['NAME']:
        tb = SummaryWriter(log_dir=f'runs/{exp_name}')
        
    if not exp_params['MODEL']['NAME'] in ['gcad','madan']:
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
    
    for epoch in range(int(exp_params['MODEL']['EPOCH'])):
        epoch_l,iter = 0,0
        if exp_params['MODEL']['NAME'] == 'gcad': break
        #edge_anom_mats,node_anom_mats,recons_a,res_a_all = init_recons_agg(adj.number_of_nodes(),adj.ndata['feature'].shape[1],exp_params)
        if model_loaded: break
        edge_ids=[]

        # unpack each of the dataloaders
        for batch,data_inds in enumerate(zip(*dataloader)):
            
            
            if load_data:
                loaded_input,recons_label=dataloading.load_batch(batch,'train')
            else:
                loaded_input = data_inds
            g_batch,pos_edges,neg_edges = zip(*loaded_input)

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

            
            if 'cora' not in exp_params['DATASET']['NAME']: print('size of batch',g_batch[0].num_dst_nodes(),'nodes')
            #vis = True if (epoch == 0 and iter == 0 and exp_params['VIS']['VIS_FILTERS'] == True) else False

            #edges, feats = process_graph(lbls)
            
            if exp_params['DATASET']['DATASAVE']:
                dataloading.save_batch(loaded_input,recons_label,iter,'train')
                continue
            if struct_model:
                optimizer.zero_grad()
            for i in loaded_input: del i
            torch.cuda.empty_cache() ; gc.collect()
            #if exp_params['VIS']['VIS_CONCENTRATION'] == True and 'madan' in exp_params['MODEL']['NAME']:
            #    madan_analysis(adj,lbl,sc_label,anoms,exp_params)
            check_gpu_usage('running model')
            if exp_params['MODEL']['IND'] is None:
                A_hat,X_hat,res_a,entropies = struct_model(lbls,edge_ids,feats,edge_ids,vis=False,vis_name='epoch1',clusts=clusts)
            else:
                A_hat,X_hat,res_a,entropies = struct_model(lbls,[pos_edges[exp_params['MODEL']['IND']]],feats,[edge_ids[exp_params['MODEL']['IND']]],vis=False,vis_name='epoch1',clusts=clusts)
            torch.cuda.empty_cache() ; gc.collect()
            if A_hat is None:
                print('result is none?')
                import ipdb ; ipdb.set_trace()
            check_gpu_usage('recons label collected, starting loss')
            #batch_clusts = torch.gather(clusts.to(in_nodes.device),1,in_nodes)
            loss, struct_loss, feat_cost,regloss,clustloss,nonclustloss,sc_idx_inside,sc_idx_outside = LossFunc.calc_loss(recons_label,A_hat, None, res_a, edge_ids, attract_edges_sel, repel_edges_sel,clusts)
            check_gpu_usage('loss collected')
            if exp_params['MODEL']['NAME'] == 'gradate': loss = A_hat[0]
            
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
            '''
            if exp_params['VIS']['VIS_LOSS']:
                visualizer.sc_label = batch_sc_label
                #edge_ids = torch.cat((pos_edges,neg_edges),axis=2).to(pos_edges.device)[:scales]
                edge_ids_ = edge_ids
                node_ids_ = g_batch.nodes()

                node_anom_mats,edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,struct_loss,feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats,recons_a,res_a_all,exp_params)
            '''
            del g_batch, pos_edges, neg_edges, l, res_a,X_hat#,A_hat
            torch.cuda.empty_cache() ; gc.collect()
            
        '''
        print("Seconds since epoch =", (time.time()-seconds)/60)
        seconds = time.time()
        if exp_params['MODEL']['NAME'] != 'madan' and 'multi-scale' in exp_params['MODEL']['NAME']:
            print("Epoch:", '%04d' % (epoch), "train_loss=", round(torch.sum(loss),3), "losses=",torch.round(loss,decimals=4).detach().cpu())
        else:
             print("Epoch:", '%04d' % (epoch), "train_loss=", round(epoch_l.item(),3))
        #print('avg loss',torch.mean(epoch_l/dataloader.__len__()))
        '''
        if exp_params['DATASET']['DATASAVE']: continue 
                
        print('epoch done',epoch,loss)
        #if 'weibo' in exp_params['DATASET']['NAME']:
        #import ipdb; ipdb.set_trace()
        if True:
            if epoch == 0 and iter == 1:
                tb_writers = TBWriter(tb, edge_ids, attract_edges_sel, repel_edges_sel, sc_label_new,clusts,anoms,exp_params)
            labels = []
            mean_intras_tot,mean_inters_tot=[],[]
            for sc,l in enumerate(loss):
                tb.add_scalar(f'Loss_{sc}', l, epoch)
                tb.add_scalar(f'RegLoss_{sc}', regloss[sc].sum(), epoch)
                tb.add_scalar(f'ClustLoss_{sc}', clustloss[sc].sum(), epoch)
                tb.add_scalar(f'NonClustLoss_{sc}', nonclustloss[sc].sum(), epoch)
                mean_intras,mean_inters=tb_writers.tb_write_anom(tb,lbls,sc_label_new,edge_ids[exp_params['MODEL']['IND']],A_hat[sc], struct_loss[sc], sc,epoch, regloss[sc],clustloss[sc],nonclustloss[sc],clusts,sc_idx_inside,sc_idx_outside,entropies)
                #labels.append(label)
                if sc == 0:
                    mean_intras_tot = mean_intras.unsqueeze(0)
                    mean_inters_tot = mean_inters.unsqueeze(0)
                else:
                    try:
                        mean_intras_tot = torch.cat((mean_intras_tot,mean_intras.unsqueeze(0)),axis=0)
                        mean_inters_tot = torch.cat((mean_inters_tot,mean_inters.unsqueeze(0)),axis=0)
                    except Exception as e:
                        print(e)
                        import ipdb ; ipdb.set_trace()
                '''
                import ipdb ; ipdb.set_trace()
                #e,U = get_spectrum(torch.tensor(label).to(torch.float64).to(adj.device).to_sparse(),n_eig=64)
                #e = e.to(adj.device) ; U = U.to(adj.device)
                py_g = pygsp_.graph.MultiScale(label)
                py_g.compute_fourier_basis(128)
                visualizer.plot_spectral_gap(py_g.e.cpu(),f'test{sc}')
                plt.figure()
                x_labelvis,y_labelvis=visualizer.plot_spectrum(e,U,feats.to(U.dtype))
                plt.savefig(f'test{sc}.png')
                '''
            # intra -> 1-s
            # inter -> s-0
            # plot adapted persistence diagrams
            sc_labels = np.unique(sc_label_new)
            '''
            for ind,i in enumerate(sc_labels):
                plt.figure()
                import ipdb ; ipdb.set_trace()
                plt.scatter(mean_intras_tot[:,i],mean_inters_tot[:,i])
                plt.plot(0,np.maximum(mean_intras_tot[:,i].max(),mean_inters_tot[:,i].max())+.5)
                plt.savefig(f'persistence_{exp_name}_sc{i}.png')

            '''
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
            if struct_model.final_attn is not None:
                tb.add_histogram(f'Filt_att',F.softmax(struct_model.final_attn,0),epoch)

        del loss,struct_loss,A_hat

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

    LossFunc = loss_func(adj,adj.ndata['feature'],exp_params,sample=True, recons='struct', alpha=None, clusts=None, regularize=True)

    # accumulate node-wise anomaly scores via model evaluation
    if exp_params['MODEL']['NAME'] not in ['madan','gcad'] and struct_model: struct_model.eval()

    edges=adj.edges('eid')
    #dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,device=exp_params['DEVICE'])
    if exp_params['MODEL']['NAME'] != 'gcad' and exp_params['DATASET']['DATASAVE'] == False:
        struct_scores, feat_scores = torch.zeros(scales,adj.number_of_nodes()).to(exp_params['DEVICE']),torch.zeros(exp_params['MODEL']['D'],adj.number_of_nodes()).to(exp_params['DEVICE'])
    iter = 0

    edge_anom_mats,node_anom_mats,recons_a,res_a_all = init_recons_agg(adj.number_of_nodes(),adj.ndata['feature'].shape[1],exp_params)

    all_samps = []
    if (exp_params['DATASET']['DATADIR'] is not None and exp_params['DATASET']['DATALOAD']):
        pass
        #random.shuffle(dataloader)
    else:
        dataloader = [dataloading.fetch_dataloader(x,y,z) for x,y,z in zip(lbls,pos_edges_full,neg_edges_full)]
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
        if struct_model and exp_params['MODEL']['NAME'] != 'gcad':
            # TODO: CHANGED FIRST ARG TO EDGE IDS
            A_hat,X_hat,res_a,entropies = struct_model(lbls,edge_ids,feats,edge_ids,vis=vis,vis_name='test',clusts=clusts)
            #import ipdb ; ipdb.set_trace()

        if exp_params['MODEL']['NAME'] == 'gcad':
            adj_ = dgl_to_mat(g_batch)
            adj_=adj_.sparse_resize_((g_batch.num_src_nodes(),g_batch.num_src_nodes()), adj_.sparse_dim(),adj_.dense_dim())
            feat = g_batch.ndata['feature']
            if type(feat) == dict:
                feat = feat['_N']
            struct_loss = struct_model.fit_transform(adj_,feat)
            struct_loss = np.array((struct_loss.T)[g_batch.dstnodes().detach().cpu().numpy()].T)

        # collect anomaly scores
        #edge_ids = torch.cat((pos_edges,neg_edges),axis=2).to(pos_edges.device)[:scales]
        
        edge_ids = torch.cat((pos_edges[0],neg_edges[0]),axis=1).unsqueeze(0).tile((scales,1,1))
        edge_ids_ = edge_ids
        node_ids_ = g_batch.nodes()

        loss, struct_loss, feat_cost,regloss,clustloss,nonclustloss,sc_idx_inside,sc_idx_outside = LossFunc.calc_loss(recons_label,A_hat, None, None, edge_ids, attract_edges_sel, repel_edges_sel,clusts)
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
        elif exp_params['MODEL']['NAME'] in ['gcad']:
            l = np.mean(struct_loss)
            
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
        a_clf.calc_prec(None, [np.mean(edge_anom_mats[0],axis=1)], truth, sc_label_new, train_attn_w, cluster=False, input_scores=True)
        print('feat scores')
        a_clf.calc_prec(None, [np.mean(node_anom_mats[0],axis=1)], truth, sc_label_new, train_attn_w, cluster=False, input_scores=True)
        print('combined scores')
        a_clf.calc_prec(None, loss.detach().cpu().numpy(), truth, sc_label_new, train_attn_w, cluster=False, input_scores=True)
    else:
        if exp_params['DATASET']['BATCH_TYPE'] == 'node':
            if -1 in node_anom_mats[0]:
                raise('node not sampled')
            a_clf.calc_prec(None,node_anom_mats, truth, sc_label_new, cluster=False, input_scores=True)
        else:
            if 'multi-scale' in exp_params['MODEL']['NAME']:
                #attns = F.softmax(train_attn_w[-1],1).numpy()
                attns = train_attn_w[-1].numpy()
            else:
                attns = None
            
            if not exp_params['MODEL']['SAMPLE_TEST']:
                a_clf.calc_prec(dgl_to_mat(adj), struct_loss.detach().cpu().numpy(), truth, sc_label_new, attns, cluster=False, input_scores=True)
            else:
                a_clf.title = 'loss'
                a_clf.calc_prec(dgl_to_nx(adj)[0], edge_anom_mats, truth, sc_label_new, attns, clusts, cluster=False)
                

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