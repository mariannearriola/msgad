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
from torch.utils.tensorboard import SummaryWriter
from label_analysis import LabelAnalysis
from itertools import combinations
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
            pos_edges= torch.tensor([(x, y) for nodes in clusters.values() for x, y in combinations(nodes, 2)])
            # for each positive edge, replace connection with a negative edge. during dataloading, index both simultaneously
            #neg_edges = pos_edges
            
            pos_clusts = clust[pos_edges[:,1]]
            clust_offset=np.random.randint(1,(clust.max()),pos_clusts.shape[0])
            pos_clusts += clust_offset ; pos_clusts = pos_clusts % (clust.max()+1)
            assert(torch.where(clust[pos_edges[:,0]]==clust[pos_edges[:,1]])[0].shape[0]==pos_edges.shape[0])

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

    dataloader = [dataloading.fetch_dataloader(lbls[i],pos_edges_full[i],i) for i in range(len(clusts))]

    struct_model=None,None
    struct_model,params,model_loaded = init_model(feats.size(1),exp_params,args)

    tb = SummaryWriter(log_dir=f'runs/{exp_name}')
        
    optimizer = torch.optim.Adam(struct_model.parameters(), lr = float(exp_params['MODEL']['LR']))

    # begin model training
    if dataloader is not None:
        print(dataloader.__len__(),'batches')
    else:
        dataloader = [None]

    LossFunc = loss_func(adj,adj.ndata['feature'],exp_params,sample=True, recons='struct', alpha=None, clusts=None, regularize=True)
    seconds = time.time()
    for epoch in range(int(exp_params['MODEL']['EPOCH'])):
        epoch_l,iter = 0,0
        if model_loaded: break
        edge_ids=[]

        # unpack each of the dataloaders
        for batch,data_inds in enumerate(zip(*dataloader)):
            if load_data:
                try:
                    loaded_input,recons_label=dataloading.load_batch(batch,'train')
                except:
                    raise 'error loading batch'
            else:
                loaded_input = data_inds
            g_batch,pos_edges,neg_edges = zip(*loaded_input)
            if load_data: pos_edges = [i.to(device) for i in pos_edges] ; neg_edges = [i.to(device) for i in neg_edges] ; lbls = [i.to(device) for i in lbls]

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

            for ind,score in enumerate(anom_scores_all):
                _,prec1,ra1=tb_writers.a_clf.calc_prec(anom_scores_all[ind][ind].detach().cpu()[np.newaxis,...],truth,sc_label_new,verbose=False,log=False)

                for anom,prec in enumerate(prec1[0]):
                    tb.add_scalar(f'Precsc{ind+1}/anom{anom}', prec, epoch)
                    tb.add_scalar(f'ROC{ind+1}/anom{anom}', ra1[0][anom], epoch)
        #del loss,struct_loss,A_hat

        if epoch == 0:
            tot_loss = epoch_l.sum(0).unsqueeze(0)
        else:
            tot_loss = torch.cat((tot_loss,epoch_l.sum(0).unsqueeze(0)),dim=0)
        epoch_l = torch.sum(epoch_l)
        for name, param in struct_model.named_parameters():
            tb.add_histogram(name, param.flatten(), epoch)

        # save/load trained model model
        if epoch > 1 and epoch == int(exp_params['MODEL']['EPOCH'])-1:
            torch.save(struct_model,f'{exp_name}.pt')

    print('done..')
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