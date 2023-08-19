
import torch
import numpy as np
import argparse
import networkx as nx
from utils import *
from anom_detector import *
from dataloading import *
from loss_utils import *
import time
import gc
from model import *
from torch.utils.tensorboard import SummaryWriter
from label_analysis import LabelAnalysis
import warnings ; warnings.filterwarnings("ignore")

def graph_anomaly_detection(exp_params):
    """Pipeline for autoencoder-based graph anomaly detection"""
    # load data
    exp_name = exp_params['EXP'] ; scales = exp_params['MODEL']['SCALES'] ; dataset = exp_params['DATASET']['NAME'] ; device = exp_params['DEVICE']
    load_data = (exp_params['DATASET']['DATADIR'] is not None and exp_params['DATASET']['DATALOAD'])
    seed_everything(82)
    dataloading = DataLoading(exp_params)

    sp_adj, edge_idx, feats, truth = dataloading.load_anomaly_detection_dataset()
    anoms,norms=np.where(truth==1)[0],np.where(truth==0)[0]
    if sp_adj is not None:
        adj = sparse_matrix_to_dgl(sp_adj,feats)
    
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

    sc_label_new,clusts=la.run_dend(g_nx,scales,return_all=True)

    lbls,neg_lbls,pos_edges_full = get_labels(adj,feats,clusts,exp_params)
    dataloader = [dataloading.fetch_dataloader(lbls[i],neg_lbls[i],pos_edges_full[i],i) for i in range(len(clusts))]
    #dataloader = [dataloading.fetch_dataloader(lbls[i],pos_edges_full[i]) for i in range(len(clusts))]

    struct_model=None,None
    struct_model,params,model_loaded = init_model(feats.size(1),exp_params,args)

    tb = SummaryWriter(log_dir=f'runs/{exp_name}')
        
    optimizer = torch.optim.Adam(struct_model.parameters(), lr = float(exp_params['MODEL']['LR']))

    # begin model training
    print(dataloader.__len__(),'batches')
    
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
                    loaded_input=dataloading.load_batch(batch,'train')
                except:
                    raise 'error loading batch'
            else:
                loaded_input = data_inds
            g_batch,pos_edges,neg_edges,batch_nodes = zip(*loaded_input)
            if load_data: pos_edges = [i.to(device) for i in pos_edges] ; neg_edges = [i.to(device) for i in neg_edges] ; lbls = [i.to(device) for i in lbls]

            edge_ids = [torch.cat((pos_edges[i],neg_edges[i]),axis=0) for i in range(len(pos_edges))]
            
            check_batch(pos_edges,neg_edges,clusts)
            
            print('size of batch',g_batch[0].num_dst_nodes(),'nodes')
            if exp_params['DATASET']['DATASAVE']:
                dataloading.save_batch(loaded_input,iter,'train') ; continue
            if struct_model:
                optimizer.zero_grad()
            for i in loaded_input: del i
            torch.cuda.empty_cache() ; gc.collect()

            check_gpu_usage('running model')
            
            pos_edges_og = [edge_ids[ind][adj.has_edges_between(edge_ids[ind][:,0],edge_ids[ind][:,1]).nonzero().T[0]] for ind in range(len(edge_ids))]
            A_hat,res_a = struct_model(adj,pos_edges_og,feats,edge_ids)
            
            torch.cuda.empty_cache() ; gc.collect()
            check_gpu_usage('recons label collected, starting loss')
            loss,struct_loss,regloss,clustloss,nonclustloss = LossFunc.calc_loss(adj, A_hat, edge_ids, clusts, batch_nodes)
            
            
            pos_counts = np.stack([get_counts(clusts[ind],edge_ids[ind])[-2] for ind in range(len(clusts))])
            batch_scores_sc = torch.stack([gather_clust_info(torch.nan_to_num(clustloss[i].detach().cpu()/pos_counts[i]),clusts[i],'std') for i in range(len(clustloss))])
            batch_scores = batch_scores_sc if batch == 0 else batch_scores + batch_scores_sc

            l = torch.sum(loss) if 'multi-scale' in exp_params['MODEL']['NAME'] else torch.mean(loss)
            epoch_l = loss.unsqueeze(0) if iter == 0 else torch.cat((epoch_l,l.unsqueeze(0)))
            if exp_params['MODEL']['DEBUG'] and iter % 100 == 0:
                print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=', round(l.item(),3))
                
            iter += 1
            print('iter',iter)
            
            check_gpu_usage('about to backward')
            l.backward()
            optimizer.step()
            #del pos_edges, neg_edges, l, res_a,X_hat#,A_hat
            torch.cuda.empty_cache() ; gc.collect()
            
        print("Seconds since epoch =", (time.time()-seconds)/60)
        seconds = time.time()
        print("Epoch:", '%04d' % (epoch), "train_loss=", torch.round(torch.sum(loss),decimals=3).detach().cpu().item(), "losses=",torch.round(loss,decimals=4).detach().cpu())
        
        #import ipdb ; ipdb.set_trace()
        #batch_scores /= len(dataloader)
        if exp_params['DATASET']['DATASAVE']: continue 
                
        print('epoch done',epoch,loss.detach().cpu())

        if epoch == 0 and iter == 1: tb_writers = TBWriter(tb,sc_label_new,clusts,anoms,norms,exp_params)

        anom_scores_all = score_multiscale_anoms(clustloss,nonclustloss, clusts, res_a)
        for sc,l in enumerate(loss):       
            tb_writers.tb_write_anom(sc_label_new,edge_ids,A_hat[sc], anom_scores_all, struct_loss,sc,epoch,clustloss,nonclustloss,clusts,anom_wise=False,scores_only=False)
        log = True if epoch == int(exp_params['MODEL']['EPOCH'])-1 else False
        if exp_params['DATASET']['BATCH_SIZE'] > 0:
            _,prec1,ra1=tb_writers.a_clf.calc_prec(batch_scores.detach().cpu(),truth,sc_label_new,verbose=False,log=False)
        else:
            _,prec1,ra1=tb_writers.a_clf.calc_prec(anom_scores_all.detach().cpu(),truth,sc_label_new,verbose=False,log=log)
        for sc in range(len(prec1)):
            for anom,prec in enumerate(prec1[sc]):
                tb.add_scalar(f'Precsc{sc+1}/anom{anom+1}', prec, epoch)
                tb.add_scalar(f'ROC{sc+1}/anom{anom+1}', ra1[sc][anom], epoch)
        del loss,struct_loss,A_hat,res_a,clustloss,nonclustloss,edge_ids ; torch.cuda.empty_cache()

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