from scipy.sparse import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import numpy as np
import scipy.sparse as sp
import scipy.io
from datetime import datetime
import argparse
import scipy
import networkx as nx
import torch_geometric
from torch.utils.data import DataLoader
from scipy import stats
from utils import *
from loss_utils import *
import pickle as pkl
import random 
import time
from models.gcad import *
import MADAN.Madan as md
import warnings
warnings.filterwarnings("ignore")

def graph_anomaly_detection(args):
    """Pipeline for autoencoder-based graph anomaly detection"""
    # load data
    sp_adj, edge_idx, feats, truth, sc_label = load_anomaly_detection_dataset(args.dataset, args.scales)
    anoms,norms=np.where(truth==1)[0],np.where(truth==0)[0]
    if sp_adj is not None:
        adj = sparse_matrix_to_tensor(sp_adj,feats)
    lbl=None

    # initialize data loading
    if edge_idx is not None:
        adj = dgl.graph((edge_idx[0],edge_idx[1]),num_nodes=feats.shape[0])
        adj.ndata['feature'] = feats
    edges=adj.edges('eid')

    if args.dataload:
        dataloader = np.arange(len(os.listdir(f'{args.datadir}/{args.dataset}/train')))
    else:
        dataloader = fetch_dataloader(adj, edges, args)
    print('sample train',args.sample_train,'sample test',args.sample_test)

    # intialize model (on given device)
    adj = adj.to(args.device)
    struct_model,feat_model=None,None
    struct_model,params = init_model(feats.size(1),args)

    if not args.model in ['gcad','madan']:
        optimizer = torch.optim.Adam(params, lr = args.lr)
    
    # begin model training
    best_loss = torch.tensor(float('inf')).to(args.device)    
    A_hat, X_hat = None,None
    seconds = time.time()
    for epoch in range(args.epoch):
        epoch_l = 0
        if epoch > 0:
            random.shuffle(dataloader)
        if args.model == 'gcad': break
        iter=0
        for data_ind in dataloader:
            if (args.datadir is not None and args.dataload) or epoch > 0:
                loaded_input,lbl=load_batch(data_ind,'train',args)
            else:
                loaded_input = data_ind
            if args.batch_type == 'node':
                all_nodes,in_nodes,g_batch= loaded_input
            elif args.batch_type == 'edge_rw':
                in_nodes, sub_graph_pos, sub_graph_neg, block = loaded_input
                g_batch = block
                pos_edges = sub_graph_pos.edges()
                neg_edges = sub_graph_neg.edges()
            else:
                in_nodes, pos_edges, neg_edges, g_batch, last_batch_node = get_edge_batch(loaded_input)
                #g_batch.add_edges(pos_edges[:,0],pos_edges[:,1])
                if args.datasave:
                    g_batch = g_batch[0]
            
            if args.model == 'madan':
                if args.debug:
                    if iter % 100 == 0:
                        print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%')
                if iter == 0:
                    adj = adj.adjacency_matrix()
                    adj = adj.sparse_resize_((adj.size(0), adj.size(0)), adj.sparse_dim(), adj.dense_dim())
                idx=adj.coalesce().indices()
                nx_graph=nx.from_edgelist([(i[0].item(),i[1].item()) for i in idx.T])
                node_dict = None
                
                nodes = list(max(nx.connected_components(nx_graph), key=len))

                nx_graph = nx.subgraph(nx_graph, nodes)
                nx_graph = nx.from_numpy_matrix(nx.adjacency_matrix(nx_graph))
                feats_ = feats[nodes]

                #node_dict = {k.item():v for k,v in zip(list(in_nodes.detach().cpu()),np.arange(len(list(nx_graph.nodes))))}
                node_dict = {k:v for k,v in zip(nodes,np.arange(len(list(nx_graph.nodes))))}
                rev_node_dict = {v: k for k, v in node_dict.items()}
                try:
                    madan = md.Madan(nx_graph, attributes=feats_, sigma=0.08)
                except Exception as e:
                    print('eigenvalues dont converge',e)
                    continue
                
                time_scales   =   np.concatenate([np.array([0]), 10**np.linspace(0,5,5)])
                batch_anoms=[node_dict[j] for j in np.intersect1d(anoms,np.array(list(node_dict.keys()))).tolist()]
                #madan.anomalous_nodes=[node_dict[j] for j in np.intersect1d(anoms,np.array(list(node_dict.keys()))).tolist()]
                #madan.anomalous_nodes=[node_dict[j] for j in anoms]
                #madan.scanning_relevant_context_time(time_scales)
                #import ipdb ; ipdb.set_trace()
                madan.compute_concentration(10000000)
                madan.compute_context_for_anomalies()
                madan.plot_concentration()
                print('anoms detected',np.intersect1d(np.where(madan.concentration==1)[0],batch_anoms).shape[0]/np.where(madan.concentration==1)[0].shape[0])
                #import ipdb ; ipdb.set_trace()
                madan.interp_com
                anoms_detected=madan.anomalous_nodes
                if node_dict and len(anoms_detected)>0:
                    anoms_detected = rev_node_dict[anoms_detected]
                if len(anoms_detected)>0:
                    print('anom found')
                    import ipdb ; ipdb.set_trace()
                iter += 1
                continue
            
            optimizer.zero_grad()
    
            if struct_model:
                A_hat,model_lbl = struct_model(g_batch,last_batch_node,pos_edges,neg_edges)
                if args.datasave:
                    lbl = model_lbl
                    
            #print('alcing loss')
            if args.batch_type == 'node':
                if args.model == 'gradate':
                    loss = A_hat[0]
                else:
                    loss, struct_loss, feat_cost = loss_func(g_batch, A_hat, X_hat, None, None, sample=False, recons=args.recons)
            else:
                recons_label = g_batch if lbl is None else lbl
                loss, struct_loss, feat_cost = loss_func(recons_label, A_hat, X_hat, pos_edges, neg_edges, sample=args.sample_test, recons=args.recons)

            if 'multi-scale' in args.model:
                l = torch.sum(torch.mean(loss,dim=1))
            else:
                l = torch.mean(loss)
            '''
            if l < best_loss:
                best_loss = dl
                torch.save(model,'best_model.pt')
            '''
            if iter == 0:
                epoch_l = l.unsqueeze(0)
            else:
                epoch_l = torch.cat((epoch_l,l.unsqueeze(0)))

            # save batch info
            if args.datasave:
                save_batch(loaded_input,lbl,iter,'train',args)
            if args.debug:
                if iter % 100 == 0:
                    print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=', round(l.item(),3))
                
            iter += 1
        epoch_l = torch.sum(epoch_l)
        epoch_l.backward()
        optimizer.step()
        print("Seconds since epoch =", (time.time()-seconds)/60)
        seconds = time.time()

        if args.model != 'madan':
            print("Epoch:", '%04d' % (epoch), "train_loss=", round(epoch_l.item(),3), "losses=",torch.round(torch.mean(loss,dim=1),decimals=4).detach().cpu())
            print('avg loss',torch.mean(epoch_l/dataloader.__len__()))

    print('best loss:', best_loss)
    
    #model = torch.load('best_model.pt')

    # accumulate node-wise anomaly scores via model evaluation
    if args.model not in ['madan','gcad']:
        if struct_model: struct_model.eval()
    
    edges=adj.edges('eid')
    #dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,device=args.device)
    if args.model != 'gcad':
        struct_scores, feat_scores = torch.zeros(len(A_hat),adj.number_of_nodes()).to(args.device),torch.zeros(args.d,adj.number_of_nodes()).to(args.device)
    iter = 0

    edge_anom_mats,node_anom_mats = [],[]
    scales = 3 if 'multi-scale' in args.model else 1
    for i in range(scales):
        am = np.zeros((adj.number_of_nodes(),adj.number_of_nodes()))
        edge_anom_mats.append(am)
        node_anom_mats.append(np.full((adj.number_of_nodes(),),-1.))
    all_samps = []
    if args.datadir is not None and args.dataload:
        random.shuffle(dataloader)
    else:
        dataloader = fetch_dataloader(adj, edges, args)
    #import ipdb ; ipdb.set_trace()
    for loaded_input in dataloader:
        if args.dataload:
            loaded_input,lbl=load_batch(loaded_input,'test',args)
        # collect input
        if args.batch_type == 'node':
            all_nodes,in_nodes,g_batch= loaded_input
            pos_edges,neg_edges=None,None
        elif args.batch_type == 'edge_rw':
            in_nodes, sub_graph_pos, sub_graph_neg, block = loaded_input
            g_batch = block
            pos_edges = sub_graph_pos.edges()
            neg_edges = sub_graph_neg.edges()
        else:
            in_nodes, pos_edges, neg_edges, g_batch, last_batch_node = get_edge_batch(loaded_input)
            if args.datasave:
                g_batch = g_batch[0]
            #g_batch.add_edges(pos_edges[:,0],pos_edges[:,1])
    
        # run evaluation
        if struct_model and args.model != 'gcad':
            A_hat,model_lbl = struct_model(g_batch,last_batch_node,pos_edges,neg_edges)
            if args.datasave:
                lbl = model_lbl
        if args.model == 'gcad':
            adj_ = g_batch.adjacency_matrix() 
            adj_=adj_.sparse_resize_((g_batch.num_src_nodes(),g_batch.num_src_nodes()), adj_.sparse_dim(),adj_.dense_dim())
            feat = g_batch.ndata['feature']
            if type(feat) == dict:
                feat = feat['_N']
            struct_loss = struct_model.fit_transform(adj_,feat)
            struct_loss = np.array((struct_loss.T)[g_batch.dstnodes().detach().cpu().numpy()].T)

        # collect anomaly scores
        edge_ids_,node_ids_ = collect_batch_scores(in_nodes,g_batch,pos_edges,neg_edges,args)
        
        # save batch info
        if args.datasave:
            save_batch(loaded_input,lbl,iter,'test',args)


        if 'edge' in args.batch_type and args.model not in ['gcad']:
            recons_label = g_batch if lbl is None else lbl
            loss, struct_loss, feat_cost = loss_func(recons_label, A_hat, X_hat, pos_edges, neg_edges, sample=args.sample_test, recons=args.recons)

        if args.model in ['gradate']:
            loss = A_hat[0]
            struct_loss = [A_hat[1]]
            if args.sample_test:
                l = torch.sum(loss)
            else:
                l = torch.mean(loss)
        elif args.model in ['gcad']:
            l = np.mean(struct_loss)

        
        for sc in range(scales):
            if args.sample_test:
                if args.batch_type == 'node' or args.model in ['gcad']:
                    node_anom_mats[sc][node_ids_.detach().cpu().numpy()] = struct_loss[sc]
                else:
                    edge_anom_mats[sc][tuple(edge_ids_)] = struct_loss[sc].detach().cpu().numpy()
                    edge_anom_mats[sc][tuple(np.flip(edge_ids_,axis=0))] = edge_anom_mats[sc][tuple(edge_ids_)]

            else:
                if args.batch_type == 'node':
                    if args.model in ['gcad','gradate']:
                        node_anom_mats[sc][node_ids_.detach().cpu().numpy()[:struct_loss[sc].shape[0]]] = struct_loss[sc]

                else:
                    edge_anom_mats[sc][tuple(edge_ids_)] = struct_loss[sc][edge_ids_[0],edge_ids_[1]].detach().cpu().numpy()
                    edge_anom_mats[sc][tuple(np.flip(edge_ids_,axis=0))] = edge_anom_mats[sc][tuple(edge_ids_)]
        
        if iter % 100 == 0 and args.debug:
            print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=',round(l.item(),3))
        
        iter += 1
            
    # anomaly detection with final scores
    print('structure scores')
    if args.batch_type == 'node':
        if -1 in node_anom_mats[0]:
            raise('node not sampled')
        detect_anomalies(None,node_anom_mats, truth, sc_label, args.dataset, sample=args.sample_test, cluster=False, input_scores=True)
    else:
        detect_anomalies(adj.adjacency_matrix().to_dense(), edge_anom_mats, truth, sc_label, args.dataset, sample=args.sample_test, cluster=False)

    if X_hat: # TODO
        print('feat scores')
        detect_anomalies(feat_scores, truth, sc_label, args.dataset)
 
    for i in struct_model.module_list:
        print(torch.sigmoid(i.lam))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora_triple_sc_all', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--datadir', default=None, type=str, help='directory to load data from')
    parser.add_argument('--datasave', default=False, type=bool, help='whether to save data')
    parser.add_argument('--dataload', default=False, type=bool, help='whether to load data')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=15, help='Training epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--scales', default='4', type=int, help='number of scales for multi-scale analysis')
    parser.add_argument('--batch_size', type=int, default=32, help='number of edges to use for batching (default: 32)')
    parser.add_argument('--recons', default='struct', type=str, help='reconstruct features or structure')
    parser.add_argument('--cutoff',default=50, type=float, help='validation cutoff')
    parser.add_argument('--d', default=5, type=int, help='d parameter for BWGNN filters')
    parser.add_argument('--model', default='multi-scale-amnet', type=str, help='encoder to use')
    parser.add_argument('--sample_train', default=True, type=bool, help="whether or not to sample edges in training")
    parser.add_argument('--sample_test', default=True, type=bool, help="whether or not to sample edges in testing")
    parser.add_argument('--batch_type', default='edge', type=str, help="node or edge sampling for batching")
    parser.add_argument('--debug', default=False, type=bool, help="if true, prints intermediate output")
    parser.add_argument('--label_type', default='single', type=str, help="if true, prints intermediate output")

    args = parser.parse_args()

    graph_anomaly_detection(args)
