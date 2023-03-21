from scipy.sparse import data
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from model import GraphReconstruction
from utils import *
from loss_utils import *
import random 
import time
from models.gcad import *
import MADAN.Madan as md
import warnings
warnings.filterwarnings("ignore")

def graph_anomaly_detection(args):
    # load data
    sp_adj, feats, truth, sc_label = load_anomaly_detection_dataset(args.dataset, args.scales)
    anoms,norms=np.where(truth==1)[0],np.where(truth==0)[0]
    adj = sparse_matrix_to_tensor(sp_adj,feats)
    lbl=None

    # initialize data loading
    if args.batch_type == 'edge':
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(4)
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler,negative_sampler=neg_sampler)
        edges=adj.edges('eid')
        batch_size = args.batch_size if args.batch_size > 0 else adj.number_of_edges()
        if args.device == 'cuda':
            dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, device=args.device)
        else:
            dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=6, device=args.device)

    elif args.batch_type == 'node':
        batch_size = args.batch_size if args.batch_size > 0 else adj.number_of_nodes()
        sampler = dgl.dataloading.SAINTSampler(mode='walk',budget=[int(args.batch_size/3),args.batch_size])
        dataloader = dgl.dataloading.DataLoader(adj, adj.nodes(), sampler, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0, device=args.device)
    print('sample train',args.sample_train,'sample test',args.sample_test)


    # intialize model (on GPU)
    struct_model,feat_model=None,None
    if args.recons == 'struct' or args.recons == 'both':
        if args.model == 'gcad':
            gcad_model = GCAD(2,100,1)
        elif args.model == 'madan':
            pass
        else:
            struct_model = GraphReconstruction(in_size = feats.size(1), hidden_size=args.hidden_dim, batch_size=batch_size, scales = args.scales, recons = 'struct', d = args.d, model_str = args.model, batch_type = args.batch_type)
    if args.recons == 'feat' or args.recons == 'both':
        feat_model = GraphReconstruction(in_size = feats.size(1), hidden_size=args.hidden_dim, batch_size=batch_size, scales = args.scales, recons = 'feat', d = args.d, model_str = args.model)

    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        if struct_model:
            struct_model = struct_model.to(args.device) ; struct_model.train()
        if feat_model:
            feat_model = feat_model.to(args.device) ; feat_model.train()
    
    if args.model == 'gcad':
        gcad = GCAD(2,100,4)
    elif args.model == 'madan':
        pass
    elif args.recons == 'struct':
        params = struct_model.parameters()
    elif args.recons == 'feat':
        params = feat_model.parameters()
    elif args.recons == 'both':
        params = list(struct_model.parameters()) + list(feat_model.parameters())

    if not args.model in ['gcad','madan']:
        optimizer = torch.optim.Adam(params, lr = args.lr)
    
    # begin model training
    best_loss = torch.tensor(float('inf')).to(args.device)    
    A_hat, X_hat = None,None
    
    for epoch in range(args.epoch):
        #for iter, bg in enumerate(data_loader):
        if args.model == 'gcad': break
        iter=0
        
        with dataloader.enable_cpu_affinity():
            for loaded_input in dataloader:
                if args.batch_type == 'node':
                    sub_graph = loaded_input
                    in_nodes = sub_graph.nodes()
                    g_batch = sub_graph
                else:
                    in_nodes, sub_graph_pos, sub_graph_neg, block = loaded_input
                    pos_edges = sub_graph_pos.edges()
                    neg_edges = sub_graph_neg.edges()
                    g_batch = block[0]

                # TODO: PUT IN A FILE
                if args.model == 'madan':
                    if args.debug:
                        if iter % 100 == 0:
                            print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%')
                    adj = g_batch.adjacency_matrix()
                    adj = adj.sparse_resize_((adj.size(0), adj.size(0)), adj.sparse_dim(), adj.dense_dim())
                    idx=adj.coalesce().indices()
                    nx_graph=nx.from_edgelist([(i[0].item(),i[1].item()) for i in idx.T])
                    node_dict = None
                    if not nx.is_connected(nx_graph):
                        nodes = list(max(nx.connected_components(nx_graph), key=len))
                        node_dict = {k:v for k,v in zip(list(nx_graph.nodes),nodes)}
                        nx_graph = nx.subgraph(nx_graph, nodes)
                    try:
                        madan = md.Madan(nx_graph, attributes=g_batch.ndata['feature']['_N'], sigma=0.08)
                    except:
                        print('eigenvalues dont converge')
                        continue
                    madan.compute_concentration(100)
                    time_scales   =   np.concatenate([np.array([0]), 10**np.linspace(0,5,500)])
                    anoms_detected=madan.anomalous_nodes
                    if node_dict and len(anoms_detected)>0:
                        anoms_detected = node_dict[anoms_detected]
                    if len(anoms_detected)>0:
                        print('anom found')
                        import ipdb ; ipdb.set_trace()
                        print('hi')
                    iter += 1
                    continue
                
                optimizer.zero_grad()
        
                if struct_model:
                    A_hat,lbl = struct_model(g_batch)
                if feat_model: # TODO
                    X_hat,lbl = feat_model(g_batch)
                    
                if args.batch_type == 'node':
                    if args.model == 'gradate':
                        loss = A_hat[0]
                    else:
                        loss, struct_loss, feat_cost = loss_func(g_batch, A_hat, X_hat, None, None, sample=False, recons=args.recons)
                else:
                    if lbl is None:
                        loss, struct_loss, feat_cost = loss_func(g_batch, A_hat, X_hat, pos_edges, neg_edges, sample=args.sample_train, recons=args.recons)
                    else:
                        loss, struct_loss, feat_cost = loss_func(lbl, A_hat, X_hat, pos_edges, neg_edges, sample=args.sample_train, recons=args.recons)


                if args.sample_train:
                    l = torch.mean(loss,1)
                    l = torch.sum(l)
                else:
                    # TODO: scales
                    l = torch.sum(loss)
                '''
                if l < best_loss:
                    best_loss = dl
                    torch.save(model,'best_model.pt')
                '''
                l.backward()
                optimizer.step()
                if args.debug:
                    if iter % 100 == 0:
                        if args.model not in ['multi-scale','multiscale']:
                            print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=', round(l.item(),3))
                        else:
                            print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=', round(l.item(),3),torch.mean(loss,dim=1))

                iter += 1
        #num_nonzeros=[]
        #for sc, sc_pred in enumerate(A_hat):
        #    num_nonzeros.append(round((torch.where(sc_pred < 0.5)[0].shape[0])/(sc_pred.shape[0]**2),2))
        print("Epoch:", '%04d' % (epoch), "train_loss=", round(l.item(),3))#, "losses=",torch.round(loss,decimals=4).detach().cpu())#, "Non-edges:",num_nonzeros)

    print('best loss:', best_loss)
    
    #model = torch.load('best_model.pt')

    # accumulate node-wise anomaly scores via model evaluation
    if struct_model: struct_model.eval()
    if feat_model: feat_model.eval()
    
    edges=adj.edges('eid')
    #dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,device=args.device)
    if args.model != 'gcad':
        struct_scores, feat_scores = torch.zeros(len(A_hat),adj.number_of_nodes()).to(args.device),torch.zeros(args.d,adj.number_of_nodes()).to(args.device)
    iter = 0

    # TODO: REFACTOR
    edge_anom_mats = []
    node_anom_mats = []
    scales = args.d
    if args.model in ['multi-scale','multiscale']:
        scales = 3
    for i in range(scales):
        if args.batch_type == 'edge':
            am = scipy.sparse.csr_matrix((adj.number_of_nodes(),adj.number_of_nodes()))
            edge_anom_mats.append(am)
        elif args.batch_type == 'node':
            node_anom_mats.append(np.full((adj.number_of_nodes(),),-1.))
    all_samps = []
    with dataloader.enable_cpu_affinity():
        for loaded_input in dataloader:
            
            if args.batch_type == 'node':
                sub_graph = loaded_input
                in_nodes = sub_graph.nodes()
                g_batch = sub_graph
                for i in g_batch.ndata['_ID']:
                    if i not in all_samps:
                        all_samps.append(i)
            
            elif args.batch_type == 'edge':
                in_nodes, sub_graph_pos, sub_graph_neg, block = loaded_input
                pos_edges = sub_graph_pos.edges()
                neg_edges = sub_graph_neg.edges()
                g_batch = block[0]
                if args.batch_type == 'edge':
                    for i in g_batch.ndata['_ID']['_N']:
                        if i.item() not in all_samps:
                            all_samps.append(i.item())

            if struct_model:
                A_hat,lbl = struct_model(g_batch)
            elif feat_model: # TODO
                X_hat = feat_model(g_batch)
            else:
                adj_ = g_batch.adjacency_matrix() 
                adj_=adj_.sparse_resize_((g_batch.num_src_nodes(),g_batch.num_src_nodes()), adj_.sparse_dim(),adj_.dense_dim())
                feat = g_batch.ndata['feature']#['_N']
                struct_loss = gcad_model.fit_transform(adj_,feat)
                struct_loss = np.array((struct_loss.T)[g_batch.dstnodes().detach().cpu().numpy()].T)
            if args.batch_type == 'edge':
                # construct edge reconstruction error matrix for anom detection
                node_ids_score = in_nodes[:block[0].num_dst_nodes()]
                #import ipdb ; ipdb.set_trace()
                pos_edges = torch.vstack((pos_edges[0],pos_edges[1]))
                neg_edges = torch.vstack((neg_edges[0],neg_edges[1]))
                edge_ids = torch.cat((pos_edges.T,neg_edges.T)).detach().cpu().numpy().T
                edge_id_dict = {k.item():v.item() for k,v in zip(torch.arange(node_ids_score.shape[0]),node_ids_score)}
                edge_ids_=np.vectorize(edge_id_dict.get)(edge_ids)
            else:
                node_ids_ = g_batch.ndata['_ID']
            if args.batch_type == 'edge' and args.model not in ['gcad']:
                if lbl is None:
                    loss, struct_loss, feat_cost = loss_func(g_batch, A_hat, X_hat, pos_edges, neg_edges, sample=args.sample_test, recons=args.recons)
                else:
                    loss, struct_loss, feat_cost = loss_func(lbl, A_hat, X_hat, pos_edges, neg_edges, sample=args.sample_test, recons=args.recons)
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
                #if args.batch_size > 0:
                if args.sample_test:
                    if args.batch_type == 'node' or args.model in ['gcad']:
                        #import ipdb ; ipdb.set_trace()
                        node_anom_mats[sc][node_ids_.detach().cpu().numpy()] = struct_loss[sc]
                    else:
                        edge_anom_mats[sc][tuple(edge_ids_)] = struct_loss[sc].detach().cpu().numpy()
                        # symmetrize
                        edge_anom_mats[sc][tuple(np.flip(edge_ids_,axis=0))] = edge_anom_mats[sc][tuple(edge_ids_)]
                else:
                    if args.batch_type == 'node' or args.model in ['gcad']:
                        node_anom_mats[sc][node_ids_.detach().cpu().numpy()[:struct_loss[sc].shape[0]]] = struct_loss[sc]
                    else:
                        edge_anom_mats[sc][tuple(edge_ids)] = struct_loss[sc][edge_ids[0],edge_ids[1]].detach().cpu().numpy()
                        # symmetrize
                        edge_anom_mats[sc][tuple(np.flip(edge_ids,axis=0))] = edge_anom_mats[sc][tuple(edge_ids)]
            
            if iter % 100 == 0 and args.debug:
                if args.batch_type == 'edge':
                    print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=',round(l.item(),3))
                elif args.batch_type == 'node':
                    print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=',round(l.item(),3))
            
            iter += 1
            
    # anomaly detection with final scores
    if A_hat or args.batch_type == 'node':
        print('structure scores')
        if args.batch_type == 'node':
            if -1 in node_anom_mats[0]:
                print('node not sampled ..?')
            detect_anomalies(None,node_anom_mats, truth, sc_label, args.dataset, sample=args.sample_test, cluster=False, input_scores=True)
        else:
            detect_anomalies(sp_adj, edge_anom_mats, truth, sc_label, args.dataset, sample=args.sample_test, cluster=False)

    if X_hat: # TODO
        print('feat scores')
        detect_anomalies(feat_scores, truth, sc_label, args.dataset)
    
    if args.debug:
        print('finished, ready to debug')
        import ipdb ; ipdb.set_trace()
        for i in struct_model.module_list:
            print(torch.sigmoid(i.lam))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora_triple_sc_all', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--scales', default='4', type=int, help='number of scales for multi-scale analysis')
    parser.add_argument('--batch_size', type=int, default=32, help='number of edges to use for batching (default: 32)')
    parser.add_argument('--recons', default='struct', type=str, help='reconstruct features or structure')
    parser.add_argument('--cutoff',default=50, type=float, help='validation cutoff')
    parser.add_argument('--d', default=1, type=int, help='d parameter for BWGNN filters')
    parser.add_argument('--model', default='multi-scale', type=str, help='encoder to use')
    parser.add_argument('--sample_train', default=False, type=bool, help="whether or not to sample edges in training")
    parser.add_argument('--sample_test', default=False, type=bool, help="whether or not to sample edges in testing")
    parser.add_argument('--batch_type', default='edge', type=str, help="node or edge sampling for batching")
    parser.add_argument('--debug', default=False, type=bool, help="if true, prints intermediate output")
    args = parser.parse_args()

    graph_anomaly_detection(args)
