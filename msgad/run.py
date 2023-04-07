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
    """Pipeline for autoencoder-based graph anomaly detection"""
    # load data
    sp_adj, feats, truth, sc_label = load_anomaly_detection_dataset(args.dataset, args.scales)
    anoms,norms=np.where(truth==1)[0],np.where(truth==0)[0]
    adj = sparse_matrix_to_tensor(sp_adj,feats)
    lbl=None

    # initialize data loading
    if args.batch_type == 'edge':
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(4)
        #sampler = dgl.dataloading.NeighborSampler([5,5,5])
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler,negative_sampler=neg_sampler)
        edges=adj.edges('eid')
        batch_size = args.batch_size if args.batch_size > 0 else adj.number_of_edges()
        if args.device == 'cuda':
            dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, device=args.device)#,pin_prefetcher=True)
        else:
            dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=6, device=args.device)
    elif args.batch_type == 'edge_rw':
        batch_size = args.batch_size if args.batch_size > 0 else adj.number_of_nodes()
        
        #sampler = dgl.dataloading.SAINTSampler(mode='walk',budget=[int(batch_size/3),batch_size])

        sampler = dgl.dataloading.ShaDowKHopSampler([4])
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler,negative_sampler=neg_sampler)
        edges=adj.edges('eid')
        dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0, device=args.device)
    elif args.batch_type == 'node':
        batch_size = args.batch_size if args.batch_size > 0 else adj.number_of_nodes()
        #sampler = dgl.dataloading.SAINTSampler(mode='walk',budget=[int(batch_size/3),batch_size])
        sampler = dgl.dataloading.ShaDowKHopSampler([4])
        if args.device == 'cuda':
            num_workers = 0
        else:
            num_workers = 4
        dataloader = dgl.dataloading.DataLoader(adj, adj.nodes(), sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, device=args.device)
    print('sample train',args.sample_train,'sample test',args.sample_test)


    # intialize model (on GPU)
    struct_model,feat_model=None,None
    if args.recons == 'struct' or args.recons == 'both':
        if args.model == 'gcad':
            gcad_model = GCAD(2,100,1)
        elif args.model == 'madan':
            pass
        else:
            struct_model = GraphReconstruction(in_size = feats.size(1), hidden_size=args.hidden_dim, batch_size=args.batch_size, scales = args.scales, recons = 'struct', d = args.d, model_str = args.model, batch_type = args.batch_type, label_type=args.label_type)
    if args.recons == 'feat' or args.recons == 'both':
        feat_model = GraphReconstruction(in_size = feats.size(1), hidden_size=args.hidden_dim, batch_size=args.batch_size, scales = args.scales, recons = 'feat', d = args.d, model_str = args.model)

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
    torch.autograd.set_detect_anomaly(True) 
    best_loss = torch.tensor(float('inf')).to(args.device)    
    A_hat, X_hat = None,None
    seconds = time.time()
    for epoch in range(args.epoch):
        if args.model == 'gcad': break
        iter=0
        with dataloader.enable_cpu_affinity():
            for loaded_input in dataloader:
                if args.batch_type == 'node':
                    all_nodes,in_nodes,g_batch= loaded_input
                elif args.batch_type == 'edge_rw':
                    in_nodes, sub_graph_pos, sub_graph_neg, block = loaded_input
                    g_batch = block
                    pos_edges = sub_graph_pos.edges()
                    neg_edges = sub_graph_neg.edges()
                else:
                    in_nodes, sub_graph_pos, sub_graph_neg, block = loaded_input
                    pos_edges = sub_graph_pos.edges()
                    neg_edges = sub_graph_neg.edges()
                    pos_edges=torch.vstack((pos_edges[0],pos_edges[1])).T
                    neg_edges=torch.vstack((neg_edges[0],neg_edges[1])).T
                    last_batch_node = torch.max(neg_edges[1])
                    '''
                    tgt_pos_nodes,tgt_neg_nodes = in_nodes[sub_graph_pos.nodes()],in_nodes[sub_graph_neg.nodes()]
                    for ind in range(pos_edges[0].shape[0]):
                        pos_edges[0][ind] = tgt_pos_nodes[pos_edges[0][ind]]
                        pos_edges[1][ind] = tgt_pos_nodes[pos_edges[1][ind]]
                        neg_edges[0][ind] = tgt_pos_nodes[neg_edges[0][ind]]
                        neg_edges[1][ind] = tgt_pos_nodes[neg_edges[1][ind]]
                    '''

                    #if not torch.equal(tgt_pos_nodes,tgt_neg_nodes):
                    #    print('not equal')
                    #    import ipdb ; ipdb.set_trace()
                    g_batch = block[0]
                    '''
                    node_dict = {k.item():v for k,v in zip(g_batch.dstnodes(),np.arange(g_batch.num_dst_nodes()))}
                    rev_node_dict = {v: k for k, v in node_dict.items()}
                    anom_batch = np.intersect1d(anoms,g_batch.dstnodes().detach().cpu().numpy())
                    anom_batch = [node_dict[i] for i in anom_batch]
                    norm_batch = np.setdiff1d(g_batch.dstnodes().detach().cpu().numpy(),anom_batch)
                    '''
                    
                if args.model == 'madan':
                    if args.debug:
                        if iter % 100 == 0:
                            print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%')
                    #adj = g_batch.adjacency_matrix()
                    if iter == 0:
                        adj = adj.adjacency_matrix()
                        adj = adj.sparse_resize_((adj.size(0), adj.size(0)), adj.sparse_dim(), adj.dense_dim())
                    idx=adj.coalesce().indices()
                    nx_graph=nx.from_edgelist([(i[0].item(),i[1].item()) for i in idx.T])
                    #feats = g_batch.ndata['feature'].cpu()
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
        
                if args.model == 'ho-gat':
                    A_hat,lbl,_ = struct_model(g_batch)
                elif struct_model:
                    #print('running')
                    A_hat,lbl = struct_model(g_batch,last_batch_node,pos_edges,neg_edges)
                if feat_model: # TODO
                    X_hat,lbl = feat_model(g_batch)
                #print('alcing loss')
                if args.batch_type == 'node':
                    if args.model == 'gradate':
                        loss = A_hat[0]
                    elif args.model == 'ho-gat':
                        lbl = torch_geometric.utils.to_dense_adj(lbl)
                    
                        loss, struct_loss, feat_cost = loss_func(lbl, [A_hat], X_hat, None, None, sample=False, recons=args.recons)
                    else:
                        loss, struct_loss, feat_cost = loss_func(g_batch, A_hat, X_hat, None, None, sample=False, recons=args.recons)
                else:
                    if lbl is None:
                        loss, struct_loss, feat_cost = loss_func(g_batch, A_hat, X_hat, pos_edges, neg_edges, sample=args.sample_train, recons=args.recons)
                    else:
                        loss, struct_loss, feat_cost = loss_func(lbl, A_hat, X_hat, pos_edges, neg_edges, sample=args.sample_train, recons=args.recons)
                #print('done')
                if 'multi-scale' in args.model:
                    l = torch.sum(torch.mean(loss,dim=1))
                else:
                    l = torch.mean(loss)
                '''
                if l < best_loss:
                    best_loss = dl
                    torch.save(model,'best_model.pt')
                '''
                l.backward()
                #print('backwarded')
                optimizer.step()
                if args.debug:
                    if iter % 100 == 0:
                        if 'multi-scale' not in args.model:
                            print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=', round(l.item(),3))
                        else:
                            print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=', round(l.item(),3))#,torch.sum(loss,dim=1))

                iter += 1
            print("Seconds since epoch =", (time.time()-seconds)/60)
            seconds = time.time()

        #num_nonzeros=[]
        #for sc, sc_pred in enumerate(A_hat):
        #    num_nonzeros.append(round((torch.where(sc_pred < 0.5)[0].shape[0])/(sc_pred.shape[0]**2),2))
        if args.model != 'madan':
            print("Epoch:", '%04d' % (epoch), "train_loss=", round(l.item(),3), "losses=",torch.round(torch.mean(loss,dim=1),decimals=4).detach().cpu())#, "Non-edges:",num_nonzeros)

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
    clust_anom_mats = []
    node_anom_mats = []
    clust_inds = []
    scales = args.d
    if 'multi-scale' in args.model:
        scales = 3
    else:
        scales = 1
    for i in range(scales):
        am = np.zeros((adj.number_of_nodes(),adj.number_of_nodes()))
        edge_anom_mats.append(am)
        node_anom_mats.append(np.full((adj.number_of_nodes(),),-1.))
    all_samps = []
    with dataloader.enable_cpu_affinity():
        for loaded_input in dataloader:
            
            if args.batch_type == 'node':
                all_nodes,in_nodes,g_batch= loaded_input
                '''
                sub_graph = loaded_input
                in_nodes = sub_graph.nodes()
                g_batch = sub_graph
                '''
                for i in g_batch.ndata['_ID']:
                    if i not in all_samps:
                        all_samps.append(i)
                
            elif args.batch_type == 'edge_rw':
                in_nodes, sub_graph_pos, sub_graph_neg, block = loaded_input
                g_batch = block
                pos_edges = sub_graph_pos.edges()
                neg_edges = sub_graph_neg.edges()
                k_,v_=torch.arange(g_batch.number_of_nodes()),g_batch.ndata['_ID']
            elif args.batch_type == 'edge':
                in_nodes, sub_graph_pos, sub_graph_neg, block = loaded_input

                pos_edges = sub_graph_pos.edges()
                neg_edges = sub_graph_neg.edges()
                pos_edges=torch.vstack((pos_edges[0],pos_edges[1])).T
                neg_edges=torch.vstack((neg_edges[0],neg_edges[1])).T

                last_batch_node = torch.max(neg_edges[1])
                g_batch = block[0]
                if args.batch_type == 'edge':
                    for i in g_batch.ndata['_ID']['_N']:
                        if i.item() not in all_samps:
                            all_samps.append(i.item())

                k_,v_=torch.arange(g_batch.number_of_nodes()),g_batch.ndata['_ID']
                

            if args.model == 'ho-gat':
                A_hat,lbl,clust_ind = struct_model(g_batch)
            elif struct_model:
                A_hat,lbl = struct_model(g_batch,last_batch_node,pos_edges,neg_edges)

            elif feat_model: # TODO
                X_hat = feat_model(g_batch)
            else:
                adj_ = g_batch.adjacency_matrix() 
                adj_=adj_.sparse_resize_((g_batch.num_src_nodes(),g_batch.num_src_nodes()), adj_.sparse_dim(),adj_.dense_dim())
                feat = g_batch.ndata['feature']
                struct_loss = gcad_model.fit_transform(adj_,feat)
                struct_loss = np.array((struct_loss.T)[g_batch.dstnodes().detach().cpu().numpy()].T)
            if args.batch_type == 'edge':
                # construct edge reconstruction error matrix for anom detection
                node_ids_score = in_nodes[:block[0].num_dst_nodes()]
                edge_ids = torch.cat((pos_edges,neg_edges)).detach().cpu().numpy().T
                edge_id_dict = {k.item():v.item() for k,v in zip(torch.arange(node_ids_score.shape[0]),node_ids_score)}
                rev_id_dict = {v: k for k, v in edge_id_dict.items()}
                edge_ids_=np.vectorize(edge_id_dict.get)(edge_ids)
                #edge_ids = np.vectorize(rev_id_dict.get)(edge_ids)
            elif args.batch_type == 'edge_rw':
                
                pos_edges = torch.vstack((pos_edges[0],pos_edges[1])).to(torch.long)
                neg_edges = torch.vstack((neg_edges[0],neg_edges[1])).to(torch.long)
                
                edge_ids = torch.cat((pos_edges.T,neg_edges.T)).detach().cpu().numpy().T
                edge_id_dict = {k.item():v.item() for k,v in zip(k_,v_)}
                edge_ids_=np.vectorize(edge_id_dict.get)(edge_ids)

                '''
                node_ids_score = in_nodes[:block.num_dst_nodes()]
                pos_edges = torch.vstack((pos_edges[0],pos_edges[1]))
                neg_edges = torch.vstack((neg_edges[0],neg_edges[1]))
                edge_ids = torch.cat((pos_edges.T,neg_edges.T)).detach().cpu().numpy().T
                edge_id_dict = {k.item():v.item() for k,v in zip(torch.arange(node_ids_score.shape[0]),node_ids_score)}
                edge_ids_=np.vectorize(edge_id_dict.get)(edge_ids)
                '''
            else:
                node_ids_ = g_batch.ndata['_ID']
            if 'edge' in args.batch_type and args.model not in ['gcad']:
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
            elif args.model == 'ho-gat':
                lbl = torch_geometric.utils.to_dense_adj(lbl)
                loss, struct_loss, feat_cost = loss_func(lbl, [A_hat], X_hat, None, None, sample=False, recons=args.recons)
 
            
            for sc in range(scales):
                #if args.batch_size > 0:
                if args.sample_test:
                    if args.batch_type == 'node' or args.model in ['gcad']:
                        node_anom_mats[sc][node_ids_.detach().cpu().numpy()] = struct_loss[sc]
                    else:
                        try:
                            edge_anom_mats[sc][tuple(edge_ids_)] = struct_loss[sc].detach().cpu().numpy()
                            edge_anom_mats[sc][tuple(np.flip(edge_ids_,axis=0))] = edge_anom_mats[sc][tuple(edge_ids_)]
                        except Exception as e:
                            import ipdb ; ipdb.set_trace()
                            print(e)
                else:
                    if args.batch_type == 'node':
                        if args.model in ['gcad','gradate']:
                            node_anom_mats[sc][node_ids_.detach().cpu().numpy()[:struct_loss[sc].shape[0]]] = struct_loss[sc]
                        else:
                            if args.model in ['ho-gat']:
                                edge_anom_mats[sc][node_ids_.detach().cpu().numpy()][:,node_ids_.detach().cpu().numpy()] = struct_loss[0][sc][:node_ids_.shape[0]][:,:node_ids_.shape[0]].detach().cpu().numpy()
                                clust_anom_mats.append(struct_loss[0][sc][node_ids_.shape[0]:])
                                if len(clust_inds) == 0:
                                    clust_inds = clust_ind
                                else:
                                    clust_inds = torch.vstack((clust_inds,clust_ind))
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
    print('structure scores')
    if args.batch_type == 'node':
        if -1 in node_anom_mats[0]:
            print('node not sampled ..?')
        if args.model in ['ho-gat']:
            detect_anomalies(adj.adjacency_matrix().to_dense(),edge_anom_mats, truth, sc_label, args.dataset, sample=args.sample_test, cluster=False, clust_anom_mats=clust_anom_mats,clust_inds=clust_inds)
        else:
            detect_anomalies(None,node_anom_mats, truth, sc_label, args.dataset, sample=args.sample_test, cluster=False, input_scores=True)
    else:
        detect_anomalies(adj.adjacency_matrix().to_dense(), edge_anom_mats, truth, sc_label, args.dataset, sample=args.sample_test, cluster=False)

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
    parser.add_argument('--model', default='multi-scale-amnet', type=str, help='encoder to use')
    parser.add_argument('--sample_train', default=False, type=bool, help="whether or not to sample edges in training")
    parser.add_argument('--sample_test', default=False, type=bool, help="whether or not to sample edges in testing")
    parser.add_argument('--batch_type', default='edge', type=str, help="node or edge sampling for batching")
    parser.add_argument('--debug', default=False, type=bool, help="if true, prints intermediate output")
    parser.add_argument('--label_type', default='single', type=str, help="if true, prints intermediate output")

    args = parser.parse_args()

    graph_anomaly_detection(args)
