import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import networkx as nx
from igraph import Graph
import copy

def load_anomaly_detection_dataset(dataset, sc, mlp, parity, datadir='data'):
    # TODO: get edge list, sample from edge list, how to get non-edges?
    data_mat = sio.loadmat(f'data/{dataset}.mat')
    mlp=True
    if mlp:
        if 'cora' in dataset:
            feats = [torch.FloatTensor(data_mat['Attributes'].toarray())]
        else:
            feats = [torch.FloatTensor(data_mat['Attributes'])]
    else:
        feats = [] 
    if parity is None and dataset != 'weibo':
        scales=[1,2,3,4,5,6]
    elif parity=='even':
        scales=[2,4,6]
    elif parity=='odd':
        scales=[1,3,5]
    else:
        scales=[]
    
    #for scales in scales:
    #    feats.append(torch.FloatTensor(sio.loadmat(f'../smoothed_graphs/{scales}_{dataset}.mat')['Attributes'].toarray()))
    
    adj = data_mat['Network']
    #feat = data_mat['Attributes']
    truth = data_mat['Label'].flatten()
    
    adj_no_loop = copy.deepcopy(adj)
    data_mat = sio.loadmat(f'data/cora_triple_sc_all.mat')

    if dataset != 'weibo':
        sc_label = data_mat['scale_anomaly_label']
    else:
        sc_label = data_mat['scale_anomaly_label'][0]

    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    adj = adj_no_loop
    if dataset == 'weibo':
        adj = adj.toarray()
    edge_list = Graph.Adjacency(adj_no_loop.toarray()).get_edgelist()
    # edge removal
    train_graph = nx.from_numpy_matrix(adj_no_loop.toarray())
    num_edges = np.count_nonzero(adj_no_loop.toarray())/2
    val_found = False
    random.shuffle(edge_list)
    num_edges_val = int(num_edges*0.2)
    edges_rem, edges_keep = edge_list[:num_edges_val],edge_list[num_edges_val:]
    nodes_val = np.unique(list(sum(edges_rem,())))
    nodes_train = np.unique(list(sum(edges_keep,())))

    val_graph = copy.deepcopy(train_graph)
    val_graph.remove_edges_from(edges_keep)

    val_adj = nx.adjacency_matrix(val_graph).todense()
    train_adj = nx.adjacency_matrix(train_graph).todense()
    val_adj = val_adj + np.eye(val_adj.shape[0])
    val_adj = val_adj[nodes_val][:,nodes_val]
    val_feats = [feats[0][nodes_val]]
    train_feats = [feats[0][nodes_train]]
    return adj_norm, feats, truth, adj_no_loop.toarray(), sc_label, train_adj, train_feats, val_adj, val_feats

def spectral_norm(w):
    evals,_ = torch.eig(torch.mm(w.T,w))
    w_eval = torch.max(evals)
    assert(w_eval >= 0)
    norm_w = w / torch.sqrt(w_eval)
    return norm_w

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

