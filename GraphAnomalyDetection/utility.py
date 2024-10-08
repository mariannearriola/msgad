import numpy as np
import scipy.io as scio
import os
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
import torch
import torch_geometric
import dgl
import networkx as nx

def load_graph(filename):
    '''
    dir = 'dataset'
    graph = scio.loadmat(os.path.join(dir, 'cora'))
    attr = graph['Attributes'].A
    adj = graph['Network']
    label = graph['Label']
    '''
    
    dir = '../msgad/data'
    data_mat = scio.loadmat(f'{dir}/{filename}.mat')
    if 'cora' in filename or 'yelp' in filename:
        feats = data_mat['Attributes'].toarray()
    else:
        feats = data_mat['Attributes']
    adj,edge_idx=None,None
    if 'Edge-index' in data_mat.keys():
        edge_idx = torch.tensor(data_mat['Edge-index'])
        adj = torch_geometric.utils.to_dense_adj(edge_idx)
        dgl_graph = dgl.graph((edge_idx[0],edge_idx[1]))
        nx_graph = nx.to_undirected(dgl.to_networkx(dgl_graph))
        nx_nodes = np.array(list(max(nx.connected_components(nx_graph), key=len)))
        
        feats = feats[nx_nodes] ; adj = adj[0][nx_nodes][:,nx_nodes].numpy()#.to_sparse()
    elif 'Network' in data_mat.keys():
        adj = data_mat['Network']
        edge_idx = torch.tensor(np.stack(adj.nonzero()))
    edge_idx = edge_idx.to(torch.long)
    truth = torch.tensor(data_mat['Label'].flatten())
    anomaly_flag = truth.to(bool).numpy()[...,np.newaxis]
    attr,adj,label=feats,adj,anomaly_flag
    
    return attr, adj, nx_nodes, label