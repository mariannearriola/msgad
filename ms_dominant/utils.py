import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random


def load_anomaly_detection_dataset(dataset, sc, datadir='data'):
    data_mat = sio.loadmat(f'data/{dataset}.mat')
    feats = [torch.FloatTensor(data_mat['Attributes'].toarray())]
    #feats = []
    for scales in range(1,sc+1):
        feats.append(torch.FloatTensor(sio.loadmat(f'../smoothed_graphs/{scales}_{dataset}.mat')['Attributes'].toarray()))

    adj = data_mat['Network']
    #feat = data_mat['Attributes']
    truth = data_mat['Label']
    truth = truth.flatten()

    sc_label = data_mat['scale_anomaly_label']
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.toarray()
    #feat = feat.toarray()
    return adj_norm, feats, truth, adj, sc_label



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
