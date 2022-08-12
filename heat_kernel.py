# CITATIONS
# Simple Graph Convolution (SGC)

import numpy as np
import scipy.sparse as sp
import random
import scipy.io as sio
import argparse
import pickle as pkl
import networkx as nx
import sys
import os
import time
import os.path as osp
from sklearn import preprocessing
from scipy.spatial.distance import euclidean
import scipy.sparse as sparse
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
import networkx
import torch

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).transpose().tocoo()

def adj_distr(adj,scale):
    # edges = upper_tri_vec(adj)
    plt.figure()
    plt.imshow(adj)
    plt.colorbar()
    fname = 'figs/adj'+str(scale)+'.png'
    plt.savefig(fname)
    return

def sgc_precompute(features, adj, degree, mp_weights):
    t = time.perf_counter()
    print('started...')
    for i in range(degree):
        print('started degree',i)
        # weight = torch.diag(torch.diag(torch.full((features.shape[1],features.shape[1]),mp_weights[i],dtype=torch.float64)))
        # x = torch.spmm(features,weight)
        # print('features',features)
        # print('x',x)
        # print(torch.equal(features,x))
        # new_adj = torch.spmm(adj,weight)
        features = torch.spmm(adj, features)
        
    precompute_time = time.perf_counter()-t
    print('time to compute',precompute_time)
    return features, precompute_time

ks = [1,2,3]
mp_weights = [1.,.7,.3]

savedir = 'anom_data/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
matfile = 'cora_sparse_anoms.mat'
path = savedir + matfile

smoothdir = 'smoothed_graphs/'
if not os.path.exists(smoothdir):
    os.makedirs(smoothdir)

mat = sio.loadmat(path)
network,attr = mat['Network'].todense(),torch.tensor(mat['Attributes'].todense())
smoothed_attrs = []

for ind,k in enumerate(ks):
    norm_network = torch.tensor(normalize_adj(network).todense())
    smoothed_attr = sgc_precompute(attr,norm_network,k,mp_weights)[0]
    newpath = smoothdir + str(k) + '_' + matfile
    print('k',k,smoothed_attr)
    print(type(mat['Network']))
    sio.savemat('{}'.format(newpath),\
                {'Network': mat['Network'], 'Label': mat['Label'], 'Attributes': csr_matrix(smoothed_attr.numpy()),\
                 'Class': mat['Class'], 'str_anomaly_label': mat['str_anomaly_label'], 'attr_anomaly_label':mat['attr_anomaly_label']})
    print('Done. The file is save as: {} \n'.format(newpath))

