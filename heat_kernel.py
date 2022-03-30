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

def sgc_precompute(features, adj, degree):
    t = time.perf_counter()
    print('started...')
    for i in range(degree):
        print('started degree',i)
        features = torch.spmm(adj, features)
    precompute_time = time.perf_counter()-t
    print('time to compute',precompute_time)
    return features, precompute_time

ks = [1,2,3]

savedir = 'anom_data/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
matfile = 'BlogCatalog.mat'
path = savedir + matfile

smoothdir = 'smoothed_graphs/'
if not os.path.exists(smoothdir):
    os.makedirs(smoothdir)

mat = sio.loadmat(path)
network,attr = mat['Network'].todense(),torch.tensor(mat['Attributes'].todense())
smoothed_attrs = []

for k in ks:
    norm_network = torch.tensor(normalize_adj(network).todense())
    smoothed_attr = sgc_precompute(attr,norm_network,k)[0]
    newpath = smoothdir + str(k) + '_' + matfile
    sio.savemat('{}'.format(newpath),\
                {'Network': mat['Network'], 'Label': mat['Label'], 'Attributes': smoothed_attr,\
                 'Class': mat['Class'], 'str_anomaly_label': mat['str_anomaly_label'], 'attr_anomaly_label':mat['attr_anomaly_label']})
    print('Done. The file is save as: {} \n'.format(newpath))

