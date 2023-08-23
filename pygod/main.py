# train a dominant detector
from pygod.detector import DOMINANT, AnomalyDAE
import scipy.io as sio
import torch
import numpy as np
from torch_geometric.data import Data
import argparse
import sys
import scipy
import scipy.io as sio
sys.path.append('../../msgad')
from msgad.anom_detector import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='.')
parser.add_argument('--dataset', type=str, default='random')
parser.add_argument('--embedding_channels', type=int, default=64)
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--model', type=str, default='dominant')
parser.add_argument('--weight', type=float, default=0.5)
parser.add_argument('--scales', type=int, default=3, help='for msgad comparison: which multi-scale labels to use')
args = parser.parse_args()

DATA_DIR = args.data_dir
data_mat = sio.loadmat(f'{DATA_DIR}/{args.dataset}.mat')
if 'cora' in args.dataset or 'yelp' in args.dataset:
    feats = torch.FloatTensor(data_mat['Attributes'].toarray())
else:
    feats = torch.FloatTensor(data_mat['Attributes'])
adj,edge_idx=None,None
if 'Edge-index' in data_mat.keys():
    edge_idx = torch.tensor(data_mat['Edge-index'])
elif 'Network' in data_mat.keys():
    adj = data_mat['Network']
    edge_idx = torch.tensor(np.stack(adj.nonzero()))
edge_idx = edge_idx.to(torch.long)
truth = torch.tensor(data_mat['Label'].flatten())

import dgl
import networkx as nx
nx_nodes = None
if 'elliptic' not in args.dataset:
    dgl_graph = dgl.graph((edge_idx[0],edge_idx[1]))
    nx_graph = nx.to_undirected(dgl.to_networkx(dgl_graph))
    nx_nodes = np.array(list(max(nx.connected_components(nx_graph), key=len)))
    feats = feats[nx_nodes] ; edge_idx = torch.stack(dgl_graph.subgraph(nx_nodes).edges())

data = Data(x=feats, edge_index=edge_idx)
anomaly_flag = truth.to(bool).numpy()

if args.model == 'dominant':
    model = DOMINANT(num_layers=4, epoch=200, weight=args.weight)  # hyperparameters can be set here
elif args.model == 'anomalydae':
    model = AnomalyDAE(num_layers=4, epoch=args.epochs, alpha=args.weight)
else:
    raise "model not found"

model.fit(data)  # input data is a PyG data object
# get outlier scores on the training data (transductive setting)
import ipdb ; ipdb.set_trace()
score = torch.zeros(truth.shape[0])
if nx_nodes is not None:
    score[nx_nodes] = model.decision_score_
else:
    score = model.decision_score_
print('roc_auc_score:%.3f'%(roc_auc_score(anomaly_flag, score.numpy())))
# predict labels and scores on the testing data (inductive setting)
a_clf = anom_classifier(None,args.scales,'../msgad/output',args.dataset,args.epochs,args.model,args.model)
with open(f'../msgad/batch_data/labels/{args.dataset}_labels_{args.scales}.mat','rb') as fin:
    mat = pkl.load(fin)
sc_all,clusts = mat['labels'],mat['clusts']
a_clf.calc_anom_stats(score[np.newaxis,...].numpy(),anomaly_flag,sc_all,clusts)