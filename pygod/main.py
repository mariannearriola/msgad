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
parser.add_argument('--data_flag', type=str, default='structure_anomaly')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--real_world_name', type=str, default='email')
parser.add_argument('--msgad_name', type=str, default='weibo')
parser.add_argument('--dataset', type=str, default='random')
parser.add_argument('--anomaly_type', type=str, default='chain')
parser.add_argument('--size', type=int, default=1000)
parser.add_argument('--anomaly_ratio', type=float, default=0.02)
parser.add_argument('--dim', type=int, default=50)
parser.add_argument('--anomaly_scale', type=float, default=0.3)
parser.add_argument('--anomaly_attr_ratio', type=float, default=0.2)
parser.add_argument('--diff_ratio', type=int, default=5)
parser.add_argument('--half_num', type=int, default=10)
parser.add_argument('--random_seed', type=int, default=12345)
parser.add_argument('--num_anchors', type=int, default=100)
parser.add_argument('--embedding_channels', type=int, default=64)
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup', type=int, default=90)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--model', type=str, default='dominant')
parser.add_argument('--weight', type=float, default=0.5)
parser.add_argument('--q', type=int, default=90)
parser.add_argument('--convergence', type=float, default=1e-4)
parser.add_argument('--ending_rounds', type=int, default=1)
parser.add_argument('--scales', type=int, default=3, help='for msgad comparison: which multi-scale labels to use')
args = parser.parse_args()

DATA_DIR = args.data_dir
data_mat = sio.loadmat(f'{DATA_DIR}/{args.msgad_name}.mat')
if 'cora' in args.msgad_name or 'yelp' in args.msgad_name:
    feats = torch.FloatTensor(data_mat['Attributes'].toarray())
else:
    feats = torch.FloatTensor(data_mat['Attributes'])
adj,edge_idx=None,None
if 'Edge-index' in data_mat.keys():
    edge_idx = data_mat['Edge-index']
elif 'Network' in data_mat.keys():
    adj = data_mat['Network']
    edge_idx = torch.tensor(np.stack(adj.nonzero()))
edge_idx = edge_idx.to(torch.long)
truth = torch.tensor(data_mat['Label'].flatten())
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
score = model.decision_score_
print('roc_auc_score:%.3f'%(roc_auc_score(anomaly_flag, score.numpy())))
# predict labels and scores on the testing data (inductive setting)

'''
pred, score = model.predict(data, return_score=True)
a_clf = anom_classifier(None,args.scales,'../output',args.msgad_name,args.epochs,'pygod','pygod','struct','')
mat = sio.loadmat(f'../msgad/batch_data/labels/{args.msgad_name}_labels.mat')
sc_all,clusts = mat['labels'][0],mat['clusts']
a_clf.calc_prec(score[np.newaxis,...].numpy(),anomaly_flag,sc_all,clusts,cluster=False,input_scores=True)
'''