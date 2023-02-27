from scipy.sparse import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.io
from sklearn.metrics import roc_auc_score
from datetime import datetime
import argparse
import scipy
import networkx as nx
import torch_geometric
from torch.utils.data import DataLoader
from scipy import stats
from model_tf import EGCN
from utils_tf import *
from scipy.spatial.distance import euclidean
import random 

def train_dominant(args):

    adj, attrs_det, label, adj_label, sc_label, adj_train, attrs_train, adj_val, attrs_val = load_anomaly_detection_dataset(args.dataset, args.scales, args.mlp, args.parity)
    adj, adj_label = sparse_matrix_to_tensor(adj,attrs_det[0]), sparse_matrix_to_tensor(adj_label,attrs_det[0])
    model = EGCN(in_size = attrs_det[0].size(1), out_size = args.hidden_dim, scales = args.scales, recons = args.recons, mlp = args.mlp, d = args.d)
    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    data_loader = DataLoader(torch.arange(adj.num_edges()), batch_size=args.batch_size,shuffle=False) 
    
    best_loss = torch.tensor(float('inf')).cuda()
    model.train()
    for epoch in range(args.epoch):
        for iter, bg in enumerate(data_loader):
            if iter % 1000 == 0: print(iter, '/', len(data_loader))
            if iter == 1: break
            batch_adj, pos_edges, neg_edges, _ = load_batch(adj_label,bg,args.device)
            batch_adj_norm=dgl.add_self_loop(batch_adj)
            optimizer.zero_grad()
            A_hat_scales, X_hat = model(batch_adj_norm)
            loss, struct_loss = loss_func(batch_adj, A_hat_scales, pos_edges, neg_edges)
            l = torch.mean(loss)
            '''
            if l < best_loss:
                best_loss = l
                torch.save(model,'best_model.pt')
            '''
            l.backward()
            optimizer.step()

        num_nonzeros=[]
        for sc, A_hat in enumerate(A_hat_scales):
            num_nonzeros.append(round((torch.where(A_hat < 0.5)[0].shape[0])/(A_hat.shape[0]**2),2))
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "losses=",loss, "Non-edges:",num_nonzeros)

    print('best loss:', best_loss)
    adj, attrs_det, label, adj_label, sc_label, train_adj, train_attrs, val_adj, val_attrs = load_anomaly_detection_dataset(args.dataset, args.scales, args.mlp, args.parity)
    adj, adj_label = sparse_matrix_to_tensor(adj,attrs_det[0]), sparse_matrix_to_tensor(adj_label,attrs_det[0])
    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        
    #model = torch.load('best_model.pt')
    model.eval()
    data_loader = DataLoader(torch.arange(adj.num_edges()), batch_size=args.batch_size,shuffle=False)

    scores, l_scores = torch.zeros(adj.number_of_nodes(),args.d+1).cuda(),torch.zeros((adj.number_of_nodes(),args.d+1))
    for iter, bg in enumerate(data_loader):
        if iter == 1: break
        batch_adj, pos_edges, neg_edges, batch_dict = load_batch(adj_label,bg,args.device)
        batch_adj_norm=dgl.add_self_loop(batch_adj)
        A_hat_scales, X_hat = model(batch_adj_norm)
        loss, struct_loss = loss_func(batch_adj, A_hat_scales, pos_edges, neg_edges)
        scores = struct_loss
        '''
        for sc, A_hat in enumerate(A_hat_scales):
            scores[list(batch_dict.values()),sc] = struct_loss[:,sc]
        '''
    print('CHECK SCORES HERE')
    for sc, A_hat in enumerate(A_hat_scales):
        scores_recons = scores[:,sc].detach().cpu().numpy()
        sorted_errors = np.argsort(-scores_recons)
        rev_sorted_errors = np.argsort(scores_recons)
        rankings = []
        import ipdb ; ipdb.set_trace()
        for error in sorted_errors:
            rankings.append(label[error])
        rankings = np.array(rankings)

        print(f'SCALE {sc+1}',np.mean(scores_recons))
        try:
            print('AP',detect_anom_ap(scores_recons,label))
        except:
            import ipdb ; ipdb.set_trace()
        print('RECONSTRUCTION')
        print(detect_anom(sorted_errors, sc_label[0], 1))
        print('reverse')
        print(detect_anom(rev_sorted_errors, sc_label[0], 1))
        print('')
        
    with open('output/{}-ranking_{}.txt'.format(args.dataset, sc), 'w+') as f:
        for index in sorted_errors:
            f.write("%s\n" % label[index])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tfinance', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--scales', default='4', type=int, help='number of scales for multi-scale analysis')
    parser.add_argument('--batch_size', type=int, default=32, help='number of edges to use for batching (default: 32)')
    parser.add_argument('--recons', default='struct', type=str, help='reconstruct features or structure')
    parser.add_argument('--mlp', default=False, type=bool, help='include features for mlp or not')
    parser.add_argument('--parity', default=None, type=str, help='even, odd, or regular scales')
    parser.add_argument('--cutoff',default=50, type=float, help='validation cutoff')
    parser.add_argument('--d', default=4, type=int, help='d parameter for BWGNN filters')
    args = parser.parse_args()

    train_dominant(args)
