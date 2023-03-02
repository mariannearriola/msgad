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
import random 

def graph_anomaly_detection(args):
    # load data
    adj, feats, truth, sc_label = load_anomaly_detection_dataset(args.dataset, args.scales)
    adj = sparse_matrix_to_tensor(adj,feats)

    # intialize model (on GPU)
    struct_model,feat_model=None,None
    if args.recons == 'struct' or args.recons == 'both':
        struct_model = GraphReconstruction(in_size = feats.size(1), hidden_size=args.hidden_dim, batch_size=args.batch_size, scales = args.scales, recons = 'struct', d = args.d, model_str = args.model)
    if args.recons == 'feat' or args.recons == 'both':
        feat_model = GraphReconstruction(in_size = feats.size(1), hidden_size=args.hidden_dim, batch_size=args.batch_size, scales = args.scales, recons = 'feat', d = args.d, model_str = args.model)

    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        if struct_model:
            struct_model = struct_model.cuda()
            struct_model.train()
        if feat_model:
            feat_model = feat_model.cuda()
            feat_model.train()
    
    if args.recons == 'struct':
        params = struct_model.parameters()
    elif args.recons == 'feat':
        params = feat_model.parameters()
    elif args.recons == 'both':
        params = list(struct_model.parameters()) + list(feat_model.parameters())
        
    optimizer = torch.optim.Adam(params, lr = args.lr)

    # initialize data loading
    if args.batch_type == 'edge':
        sampler = dgl.dataloading.NeighborSampler([25])
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler)
    elif args.batch_type == 'node':
        sampler = dgl.dataloading.SAINTSampler(mode='node',budget=args.batch_size)
    dataloader = dgl.dataloading.DataLoader(adj, adj.nodes(), sampler,batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)

    # begin model training
    best_loss = torch.tensor(float('inf')).cuda()    
    A_hat, X_hat = None,None
    for epoch in range(args.epoch):
        #for iter, bg in enumerate(data_loader):
        iter=0
        with dataloader.enable_cpu_affinity():
            for loaded_input in dataloader:
                if args.batch_type == 'node':
                    sub_graph = loaded_input
                    in_nodes = sub_graph.nodes()
                elif args.batch_type == 'edge':
                    in_nodes,sub_graph,block = loaded_input
                optimizer.zero_grad()
                neg_edges = dgl.sampling.global_uniform_negative_sampling(sub_graph, args.batch_size)
                g_batch = adj.subgraph(in_nodes)
                if struct_model:
                    A_hat = struct_model(g_batch)
                if feat_model:
                    X_hat = feat_model(g_batch)
                loss, struct_loss, feat_cost = loss_func(g_batch, A_hat, X_hat, sub_graph.edges(), neg_edges, recons=args.recons)
                l = torch.sum(loss)
                '''
                if l < best_loss:
                    best_loss = dl
                    torch.save(model,'best_model.pt')
                '''
                
                l.backward()
                optimizer.step()
                iter += 1
                
                if args.debug:
                    if iter % 50 == 0 and iter != 0:
                        print(iter,l.item())
                        
                        num_nonzeros=[]
                        for sc, sc_pred in enumerate(A_hat):
                            num_nonzeros.append(round((torch.where(sc_pred < 0.5)[0].shape[0])/(sc_pred.shape[0]**2),10))
                        avg_non_edges = []
                        avg_pos_edges = []
                        for node in g_batch.adjacency_matrix().to_dense():
                            avg_non_edges.append(torch.where(node==1)[0].shape[0])
                            avg_pos_edges.append(torch.where(node==0)[0].shape[0])
                        num_nonzeros_adj=round((torch.where(g_batch.adjacency_matrix().to_dense() < 0.5)[0].shape[0])/(g_batch.number_of_nodes()**2),3)
                        num_nonzeros_ahat=round((torch.where(A_hat[0] < 0.5)[0].shape[0])/(g_batch.number_of_nodes()**2),10)
                        print(iter, round(l.item(),4),loss,num_nonzeros_adj,num_nonzeros_ahat)
                        
                
        #num_nonzeros=[]
        #for sc, sc_pred in enumerate(A_hat):
        #    num_nonzeros.append(round((torch.where(sc_pred < 0.5)[0].shape[0])/(sc_pred.shape[0]**2),2))
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "losses=",torch.round(loss,decimals=4).detach().cpu())#, "Non-edges:",num_nonzeros)

    print('best loss:', best_loss)
    
    #model = torch.load('best_model.pt')

    # accumulate node-wise anomaly scores via model evaluation
    if struct_model: struct_model.eval()
    if feat_model: feat_model.eval()
    struct_scores, feat_scores = torch.zeros(args.d,adj.number_of_nodes()).cuda(),torch.zeros(args.d,adj.number_of_nodes()).cuda()
    with dataloader.enable_cpu_affinity():
        for in_nodes,sub_graph,block in dataloader:
            if args.debug:
                if iter % 50 == 0: print(iter, l.item(),loss.item())
            neg_edges = dgl.sampling.global_uniform_negative_sampling(sub_graph, args.batch_size)
            g_batch = adj.subgraph(in_nodes)
            if struct_model:
                A_hat = struct_model(g_batch)
            if feat_model:
                X_hat = feat_model(g_batch)
            #import ipdb ; ipdb.set_trace()
            loss, struct_loss, feat_loss = loss_func(g_batch, A_hat, X_hat, sub_graph.edges(), neg_edges, recons=args.recons)
            l = torch.sum(loss)
            for sc in range(args.d):
                if A_hat:
                    struct_scores[sc,in_nodes] = struct_loss[sc,:]
                if X_hat:
                    feat_scores[sc,in_nodes] = feat_loss[sc,:]
    
    # anomaly detection with final scores
    if A_hat:
        print('structure scores')
        detect_anomalies(struct_scores, truth, sc_label, args.dataset)
    if X_hat:
        print('feat scores')
        detect_anomalies(feat_scores, truth, sc_label, args.dataset)

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
    parser.add_argument('--cutoff',default=50, type=float, help='validation cutoff')
    parser.add_argument('--d', default=1, type=int, help='d parameter for BWGNN filters')
    parser.add_argument('--model', default='multi-scale', type=str, help='encoder to use')
    parser.add_argument('--sample_train', default=False, type=bool, help="whether or not to sample edges in training")
    parser.add_argument('--sample_test', default=False, type=bool, help="whether or not to sample edges in testing")
    parser.add_argument('--batch_type', default='edge', type=str, help="node or edge sampling for batching")
    parser.add_argument('--debug', default=False, type=bool, help="if true, prints intermediate output")
    args = parser.parse_args()

    graph_anomaly_detection(args)
