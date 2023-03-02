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
    adj, feats, truth, sc_label = load_anomaly_detection_dataset(args.dataset, args.scales)
    adj = sparse_matrix_to_tensor(adj,feats)
    model = GraphReconstruction(in_size = feats.size(1), hidden_size=args.hidden_dim, scales = args.scales, recons = args.recons, d = args.d, model_str = args.model)
    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    sampler = dgl.dataloading.NeighborSampler([50])
    sampler = dgl.dataloading.as_edge_prediction_sampler(sampler)
    dataloader = dgl.dataloading.DataLoader(adj, adj.nodes(), sampler,batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    data_loader = DataLoader(torch.arange(adj.num_edges()), batch_size=args.batch_size,shuffle=True) 
    best_loss = torch.tensor(float('inf')).cuda()
    model.train()
    
    for epoch in range(args.epoch):
        #for iter, bg in enumerate(data_loader):
        iter=0
        with dataloader.enable_cpu_affinity():
            for in_nodes,sub_graph,block in dataloader:
                optimizer.zero_grad()
                neg_edges = dgl.sampling.global_uniform_negative_sampling(sub_graph, args.batch_size)
                g_batch = adj.subgraph(in_nodes)
                A_hat_scales, X_hat = model(g_batch)
                loss, struct_loss = loss_func(g_batch, A_hat_scales, sub_graph.edges(), neg_edges)
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
                    if iter % 10 == 0 and iter != 0:
                        num_nonzeros=[]
                        for sc, A_hat in enumerate(A_hat_scales):
                            num_nonzeros.append(round((torch.where(A_hat < 0.5)[0].shape[0])/(A_hat.shape[0]**2),10))
                        avg_non_edges = []
                        avg_pos_edges = []
                        for node in g_batch.adjacency_matrix().to_dense():
                            avg_non_edges.append(torch.where(node==1)[0].shape[0])
                            avg_pos_edges.append(torch.where(node==0)[0].shape[0])
                        num_nonzeros_adj=round((torch.where(g_batch.adjacency_matrix().to_dense() < 0.5)[0].shape[0])/(g_batch.number_of_nodes()**2),3)
                        print(iter, '/', len(data_loader), l.item(),loss,np.mean(np.array(avg_non_edges)),np.mean(np.array(avg_pos_edges)))
                
        num_nonzeros=[]
        for sc, A_hat in enumerate(A_hat_scales):
            num_nonzeros.append(round((torch.where(A_hat < 0.5)[0].shape[0])/(A_hat.shape[0]**2),2))
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "losses=",torch.round(loss,decimals=4).detach().cpu(), "Non-edges:",num_nonzeros)

    print('best loss:', best_loss)
    
    #model = torch.load('best_model.pt')
    model.eval()
    data_loader = DataLoader(torch.arange(adj.num_edges()), batch_size=args.batch_size,shuffle=True)

    scores, l_scores = torch.zeros(args.d,adj.number_of_nodes()).cuda(),torch.zeros((adj.number_of_nodes(),args.d+1))
    with dataloader.enable_cpu_affinity():
        for in_nodes,sub_graph,block in dataloader:
            if args.debug:
                if iter % 50 == 0: print(iter, '/', len(data_loader), l.item(),loss.item())
            neg_edges = dgl.sampling.global_uniform_negative_sampling(sub_graph, args.batch_size)
            g_batch = adj.subgraph(in_nodes)
            A_hat_scales, X_hat = model(g_batch)
            loss, struct_loss = loss_func(g_batch, A_hat_scales, sub_graph.edges(), neg_edges)
            for sc, A_hat in enumerate(A_hat_scales):
                scores[sc,in_nodes] = struct_loss[sc,:]
    
    detect_anomalies(scores, truth, sc_label, args.dataset)

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
    parser.add_argument('--d', default=4, type=int, help='d parameter for BWGNN filters')
    parser.add_argument('--model', default='multi-scale', type=str, help='encoder to use')
    parser.add_argument('--sample_train', default=False, type=bool, help="whether or not to sample edges in training")
    parser.add_argument('--sample_test', default=False, type=bool, help="whether or not to sample edges in testing")
    parser.add_argument('--debug', default=False, type=bool, help="if true, prints intermediate output")
    args = parser.parse_args()

    graph_anomaly_detection(args)
