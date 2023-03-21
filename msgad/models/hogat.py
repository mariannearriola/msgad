import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import random
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import numpy as np
from torch_geometric.nn import GATConv

import dgl
import torch

from dgl.data import CoraGraphDataset

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj, graph):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        #edge = adj.coalesce().indices()
        #edge = adj.nonzero().t()
        edge = graph.edges()
        edge = torch.vstack((edge[0],edge[1]))

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        # TODO: issue, contains 0
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        import ipdb ; ipdb.set_trace()
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class HOGAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super().__init__()
        self.dropout = dropout
        self.attention_1 = GATConv(nfeat, nhid*2)
        self.attention_2 = GATConv(nhid*2, nhid)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        '''
        self.attention_1 = SpGraphAttentionLayer(nfeat, 
                                                 nhid*2, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True)
        self.attention_2 = SpGraphAttentionLayer(nhid*2, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True)
        '''
        self.linear = nn.Linear(nhid, nfeat)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
        self.sigmoid = nn.Sigmoid()
    
    def generate_rw_subgraph(self, graph_, adj, nb_nodes, subgraph_size):
        """Generate subgraph with random walk algorithm."""
        #src, dst = adj.coalesce().indices()
        #src,dst=torch.nonzero(adj[0]).T
        src,dst=adj.coalesce().indices()
        graph = dgl.graph((src, dst)).to(graph_.device)
        all_idx = torch.tensor(list(range(nb_nodes)))

        traces = dgl.sampling.random_walk(graph, all_idx.to(graph_.device), length=3)
        subv = traces[0]#.tolist()
        return subv

    def get_motif_from_adj(self):
        '''
        FOR EACH NODE, get connected nodes that are greater and add these nodes to the motif
        changed to random walk for efficiency...
        '''
        motif_list = []
        for node0 in range(len(self.adj)):
            for node1 in self.adj[node0]._indices()[0]:
                if node1 <= node0:
                    continue
                for node2 in self.adj[node1]._indices()[0]:
                    if node2 <=node0 or node2 <= node1:
                        continue
                    if node2 in self.adj[node0]._indices()[0]:
                        motif_list.append([node0, int(node1), int(node2)])
        return motif_list
    
    def get_augmented_features(self):
        '''Add mean motif features to feature matrix'''
        motif_features = []
        augmented_features = self.features
        return torch.cat([self.features,torch.mean(self.features[self.motif_list],dim=1)])
        '''
        for motif in self.motif_list:
            motif_feat = torch.mean(self.features[motif],dim=0)
            #motif_feat = torch.mean(torch.stack([self.features[motif[0]], self.features[motif[1]], self.features[motif[2]]]), dim = 0)
            augmented_features=torch.cat((augmented_features,motif_feat.unsqueeze(0)))
        #motif_features = torch.stack(motif_features)
        #augmented_features = torch.cat([self.features, motif_features])
        return augmented_features
        '''

    def get_augmented_adj(self):
        '''For each motif node, '''
        motif_edges = torch.vstack((self.graph.edges()[0],self.graph.edges()[1]))
        motif_src_nodes = []
        motif_dst_nodes = []
        #idx = self.adj.coalesce().indices()
        #for node_index, motif in zip(self.motif_index, self.motif_list):
        
        for node_index,motif in zip(self.motif_index,self.motif_list):
            motif_nodes = torch.unique(motif)
            in_e,out_e=dgl.sampling.sample_neighbors(self.graph_,motif_nodes,-1).edges()
            all_e = torch.vstack((in_e,out_e)).T
            for eid,edge in enumerate(all_e):
                if edge[0] in motif_nodes:
                    all_e[eid][0] = node_index
                if edge[1] in motif_nodes:
                    all_e[eid][1] = node_index
            all_e = all_e.T
            all_e_sym = torch.vstack((all_e[1],all_e[0]))
            all_e = torch.cat((all_e,all_e_sym),dim=1).unique(dim=1)
            
            motif_edges = torch.cat((motif_edges,all_e),dim=1)
            '''
            neighbor_set = set()
            for node in motif:
                for neighbor in self.adj[node]._indices()[0]:
                    neighbor_set.add(int(neighbor))
                neighbor_set.add(node)
            import ipdb ; ipdb.set_trace()
            motif_edge = torch.tensor((list(neighbor_set),[node_index]*len(neighbor_set))).to(self.graph.device)
        
            motif_edges = torch.hstack((motif_edges,motif_edge))
            #motif_dst_nodes.append(torch.tensor(list(neighbor_set)))
            #motif_src_nodes.append(torch.tensor([node_index]*len(neighbor_set)))
        #motif_src_nodes = torch.cat(motif_src_nodes).to(graph.device)
        #motif_dst_nodes = torch.cat(motif_dst_nodes).to(graph.device)
        '''
        #src_nodes = torch.cat([self.graph.edges()[0], motif_src_nodes, motif_dst_nodes])
        #dst_nodes = torch.cat([self.graph.edges()[1], motif_dst_nodes, motif_src_nodes])

        augmented_num_edges = self.num_edges + motif_edges.shape[-1]
        #augmented_num_edges = self.num_edges + 2*len(motif_src_nodes)
        #w = torch.ones(augmented_num_edges).to(graph.device)
        #adj = coo_matrix((w, (src_nodes, dst_nodes)),
        #                 shape=(self.num_nodes, self.num_nodes))
        
        return augmented_num_edges, motif_edges
        return augmented_num_edges, torch.vstack((src_nodes,dst_nodes))
        #adj = torch.sparse_coo_tensor(torch.vstack((src_nodes,dst_nodes)),w)
        #return augmented_num_edges, adj

    def forward(self, x, graph):
        self.graph,self.adj,self.features = graph.to(x.device),graph.adjacency_matrix().to(graph.device),x
        self.num_edges = graph.num_edges()
        self.num_nodes = graph.number_of_dst_nodes()
        self.adj = graph.adjacency_matrix()
        self.adj=self.adj.sparse_resize_((graph.num_src_nodes(),graph.num_src_nodes()), self.adj.sparse_dim(),self.adj.dense_dim())
        u,v = self.adj.coalesce().indices()
        self.graph_ = dgl.graph((u,v)).to(self.graph.device)
        #self.motif_list = self.get_motif_from_adj()
        subgraph_size = 4
        subgraphs = int(self.graph_.number_of_nodes()/subgraph_size)

        self.motif_list = self.generate_rw_subgraph(self.graph,self.adj,subgraphs,subgraph_size-1)

        self.motif_index = list(range(self.num_nodes, self.num_nodes + len(self.motif_list)))
        self.num_motifs = len(self.motif_list)
        self.motif_start_id = self.num_nodes
        self.num_nodes = self.num_nodes + self.num_motifs
        self.features = self.get_augmented_features() 
        self.num_edges, self.adj = self.get_augmented_adj()

        
        x, adj, graph = self.features, self.adj, self.graph
        
        x = self.attention_1(x, adj)
        x = self.attention_2(x, adj)

        #x = self.attention_1(x, adj, graph)
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = self.attention_2(x, adj, graph)
        rec_adj = self.sigmoid(torch.matmul(x, x.T))
        rec_feature = self.sigmoid(self.linear(x))
        #x = F.elu(self.out_att(x, adj))
        return rec_feature, rec_adj