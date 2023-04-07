import scipy
import scipy.sparse as sp
from torch_geometric.nn.conv import APPNP
import sympy
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
import torch
import numpy as np

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

class BWGNN(nn.Module):
    """
    Beta wavelet autoencoder

    Parameters
    ----------
    in_dim : int
        Input feature dimension
    h_dim: int
        Hidden dimension for feature transformation
    d : int
        Number of filters to use for embedding
    """
    def __init__(self, in_dim, h_dim, d=4):
        super(BWGNN, self).__init__()
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(h_dim, h_dim, d+1, i, self.thetas[i]))
        self.linear = nn.Linear(in_dim, h_dim*2)
        self.linear2 = nn.Linear(h_dim*2, h_dim)
        self.linear3 = nn.Linear(h_dim*(d+1), h_dim)
        self.act = nn.LeakyReLU()
        self.d = d
        
    def forward(self, graph, in_feat, dst_nodes):
        #graph = dgl.block_to_graph(graph)
        #graph.add_edges(graph.dstnodes(),graph.dstnodes())
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        
        for ind,conv in enumerate(self.conv):
            h0 = conv(graph, h)
            if ind == 0:
                all_h = h0
            else:
                all_h = torch.cat((all_h,h0),dim=1)
            
        x = self.linear3(all_h)[dst_nodes]
        x = x@x.T
        #x = torch.mm(x,torch.transpose(x,0,1))
        #x = torch.sparse.mm(x.to_sparse(),torch.transpose(x.to_sparse(),0,1)).to_dense()
        #return x
        return torch.sigmoid(x)

class PolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 d,
                 i,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._d = d
        self._i = i
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self._theta = theta
        self._k = len(self._theta)

    def forward(self, graph, feat):
        def unnLaplacian(h, D_invsqrt, graph):
            h_src = h
            h_dst = h[:graph.number_of_dst_nodes()]
            graph.srcdata['h'] = h_src * D_invsqrt[graph.srcnodes()]
            graph.dstdata['h'] = h_dst * D_invsqrt[graph.dstnodes()]

            # message pass src -> dst
            graph.update_all(fn.copy_u('h','m'), fn.sum('m','h'))

            #graph.dstdata['self'] = h_dst * D_invsqrt[graph.dstnodes()]
            # message pass src -> dst
            #graph.update_all(fn.copy_u('self','m'), fn.sum('m','self'))

            # are srcs updated in message passing? read tutorials
            return h_dst - graph.dstdata.pop('h') * D_invsqrt[graph.dstnodes()]

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.out_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat[graph.dstnodes()]
            for k in range(1, self._k):
                feat[graph.dstnodes()] = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat[graph.dstnodes()]
        return h # return thetas for weighting a**2,3,... # theta[0]*a,theta[1]*a^2,...


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas
