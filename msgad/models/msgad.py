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


class MSGAD(nn.Module):
    """
    Multi-scale autoencoder leveraging beta wavelets

    Parameters
    ----------
    in_dim : int
        Input feature dimension
    h_dim: int
        Hidden dimension for feature transformation
    d : int
        Number of scale representations to learn
    """
    def __init__(self, in_dim, h_dim, d=4):
        super(MSGAD, self).__init__()
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(h_dim, h_dim, d+1, i, self.thetas[i]))
        self.linear = nn.Linear(in_dim, h_feats)
        self.act = nn.LeakyReLU()
        self.d = d
        
    def forward(self, graph, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        all_h = []
        for ind,conv in enumerate(self.conv):
            h0 = conv(dgl.add_self_loop(graph), h)
            #h0 = conv(graph, h)
            all_h.append(h0)
        
        recons = []
        for x in all_h:
            recons.append(torch.sparse.mm(x.to_sparse(),torch.transpose(x.to_sparse(),0,1)))
        #recons = torch.sigmoid(prod)
        return recons

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
        def unnLaplacian(feat, D_invsqrt, graph):
            graph.srcdata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h','m'), fn.sum('m','h'))
            return feat - graph.srcdata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.out_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            #adj = normalize_adj(graph.adjacency_matrix().to_dense()).todense()
            #adj = torch.FloatTensor(adj).cuda()
            h = self._theta[0]*feat#(adj@feat)
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                #feat = adj@feat
                h += self._theta[k]*feat
        return h

def calculate_theta2(d):
    thetas = []
    eval_max,offset=2,0
    x = sympy.symbols('x')
    for i in range(offset,d+offset,1):
        f = sympy.poly((x/eval_max) ** i * (1 - x/eval_max) ** (d-i+offset) / (eval_max*scipy.special.beta(i+1, d+1-i+offset)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(0,d+offset,1):
            inv_coeff.append(float(coeff[d-i+offset]))
        thetas.append(inv_coeff)
    return thetas
    