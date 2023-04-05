import scipy
import scipy.sparse as sp
from torch_geometric.nn.conv import APPNP
import sympy
import torch.nn as nn
import torch.nn.functional as F
import math
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
        for i in range(len(self.thetas)): # exclude high pass
            self.conv.append(PolyConv(h_dim, h_dim, d+1, i, self.thetas[i]))
        self.linear = nn.Linear(in_dim, h_dim)
        self.act = nn.LeakyReLU()
        self.d = d
        self.lam = nn.Parameter(data=torch.normal(mean=torch.full((d,),0.),std=1))#.cuda())#, requires_grad=True).cuda()
        self.relu = torch.nn.ReLU()
        
    def forward(self, graph, in_feat):
        if graph.is_block: graph = dgl.block_to_graph(graph)

        # add self-loop
        graph.add_edges(graph.dstnodes(),graph.dstnodes())

        # feature transformer
        h = self.linear(in_feat)
        h = self.act(h)
        
        # scale-wise embeddings via multi-frequency graph wavelet
        all_h = []
        for ind,conv in enumerate(self.conv):
            h0 = conv(graph, h)
            all_h.append(h0)
    
        # learned linear combo of multi-frequency embeddings
        recons = torch.sigmoid(self.lam[0])*all_h[0]
        for ind,x in enumerate(all_h[1:]):
            recons += torch.sigmoid(self.lam[ind])*x

        # inner-product decoder
        #recons = torch.mm(recons,torch.transpose(recons,0,1))
        recons = recons@recons.T
        #return recons
        return torch.sigmoid(recons)

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

            return h_dst - graph.dstdata.pop('h') * D_invsqrt[graph.dstnodes()]

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.out_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat[graph.dstnodes()]
        
            #n,m=h.shape
            #thr = (math.sqrt(2 * math.log(n)) / math.sqrt(n) * 1) * torch.unsqueeze(torch.tensor(self._theta[0]), dim=0).repeat(m)
            #h = torch.mul(torch.sign(h), (((torch.abs(h) - thr.cuda()) + torch.abs(torch.abs(h) - thr.cuda())) / 2))
            
            # add self loop to dst nodes ?
            for k in range(1, self._k):
                feat[graph.dstnodes()] = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat[graph.dstnodes()]
            # SHRINKAGE
            #n,m=h.shape
            #thr = (math.sqrt(2 * math.log(n)) / math.sqrt(n) * sigma) * torch.unsqueeze(scale, dim=0).repeat(m)
            # self.threshold = (math.sqrt(2*math.log(h.shape)))
            # = torch.mul(torch.sign(x), (((torch.abs(x) - self.threshold) + torch.abs(torch.abs(x) - self.threshold)) / 2))

        return h

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
