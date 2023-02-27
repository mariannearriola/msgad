import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import torch
import sympy
import scipy
import networkx as nx
from layers import GraphConvolution
import numpy as np

class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, d=4):
        super(BWGNN, self).__init__()
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(h_feats, h_feats, d+1, i, self.thetas[i]))
        self.linear = nn.Linear(in_feats, h_feats)
        self.act = nn.LeakyReLU()#negative_slope=0.01)
        self.d = d
        
    def forward(self, graph):
        in_feat = graph.ndata['feature']
        h = self.linear(in_feat)
        h = self.act(h)
        for ind,conv in enumerate(self.conv):
            h0 = conv(graph, h)
            if ind == 0:
                all_h = h0
            else:
                all_h = torch.cat((all_h,h0),dim=1)
        return all_h

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
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        return h

def calculate_theta2(d):
    thetas = []
    eval_max=2
    x = sympy.symbols('x')
    offset=0
    for i in range(offset,d+offset,1):
        f = sympy.poly((x/eval_max) ** i * (1 - x/eval_max) ** (d-i+offset) / (eval_max*scipy.special.beta(i+1, d+1-i+offset)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(0,d+offset,1):
            inv_coeff.append(float(coeff[d-i+offset]))
        thetas.append(inv_coeff)
    return thetas

class EGCN(nn.Module):
    def __init__(self, in_size, out_size, scales, recons, mlp, d):
        super(EGCN, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.d = d
        self.recons = recons
        hidden_size = 128
        self.hidden_size=hidden_size
        in_size = 1433
        out_size = hidden_size
        self.final_size=64
        self.linear = torch.nn.Linear(hidden_size*(self.d),hidden_size*(self.d))
        self.conv = BWGNN(in_size, hidden_size, out_size, d=self.d)
        self.act = nn.LeakyReLU()

    def forward(self,graph):
        '''
        Input:
            graph: input dgl graph
        Output:
            recons: scale-wise adjacency reconstructions
        '''
        A_hat_scales = self.conv(graph)
        #A_hat_scales = self.linear(self.act(A_hat_scales))
        sp_size=self.hidden_size
        A_hat_emb = [A_hat_scales[:,:sp_size].to_sparse()]
        for i in range(1,self.d):
            A_hat_emb.append(A_hat_scales[:,i*sp_size:(i+1)*sp_size].to_sparse())

        A_hat_ret = []
        embs = []
        for A_hat in A_hat_emb:
            embs.append(torch.sigmoid(torch.sparse.mm(A_hat,torch.transpose(A_hat,0,1)).to_dense()))
        return embs,embs