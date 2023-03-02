import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import torch
import sympy
import scipy
import scipy.sparse as sp
import networkx as nx
from layers import GraphConvolution
import numpy as np
import torch_geometric.nn.conv as conv
from torch_geometric.nn import MLP
from torch_geometric.nn.conv import GATConv, GCNConv, APPNP, MessagePassing

class SimpleGNN(nn.Module):
    def __init__(self,in_dim,embed_dim,model_str,recons,hops=2,dropout=0.2,act=nn.LeakyReLU()):
        super(SimpleGNN, self).__init__()
        self.model_str = model_str
        self.recons = recons
        self.dense=MLP(in_channels=in_dim,hidden_channels=embed_dim*2,out_channels=embed_dim,num_layers=1)
        self.linear = False
        self.decoder, self.encoder_act, self.decoder_mlp = None,None,None

        if model_str == 'anomalydae':
            self.linear = True
            self.encoder=GATConv(embed_dim, embed_dim)
            if self.recons == 'feat':
                self.decoder=GATConv(embed_dim,in_dim)
        elif model_str == 'dominant':
            self.linear = True
            self.encoder=GCNConv(embed_dim, embed_dim)
            if self.recons == 'feat':
                self.decoder=GCNConv(embed_dim,in_dim)
            self.encoder_act = act
        elif model_str == 'appnp':
            self.linear = True
            self.encoder=APPNP(hops,0.1)
        elif model_str == 'mlpae':
            self.linear=True 
            self.encoder=None
            self.dense=MLP(in_channels=in_dim,hidden_channels=embed_dim,out_channels=in_dim,num_layers=3)
        elif model_str == 'adone':
            self.linear = False ; self.encoder = MessagePassing()
            self.decoder = MLP(in_channels=in_dim,hidden_channels=embed_dim,out_channels=in_dim,num_layers=3)
            self.encoder_act = act
        elif model_str == 'gaan':
            self.linear = False
            self.dense=MLP(in_channels=in_dim,hidden_channels=embed_dim,out_channels=in_dim,num_layers=3,act_first=True)
            self.decoder_linear=nn.Linear(1, 1)

        self.dropout = dropout
        self.act = act
        
    def forward(self,graph):
        x = graph.ndata['feature']
        if self.linear:
            x = self.act(self.dense(x))
        edge_list = torch.vstack((graph.edges()[0],graph.edges()[1]))

        if self.encoder:
            x = self.encoder(x,edge_list)

        if self.encoder_act:
            x = self.encoder_act(x)
        
        if self.decoder:
            recons = self.decoder(x)
        else:
            recons = torch.sigmoid(torch.sparse.mm(x.to_sparse(),torch.transpose(x.to_sparse(),0,1)).to_dense())
        
        if self.model_str == 'gaan':
            edge_prob = torch.reshape(recons[edge_list[0], edge_list[1]],
                                  [edge_list.shape[1], 1])
            recons = torch.sigmoid(self.decoder_linear(edge_prob))

        return recons

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, d=4):
        super(BWGNN, self).__init__()
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(h_feats, h_feats, d+1, i, self.thetas[i]))
        self.linear = nn.Linear(in_feats, h_feats)
        #self.linear2 = nn.Linear(h_feats*2, h_feats)
        self.act = nn.LeakyReLU()#negative_slope=0.01)
        self.d = d
        
    def forward(self, graph):
        in_feat = graph.ndata['feature']
        h = self.linear(in_feat)
        h = self.act(h)
        #h = self.linear2(h)
        for ind,conv in enumerate(self.conv):
            h0 = conv(dgl.add_self_loop(graph), h)
            if ind == 0:
                all_h = h0
            else:
                all_h = torch.cat((all_h,h0),dim=1)
        x = all_h
        recons = torch.sigmoid(torch.sparse.mm(x.to_sparse(),torch.transpose(x.to_sparse(),0,1)).to_dense())
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
            adj = normalize_adj(graph.adjacency_matrix().to_dense()).todense()
            adj = torch.FloatTensor(adj).cuda()
            h = self._theta[0]*feat#*(adj@feat)
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
        # NOTE: DIFFERENCE IS THAT D NOT ADDED BY 1
        for i in range(0,d+offset,1):
            inv_coeff.append(float(coeff[d-i+offset]))
        thetas.append(inv_coeff)
    return thetas
    
class GraphReconstruction(nn.Module):
    def __init__(self, in_size, hidden_size, scales, recons, d, model_str):
        super(GraphReconstruction, self).__init__()
        self.in_size = in_size
        out_size = in_size # for reconstruction
        self.d = d
        self.recons = recons
        self.model_str = model_str
        if model_str == 'multi_scale' or model_str == 'multi-scale':
            '''
            self.conv = BWGNN(in_size, hidden_size, out_size, d=self.d+1)
            self.conv2 = BWGNN(in_size, hidden_size, out_size, d=self.d+2)
            self.conv3 = BWGNN(in_size, hidden_size, out_size, d=self.d+3)
            '''
            self.conv = SimpleGNN(in_size,hidden_size,'dominant',recons,hops=5)
            self.conv2 = SimpleGNN(in_size,hidden_size,'dominant',recons,hops=10)
            self.conv3 = SimpleGNN(in_size,hidden_size,'dominant',recons,hops=15)
        else:
            self.conv = SimpleGNN(in_size,hidden_size,model_str,recons)
        self.act = nn.LeakyReLU()

    def forward(self,graph):
        '''
        Input:
            graph: input dgl graph
        Output:
            recons: scale-wise adjacency reconstructions
        '''
        '''
        # NOTE: BWGNN
        A_hat_scales = self.conv(graph)
        #A_hat_scales = self.linear(self.act(A_hat_scales))
        sp_size=self.hidden_size
        A_hat_emb = [A_hat_scales[:,:sp_size].to_sparse()]
        for i in range(1,self.d):
            A_hat_emb.append(A_hat_scales[:,i*sp_size:(i+1)*sp_size].to_sparse())
        '''
        if self.model_str == 'multi_scale' or self.model_str == 'multi-scale':
            recons = [self.conv(graph),self.conv2(graph),self.conv3(graph)]
        else:
            recons = [self.conv(graph)]
        return recons,recons