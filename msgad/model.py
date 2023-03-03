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
from models.dominant import *

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
        self.linear2 = nn.Linear(h_feats*d, h_feats)
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
        x = self.linear2(all_h)
        recons = torch.sparse.mm(x.to_sparse(),torch.transpose(x.to_sparse(),0,1)).to_dense()
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
            adj = normalize_adj(graph.adjacency_matrix().to_dense()).todense()
            adj = torch.FloatTensor(adj).cuda()
            h = self._theta[0]*(adj@feat)
            for k in range(1, self._k):
                #feat = unnLaplacian(feat, D_invsqrt, graph)
                feat = adj@feat
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
    def __init__(self, in_size, hidden_size, batch_size, scales, recons, d, model_str, act = nn.LeakyReLU()):
        super(GraphReconstruction, self).__init__()
        self.in_size = in_size
        out_size = in_size # for reconstruction
        self.d = d
        self.recons = recons
        self.model_str = model_str
        dropout = 0
        self.embed_act = act
        self.decode_act = torch.sigmoid
        self.weight_decay = 0.01
        if model_str == 'multi_scale' or model_str == 'multi-scale':
            self.conv = BWGNN(in_size, hidden_size, out_size, d=self.d)
            #self.conv2 = BWGNN(in_size, hidden_size, out_size, d=self.d+2)
            #self.conv3 = BWGNN(in_size, hidden_size, out_size, d=self.d+3)
            '''
            self.conv = SimpleGNN(in_size,hidden_size,'dominant',recons,hops=5)
            self.conv2 = SimpleGNN(in_size,hidden_size,'dominant',recons,hops=10)
            self.conv3 = SimpleGNN(in_size,hidden_size,'dominant',recons,hops=15)
            '''
        elif model_str == 'adone': # x, s, e
            self.conv = AdONE_Base(in_size,batch_size,hidden_size,4,dropout,act)
        elif model_str == 'anomalous': # x
            raise NotImplementedError
            w_init = torch.randn_like(torch.tensor(in_size,batch_size))
            r_init = torch.inverse((1 + self.weight_decay)
                * torch.eye(batch_size).cuda() + self.gamma * l) @ x #TODO?
            self.conv = ANOMALOUS_Base(w_init,r_init)
        elif model_str in ['anomalydae','anomaly_dae']: # x, e, batch_size (0 for no batching)
            self.conv = AnomalyDAE_Base(in_size,batch_size,hidden_size,out_size,dropout,act)
        elif model_str == 'dominant':
            self.conv = DOMINANT_Base(in_size,hidden_size,3,dropout,act)
        elif model_str == 'done': # x, s, e
            self.conv = DONE_Base(in_size,batch_size,hidden_size,4,dropout,act)
        elif model_str == 'gaan': # x, noise, e
            self.conv = GAAN_Base(in_size,16,2,2,dropout,act)
        elif model_str == 'gcnae': # x, e
            self.conv = GCN(in_size,hidden_size,2,batch_size,dropout,act)
        elif model_str == 'guide': # x, s, e
            self.conv = GUIDE_Base(in_size,batch_size,32,4,4,dropout,act)
        elif model_str == 'mlpae': # x
            self.conv = MLP(in_channels=in_size,hidden_channels=hidden_size,out_channels=batch_size,num_layers=3)
        elif model_str == 'ogcnn': # x, s, e
            self.conv = GCN_base(in_size,hidden_size,4,dropout,act)
        elif model_str == 'radar': # x
            raise NotImplementedError
            w_init = torch.randn_like(torch.tensor(in_size,batch_size))
            r_init = torch.inverse((1 + self.weight_decay)
                * torch.eye(x.shape[0]).to(self.device) + self.gamma * l) @ x #TODO?
            self.conv = Radar_Base(w_init,r_init)
        elif model_str == 'conad': # x, e
            self.conv = CONAD_Base(in_size,hidden_size,4,dropout,act)
        else:
            raise('model not found')
            #self.conv = SimpleGNN(in_size,hidden_size,model_str,recons)

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
        edges = torch.vstack((graph.edges()[0],graph.edges()[1]))
        if self.model_str in ['adone','done','guide','ogcnn']: # x, s, e
            recons = [self.conv(graph.ndata['feature'], graph.adjacency_matrix().to_dense(), edges)]
        elif self.model_str in ['anomalous','mlpae','radar']: # x
            recons = [self.conv(graph.ndata['feature'])]
            if self.model_str == 'mlpae':
                recons = [self.decode_act(recons[0])]
        elif self.model_str in ['conad','gcnae','dominant']: #x, e
            recons = [self.conv(graph.ndata['feature'], edges)]
            if self.model_str == 'dominant':
                recons_ind = 0 if self.recons == 'feat' else 1
                recons = [self.conv(graph.ndata['feature'], edges)[recons_ind]]
        elif self.model_str in ['gaan']: # x, noise, e
            gaussian_noise = torch.randn(graph.number_of_nodes(), self.noise_dim).cuda()
            recons = [self.conv(graph.ndata['feature'], gaussian_noise, edges)]
        elif self.model_str in ['anomaly_dae','anomalydae']: #x, e, batch_size
            recons = [self.conv(graph.ndata['feature'], edges,0)]
        elif self.model_str in ['multi_scale','multi-scale']: # g
            recons = [self.conv(graph)]#,self.conv2(graph),self.conv3(graph)]
        return recons