import torch.nn as nn
import torch.nn.functional as F
import torch
import sympy
import scipy
import networkx as nx
from layers import GraphConvolution
import numpy as np
from torch_geometric.nn import GATConv

class AnomalyDAE(nn.Module):
    def __init__(self,in_node_dim,embed_dim,out_dim,dropout=0.2,act=nn.LeakyReLU()):
        super(AnomalyDAE, self).__init__()
        self.dense=nn.Linear(in_node_dim,embed_dim)
        self.attention_layer=GATConv(embed_dim, out_dim)
        self.dropout = dropout
        self.act = act

    def forward(self,x,adj):
        x = self.act(self.dense(x))
        #x = self.dense(x)
        #x = F.dropout(x,self.dropout)
        x = self.attention_layer(x,adj.nonzero().t().contiguous())
        embed_x = x
        return [embed_x],[embed_x]


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, d=4):
        super(BWGNN, self).__init__()
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(h_feats, h_feats, d+1, i, self.thetas[i]))
        self.linear = nn.Linear(in_feats, h_feats)
        #self.linear2 = nn.Linear(h_feats, h_feats)
        #self.linear3 = nn.Linear(h_feats*len(self.conv),h_feats*len(self.conv))
        #self.linear4 = nn.Linear(h_feats, 64)
        
        #self.act = nn.ReLU()
        self.act = nn.LeakyReLU()#negative_slope=0.01)
        self.d = d
        
    def forward(self, in_feat, adj):
        def unnLaplacian(adj):
            G = nx.from_numpy_matrix(adj.detach().cpu().numpy())
            l = torch.tensor(nx.laplacian_matrix(G).toarray()).cuda()
            return l
        
        #lapl = unnLaplacian(adj).float().cuda()
        h = self.linear(in_feat)
        h = self.act(h)
        #h = self.linear2(h)
        #h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0]).cuda()
        all_h = []
        for ind,conv in enumerate(self.conv):
            h0 = conv(adj, h, h)
            if ind == 0:
                all_h = h0
            else:
                all_h = torch.cat((all_h,h0),dim=1)
        #all_h = self.linear3(all_h)        
                
        return all_h,h
        
        h_final = self.linear3(all_h)
        #h_final = self.act(h_final)
        #h_final = self.linear4(h_final)
        return h_final
        '''  
        #import ipdb ; ipdb.set_trace()
        # h
        #import ipdb ; ipdb.set_trace()
        h = self.linear3(h_final)
        #h = self.act(h)
        #h = self.linear4(h_final[0])
        #h = self.act(h_final[0])
        h = self.act(self.linear4(h))
        # TODO: remove if not reconstructing adj
        #h = torch.sigmoid(self.linear4(h))
        '''

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
        self.theta = theta

    def forward(self, adj, in_feat, lapl):
        '''
        Input:
            feat: transformed features
            adj: normalized adjacency matrix
        '''
       
        #l = unnLaplacian(adj)
        #w = (l/2)**self._i * (torch.eye(l.shape[0]).cuda()-l/2)**(self._d-self._i)
        #w /= scipy.special.beta(self._i+1,self._d+1-self._i)
        
        feat=in_feat
        h = self.theta[0]*(adj@feat)
        for theta in self.theta[1:]:
            '''   
            #for ind,theta in enumerate(self.theta):    
            p,q=self._i,self._d-self._i-1
            #print(p,q)
            #import ipdb ; ipdb.set_trace()
            #beta_cons = np.math.factorial(p+1)*(np.math.factorial(p+1)/np.math.factorial(p+q+1+2))
            beta_cons = scipy.special.beta(p+1,q+1)
            out = (((lapl/2)**p)@(torch.eye(lapl.shape[0]).cuda()-(lapl/2))**q)
            out /= 2*beta_cons
            out = out @ feat
            h=out
            '''
            
            feat = adj@feat
            h += theta*feat
            
            #h = w.to(torch.float32)@feat
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
        mlp=True
        if mlp:
            self.scales = scales+1
        else:
            self.scales = scales
        if recons == 'struct':
            in_size = 1433
            out_size = 128
        elif recons == 'feat':
            in_size = 1433
            out_size = 1433
        self.final_size=64
        #self.linear = torch.nn.Linear(hidden_size*(self.d+1),hidden_size*2)#*(self.d+1))
        #self.linear2 = torch.nn.Linear(hidden_size*(self.d+1),hidden_size)
        self.conv = BWGNN(in_size, hidden_size, out_size, d=self.d)
        #self.conv = AnomalyDAE(in_size,hidden_size,out_size)
        self.act = nn.LeakyReLU()
    '''
    def reset_parameters(self):
        import math
        stdv = 1. / math.sqrt(self.weights[0].size(1))
        for weight_ind in range(len(self.weights)):
            self.weights[weight_ind].data.uniform_(-stdv, stdv)
    '''
    def forward(self, x, adj,label):
        A_hat_scales = []
        #import ipdb ; ipdb.set_trace
        A_hat_scales,h = self.conv(x[0],adj)
        
        #A_hat_ret = self.linear(self.act(A_hat_scales))
        A_hat_ret = []
        sp_size=self.hidden_size
        A_hat_emb = [A_hat_scales[:,:sp_size]]
        for i in range(1,self.d):
            A_hat_emb.append(A_hat_scales[:,i*sp_size:(i+1)*sp_size])
        
        #A_hat_emb = h
        #import ipdb ; ipdb.set_trace()
        #A_hat_scales = [A_hat_ret]
        A_hat_ret = []
        #A_hat_scales = [self.linear2(A_hat_scales)] 
        recons,embs = [],[]
        for A_hat in A_hat_scales:
            A_hat_ret.append(torch.sigmoid(A_hat@A_hat.T))
        for A_hat in A_hat_emb:
            embs.append(A_hat)
            recons.append(torch.sigmoid(A_hat@A_hat.T))
        #return recons,A_hat_ret
        #return A_hat_ret,A_hat_ret
        return recons, recons
