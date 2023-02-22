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
        #ret_x = torch.sigmoid(x@x.T)
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


class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, recons):
        super(Encoder, self).__init__()
        out = 8189
        self.gc1 = GraphConvolution(nfeat, nhid, bias=False)
        self.gc2 = GraphConvolution(nhid, nhid, bias=False)
        self.gc2 = GraphConvolution(nhid, out, bias=False)
        self.gc3 = GraphConvolution(nhid, nhid, bias=False)
        self.dropout = dropout
        self.recons = recons
        
    def forward(self, x, adj, w):
        #x = F.relu(self.gc1(x, adj))
        #x = F.leaky_relu(self.gc1(x, w),negative_slope=0.1)
        x = x@w
        #x = self.gc1(x,adj,w)
        #x = F.relu(x)
        x = -F.leaky_relu(-x,negative_slope=0.1)
        
        # added
        #x = F.relu(self.gc3(x, adj))
        #x = F.dropout(x, self.dropout, self.training)
        return x

class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        return x

class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()
        self.training = True
        self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x):#,label):
        if False:
            self.embed_sim(x,label)
        #import ipdb ; ipdb.set_trace()
        x = x @ x.T
        x = torch.sigmoid(x)
        return x

def embed_sim(embeds,label):
    import numpy as np
    #import ipdb ; ipdb.set_trace()
    anom_sc1 = label[0][0]#[0]
    anom_sc2 = label[1][0]#[0]
    anom_sc3 = label[2][0]#[0] 
    anoms_cat = np.concatenate((anom_sc1,np.concatenate((anom_sc2,anom_sc3),axis=None)),axis=None) 
    all_anom = [anom_sc1,anom_sc2,anom_sc3]
    '''
    for h in all_h:
        if not self.training:
            embeds = h
            import numpy as np
            #import ipdb ; ipdb.set_trace()
            anom_sc1 = label[0][0]#[0]
            anom_sc2 = label[1][0]#[0]
            anom_sc3 = label[2][0]#[0] 
            anoms_cat = np.concatenate((anom_sc1,np.concatenate((anom_sc2,anom_sc3),axis=None)),axis=None) 
            all_anom = [anom_sc3,anom_sc2,anom_sc3]
            # get max embedding diff for normalization
            max_diff = 0
            #import ipdb ; ipdb.set_trace()
            for ind,embed in enumerate(embeds):
                for ind_,embed_ in enumerate(embeds):
                    if ind_>= ind:
                        break
                    max_diff = torch.norm(embed-embed_) if torch.norm(embed-embed_) > max_diff else max_diff
            #import ipdb ; ipdb.set_trace()
            # get anom embeds differences
            all_anom_diffs = []
            for anoms in all_anom:
                try:
                    anoms_embs = embeds[anoms]
                except:
                    import ipdb ; ipdb.set_trace()
                anom_diffs = []
                for ind,embed in enumerate(anoms_embs):
                    for ind_,embed_ in enumerate(anoms_embs):
                        #if len(anom_diffs) == len(anoms): continue
                        if ind_ >= ind: break
                        anom_diffs.append(torch.norm(embed-embed_)/max_diff)
                all_anom_diffs.append(anom_diffs)
            
            # TODO: find normal clusters
            # get normal embeds differences
            normal_diffs = []
            for ind,embed in enumerate(embeds):
                if ind in anoms_cat: continue
                if len(normal_diffs) == len(all_anom_diffs):
                    break
                for ind_,embed_ in enumerate(embeds):
                    if ind_ >= ind: break
                    if ind_ in anoms_cat: continue
                    normal_diffs.append(torch.norm(embed-embed_)/max_diff)
            
        # TODO: only get connected node embeddings?
            # get normal vs anom embeds differences
            all_norm_anom_diffs = []
            for anoms in all_anom:
                norm_anom_diffs=[]
                for ind, embed in enumerate(embeds):
                    if ind in anoms_cat: continue
                    for ind_,anom in enumerate(embeds[anoms]):
                        #if len(norm_anom_diffs) == len(anoms): continue 
                        norm_anom_diffs.append(torch.norm(embed-anom)/max_diff)
                all_norm_anom_diffs.append(norm_anom_diffs)


            print('normal-normal',sum(normal_diffs)/len(normal_diffs))
            print('anom-anom',sum(all_anom_diffs[0])/len(all_anom_diffs[0]),sum(all_anom_diffs[1])/len(all_anom_diffs[1]),sum(all_anom_diffs[2])/len(all_anom_diffs[2])) 
            print('anom-normal',sum(all_norm_anom_diffs[0])/len(all_norm_anom_diffs[0]),sum(all_norm_anom_diffs[1])/len(all_norm_anom_diffs[1]),sum(all_norm_anom_diffs[2])/len(all_norm_anom_diffs[2]))
            #import ipdb ; ipdb.set_trace()
            print('----')
    '''
    print((sum(all_anom_diffs[0])/len(all_anom_diffs[0])).item(),(sum(all_anom_diffs[1])/len(all_anom_diffs[1])).item(),(sum(all_anom_diffs[2])/len(all_anom_diffs[2])).item()) 

class Dominant(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout, recons):
        super(Dominant, self).__init__()
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout, recons)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)
        self.recons = recons
    def forward(self, x, adj, w_feat):#, label):
        # encode
        
        x = self.shared_encoder(x, adj, w_feat)
        x = torch.sigmoid(x@x.T)
        return x, x

def glorot_init(in_size, out_size):
    import numpy as np
    import math
    stdv = 1. / math.sqrt(in_size)
    #init_range = np.sqrt(6.0/(in_size+out_size))
    initial = torch.rand(in_size, out_size)*(2*stdv)
    resh_initial = initial[None, :, :]
    return resh_initial.cuda()

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
