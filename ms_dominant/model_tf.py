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
            #self.conv.append(PolyConv(in_feats, h_feats, d+1, i))
            self.conv.append(PolyConv(h_feats, h_feats, d+1, i, self.thetas[i]))
        self.linear = nn.Linear(in_feats, h_feats)
        #self.linear2 = nn.Linear(h_feats, h_feats)
        #import ipdb ; ipdb.set_trace()
        self.linear3 = nn.Linear(h_feats*len(self.conv),h_feats*len(self.conv))
        
        #self.act = nn.ReLU()
        self.act = nn.LeakyReLU()#negative_slope=0.01)
        self.d = d
        #import ipdb ; ipdb.set_trace()
    def forward(self, adj):
        in_feat = adj.ndata['feature']
        h = self.linear(in_feat)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0]).cuda()
        all_h = []
        for ind,conv in enumerate(self.conv):
            h0 = conv(adj, h)
            if ind == 0:
                all_h = h0
            else:
                all_h = torch.cat((all_h,h0),dim=1)
        return all_h
        '''
        all_h = all_h.reshape(len(self.conv),2708,1433)
        for ind,h in enumerate(all_h):
            if ind == 0:
                reverse_h = h.unsqueeze(1)
            else:
                reverse_h = torch.cat((reverse_h,h.unsqueeze(1)),dim=1)
        final = self.gru(reverse_h)[0]
        for ind,h in enumerate(final):
            if ind == 0:
                reverse_final = h.unsqueeze(1)
            else:
                reverse_final = torch.cat((reverse_final,h.unsqueeze(1)),dim=1) 
            
        return self.act(reverse_final)
        # h_final = self.act(self.linear3(h_final))
        
        #for ind, conv in enumerate(self.conv):
        #    all_h.append(
        #import ipdb ; ipdb.set_trace()
        #return [-self.act(-self.linear3(h_final))]
        #import ipdb ; ipdb.set_trace()
        '''
        '''
        all_h = all_h.reshape(len(self.conv),2708,1433)
        h_final = self.gru(all_h)[0]
        #import ipdb ; ipdb.set_trace()
        '''
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
        # dot product between nodes in the same cluster
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

        return all_h 
        return h

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
        '''
        Input:
            feat: transformed features
            adj: normalized adjacency matrix
        '''
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
    eval_max=1.4
    x = sympy.symbols('x')
    offset=2
    for i in range(offset,d+1+offset,1):
        f = sympy.poly((x/eval_max) ** i * (1 - x/eval_max) ** (d-i+offset) / (eval_max*scipy.special.beta(i+1, d+1-i+offset)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(0,d+1+offset,1):
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
    #def forward(self, x, adj):
    def forward(self, x, adj, w):#,bias,adj_bool):
        #x = F.relu(self.gc1(x, adj))
        #x = F.leaky_relu(self.gc1(x, w),negative_slope=0.1)
        x = x@w
        #import ipdb ; ipdb.set_trace()
        # TODO: use for features
        #x = self.gc1(x,adj,w)#,bias.expand(adj.shape[0],-1),adj_bool)
        #x = F.relu(x)
        x = -F.leaky_relu(-x,negative_slope=0.1)
        #x = self.gc2(x,adj,w2)
        #x = F.relu(x)
        #x = self.gc1(x, w)
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc2(x, adj))
        #import ipdb ; ipdb.set_trace() 

        '''
        if self.recons == 'struct':
            x = x@x.T
            x = torch.sigmoid(x)
        '''
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

    #def forward(self, x, adj):
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
    # get max embedding diff for normalization
    '''
    max_diff = 0
    #import ipdb ; ipdb.set_trace()
    for ind,embed in enumerate(embeds):
        for ind_,embed_ in enumerate(embeds):
            if ind_>= ind:
                break
            max_diff = torch.norm(embed-embed_) if torch.norm(embed-embed_) > max_diff else max_diff
    '''
    # get anom embeds differences
    all_anom_diffs = []
    for anoms in all_anom:
        anoms_embs = embeds[anoms]
        anom_diffs = []
        for ind,embed in enumerate(anoms_embs):
            for ind_,embed_ in enumerate(anoms_embs):
                #if len(anom_diffs) == len(anoms): continue
                if ind_ >= ind: break
                #anom_diffs.append(torch.norm(embed-embed_)/max_diff)
                anom_diffs.append(embed@embed_.T)
        all_anom_diffs.append(anom_diffs)
    '''
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
    def forward(self, x, adj, w_feat):#, label):#,bias,adj_bool=False):
        # encode
        
        x = self.shared_encoder(x, adj, w_feat)
        #x_hat = self.attr_decoder(x)
        #x=self.struct_decoder(x)#,label)
        x = torch.sigmoid(x@x.T)
        return x, x
        #struct_reconstructed = self.struct_decoder(x)
        
        #return struct_reconstructed, x

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
        #self.struc_weights, self.feat_weights = [], []
        #self.struc_weights=glorot_init(in_size, out_size)
        hidden_size = 128
        self.hidden_size=hidden_size
        #hidden_size = 18
        #import ipdb ; ipdb.set_trace()
        mlp=True
        if mlp:
            self.scales = scales+1
        else:
            self.scales = scales
        if recons == 'struct':
            in_size = 10
            out_size = 128
        elif recons == 'feat':
            in_size = 1433
            out_size = 1433
        self.final_size=64
        #self.bias=torch.nn.Parameter(torch.FloatTensor(hidden_size)[None,None,:])
        #self.bias2=torch.nn.Parameter(torch.FloatTensor(2708,out_size))
        #self.bias.data.fill_(0.0)
        #self.bias2.data.fill_(0.0) 
        #self.struc_weights.requires_grad, self.feat_weights.requires_grad = True, True
        if self.recons == 'feat':
            self.feat_weights=torch.nn.Parameter(glorot_init(in_size, out_size))
            self.feat_weights.requires_grad = True
            '''
            #self.feat_weights2=torch.nn.Parameter(glorot_init(hidden_size, out_size))
            #self.feat_weights2.requires_grad = True
            self.feat_weights_2=torch.nn.Parameter(glorot_init(in_size, out_size))
            self.feat_weights_2.requires_grad = True
            self.feat_weights_3=torch.nn.Parameter(glorot_init(in_size, out_size))
            self.feat_weights_3.requires_grad = True
            self.feat_weights_4=torch.nn.Parameter(glorot_init(in_size, out_size))
            self.feat_weights_4.requires_grad = True
            '''
        '''
        elif self.recons == 'struct':
            self.feat_weights=torch.nn.Parameter(glorot_init(in_size, out_size))
            self.feat_weights.requires_grad = True
            
            self.feat_weights_2=torch.nn.Parameter(glorot_init(in_size, out_size))
            self.feat_weights_2.requires_grad = True
            self.feat_weights_3=torch.nn.Parameter(glorot_init(in_size, out_size))
            self.feat_weights_3.requires_grad = True
            #self.feat_weights_extra=torch.nn.Parameter(glorot_init(in_size, out_size))
            #self.feat_weights_extra.requires_grad = True
        
        #self.bias.requires_grad = True
        if scales > 1:
            #pass
            #self.bias2.requires_grad = True
            #self.reset_parameters()
            #import ipdb ; ipdb.set_trace()
            #self.struc_gru = torch.nn.GRU(input_size=self.in_size, hidden_size=self.out_size, num_layers=scales)
            if self.recons == 'feat':
                self.feat_gru_1 = torch.nn.GRU(input_size=in_size, hidden_size=out_size, num_layers=1)
                self.feat_gru_2 = torch.nn.GRU(input_size=in_size, hidden_size=out_size, num_layers=1)
            else:
                self.feat_gru_1 = torch.nn.GRU(input_size=out_size, hidden_size=out_size, num_layers=1)
                self.feat_gru_2 = torch.nn.GRU(input_size=out_size, hidden_size=out_size, num_layers=1)
                self.feat_gru_3 = torch.nn.GRU(input_size=out_size, hidden_size=out_size, num_layers=self.scales)
                
            #self.feat_gru2 = torch.nn.GRU(input_size=self.out_size, hidden_size=self.in_size, num_layers=scales)
            #self.bias_gru = torch.nn.GRU(input_size=self.out_size, hidden_size=self.out_size, num_layers=scales)
            for param in self.feat_gru_1.parameters():
                param.requires_grad = True
            for param in self.feat_gru_2.parameters():
                param.requires_grad = True
            #for param in self.feat_gru_3.parameters():
            #    param.requires_grad = True
            
            #for param in self.bias_gru.parameters():
            #    param.requires_grad = True
        '''
        #self.conv = Dominant(in_size, hidden_size, 0.3, recons)
        #self.linear = torch.nn.Linear(2048, 256)
        #self.linear = torch.nn.Linear(self.scales*hidden_size, self.out_size)
        #self.conv2 = Dominant(hidden_size, out_size, 0.3, recons)
        #self.conv3 = Dominant(in_size,out_size,0.3,recons) 
        self.linear = torch.nn.Linear(hidden_size*(self.d+1),hidden_size)#*(self.d+1))
        self.conv = BWGNN(in_size, hidden_size, out_size, d=self.d)
        #self.conv2 = BWGNN(in_size, hidden_size, out_size,d=4)
        #self.conv3 = BWGNN(in_size, hidden_size, out_size,d=6)
        #self.conv4 = BWGNN(in_size, hidden_size, out_size,d=5) 
        #self.conv5 = BWGNN(in_size, hidden_size, out_size,d=6)
        #self.gru = torch.nn.GRU(input_size=hidden_size,hidden_size=64,num_layers=3)
        self.act = nn.LeakyReLU()
        #for param in self.gru.parameters():
        #    param.requires_grad = True
    '''
    def reset_parameters(self):
        import math
        stdv = 1. / math.sqrt(self.weights[0].size(1))
        for weight_ind in range(len(self.weights)):
            self.weights[weight_ind].data.uniform_(-stdv, stdv)
    '''
    def forward(self,adj,label):
        # updating first set of weights?
        A_hat_scales = []
        #_, w_out_struc = self.struc_gru(self.struc_weights)
        #import ipdb ; ipdb.set_trace()
        '''
        if self.scales > 1:
            _, w_out_feat = self.feat_gru(self.feat_weights)
            #_, w_out_bias = self.bias_gru(self.bias)
        '''
        
        #import ipdb ; ipdb.set_trace()
        w_out_feat,w_out_feat_2,w_out_feat_3=None,None,None
    
        #import ipdb ; ipdb.set_trace()
        #A_hat_found = [self.conv(x[0],adj)]#,self.conv2(x[0],adj)]#,self.conv3(x[0],adj)]#,self.conv4(x[0],adj),self.conv5(x[0],adj)]
        A_hat_scales = self.conv(adj)
        '''
        shapes=[]
        for ind,A_hat in enumerate(A_hat_found):
            if self.recons == 'struct':
                #A_hat_scales.append(torch.sigmoid(A_hat@A_hat.T))
                
                if ind == 0:
                    A_hat_scales = A_hat#.unsqueeze(0)
                    shapes.append(A_hat.shape[1])
                else:
                    #import ipdb ; ipdb.set_trace()
                    A_hat_scales = torch.cat((A_hat_scales,A_hat),dim=1)
                    shapes.append(A_hat.shape[1])
            elif self.recons == 'feat':
                A_hat_scales.append(A_hat)
        '''
        # TODO: need to batch for reconstruction
        A_hat_ret = self.linear(self.act(A_hat_scales))
        sp_size=self.hidden_size#self.final_size
        A_hat_emb = [A_hat_scales[:,:sp_size]]
        for i in range(1,self.d+1):
            A_hat_emb.append(A_hat_scales[:,i*sp_size:(i+1)*sp_size])

        #A_hat_scales = [A_hat_scales[:,:sp_size],A_hat_scales[:,sp_size:2*sp_size],A_hat_scales[:,2*sp_size:3*sp_size],A_hat_scales[:,3*sp_size:4*sp_size]]#,A_hat_scales[:,4*sp_size:]]#,A_hat_scales[:,512:]]
        #A_hat_scales = self.gru(A_hat_scales)[0]
        A_hat_scales = [A_hat_ret]
        A_hat_ret = []
         
        embs = []
        for A_hat in A_hat_scales:
            A_hat_ret.append(torch.sigmoid(A_hat@A_hat.T))
        for A_hat in A_hat_emb:
            embs.append(torch.sigmoid(A_hat@A_hat.T))
        #import ipdb ; ipdb.set_trace()
        return embs,embs
        #return A_hat_ret,embs
        
        #import ipdb ; ipdb.set_trace() 
        #return [A_hat_scales[-1]], [A_hat_scales[-1]]
        '''
        A_hat = self.conv2(A_hat,adj,label)
        if self.recons == 'struct':
            A_hat_scales.append(torch.sigmoid(A_hat@A_hat.T))
        elif self.recons == 'feat':
            A_hat_scales.append(A_hat)
        
        A_hat = self.conv3(A_hat,adj,label)
        if self.recons == 'struct':
            A_hat_scales.append(torch.sigmoid(A_hat@A_hat.T))
        elif self.recons == 'feat':
            A_hat_scales.append(A_hat)
        '''
        ''' 
        A_hat, X_hat = self.conv(x[0],adj,self.feat_weights[0],label)
        if self.recons == 'struct':
            A_hat_scales.append(torch.sigmoid(A_hat@A_hat.T))
        elif self.recons == 'feat':
            A_hat_scales.append(A_hat)
        
        A_hat, X_hat = self.conv(adj@A_hat,adj,self.feat_weights_2[0],label)
        if self.recons == 'struct':
            A_hat_scales.append(torch.sigmoid(A_hat@A_hat.T))
        elif self.recons == 'feat':
            A_hat_scales.append(A_hat)
        A_hat, X_hat = self.conv(adj@A_hat,adj,self.feat_weights_3[0],label)
        if self.recons == 'struct':
            A_hat_scales.append(torch.sigmoid(A_hat@A_hat.T))
        elif self.recons == 'feat':
            A_hat_scales.append(A_hat)
        '''
        '''
        A_hat, X_hat = self.conv(adj@A_hat,adj,self.feat_weights_4[0],label)
        A_hat_scales.append(A_hat)
        '''
        '''
        _, w_out_feat = self.feat_gru_1(self.feat_weights)
        A_hat_scales.append(A_hat)
        A_hat, X_hat = self.conv3(adj@A_hat,adj,w_out_feat[0])
        _, w_out_feat = self.feat_gru_1(w_out_feat,w_out_feat)
        A_hat_scales.append(A_hat)
        A_hat, X_hat = self.conv3(adj@A_hat,adj,w_out_feat[0])
        _, w_out_feat = self.feat_gru_1(w_out_feat,w_out_feat)
        A_hat_scales.append(A_hat)
        A_hat, X_hat = self.conv3(adj@A_hat,adj,w_out_feat[0])
        _, w_out_feat = self.feat_gru_1(w_out_feat,w_out_feat)
        A_hat_scales.append(A_hat)
        ''' 
        '''
        #import ipdb ; ipdb.set_trace() 
        for weight_ind in range(0,self.scales):#-1):
            if weight_ind == 0:
                if self.recons == 'feat':
                    A_hat, X_hat = self.conv3(x[weight_ind], adj, self.feat_weights[0])#,self.bias[0][0],adj_bool=True)#, self.feat_weights)
                    #A_hat, X_hat = self.conv2(A_hat, adj, self.feat_weights2[0])#,self.bias2)
                elif self.recons == 'struct':
                    A_hat, X_hat = self.conv3(x[0], adj, self.feat_weights[0])
                if self.scales > 1:
                    _, w_out_feat = self.feat_gru_1(self.feat_weights)
                #A_hat_final = A_hat
            else:
                A_hat, X_hat = self.conv3(x[0],adj,w_out_feat[0])#,w_out_bias[weight_ind][0])#, w_out_feat[weight_ind][0])
                #A_hat, X_hat = self.conv3(x[0],adj,self.feat_weights_extra[0]) 
                #_, w_out_feat = self.feat_gru_1(w_out_feat,w_out_feat)
                if weight_ind > 1:
                    if w_out_feat_2 is None:
                        #w_out_feat_2 = self.feat_weights_2
                        A_hat, X_hat = self.conv3(x[0],adj,self.feat_weights_2[0])
                        _, w_out_feat_2 = self.feat_gru_2(self.feat_weights_2)
                    else:
                        A_hat, X_hat = self.conv3(x[0],adj,w_out_feat_2[0])
                        _, w_out_feat_2 = self.feat_gru_2(w_out_feat_2,w_out_feat_2)
                if weight_ind > 2:
                    #_, w_out_feat_3 = self.feat_gru_2(w_out_feat_2,w_out_feat_2)
                    A_hat, X_hat = self.conv3(x[0],adj,self.feat_weights_3[0])
            #import ipdb ; ipdb.set_trace()
            #A_hat_scales.append(torch.sigmoid(A_hat@A_hat.T))
            A_hat_scales.append(A_hat)
            #A_hat_final = torch.cat((A_hat_final,A_hat),-1)
            #X_hat_scales.append(X_hat)
        '''
        '''
        all_sc = torch.tensor([]).cuda()
        for i in A_hat_scales:
            all_sc = torch.cat((all_sc,i.unsqueeze(1)),dim=1)
        all_sc = torch.reshape(all_sc,(self.scales,2708,self.out_size))
        new_sc = self.feat_gru_3(all_sc)[1]
        A_hat_scales=[]
        for i in new_sc:
            #embed_sim(i,label)
            #print('---')
            A_hat_scales.append(torch.sigmoid(i@i.T))
        '''
        '''
        new_sc = []
        for i in A_hat_scales:
            embed_sim(i,label)
            #print('---')
            new_sc.append(torch.sigmoid(i@i.T))
        #import ipdb ; ipdb.set_trace()
        return new_sc,new_sc
        #import ipdb ; ipdb.set_trace()
        #A_hat_final = F.relu(self.linear(A_hat_final))
        '''
        #A_hat_scales=[self.conv(x[0],adj),self.conv2(x[0],adj),self.conv3(x[0],adj)]
        # ???
        #for weight_ind in range(len(self.weights)):
        #    self.weights[weight_ind] = self.weights[weight_ind].detach()
        return A_hat_scales, A_hat_scales
