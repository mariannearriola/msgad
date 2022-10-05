import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution

class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, bias=False)
        self.gc2 = GraphConvolution(nhid, nhid, bias=False)
        self.gc3 = GraphConvolution(nhid, nhid, bias=False)
        self.dropout = dropout

    #def forward(self, x, adj):
    def forward(self, x, adj, w,bias):
        #x = F.relu(self.gc1(x, adj))
        #x = F.leaky_relu(self.gc1(x, w),negative_slope=0.1)
        x = self.gc1(x,adj,w,bias)
        #x = self.gc1(x, w)
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc2(x, adj))

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

        self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    #def forward(self, x, adj):
    def forward(self, x):
        #self.embed_sim(x)
        x = x @ x.T
        x = torch.sigmoid(x)
        return x

    def embed_sim(self, embeds):
        import numpy as np
        anom_sc1 = np.array([1653, 879, 1276]) 
        anom_sc2 = np.array([376, 804, 867])
        anom_sc3 = np.array([906, 1143, 574])
        anoms = np.concatenate((anom_sc1,np.concatenate((anom_sc2,anom_sc3),axis=None)),axis=None) 

        # get max embedding diff for normalization
        max_diff = 0
        for ind,embed in enumerate(embeds):
            for ind_,embed_ in enumerate(embeds):
                if ind_>= ind:
                    break
                max_diff = torch.norm(embed-embed_) if torch.norm(embed-embed_) > max_diff else 0
        # get anom embeds differences
        anom_diffs = []
        for ind,anom in enumerate(anoms):
            for ind_,anom_ in enumerate(anoms):
                if ind_>=ind: break
                if anom != anom_: continue
                anom_diffs.append(torch.norm(anom-anom_)/max_diff)
        # get normal embeds differences
        normal_diffs = []
        for ind,embed in enumerate(embeds):
            if ind in anoms: continue
            for ind_,embed_ in enumerate(embeds):
                if ind_ >= ind: break
                if ind_ in anoms: continue
                normal_diffs.append(torch.norm(embed-embed_)/max_diff)
        # get normal vs anom embeds differences
        norm_anom_diffs = []
        for ind, embed in enumerate(embeds):
            if ind in anoms: continue
            for ind_,anom in enumerate(embeds):
                if ind_ >= ind: break
                if ind_ not in anoms: continue
                norm_anom_diffs.append(torch.norm(embed-anom)/max_diff)
        import ipdb ; ipdb.set_trace()


class Dominant(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(Dominant, self).__init__()
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)
    
    def forward(self, x, adj, w_feat,bias):
        # encode
        x = self.shared_encoder(x, adj, w_feat,bias)
        #import ipdb ; ipdb.set_trace()
        #x_hat = self.attr_decoder(x)
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
    def __init__(self, in_size, out_size, scales):
        super(EGCN, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        #self.struc_weights, self.feat_weights = [], []
        #self.struc_weights=glorot_init(in_size, out_size)
        hidden_size = 64
        self.feat_weights=torch.nn.Parameter(glorot_init(in_size, hidden_size))
        self.feat_weights2=torch.nn.Parameter(glorot_init(hidden_size, out_size))
        self.bias=torch.nn.Parameter(torch.FloatTensor(2708,hidden_size))
        self.bias2=torch.nn.Parameter(torch.FloatTensor(2708,out_size))
        self.bias.data.fill_(0.0)
        self.bias2.data.fill_(0.0)
        self.scales = 1
        #self.struc_weights.requires_grad, self.feat_weights.requires_grad = True, True
        self.feat_weights.requires_grad = True
        self.feat_weights2.requires_grad = True
        self.bias.requires_grad = True
        self.bias2.requires_grad = True
        #self.reset_parameters()
        #self.struc_gru = torch.nn.GRU(input_size=self.in_size, hidden_size=self.out_size, num_layers=scales)
        #self.feat_gru = torch.nn.GRU(input_size=self.in_size, hidden_size=self.out_size, num_layers=scales)
        #for param in self.feat_gru.parameters():
        #    param.requires_grad = True
        self.conv = Dominant(self.in_size, hidden_size, 0.3)
        self.conv2 = Dominant(hidden_size, self.out_size, 0.3)
    
    '''
    def reset_parameters(self):
        import math
        stdv = 1. / math.sqrt(self.weights[0].size(1))
        for weight_ind in range(len(self.weights)):
            self.weights[weight_ind].data.uniform_(-stdv, stdv)
    '''
    def forward(self, x, adj):
        # updating first set of weights?
        A_hat_scales = []
        #_, w_out_struc = self.struc_gru(self.struc_weights)
        if self.scales > 1:
            _, w_out_feat = self.feat_gru(self.feat_weights)
        #import ipdb ; ipdb.set_trace()
        for weight_ind in range(self.scales):
            if weight_ind == 0:
                A_hat, X_hat = self.conv(x, adj, self.feat_weights[0],self.bias)#, self.feat_weights)
                A_hat, X_hat = self.conv2(A_hat, adj, self.feat_weights2[0],self.bias2)
            #else:
            #    A_hat, X_hat = self.conv(x[weight_ind], w_out_feat[weight_ind])#, w_out_feat[weight_ind][0])
            A_hat_scales.append(A_hat)
            #X_hat_scales.append(X_hat)
        # ???
        #for weight_ind in range(len(self.weights)):
        #    self.weights[weight_ind] = self.weights[weight_ind].detach()
        return A_hat_scales, X_hat
