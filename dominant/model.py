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
    def forward(self, x, w):
        #x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc1(x, w))
        x = F.dropout(x, self.dropout, training=self.training)
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
        #x = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x

class Dominant(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(Dominant, self).__init__()
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        #self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)
    
    #def forward(self, x, adj):
    def forward(self, x, w):
        # encode
        #x = self.shared_encoder(x, adj)
        x = self.shared_encoder(x, w)
        # decode feature matrix
        #x_hat = self.attr_decoder(x, adj)
        # decode adjacency matrix
        #struct_reconstructed = self.struct_decoder(x, adj)
        struct_reconstructed = self.struct_decoder(x)
        # return reconstructed matrices
        #return struct_reconstructed, x_hat
        return struct_reconstructed, x

def glorot_init(in_size, out_size):
    import numpy as np
    import math
    stdv = 1. / math.sqrt(in_size)
    #init_range = np.sqrt(6.0/(in_size+out_size))
    initial = torch.rand(in_size, out_size)*(2*stdv)
    resh_initial = initial[None, :, :]
    return resh_initial.cuda()

class EGCN(nn.Module):
    def __init__(self, in_size, out_size):
        super(EGCN, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        #self.weight = nn.Parameter(torch.FloatTensor(in_size, out_size))
        self.weight = glorot_init(in_size, out_size)
        self.reset_parameters()
        self.gru = torch.nn.GRU(input_size=self.in_size, hidden_size=self.out_size, num_layers=1)
        for param in self.gru.parameters():
            param.requires_grad = True
        self.conv = Dominant(self.in_size, self.out_size, 0.3)

    def reset_parameters(self):
        import math
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, w):
        if w is not None:
            self.weight = w
            #self.weight = w[None, :, :]
        #print(self.weight.shape)
        #import ipdb ; ipdb.set_trace()
        #_, w_out = self.gru(self.weight.reshape(1,self.out_size,self.in_size))#,self.weight)
        _, w_out = self.gru(self.weight)#,self.weight)
        A_hat, X_hat = self.conv(x, w_out[0])
        #import ipdb ; ipdb.set_trace() 
        
        return A_hat, X_hat, w_out
