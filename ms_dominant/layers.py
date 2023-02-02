import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = torch.sigmoid
        
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(1,out_features))
        '''
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        '''
        self.reset_parameters()
        
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj, w):#,bias,adj_bool):
        '''
        if False:
            output = F.relu(torch.mm(torch.mm(adj,input),self.weight)+self.bias)
        else:
            output = F.relu(torch.mm(input,self.weight))#+bias)
        '''
        #output = self.act(torch.mm(input,w))
        output = torch.mm(input,w)
        output = F.leaky_relu(output,negative_slope=0.1)
        #output = torch.spmm(adj, support)
        '''
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        '''
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
