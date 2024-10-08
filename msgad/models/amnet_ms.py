import torch
import torch.nn.functional as F
from itertools import chain
import torch.nn as nn
import gc
import scipy
import scipy.sparse as sp
import sympy

from typing import Optional
from torch_geometric.typing import OptTensor
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import get_laplacian
import torch
#from model_utils import *
import numpy as np
from numpy import polynomial
import math

def check_gpu_usage(tag):
    return
    allocated_bytes = torch.cuda.memory_allocated(torch.device('cuda'))
    cached_bytes = torch.cuda.memory_cached(torch.device('cuda'))

    allocated_gb = allocated_bytes / 1e9
    cached_gb = cached_bytes / 1e9
    print(f"{tag} -> GPU Memory - Allocated: {allocated_gb:.2f} GB, Cached: {cached_gb:.2f} GB")

def seed_everything(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def constant(tensor, value):
    if tensor is not None:
        tensor.data.fill_(value)

class BernConv(MessagePassing):
    def __init__(self, hidden_channels, K, bias=False, normalization=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(BernConv, self).__init__(**kwargs)
        assert K > 0
        self.K = K
        self.in_channels = hidden_channels
        self.out_channels = hidden_channels
        self.normalization = normalization
        seed_everything()

        self.reset_parameters()


    def reset_parameters(self):
        pass
        #torch.nn.init.zeros_(self.weight)


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str],
                 lambda_max, dtype: Optional[int] = None):
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        edge_weight = edge_weight #/ lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)
        assert edge_weight is not None
        return edge_index, edge_weight


    def forward(self, x, edge_index, conv_weight, edge_weight: OptTensor = None,
                lambda_max: OptTensor = None):
        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype,
                                      device=x.device)
        assert lambda_max is not None
        edge_index, norm = self.__norm__(edge_index.T, x.size(self.node_dim),
                                         edge_weight, 'sym', lambda_max, dtype=x.dtype)

        Bx_0 = x
        Bx = [Bx_0]
        Bx_next = Bx_0


        for _ in range(self.K):
            Bx_next = self.propagate(edge_index, x=Bx_next, norm=norm, size=None)
            Bx.append(Bx_next)
        bern_coeff =  BernConv.get_bern_coeff(self.K)
        for k in range(0, self.K + 1):
            coeff = bern_coeff[k]
            basis = Bx[0] * coeff[0]
            for i in range(1, self.K + 1):
                basis += Bx[i] * coeff[i]
            out = basis if k == 0 else torch.cat((out,basis),dim=1)
            del basis

        del lambda_max, Bx, bern_coeff, Bx_next, Bx_0, edge_index, norm
        torch.cuda.empty_cache()
        return out

    
    @staticmethod
    def get_bern_coeff(degree):
        
        def Bernstein(de, i):
            coefficients = [0, ] * i + [math.comb(de, i)]
            first_term = polynomial.polynomial.Polynomial(coefficients)
            second_term = polynomial.polynomial.Polynomial([1, -1]) ** (de - i)
            return first_term * second_term

        out = []

        for i in range(degree + 1):
            out.append(Bernstein(degree, i).coef)

        return out
    
class AMNet_ms(nn.Module):
    def __init__(self, in_channels, hid_channels, num_class, K, filter_num=5, dropout=0.3):
        super(AMNet_ms, self).__init__()
        self.attn_fn = nn.Tanh()
        self.K = K
        self.filters = nn.ModuleList([BernConv(hid_channels, K, normalization=False, bias=False) for _ in range(filter_num)])
        self.filter_num = filter_num
        self.reset_parameters()


    def reset_parameters(self):
        #torch.nn.init.zeros_(self.lam)
        pass

    def __norm__(self, edge_index, num_nodes: Optional[int],
                    edge_weight: OptTensor, normalization: Optional[str],
                    lambda_max, dtype: Optional[int] = None):
        
            #mat = torch_geometric.utils.to_dense_adj(edge_index,edge_attr=edge_weight)[0]#,num_nodes=num_nodes)
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
            edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                    normalization, dtype,
                                                    num_nodes)

            edge_weight = edge_weight# / lambda_max
            edge_weight.masked_fill_(edge_weight == float('inf'), 0)
            assert edge_weight is not None
            return edge_index, edge_weight

    def forward(self, x, edge_index, conv_weight, label=None):
        """
        ...
        Input:
            x: {array-like, torch tensor}, shape=[n,h]
                Graph features.
            edge_index: {array-like, torch tensor}, shape=[e,2]
                Graph edge list.
            conv_weight: {array-like, torch tensor}, shape=[...]
        Output:
            
        """
        h = self.filters[0](x, edge_index, None).unsqueeze(1)
        for i in range(1,len(self.filters)):
            h = torch.cat((h,self.filters[i](x, edge_index, None).unsqueeze(1)),dim=1)
        return h

    @torch.no_grad()
    def get_attn(self, label, train_index, test_index):
        anomaly, normal = label
        test_attn_anomaly = list(chain(*torch.mean(self.attn_score[test_index & anomaly], dim=0).tolist()))
        test_attn_normal = list(chain(*torch.mean(self.attn_score[test_index & normal], dim=0).tolist()))
        train_attn_anomaly = list(chain(*torch.mean(self.attn_score[train_index & anomaly], dim=0).tolist()))
        train_attn_normal = list(chain(*torch.mean(self.attn_score[train_index & normal], dim=0).tolist()))

        return (train_attn_anomaly, train_attn_normal), \
               (test_attn_anomaly, test_attn_normal)
