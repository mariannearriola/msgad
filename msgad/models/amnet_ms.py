import torch
import torch.nn.functional as F
from itertools import chain
import torch.nn as nn

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
import numpy as np
from numpy import polynomial
import math

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
        self.weight = Parameter(torch.Tensor(K + 1, 1))
        self.normalization = normalization

        if bias:
            self.bias = Parameter(torch.Tensor(hidden_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        zeros(self.bias)
        torch.nn.init.zeros_(self.weight)


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str],
                 lambda_max, dtype: Optional[int] = None):
        #edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        #mat = torch_geometric.utils.to_dense_adj(edge_index,edge_attr=edge_weight)[0]#,num_nodes=num_nodes)
        #edge_weight = 
        #basis_[basis.edges()] = basis.edata['w'].to(self.graph.device)
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        edge_weight = edge_weight #/ lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)
        assert edge_weight is not None
        return edge_index, edge_weight


    def forward(self, x, edge_index, edge_weight: OptTensor = None,
                lambda_max: OptTensor = None):

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype,
                                      device=x.device)
        assert lambda_max is not None

        edge_index, norm = self.__norm__(edge_index, x.size(self.node_dim),
                                         edge_weight, 'sym', lambda_max, dtype=x.dtype)
    
        mat = self.__norm__(edge_index, x.size(self.node_dim),
                                         edge_weight, 'sym', lambda_max, dtype=x.dtype)
        Bx_0 = x
        Bx = [Bx_0]
        Bx_next = Bx_0


        for _ in range(self.K):
            Bx_next = self.propagate(edge_index, x=Bx_next, norm=norm, size=None)
            #Bx_next = mat @ Bx_next
            Bx.append(Bx_next)
        bern_coeff =  BernConv.get_bern_coeff(self.K)
        eps = 1e-2
        #if self.normalization:
        #    weight = torch.sigmoid(self.weight)
        #else:
        #    weight = torch.clamp(self.weight, min = 0. + eps, max = 1. - eps)
        #print(weight)
        out = torch.zeros_like(x)
        for k in range(0, self.K + 1):
            coeff = bern_coeff[k]
            basis = Bx[0] * coeff[0]
            for i in range(1, self.K + 1):
                basis += Bx[i] * coeff[i]
            out += basis * F.softmax(self.weight)[k]
            
        # NOTE: out is just epsilon (need to add weights to paramter list of amnet ms)
        del lambda_max
        del basis
        #del weight
        del Bx
        del bern_coeff
        del Bx_next
        del Bx_0
        #del edge_index
        #del norm
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
        '''
        thetas = []
        x = sympy.symbols('x')
        for i in range(degree+1):
            f = sympy.poly((x/2) ** i * (1 - x/2) ** (degree-i) / (scipy.special.beta(i+1, degree+1-i)))
            coeff = f.all_coeffs()
            inv_coeff = []
            for i in range(degree+1):
                inv_coeff.append(float(coeff[degree-i]))
            thetas.append(inv_coeff)
        del x
        del inv_coeff
        del coeff
        del f
        torch.cuda.empty_cache()
        return thetas
        '''
        
class AMNet_ms(nn.Module):
    def __init__(self, in_channels, hid_channels, num_class, K, filter_num=5, dropout=0.3):
        super(AMNet_ms, self).__init__()
        self.act_fn = nn.ReLU()
        self.attn_fn = nn.Tanh()
        self.act_sg = nn.Sigmoid()
        self.linear_transform_in = nn.Sequential(nn.Linear(in_channels, hid_channels),
                                                 self.act_fn,
                                                 nn.Linear(hid_channels, hid_channels),
                                                 )
        self.K = K
        self.filters = nn.ModuleList([BernConv(hid_channels, K, normalization=False, bias=True) for _ in range(filter_num)])
        #self.filters = nn.ModuleList([BernConv(hid_channels, K, normalization=False, bias=True),BernConv(hid_channels, K, normalization=False, bias=True),BernConv(hid_channels, K, normalization=False, bias=True),BernConv(hid_channels, K, normalization=False, bias=True),BernConv(hid_channels, K, normalization=False, bias=True)])
        #self.filters.extend([BernConv(hid_channels, K, normalization=False, bias=True) for i in range(1, filter_num)])
        self.filter_num = filter_num

        self.W_f = nn.Sequential(nn.Linear(hid_channels, in_channels),
                                 self.attn_fn,
                                 )
        
        self.W_x = nn.Sequential(nn.Linear(hid_channels, in_channels),
                                 self.attn_fn,
                                 )
        
        #self.out_l = nn.Linear(hid_channels, hid_channels)
        #self.lam = nn.Parameter(data=torch.normal(mean=torch.full((filter_num,),0.),std=1))
        '''
        self.linear_cls_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_class))

        self.attn = list(self.W_x.parameters())
        self.attn.extend(list(self.W_f.parameters()))
        self.lin = list(self.linear_transform_in.parameters())
        self.lin.extend(list(self.linear_cls_out.parameters()))
        self.relu = torch.nn.ReLU()
        '''
        self.lam = nn.Parameter(data=torch.normal(mean=torch.full((filter_num,),0.),std=1))
        self.reset_parameters()


    def reset_parameters(self):
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

    def forward(self, x, edge_index, label=None):
        """
        :param label:
        :param x:
        :param edge_index:
        :return:
        """

        x = self.linear_transform_in(x)
        h_list = []
        lams = F.softmax(self.lam)
        for i, filter_ in enumerate(self.filters):
            #print('for filter',i,self.filters[i].weight)
            #h = filter_(x, edge_index)
            h = self.filters[i](x, edge_index)*lams[i]
            h_list.append(h)
            del filter_
            del h
        print(lams)
        torch.cuda.empty_cache()
        h_filters = torch.stack(h_list, dim=1)
        h_filters_proj = self.W_f(h_filters)
        x_proj = self.W_x(x).unsqueeze(-1)
        score_logit = torch.bmm(h_filters_proj, x_proj)
        soft_score = F.softmax(score_logit, dim=1) ; score = soft_score

        #score = score_logit # shape: [ num_nodes, K, num_filters ] ; node-wise attention
        # attention for various freq. profiles
        res = h_filters[:, 0, :] * score[:, 0]
        for i in range(1, self.filter_num):
            res += (h_filters[:, i, :] * score[:, i])
        if True in torch.isnan(res):
            import ipdb ; ipdb.set_trace()
            print('nan')
            
        #res = self.out_l(res)
        edge_index, norm = self.__norm__(edge_index, x.shape[0],
                                         None, 'sym', torch.tensor(2.0, dtype=x.dtype, device=x.device), dtype=x.dtype)

        #L = np.zeros((x.shape[0],x.shape[0]))
        #edge_index = edge_index.detach().cpu().numpy() ; norm = norm.detach().cpu().numpy()
        #L[edge_index[0],edge_index[1]]=norm
        #self.L = L
        #res_ = torch.sigmoid(res@res.T)
        #import ipdb ; ipdb.set_trace()
        res_ = res@res.T
        del h_filters
        del x
        del score_logit
        del x_proj
        del h_filters_proj
        for i in range(len(h_list)):
            del h_list[0]
        torch.cuda.empty_cache()
        #if res_.gt(0.5).nonzero().shape[0]>0:
        #    import ipdb ; ipdb.set_trace()
        return res_,res,torch.squeeze(score,-1).T



    @torch.no_grad()
    def get_attn(self, label, train_index, test_index):
        anomaly, normal = label
        test_attn_anomaly = list(chain(*torch.mean(self.attn_score[test_index & anomaly], dim=0).tolist()))
        test_attn_normal = list(chain(*torch.mean(self.attn_score[test_index & normal], dim=0).tolist()))
        train_attn_anomaly = list(chain(*torch.mean(self.attn_score[train_index & anomaly], dim=0).tolist()))
        train_attn_normal = list(chain(*torch.mean(self.attn_score[train_index & normal], dim=0).tolist()))

        return (train_attn_anomaly, train_attn_normal), \
               (test_attn_anomaly, test_attn_normal)



