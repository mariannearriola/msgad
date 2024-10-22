import torch
import torch.nn.functional as F
from itertools import chain
import torch.nn as nn

from typing import Optional
from torch_geometric.typing import OptTensor
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import get_laplacian
import torch
import numpy as np
from numpy import polynomial
import matplotlib.pyplot as plt
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
    def __init__(self, hidden_channels, K, id, bias=False, normalization=False, vis_filters=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(BernConv, self).__init__(**kwargs)
        assert K > 0
        self.K = K
        self.id = id
        self.in_channels = hidden_channels
        self.out_channels = hidden_channels
        self.weight = Parameter(torch.Tensor(K + 1, 1))
        self.normalization = normalization
        self.vis_filters = vis_filters
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
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        edge_weight = edge_weight / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)
        assert edge_weight is not None
        return edge_index, edge_weight
    def run_filter(self, bern_coeff,weight,eval):
        out_ = 0.
        for k in range(0, self.K+1):
            coeff = bern_coeff[k]
            basis_ = eval * coeff[0]
            for i in range(1, self.K+1):
                basis_ += eval * coeff[i]
            out_ += basis_ * weight[k]
        return out_
        
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

        Bx_0 = x
        Bx = [Bx_0]
        Bx_next = Bx_0


        for _ in range(self.K):
            Bx_next = self.propagate(edge_index, x=Bx_next, norm=norm, size=None)
            Bx.append(Bx_next)

        bern_coeff =  BernConv.get_bern_coeff(self.K)
        eps = 1e-2
        if self.normalization:
            weight = torch.sigmoid(self.weight)
        else:
            weight = torch.clamp(self.weight, min = 0. + eps, max = 1. - eps)

        out = torch.zeros_like(x)
        for k in range(0, self.K + 1):
            coeff = bern_coeff[k]
            basis = Bx[0] * coeff[0]
            for i in range(1, self.K + 1):
                basis += Bx[i] * coeff[i]
            out += basis * weight[k]
        if self.vis_filters:
            plt.figure()
            L = np.zeros((x.shape[0],x.shape[0]))
            edge_index = edge_index.detach().cpu().numpy() ; norm = norm.detach().cpu().numpy()
            L[edge_index[0],edge_index[1]]=norm
            e,U = np.linalg.eigh(L)
            e[0] = 0
            c = np.dot(U.transpose(), out.detach().cpu().numpy())
            M = np.zeros((15,c.shape[1]))
            
            for j in range(x.shape[0]):
                #import ipdb ; ipdb.set_trace()
                idx = min(int(e[j] / 0.1), 15-1)
                M[idx] += c[j]**2
            #for j in range(50):
            #    plt.plot(M[:,j])
            M=M/sum(M)
            plt.plot(np.mean(M,axis=1))
            plt.savefig(f'amnet_filter_vis_{self.id}.png')
        #print('hi')
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


class AMNet(nn.Module):
    def __init__(self, in_channels, hid_channels, num_class, K, filter_num=5, dropout=0.3, vis_filters=False):
        super(AMNet, self).__init__()
        self.act_fn = nn.ReLU()
        self.attn_fn = nn.Tanh()
        self.linear_transform_in = nn.Sequential(nn.Linear(in_channels, hid_channels),
                                                 self.act_fn,
                                                 nn.Linear(hid_channels, hid_channels),
                                                 )
        self.K = K
        self.vis_filters = vis_filters
        self.filters = nn.ModuleList([BernConv(hid_channels, K, id, normalization=True, bias=True, vis_filters=vis_filters) for id in range(filter_num)])
        self.filter_num = filter_num
        self.W_f = nn.Sequential(nn.Linear(hid_channels, hid_channels),
                                 self.attn_fn,
                                 )
        self.W_x = nn.Sequential(nn.Linear(hid_channels, hid_channels),
                                 self.attn_fn,
                                 )
        self.linear_cls_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_channels, num_class))


        self.attn = list(self.W_x.parameters())
        self.attn.extend(list(self.W_f.parameters()))
        self.lin = list(self.linear_transform_in.parameters())
        self.lin.extend(list(self.linear_cls_out.parameters()))
        self.reset_parameters()


    def reset_parameters(self):
        pass


    def forward(self, x, edge_index, label=None):
        """
        :param label:
        :param x:
        :param edge_index:
        :return:
        """

        x = self.linear_transform_in(x)
        h_list = []
        for i, filter_ in enumerate(self.filters):
            h = filter_(x, edge_index)
            h_list.append(h)

        h_filters = torch.stack(h_list, dim=1)
        h_filters_proj = self.W_f(h_filters)
        x_proj = self.W_x(x).unsqueeze(-1)

        score_logit = torch.bmm(h_filters_proj, x_proj)
        soft_score = F.softmax(score_logit, dim=1)
        score = soft_score

        res = h_filters[:, 0, :] * score[:, 0]
        for i in range(1, self.filter_num):
            res += h_filters[:, i, :] * score[:, i]
        if True in torch.isnan(res):
            import ipdb ; ipdb.set_trace()
            print('nan')
        
        # ADAPT TO RECONSTRUCTION
        res = res@res.T
        return res
        #return torch.sigmoid(res)
        #return 1 / (1 + math.exp(-res))
        #return torch.sigmoid(res@res.T)
        '''
        y_hat = self.linear_cls_out(res)
        marginal_loss = 0.
        return y_hat
        '''
        '''
        if self.training:
            anomaly_train, normal_train = label
            normal_bias = score[normal_train][:, 1] - score[normal_train][:, 0]
            anomaly_bias = score[anomaly_train][:, 0] - score[anomaly_train][:, 1]
            normal_bias = torch.clamp(normal_bias, -0.)
            anomaly_bias = torch.clamp(anomaly_bias, -0.)
            normal_bias = torch.mean(normal_bias)
            anomaly_bias = torch.mean(anomaly_bias)
            bias = anomaly_bias + normal_bias
            marginal_loss = bias

        if self.training:
            return y_hat,  marginal_loss
        else:
            return y_hat
        '''



    @torch.no_grad()
    def get_attn(self, label, train_index, test_index):
        anomaly, normal = label
        test_attn_anomaly = list(chain(*torch.mean(self.attn_score[test_index & anomaly], dim=0).tolist()))
        test_attn_normal = list(chain(*torch.mean(self.attn_score[test_index & normal], dim=0).tolist()))
        train_attn_anomaly = list(chain(*torch.mean(self.attn_score[train_index & anomaly], dim=0).tolist()))
        train_attn_normal = list(chain(*torch.mean(self.attn_score[train_index & normal], dim=0).tolist()))

        return (train_attn_anomaly, train_attn_normal), \
               (test_attn_anomaly, test_attn_normal)



