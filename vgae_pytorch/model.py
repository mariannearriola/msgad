import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from typing import Optional, Tuple

from torch import Tensor
from torch.nn import GRU
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import args

class VGAE(nn.Module):
    def __init__(self, adj):
        super(VGAE,self).__init__()
        # self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
        self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

    def encode(self, X, W1, W2):
        # hidden = self.base_gcn(X)
        # self.mean = self.gcn_mean(hidden)
        # self.logstd = self.gcn_logstddev(hidden)
        # print('X SHAPE',X.shape,'W SHAPE',W.shape)
        self.mean,self.logstd=self.gcn_mean(X,W1),self.gcn_logstddev(X,W2)
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X, W1, W2):
        Z = self.encode(X,W1, W2)
        A_pred = dot_product_decode(Z)
        return A_pred

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation = torch.nn.LeakyReLU(negative_slope=0.5), **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        # self.weight = glorot_init(input_dim, output_dim) 
        self.adj = adj
        self.activation = activation

    def forward(self, inputs, W):
        x = inputs
        # print('WEIGHTS SHAPE',W.shape,x.shape)
        x = torch.mm(x,W)
        #x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        # outputs = x
        return outputs


def dot_product_decode(Z):
    # A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    # sc = torch.full(A_pred.size(),-.5).cuda(args.device)
    # A_pred = A_pred + sc
    A_pred = torch.matmul(Z,Z.t())
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    # t = torch.Tensor([0.5]).cuda(args.device)
    # A_pred = (A_pred >= t).float()*1
    return A_pred

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial).cuda(args.device)

class APPNP(MessagePassing):
    r"""The approximate personalized propagation of neural predictions layer
    from the `"Predict then Propagate: Graph Neural Networks meet Personalized
    PageRank" <https://arxiv.org/abs/1810.05997>`_ paper

    .. math::
        \mathbf{X}^{(0)} &= \mathbf{X}

        \mathbf{X}^{(k)} &= (1 - \alpha) \mathbf{\hat{D}}^{-1/2}
        \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \mathbf{X}^{(k-1)} + \alpha
        \mathbf{X}^{(0)}

        \mathbf{X}^{\prime} &= \mathbf{X}^{(K)},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        K (int): Number of iterations :math:`K`.
        alpha (float): Teleport probability :math:`\alpha`.
        dropout (float, optional): Dropout probability of edges during
            training. (default: :obj:`0`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int, alpha: float, dropout: float = 0.,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        h = x
        for k in range(self.K):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            x = x * (1 - self.alpha)
            x += self.alpha * h

        return x


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.alpha})'

class GAE(nn.Module):
    def __init__(self,adj):
        super(GAE,self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj)#, activation=torch.sigmoid)
        # replace with APPNP
        # self.appnp = APPNP(.1,scale+1)

    def encode(self, X, W, adj, t):
        # hidden = self.base_gcn(X, W)
        # z = self.mean = self.gcn_mean(hidden)
        z = self.mean = self.gcn_mean(X,W)
        # if t == 1:
        #     appnp = APPNP(t,.1)
        # if t == 2:
        #     appnp = APPNP(t,.3)
        # if t == 3:
        #     appnp = APPNP(t,.5)
        # z = appnp(z,adj)
        return z

    def forward(self, X, W, adj, t):
        #Z = self.encode(X, W, adj, t)
        A_pred = self.encode(X,W,adj,t)
        #A_pred = dot_product_decode(Z)
        return A_pred


# class GraphConv(nn.Module):
# 	def __init__(self, input_dim, hidden_dim, output_dim):
# 		super(VGAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
# 		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

# 	def forward(self, X, A):
# 		out = A*X*self.w0
# 		out = F.relu(out)
# 		out = A*X*self.w0
# 		return out


class GCNConv_Fixed_W(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv_Fixed_W, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, W: torch.FloatTensor, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops)

        x = torch.matmul(x, W)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

class RecurrentGCN(torch.nn.Module):
    def __init__(self,adj,in_channels,hidden_size,num_timesteps):
        super(RecurrentGCN,self).__init__()
        # self.recurrent0 = EvolveGCNO(adj,in_channels,hidden_size,num_timesteps)
        self.recurrent1 = EvolveGCNO(adj,in_channels,hidden_size,num_timesteps)
        self.recurrent2 = EvolveGCNO(adj,in_channels,hidden_size,num_timesteps)
        self.recurrent3 = EvolveGCNO(adj,in_channels,hidden_size,num_timesteps)
        # self.GRCU_layers = [self.recurrent0.to(args.device),self.recurrent1.to(args.device),self.recurrent2.to(args.device),self.recurrent3.to(args.device)]
        self.GRCU_layers = [self.recurrent1.to(args.device),self.recurrent2.to(args.device),self.recurrent3.to(args.device)]
        self._parameters = nn.ParameterList()
        # just adds self.recurrent1 and self.recurrent2 params
        # TODO: MAKE SURE PARAMETERS ARE HERE
        self._parameters.extend(list(self.GRCU_layers[0].recurrent_layer.parameters()))
        self._parameters.extend(list(self.GRCU_layers[1].recurrent_layer.parameters()))
        self._parameters.extend(list(self.GRCU_layers[2].recurrent_layer.parameters()))
        # self._parameters.extend(list(self.GRCU_layers[3].recurrent_layer.parameters()))
    
    def parameters(self):
        return self._parameters

    def forward(self,features_list):
        print('len features list',len(features_list))
        # w0,embeds0 = self.recurrent0(features_list[0],0)
        # self.recurrent1.weight = w0[0]
        w1,embeds1 = self.recurrent1(features_list[0],0)
        self.recurrent2.weight = w1[0]
        w2,embeds2 = self.recurrent2(features_list[1],1)
        self.recurrent3.weight = w2[0]
        w3,embeds3 = self.recurrent3(features_list[2],2)
        # w1,embeds1 = self.recurrent1(features_list[1],0)
        # self.recurrent2.weight = w1[0]
        # w2,embeds2 = self.recurrent2(features_list[2],1)
        # self.recurrent3.weight = w2[0]
        # w3,embeds3 = self.recurrent3(features_list[3],2)
        # return recons arr
        # TODO: check shapes...
        # a0_recons = dot_product_decode(embeds0)
        a1_recons = dot_product_decode(embeds1)
        a2_recons =  dot_product_decode(embeds2)
        a3_recons = dot_product_decode(embeds3)
        recons_arr = [a1_recons,a2_recons,a3_recons]
        # recons_arr = [a0_recons,a1_recons,a2_recons,a3_recons]
        # recons_arr = [a1_recons]
        return recons_arr


class EvolveGCNO(torch.nn.Module):
    r"""An implementation of the Evolving Graph Convolutional without Hidden Layer.
    For details see this paper: `"EvolveGCN: Evolving Graph Convolutional
    Networks for Dynamic Graph." <https://arxiv.org/abs/1902.10191>`_
    Args:
        in_channels (int): Number of filters.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
    """

    def __init__(
        self,
        adj: torch.Tensor,
        in_channels: int,
        improved: bool = False,
        cached: bool = False,
        normalize: bool = True,
        add_self_loops: bool = True,
    ):
        super(EvolveGCNO, self).__init__()
        self.adj = adj
        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.weight = None
        self.initial_weight = torch.nn.Parameter(torch.Tensor(in_channels, in_channels))
        self._create_layers()
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.initial_weight)

    def _create_layers(self):

        self.recurrent_layer = GRU(
            input_size=self.in_channels, hidden_size=self.in_channels, num_layers=1
        )
        for param in self.recurrent_layer.parameters():
            param.requires_grad = True
            param.retain_grad()
            #self._parameters.extend(param)
        # just adds self.recurrent1 and self.recurrent2 params
        # TODO: MAKE SURE PARAMETERS ARE HERE


        # self.conv_layer = GCNConv_Fixed_W(
        #     in_channels=self.in_channels,
        #     out_channels=self.in_channels,
        #     improved=self.improved,
        #     cached=self.cached,
        #     normalize=self.normalize,
        #     add_self_loops=self.add_self_loops
        # )
        self.conv_layer = GAE(self.adj)
    def parameters(self):
        return self._parameters

    def forward(
        self,
        X: torch.FloatTensor,
        t: int#,
        # edge_index: torch.LongTensor,
        # edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass.
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.
        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """
        
        if self.weight is None:
            self.weight = self.initial_weight.data
            
        W = self.weight[None, :, :]
        
        _, W = self.recurrent_layer(W, W)
        # X = self.conv_layer(W.squeeze(dim=0), X)#, edge_index, edge_weight)
        
        embeds = self.conv_layer(X,W[0],self.adj.coalesce().indices(),t)
    
        return W,embeds


# class EvolveGCNO(torch.nn.Module):

#     def __init__(s
#         self,
#         adj: torch.Tensor,
#         in_channels: int,
#         num_timesteps: int,
#         hidden_size: int,
#         improved: bool = False,
#         cached: bool = False,
#         normalize: bool = True,
#         add_self_loops: bool = True,
#     ):
#         super(EvolveGCNO, self).__init__()

#         self.adj = adj
#         self.in_channels = in_channels
#         self.hidden_size = hidden_size
#         self.num_timesteps = num_timesteps
#         self.improved = improved
#         self.cached = cached
#         self.normalize = normalize
#         self.add_self_loops = add_self_loops
#         self.weight = None
#         self.initial_weight = torch.nn.Parameter(torch.Tensor(in_channels, in_channels)).cuda(args.device)
#         # self.initial_weight = glorot_init(self.in_channels,self.in_channels)
#         self._create_layers()
#         self.reset_parameters()
    
#     def reset_parameters(self):
#         glorot(self.initial_weight)
#         # self.initial_weight = glorot_init(self.in_channels,self.in_channels)


#     def _create_layers(self):

#         self.recurrent_layer = GRU(
#             input_size=self.in_channels, hidden_size=self.hidden_size#, num_layers=self.num_timesteps
#         ).cuda(args.device)
#         for param in self.recurrent_layer.parameters():
#             param.requires_grad = True
#             param.retain_grad()

#         # self.conv_layer = GCNConv_Fixed_W(
#         #     in_channels=self.in_channels,
#         #     out_channels=self.in_channels,
#         #     improved=self.improved,
#         #     cached=self.cached,
#         #     normalize=self.normalize,
#         #     add_self_loops=self.add_self_loops
#         # )
#         self.conv_layer = GAE(self.adj)

#     def forward(
#         self,
#         # X: torch.FloatTensor
#         X : list
#         # edge_index: torch.LongTensor,
#         # edge_weight: torch.FloatTensor = None,
#     ) -> torch.FloatTensor:
#         """
#         Making a forward pass.
#         Arg types:
#             * **X** *(PyTorch Float Tensor)* - Node embedding.
#             * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
#             * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.
#         Return types:
#             * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
#         """
#         print('state dict',self.recurrent_layer.state_dict())
#         if self.weight is None:
#             self.weight = self.initial_weight.data.cuda(args.device)
#         W = self.weight[None, :, :]

#         h0 = torch.zeros(self.num_timesteps, args.hidden1_dim, args.hidden2_dim).cuda(args.device).requires_grad_()
#         print('WEIGHT SHAPE',W.shape)
#         print('hidden shape',h0.shape)
#         W_all, W_last = self.recurrent_layer(W, h0.detach())
#         # X = self.conv_layer(W.squeeze(dim=0), X, edge_index, edge_weight)
#         recons_arr = []
#         for t in range(self.num_timesteps):
#             # recons_arr.append(self.conv_layer(X,W[t]))
#             recons_arr.append(self.conv_layer(X[t],W_all[t],self.adj.coalesce().indices(),t))
#             # recons_arr.append(self.conv_layer(X[0],W[t],self.adj.coalesce().indices(),t+1))

#         return recons_arr
#         # return: embeddings for timestep t