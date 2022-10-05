# -*- coding: utf-8 -*-
"""AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks"""
# Author: Xueying Ding <xding2@andrew.cmu.edu>
# License: BSD 2 clause


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv
from sklearn.utils.validation import check_is_fitted
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
'''
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
'''
from . import BaseDetector
from ..utils.metric import eval_roc_auc

class StructureAE(nn.Module):
    """
    Description
    -----------
    Structure Autoencoder in AnomalyDAE model: the encoder
    transforms the node attribute X into the latent
    representation with the linear layer, and a graph attention
    layer produces an embedding with weight importance of node 
    neighbors. Finally, the decoder reconstructs the final embedding
    to the original.

    See :cite:`fan2020anomalydae` for details.

    Parameters
    ----------
    in_dim: int
        input dimension of node data
    embed_dim: int
        the latent representation dimension of node
       (after the first linear layer)
    out_dim: int 
        the output dim after the graph attention layer
    dropout: float
        dropout probability for the linear layer
    act: F, optional
         Choice of activation function        
    Returns
    -------
    x : torch.Tensor
        Reconstructed attribute (feature) of nodes.
    embed_x : torch.Tensor
              Embedd nodes after the attention layer
    """

    def __init__(self,
                 in_dim,
                 embed_dim,
                 out_dim,
                 dropout,
                 act):
        super(StructureAE, self).__init__()
        self.dense = nn.Linear(in_dim, embed_dim)
        #self.attention_layer = GATConv(embed_dim, out_dim)
        self.attention_layer = GATConv(in_dim,out_dim,negative_slope=0.5)
        self.dropout = dropout
        self.act = F.leaky_relu
        self.sig = torch.sigmoid

    def forward(self,
                x,
                edge_index):
        # encoder
        #import ipdb ; ipdb.set_trace()
        embed_x = self.act(self.attention_layer(x, edge_index))
        # decoder
        x = self.sig(embed_x @ embed_x.T)
        return x, embed_x


class AttributeAE(nn.Module):
    """
    Description
    -----------
    Attribute Autoencoder in AnomalyDAE model: the encoder
    employs two non-linear feature transform to the node attribute
    x. The decoder takes both the node embeddings from the structure
    autoencoder and the reduced attribute representation to 
    reconstruct the original node attribute.

    Parameters
    ----------
    in_dim:  int
        dimension of the input number of nodes
    embed_dim: int
        the latent representation dimension of node
        (after the first linear layer)
    out_dim:  int
        the output dim after two linear layers
    dropout: float
        dropout probability for the linear layer
    act: F, optional
         Choice of activation function   
    Returns
    -------
    x : torch.Tensor
        Reconstructed attribute (feature) of nodes.
    """

    def __init__(self,
                 in_dim,
                 embed_dim,
                 out_dim,
                 dropout,
                 act):
        super(AttributeAE, self).__init__()
        self.dense1 = nn.Linear(in_dim, embed_dim)
        self.dense2 = nn.Linear(embed_dim, out_dim)
        self.dropout = dropout
        self.act = act
        self.dropoutf = F.dropout
    def forward(self,
                x,
                struct_embed):
        # encoder
        x = self.act(self.dense1(x.T))
        
        x = self.dropoutf(x, self.dropout)
        x = self.dense2(x)
        x = self.dropoutf(x, self.dropout)
        # decoder
        x = struct_embed @ x.T
        
        #x = x @ x.T
        return x


class AnomalyDAE_Base(nn.Module):
    """
    Description
    -----------
    AdnomalyDAE_Base is an anomaly detector consisting of a structure autoencoder,
    and an attribute reconstruction autoencoder. 

    Parameters
    ----------
    in_node_dim : int
         Dimension of input feature
    in_num_dim: int
         Dimension of the input number of nodes
    embed_dim:: int
         Dimension of the embedding after the first reduced linear layer (D1)   
    out_dim : int
         Dimension of final representation
    dropout : float, optional
        Dropout rate of the model
        Default: 0
    act: F, optional
         Choice of activation function
    """

    def __init__(self,
                 in_node_dim,
                 in_num_dim,
                 embed_dim,
                 out_dim,
                 dropout,
                 act):
        super(AnomalyDAE_Base, self).__init__()
        self.structure_AE = StructureAE(in_node_dim, embed_dim,
                                        out_dim, dropout, act)
        self.attribute_AE = AttributeAE(in_num_dim, embed_dim,
                                        out_dim, dropout, act)

    def forward(self,
                x,
                edge_index,
                adj):
        A_hat, embed_x = self.structure_AE(x, edge_index)
        X_hat = self.attribute_AE(x, embed_x)
        return A_hat, X_hat
        #return A_hat, A_hat
class EGCN(nn.Module):
    def __init__(self, num_scales,
                 in_node_dim,
                 in_num_dim,
                 embed_dim,
                 out_dim,
                 dropout,
                 act):
        super(EGCN, self).__init__()
        #import ipdb ; ipdb.set_trace() 
        ''' 
        self.model = AnomalyDAE_Base(in_node_dim=in_node_dim,
                                     in_num_dim=in_num_dim,
                                     embed_dim=embed_dim,
                                     out_dim=out_dim,
                                     dropout=dropout,
                                     act=act).to('cuda:0')
        
        '''
        #self.dense = nn.Linear(in_dim, embed_dim)
        #self.attention_layer = GATConv(embed_dim, out_dim)
        self.attention_layer = GATConv(in_node_dim,out_dim,negative_slope=0.5).to('cuda:0')
        self.dropout = dropout
        self.act = F.leaky_relu
        self.sig = torch.sigmoid
        #self.structure_AE = StructureAE(in_node_dim,embed_dim,out_dim,dropout,act)
        '''
        for param in self.model.parameters():
            param.requires_grad = False
        '''
        #self.model.eval()
        #self.num_scales = num_scales
        self.num_scales=1
        #self.gru_att_l,self.gru_att_r,self.gru_bias,self.gru_lin_l,self.gru_lin_r = [],[],[],[],[]
        
        self._parameters = nn.ParameterList()
        self.gru_att_l=torch.nn.GRU(128,128,num_layers=5).to('cuda:0')

        self.gru_att_r=torch.nn.GRU(128,128,num_layers=num_scales).to('cuda:0')
        self.gru_bias=torch.nn.GRU(128,128,num_layers=num_scales).to('cuda:0')

        self.gru_lin_l=torch.nn.GRU(1433,1433,num_layers=num_scales).to('cuda:0')

        self.gru_lin_r=torch.nn.GRU(1433,1433,num_layers=num_scales).to('cuda:0')

        #self._parameters = nn.ParameterList([self.gru_att_l.parameters(),self.gru_att_r.parameters(),self.gru_bias.parameters(),self.gru_lin_l.parameters(),self.gru_lin_r.parameters()])    
        ''' 
        for scale in range(self.num_scales):

            self.gru_att_l.append(torch.nn.GRU(128,128,num_layers=num_scales))#.cuda())
            self.gru_att_r.append(torch.nn.GRU(128,128,num_layers=num_scales))#.cuda())
            self.gru_bias.append(torch.nn.GRU(128,128,num_layers=num_scales))#.cuda())
            self.gru_lin_l.append(torch.nn.GRU(1433,1433,num_layers=num_scales))#.cuda())
            self.gru_lin_r.append(torch.nn.GRU(1433,1433,num_layers=num_scales))#.cuda())
            
            self.register(self.gru_att_l[-1],scale,0)
            self.register(self.gru_att_r[-1],scale,1)
            self.register(self.gru_bias[-1],scale,2)
            self.register(self.gru_lin_l[-1],scale,3)
            self.register(self.gru_lin_r[-1],scale,4)
        '''
        scale = 1
        self.register(self.gru_att_l,scale,0)
        #self.register(self.gru_att_r,scale,1)
        #self.register(self.gru_bias,scale,2)
        #self.register(self.gru_lin_l,scale,3)
        #self.register(self.gru_lin_r,scale,4)
    
        self.init_att_w = self.glorot_init(1,128)#.cuda()
        self.init_bias_w = self.glorot_init(1,128)#.cuda()
        self.init_lin_w = self.glorot_init(128,1433)#.cuda()
        #import ipdb ; ipdb.set_trace() 
    def register(self,mod,sc,modnum):
        for p_ind,p in enumerate(mod.parameters()):
            p.requires_grad_()
            p.retain_grad = True
            #self.register_parameter(f'{str(sc)}_{modnum}_{p_ind}',p)
            self._parameters.append(p)
    def parameters(self):
        return self._parameters

    def glorot_init(self, in_size, out_size):
        import math
        stdv = 1. / math.sqrt(in_size)
        #init_range = np.sqrt(6.0/(in_size+out_size))
        initial = torch.rand(in_size, out_size)*(2*stdv)
        resh_initial = initial[None, :, :]
        return resh_initial.to('cuda:0')


    
    def forward(self, attrs, edge_index, adj):
        all_weights = []
        #import ipdb ; ipdb.set_trace()
        #import ipdb ; ipdb.set_trace()
        for out in range(self.num_scales):
            '''
            if out == 0:
                att_l_out,_ = self.gru_att_l[out](self.init_att_w)#,self.init_att_w)
                att_r_out,_ = self.gru_att_r[out](self.init_att_w)#,self.init_att_w)
                bias_out,_ = self.gru_bias[out](self.init_bias_w)#,self.init_bias_w)
                lin_l_out,_ = self.gru_lin_l[out](self.init_lin_w)#,self.init_lin_w)
                lin_r_out,_ = self.gru_lin_r[out](self.init_lin_w)#,self.init_lin_w)
            else:
                att_l_out,_ = self.gru_att_l[out](att_l_out)#,self.init_att_w)
                att_r_out,_ = self.gru_att_r[out](att_r_out)#,self.init_att_w)
                bias_out,_ = self.gru_bias[out](bias_out)#,self.init_bias_w)
                lin_l_out,_ = self.gru_lin_l[out](lin_l_out)#,self.init_lin_w)
                lin_r_out,_ = self.gru_lin_r[out](lin_r_out)#,self.init_lin_w)
            
            '''
            #import ipdb ; ipdb.set_trace()
            if out == 0:
                att_l_out,_ = self.gru_att_l(self.init_att_w)#,self.init_att_w)
                att_r_out,_ = self.gru_att_r(self.init_att_w)#,self.init_att_w)
                bias_out,_ = self.gru_bias(self.init_bias_w)#,self.init_bias_w)
                lin_l_out,_ = self.gru_lin_l(self.init_lin_w)#,self.init_lin_w)
                lin_r_out,_ = self.gru_lin_r(self.init_lin_w)#,self.init_lin_w)
            else:
                att_l_out,_ = self.gru_attr_l(att_l_out)#,self.init_att_w)
                att_r_out,_ = self.gru_att_r(att_r_out)#,self.init_att_w)
                bias_out,_ = self.gru_bias(bias_out)#,self.init_bias_w)
                lin_l_out,_ = self.gru_lin_l(lin_l_out)#,self.init_lin_w)
                lin_r_out,_ = self.gru_lin_r(lin_r_out)#,self.init_lin_w)
            att_l_out = self.sig(att_l_out) 
            weight_dict = {}
            import ipdb ; ipdb.set_trace()

            
            #self.state_dict['structure_AE.attention_layer.att_l'] = att_l_out
            #import ipdb ; ipdb.set_trace()
            '''
            sd = self.state_dict()
            
            sd['attention_layer.att_r'] = self.act(att_r_out)
            sd['attention_layer.bias'] = self.act(bias_out[0][0])
            sd['attention_layer.lin_r.weight'] = self.act(lin_l_out[0])
            sd['attention_layer.lin_l.weight'] = self.act(lin_r_out[0])
            self.load_state_dict(sd)
            '''
            '''
            weight_dict['structure_AE.attention_layer.att_l'] = att_l_out
            weight_dict['structure_AE.attention_layer.att_r'] = att_r_out
            weight_dict['structure_AE.attention_layer.bias'] = bias_out[0][0]
            weight_dict['structure_AE.attention_layer.lin_r.weight'] = lin_l_out[0]
            weight_dict['structure_AE.attention_layer.lin_l.weight'] = lin_r_out[0]
            '''
            '''
            weight_dict['attention_layer.att_l'] = att_l_out
            weight_dict['attention_layer.att_r'] = att_r_out
            weight_dict['attention_layer.bias'] = bias_out[0][0]
            weight_dict['attention_layer.lin_r.weight'] = lin_l_out[0]
            weight_dict['attention_layer.lin_l.weight'] = lin_r_out[0]
            '''
            all_weights.append(weight_dict)
        '''
        import ipdb ; ipdb.set_trace()
        for sc in range(self.num_scales):
            # 2.5 update state dict
            
            #state_dict = self.state_dict
            state_dict = self.structure_AE.state_dict()
            for k,v in all_weights[sc].items():
                #state_dict[k] = v
                state_dict[k].copy_(v)
            self.structure_AE.load_state_dict(state_dict)
            #self.model.eval()
            import ipdb ; ipdb.set_trace()
            #self.model.train()
            #A_hat, X_hat = self.model(attrs, edge_index, adj)
        '''
        #A_hat, _ = self.structure_AE(attrs,edge_index)
        embed_x = self.act(self.attention_layer(attrs, edge_index))
        # decoder
        A_hat = self.sig(embed_x @ embed_x.T)
        #A_hat = self.sig(attrs @ attrs.T)
        #return A_hat, attrs, all_weights
        return A_hat, attrs, all_weights
class AnomalyDAE_evolve(BaseDetector):
    """
    AnomalyDAE (Dual autoencoder for anomaly detection on attributed networks):
    AnomalyDAE is an anomaly detector that. consists of a structure autoencoder 
    and an attribute autoencoder to learn both node embedding and attribute 
    embedding jointly in latent space. The structural autoencoer uses Graph Attention
    layers. The reconstruction mean square error of the decoders are defined 
    as structure anamoly score and attribute anomaly score, respectively, 
    with two additional penalties on the reconstructed adj matrix and 
    node attributes (force entries to be nonzero).

    See: cite 'fan2020anomalydae' for details.

    Parameters
    ----------
    embed_dim :  int, optional
        Hidden dimension of model. Defaults: `8``.
    out_dim : int, optional
        Dimension of the reduced representation after passing through the 
        structure autoencoder and attribute autoencoder. Defaults: ``4``.
    dropout : float, optional
        Dropout rate. Defaults: ``0.2``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Defaults: ``1e-5``.
    act : callable activation function or None, optional
        Activation function if not None.
        Defaults: ``torch.nn.functional.relu``.
    alpha : float, optional
        loss balance weight for attribute and structure.
        Defaults: ``0.5``.
    theta: float, optional
         greater than 1, impose penalty to the reconstruction error of
         the non-zero elements in the adjacency matrix
         Defaults: ``1.01``
    eta: float, optional
         greater than 1, imporse penalty to the reconstruction error of 
         the non-zero elements in the node attributes
         Defaults: ``1.01``
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Defaults: ``0.1``.
    lr : float, optional
        Learning rate. Defaults: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Defaults: ``5``.
    gpu : int
        GPU Index, -1 for using CPU. Defaults: ``0``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Defaults: ``False``.

    Examples
    --------
    >>> from pygod.models import AnomalyDAE
    >>> model = AnomalyDAE()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 embed_dim=1433,
                 out_dim=128,
                 dropout=0.,
                 weight_decay=1e-9,
                 act=torch.sigmoid,
                 alpha=0.1,
                 theta=3.,
                 eta=10.,
                 contamination=1e-10,
                 lr=0.01,
                 epoch=80,
                 gpu=0,
                 verbose=True):
        super(AnomalyDAE_evolve, self).__init__(contamination=contamination)

        # model param
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.alpha = alpha
        self.theta = theta
        self.eta = eta

        # training param
        self.lr = lr
        self.epoch = epoch
        if gpu >= 0 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(gpu)
        else:
            self.device = 'cpu'

        # other param
        self.verbose = verbose
        self.model = None


    def fit(self, G, y_true=None):
        """
        Description
        -----------
        Fit detector with input data.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        y_true : numpy.array, optional (default=None)
            The optional outlier ground truth labels used to monitor the
            training progress. They are not used to optimize the
            unsupervised model.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        attrs, adj, edge_index,y_true = self.process_graph(G)
        #import ipdb ; ipdb.set_trace()
        # 0. initialize models
    
        self.grus = EGCN(1,in_node_dim=attrs.shape[1],
                                     in_num_dim=attrs.shape[0],
                                     embed_dim=self.embed_dim,
                                     out_dim=self.out_dim,
                                     dropout=self.dropout,
                                     act=self.act)
        # TODO: get output of sequence within gru
        num_scales = 2
        ''''
        for param in self.grus.parameters():
            param.requires_grad = True
            param.requires_grad_()
        '''
        #import ipdb ; ipdb.set_trace()
        '''
        optimizer = torch.optim.Adam(self.grus.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        '''
        optimizer = torch.optim.Adam(self.grus.parameters(),
                                     lr=self.lr, weight_decay=self.weight_decay)
        #import ipdb ; ipdb.set_trace()
        self.grus.train()
        for epoch in range(self.epoch):
            scores = None
            # 1. use GRU to get paramsi
            #weights = self.grus()
            # 2. get model(s) results at each layer
            #combine_loss = 0
            '''
            for sc in range(num_scales):
                
                # 2.5 update state dict
                state_dict = self.model.state_dict()
                #import ipdb ; ipdb.set_trace()
                for k,v in weights[sc].items():
                    #state_dict[k] = v
                    state_dict[k].copy_(v)
                self.model.load_state_dict(state_dict)
                #self.model.eval()
                #self.model.train()
                optimizer.zero_grad()
                A_hat, X_hat = self.model(attrs, edge_index, adj)
                #import ipdb ; ipdb.set_trace()
                
                loss, struct_loss, feat_loss = self.loss_func(adj, A_hat,
                                                              attrs, X_hat)
                loss_mean = torch.mean(loss)
                combine_loss += loss_mean
                if sc == 0:
                    scores = loss
                else:
                    scores = torch.add(scores,loss)
            '''
            A_hat, X_hat,all_weights = self.grus(attrs,edge_index,adj)
            loss, struct_loss, feat_loss = self.loss_func(adj, A_hat, attrs, X_hat)
            loss_mean = torch.mean(loss)
            scores = loss
            optimizer.zero_grad()
            #import ipdb ; ipdb.set_trace()
            # 3. combine losses at scales
            #combine_loss.requires_grad_()
            #loss_mean.requires_grad_()
            loss_mean.backward()
            optimizer.step()
            #self.grus.forward()
            #score = loss.detach().cpu().numpy()
            if self.verbose:
                print("Epoch:", '%04d' % epoch, "train_loss=",
                      "{:.5f}".format(loss_mean.item()), "train/struct_loss=",
                      "{:.5f}".format(struct_loss.item()), "train/feat_loss=",
                      "{:.5f}".format(feat_loss.item()))#, "train/AUC=",
                      #"{:.5f}".format(eval_roc_auc(y_true,score)))
        '''
        self.model.eval()
        A_hat, X_hat = self.model(attrs, edge_index)
        loss, struct_loss, feat_loss = self.loss_func(adj, A_hat,
                                                      attrs, X_hat)
        score = loss.detach().cpu().numpy()
        '''
        self.decision_scores_ = score
        #self._process_decision_scores()
        return self
    
    def decision_function(self, G):
        """
        Description
        -----------
        Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        anomaly_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        '''
        check_is_fitted(self, ['model'])

        # get needed data object from the input data
        attrs, adj, edge_index, y_label = self.process_graph(G)

        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        A_hat, X_hat = self.model(attrs, edge_index)
        outlier_scores, _, _ = self.loss_func(adj, A_hat, attrs, X_hat)
        return outlier_scores.detach().cpu().numpy()
        '''
        pass
    def process_graph(self, G):
        """
        Description
        -----------
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        x : torch.Tensor
            Attribute (feature) of nodes.
        adj : torch.Tensor
            Adjacency matrix of the graph.
        edge_index : torch.Tensor
            Edge list of the graph.
        """
        #edge_index = G.edge_index
        edge_index = torch.tensor(G['graph']).nonzero().t().contiguous()
        #  via sparse matrix operation
        dense_adj \
            = SparseTensor(row=edge_index[0], col=edge_index[1]).to_dense()

        # adjacency matrix normalization
        rowsum = dense_adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj = (dense_adj * d_mat_inv_sqrt).T * d_mat_inv_sqrt
        #import ipdb ; ipdb.set_trace()
        edge_index = edge_index.to(self.device)
        adj = adj.to(self.device)
        x = G['feat'].to(self.device)
        # return data objects needed for the network
        return x, adj, edge_index, G['truth']

    def loss_func(self,
                  adj,
                  A_hat,
                  attrs,
                  X_hat):
        # generate hyperparameter - structure penalty
        reversed_adj = torch.ones(adj.shape).to(self.device) - adj
        import random
        #thetas = torch.where(reversed_adj > 0, reversed_adj,
        #                     torch.full(adj.shape, random.random()).to(self.device))
        thetas = torch.full(adj.shape,random.random()).to(self.device)
        # generate hyperparameter - node penalty
        reversed_attr = torch.ones(attrs.shape).to(self.device) - attrs
        '''
        etas = torch.where(reversed_attr == 1, reversed_attr,
                           torch.full(attrs.shape, self.eta).to(self.device))
       
        '''
        # Attribute reconstruction loss
        
        diff_attribute = torch.pow(X_hat -
                                   attrs, 2)# * etas
        attribute_reconstruction_errors = \
            torch.sqrt(torch.sum(diff_attribute, 1))
        attribute_cost = torch.mean(attribute_reconstruction_errors)
        
        # structure reconstruction loss
        diff_structure = torch.pow(A_hat - adj, 2) 
        structure_reconstruction_errors = \
            torch.sqrt(torch.sum(diff_structure, 1))
        structure_cost = torch.mean(structure_reconstruction_errors)
        
        cost = self.alpha * attribute_reconstruction_errors + (
                1 - self.alpha) * structure_reconstruction_errors * thetas
        import ipdb ; ipdb.set_trace()
        #loss = nn.BCELoss()
        #cost = loss(A_hat,adj) 
        loss = structure_reconstruction_errors
        return cost, structure_cost, structure_cost# attribute_cost

