# -*- coding: utf-8 -*-
""" Graph Convolutional Network Autoencoder
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import torch_geometric
import dgl
import networkx as nx
import scipy
import torch.nn.functional as F
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader

from . import BaseDetector
from .basic_nn import GCN
from ..utils.metric import eval_roc_auc


class GCNAE(BaseDetector):
    """
    Vanila Graph Convolutional Networks Autoencoder

    See :cite:`yuan2021higher` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``0``.
    num_layers : int, optional
        Total number of layers in autoencoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``0``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.models import GCNAE
    >>> model = GCNAE()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 hid_dim=128,
                 num_layers=5,
                 dropout=0.3,
                 weight_decay=0.,
                 act=F.leaky_relu,
                 contamination=0.1,
                 lr=1e-3,
                 epoch=100,
                 gpu=0,
                 verbose=True,
                 start=1,
                 end=3,
                 adj=None):
        super(GCNAE, self).__init__(contamination=contamination)
        self.start_scale = start
        self.scales = end
        # model param
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.out_size=1681
        print('layers',end-start+1)
        # training param
        self.lr = lr
        self.epoch = epoch
        #import ipdb ; ipdb.set_trace()
        if gpu >= 0 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(gpu)
        else:
            self.device = 'cpu'
        # other param
        self.verbose = verbose
        self.model = None
        self.adj = torch.tensor(adj.todense()).to(self.device)

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
        x, edge_index = self.process_graph(G)
        adj = self.adj.to(self.device).float()
        self.models = []
        params = torch.nn.ParameterList([])
        all_scales=[2] 
        #for scale in range(self.start_scale,self.scales+1):
        for scale in all_scales:
            in_size = x.shape[1]
            if scale == 1:
                in_size = x.shape[1]
            in_size=400
            out_size = self.out_size
            # TODO: for adj reconstruction, remove
            #out_size = self.hid_dim
            out_size = self.out_size
            #out_size = x.shape[1]
            model = GCN(in_channels=in_size,
                             hidden_channels=self.hid_dim,
                             num_layers=scale,
                             out_channels=out_size,
                             dropout=self.dropout,
                             act=self.act).to(self.device)
            #import ipdb ; ipdb.set_trace()
            for p in model.parameters():
                params.append(p)
            self.models.append(model)
        #self.linear = torch.nn.Linear((self.scales-self.start_scale+1)*self.hid_dim,x.shape[1]).to(self.device)
        #self.linear = torch.nn.GRU(out_size,x.shape[1]).to(self.device)
        #for p in self.linear.parameters():
        #    params.append(p)
        optimizer = torch.optim.Adam(params,
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        data_loader = DataLoader(torch.arange(x.shape[0]), batch_size=1681,shuffle=True) 
        score = None
        adj=torch_geometric.utils.to_dense_adj(edge_index)[0]
        nx_graph = nx.from_numpy_matrix(adj.numpy())
        for epoch in range(self.epoch):
            print(epoch)
            for iter, bg in enumerate(data_loader):
                print(iter)
                for model in self.models:
                    model.train()
                #self.linear.train()
                batched_x, batched_edges = x[bg],list(nx_graph.edges(bg.numpy()))
                batched_edges = torch.Tensor([[torch.where(bg==a)[0][0].item() for a,b in batched_edges if a in bg and b in bg],[torch.where(bg==b)[0][0].item() for a,b in batched_edges if b in bg and a in bg]]).long()
                for b in bg:
                    batched_edges=torch.cat((batched_edges.t(),torch.Tensor([[torch.where(bg==b)[0][0].item(),torch.where(bg==b)[0][0].item()]]))).t()
                #batched_edges=batched_adj.nonzero().t().contiguous()
                batched_adj=torch_geometric.utils.to_dense_adj(batched_edges.type(torch.int64))[0]
                for ind,model in enumerate(self.models):
                    #import ipdb ; ipdb.set_trace()
                    x_ = model(batched_x,batched_edges.long())
                    if self.out_size == 1681:
                        x_ = x_ @ x_.T
                        x_ = torch.sigmoid(x_)
                        loss_ = F.binary_cross_entropy(x_.flatten(),batched_adj.flatten())
                    else:
                        loss_=F.mse_loss(x_,x,reduction='none')
                    '''
                    if ind == 0:
                        all_x = x_
                        _,all_x = self.linear(all_x[None,:,:])
                        x_ = x_[None,:,:]
                    else:
                        #all_x = torch.cat((all_x,x_),-1)
                        x_,all_x = self.linear(x_[None,:,:],all_x)
                    ''' 
                    #import ipdb ; ipdb.set_trace()
                    #x_ = F.relu(x_)#[0])
                    #x_ = F.relu(self.linear(all_x))
                    #import ipdb ; ipdb.set_trace()

                    #loss_ = F.mse_loss(x_, x)
                    #loss_ = F.mse_loss(x_,adj,reduction='none')
                    loss_ = torch.mean(loss_)
                    #print('epoch ',epoch,ind,loss_)
                    if ind == 0:
                        loss = loss_
                    else:
                        loss += loss_
                #import ipdb ; ipdb.set_trace()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #loss=F.mse_loss(x_, x, reduction='none')
                #loss = F.mse_loss(x_, adj,reduction='none')
                if self.out_size == 1681:
                    loss = F.binary_cross_entropy(x_,batched_adj,reduction='none')#.detach().cpu().numpy()
                else:
                    loss = F.mse_loss(x_,x,reduction='none')
                #score = scipy.stats.skew(loss.detach().cpu().numpy())
                score = torch.mean(loss,axis=1).detach().cpu().numpy()
                #score = loss
            if self.verbose:
                #import ipdb ; ipdb.set_trace()
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(epoch, torch.mean(loss).item()), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, score.detach().cpu().numpy())
                    print(" | AUC {:.4f}".format(auc), end='')
                print()
        
        #self.decision_scores_ = score.detach().cpu().numpy()
        self.decision_scores_ = score
        self._process_decision_scores()
        return self

    def decision_function(self, G):
        """
        Description
        -----------
        Predict raw anomaly score using the fitted detector. Outliers
        are assigned with larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        outlier_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        #check_is_fitted(self, ['model'])
        #self.model.eval()
        x, edge_index = self.process_graph(G)
        for model in self.models:
            model.eval()
        #self.linear.eval()
        adj = self.adj.to(self.device).float()
       
        data_loader = DataLoader(torch.arange(x.shape[0]), batch_size=1681,shuffle=True) 
        adj=torch_geometric.utils.to_dense_adj(edge_index)[0]
        nx_graph = nx.from_numpy_matrix(adj.numpy()) 

        #import ipdb ; ipdb.set_trace()

        scores = torch.zeros(x.shape[0])
        #import ipdb ; ipdb.set_trace()
        for bg_ in enumerate(data_loader):
            bg = bg_[1]
            batched_x, batched_edges = x[bg],list(nx_graph.edges(bg.numpy()))
            batched_edges = torch.Tensor([[torch.where(bg==a)[0][0].item() for a,b in batched_edges if a in bg and b in bg],[torch.where(bg==b)[0][0].item() for a,b in batched_edges if b in bg and a in bg]]).long()
            for b in bg:
                batched_edges=torch.cat((batched_edges.t(),torch.Tensor([[torch.where(bg==b)[0][0].item(),torch.where(bg==b)[0][0].item()]]))).t()
            #batched_edges=batched_adj.nonzero().t().contiguous()
            #import ipdb ; ipdb.set_trace()
            batched_adj=torch_geometric.utils.to_dense_adj(batched_edges.type(torch.int64))[0]
                
            for ind,model in enumerate(self.models):
                x_ = model(batched_x, batched_edges.type(torch.int64))
                if self.out_size == 1681:
                    x_ = x_ @ x_.T
                    x_ = torch.sigmoid(x_)
                    loss = F.binary_cross_entropy(x_,batched_adj.float(),reduction='none')
                else:
                    loss_=F.mse_loss(x_,x)
                    loss_ = torch.mean(loss_)
                    loss = F.mse_loss(x_, x,reduction='none')
                #import ipdb ; ipdb.set_trace()
                #loss = F.mse_loss(x_, self.adj, reduction='none')
                #import ipdb ;ipdb.set_trace()
                
                outlier_scores = torch.mean(loss,
                                    dim=1).detach().cpu().numpy()
                #outlier_scores = loss.detach().cpu().numpy()
                #outlier_scores = scores.detach().cpu().numpy()
                #outlier_scores = scipy.stats.skew(loss.detach().cpu().numpy())
                for b in bg:
                    scores[b] = float(outlier_scores[torch.where(bg==b)[0][0].item()])
                #scores.append(outlier_scores)
        return scores


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
        edge_index = torch.tensor(G['graph'].todense()).nonzero().t().contiguous()
        #  via sparse matrix operation
        from torch_sparse import SparseTensor
        dense_adj \
            = SparseTensor(row=edge_index[0], col=edge_index[1]).to_dense()

        # adjacency matrix normalization
        rowsum = dense_adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj = (dense_adj * d_mat_inv_sqrt).T * d_mat_inv_sqrt

        edge_index = edge_index.to(self.device)
        adj = adj.to(self.device)
        x = G['feat'].to(self.device)
        # return data objects needed for the network
        #return x, adj, edge_index, #G['truth']
        return x, edge_index

