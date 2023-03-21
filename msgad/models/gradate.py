# -*- coding: utf-8 -*-
"""ANEMONE: Graph Anomaly Detection 
with Multi-Scale Contrastive Learning (ANEMONE)"""
# Author: Canyu Chen <cchen151@hawk.iit.edu>
# License: BSD 2 clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import random
import os
import dgl

from torch_geometric.utils import to_dense_adj
from torch_cluster import random_walk

from .base import BaseDetector

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).T
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def generate_rw_subgraph(graph_, adj, nb_nodes, subgraph_size):
    """Generate subgraph with random walk algorithm."""
    #src, dst = adj.coalesce().indices()
    src,dst=torch.nonzero(adj[0]).T
    graph = dgl.graph((src, dst)).to(graph_.device)
    all_idx = torch.tensor(list(range(graph.number_of_nodes())))

    traces = dgl.sampling.random_walk(graph, all_idx.to(graph_.device), length=3)
    subv = traces[0].tolist()
    return subv


class ANEMONE(BaseDetector):
    """
    ANEMONE (ANEMONE: Graph Anomaly Detection 
    with Multi-Scale Contrastive Learning)
    ANEMONE is a multi-scale contrastive self-supervised 
    learning-based method for graph anomaly detection. (beta)

    Parameters
    ----------
    lr : float, optional
        Learning rate. 
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.
    epoch : int, optional
        Maximum number of training epoch. 
    embedding_dim : 
    negsamp_ratio : 
    readout : 
    dataset : 
    weight_decay : 
    batch_size : 
    subgraph_size : 
    alpha :
    negsamp_ratio_patch :
    negsamp_ratio_context :
    auc_test_rounds : 
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``0``.

    Examples
    --------
    >>> from pygod.models import ANEMONE
    >>> model = ANEMONE()
    >>> model.fit(data)
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 lr=None,
                 verbose=False,
                 epoch=None,
                 embedding_dim=64,
                 negsamp_ratio=1,
                 readout='avg',
                 dataset='cora',
                 weight_decay=0.0,
                 batch_size=300,
                 subgraph_size=4,
                 alpha=0.1,
                 beta=0.1,
                 negsamp_ratio_patch=1,
                 negsamp_ratio_context=1,
                 auc_test_rounds=256,
                 contamination=0.1,
                 gpu=0):
        super(ANEMONE, self).__init__(contamination=contamination)

        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.negsamp_ratio = negsamp_ratio
        self.readout = readout
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.subgraph_size = subgraph_size
        self.auc_test_rounds = auc_test_rounds
        self.alpha = alpha
        self.beta = beta
        self.negsamp_ratio_patch = negsamp_ratio_patch
        self.negsamp_ratio_context = negsamp_ratio_context
        if gpu >= 0 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(gpu)
        else:
            self.device = 'cpu'

        if lr is None:
            if self.dataset in ['cora', 'citeseer', 'pubmed', 'Flickr']:
                self.lr = 1e-3
            elif self.dataset == 'ACM':
                self.lr = 5e-4
            elif self.dataset == 'BlogCatalog':
                self.lr = 3e-3
            else:
                self.lr = 1e-3

        if epoch is None:
            if self.dataset in ['cora', 'citeseer', 'pubmed']:
                self.num_epoch = 100
            elif self.dataset in ['BlogCatalog', 'Flickr', 'ACM']:
                self.num_epoch = 400
            else:
                self.num_epoch = 100

        self.verbose = verbose
        self.model = None

    def fit(self, G):
        """
        Fit detector with input data.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        x, adj, edge_index, labels = self.process_graph(G)

        adj = adj.cpu().numpy()
        x = x.cpu().numpy()

        nb_nodes = x.shape[0]
        ft_size = x.shape[1]

        adj = normalize_adj(adj)
        adj = (adj + sp.eye(adj.shape[0])).todense()

        x = torch.FloatTensor(x[np.newaxis])
        adj = torch.FloatTensor(adj[np.newaxis])

        self.model = ANEMONE_Base(ft_size,
                                  self.embedding_dim,
                                  'prelu',
                                  self.negsamp_ratio_patch,
                                  self.negsamp_ratio_context,
                                  self.readout)

        optimiser = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        b_xent_patch = nn.BCEWithLogitsLoss(reduction='none',
                                            pos_weight=torch.tensor(
                                                [self.negsamp_ratio_patch]))
        b_xent_context = nn.BCEWithLogitsLoss(reduction='none',
                                              pos_weight=torch.tensor([
                                                  self.negsamp_ratio_context]))

        batch_num = nb_nodes // self.batch_size + 1

        multi_epoch_ano_score = np.zeros((self.num_epoch, nb_nodes))

        for epoch in range(self.num_epoch):

            self.model.train()

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)

            subgraphs = generate_rw_subgraph(G, nb_nodes, self.subgraph_size)

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * self.batch_size:
                                  (batch_idx + 1) * self.batch_size]
                else:
                    idx = all_idx[batch_idx * self.batch_size:]

                cur_batch_size = len(idx)

                lbl_patch = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size),
                     torch.zeros(cur_batch_size * self.negsamp_ratio_patch))),
                    1)

                lbl_context = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size), torch.zeros(
                        cur_batch_size * self.negsamp_ratio_context))), 1)

                ba = []
                bf = []
                added_adj_zero_row = torch.zeros(
                    (cur_batch_size, 1, self.subgraph_size))
                added_adj_zero_col = torch.zeros(
                    (cur_batch_size, self.subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = x[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat(
                    (bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                logits_1, logits_2 = self.model(bf, ba)

                # Context-level
                loss_all_1 = b_xent_context(logits_1, lbl_context)
                loss_1 = torch.mean(loss_all_1)

                # Patch-level
                loss_all_2 = b_xent_patch(logits_2, lbl_patch)
                loss_2 = torch.mean(loss_all_2)

                loss = self.alpha * loss_1 + (1 - self.alpha) * loss_2

                loss.backward()
                optimiser.step()

                logits_1 = torch.sigmoid(torch.squeeze(logits_1))
                logits_2 = torch.sigmoid(torch.squeeze(logits_2))

                if self.alpha != 1.0 and self.alpha != 0.0:
                    if self.negsamp_ratio_context == 1 and \
                            self.negsamp_ratio_patch == 1:
                        ano_score_1 = - (logits_1[:cur_batch_size] -
                            logits_1[cur_batch_size:]).detach().cpu().numpy()
                        ano_score_2 = - (logits_2[:cur_batch_size] -
                            logits_2[cur_batch_size:]).detach().cpu().numpy()
                    else:
                        ano_score_1 = - (logits_1[:cur_batch_size] -
                            torch.mean(logits_1[cur_batch_size:].view(
                                cur_batch_size, self.negsamp_ratio_context),
                                dim=1)).detach().cpu().numpy()  # context
                        ano_score_2 = - (logits_2[:cur_batch_size] -
                            torch.mean(logits_2[cur_batch_size:].view(
                                    cur_batch_size, self.negsamp_ratio_patch),
                                    dim=1)).detach().cpu().numpy()  # patch
                    ano_score = self.alpha * ano_score_1 + (
                                1 - self.alpha) * ano_score_2
                elif self.alpha == 1.0:
                    if self.negsamp_ratio_context == 1:
                        ano_score = - (logits_1[:cur_batch_size] -
                            logits_1[cur_batch_size:]).detach().cpu().numpy()
                    else:
                        ano_score = - (logits_1[:cur_batch_size] -
                            torch.mean(logits_1[cur_batch_size:].view(
                                cur_batch_size, self.negsamp_ratio_context),
                                dim=1)).detach().cpu().numpy()  # context
                elif self.alpha == 0.0:
                    if self.negsamp_ratio_patch == 1:
                        ano_score = - (logits_2[:cur_batch_size] -
                            logits_2[cur_batch_size:]).detach().cpu().numpy()
                    else:
                        ano_score = - (logits_2[:cur_batch_size] -
                            torch.mean(logits_2[cur_batch_size:].view(
                                cur_batch_size, self.negsamp_ratio_patch),
                                dim=1)).detach().cpu().numpy()  # patch

                multi_epoch_ano_score[epoch, idx] = ano_score

        ano_score_final = np.mean(multi_epoch_ano_score, axis=0)

        self.decision_scores_ = ano_score_final
        self._process_decision_scores()
        return self

    def decision_function(self, G):
        """
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
        
        x, adj, edge_index, _ = self.process_graph(G)

        adj = adj.cpu().numpy()
        x = x.cpu().numpy()

        nb_nodes = x.shape[0]
        ft_size = x.shape[1]

        adj = normalize_adj(adj)
        adj = (adj + sp.eye(adj.shape[0])).todense()

        x = torch.FloatTensor(x[np.newaxis])
        adj = torch.FloatTensor(adj[np.newaxis])

        batch_num = nb_nodes // self.batch_size + 1

        multi_round_ano_score = np.zeros((self.auc_test_rounds, nb_nodes))

        # enable the evaluation mode
        self.model.eval()

        for round in range(self.auc_test_rounds):

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)

            subgraphs = generate_rw_subgraph(G, nb_nodes, self.subgraph_size)

            for batch_idx in range(batch_num):

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * self.batch_size:
                                  (batch_idx + 1) * self.batch_size]
                else:
                    idx = all_idx[batch_idx * self.batch_size:]

                cur_batch_size = len(idx)

                ba = []
                bf = []
                added_adj_zero_row = torch.zeros(
                    (cur_batch_size, 1, self.subgraph_size))
                added_adj_zero_col = torch.zeros(
                    (cur_batch_size, self.subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = x[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat(
                    (bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                with torch.no_grad():

                    test_logits_1, test_logits_2 = self.model(bf, ba)
                    test_logits_1 = torch.sigmoid(torch.squeeze(test_logits_1))
                    test_logits_2 = torch.sigmoid(torch.squeeze(test_logits_2))

                if self.alpha != 1.0 and self.alpha != 0.0:
                    if self.negsamp_ratio_context == 1 and \
                            self.negsamp_ratio_patch == 1:
                        ano_score_1 = - (test_logits_1[:cur_batch_size] -
                            test_logits_1[cur_batch_size:]).cpu().numpy()
                        ano_score_2 = - (test_logits_2[:cur_batch_size] -
                            test_logits_2[cur_batch_size:]).cpu().numpy()
                    else:
                        ano_score_1 = - (test_logits_1[:cur_batch_size] -
                            torch.mean(test_logits_1[cur_batch_size:].view(
                                cur_batch_size, self.negsamp_ratio_context),
                                dim=1)).cpu().numpy()  # context
                        ano_score_2 = - (test_logits_2[:cur_batch_size] -
                            torch.mean(test_logits_2[cur_batch_size:].view(
                                cur_batch_size, self.negsamp_ratio_patch),
                                dim=1)).cpu().numpy()  # patch
                    ano_score = self.alpha * ano_score_1 + \
                                (1 - self.alpha) * ano_score_2
                elif self.alpha == 1.0:
                    if self.negsamp_ratio_context == 1:
                        ano_score = - (test_logits_1[:cur_batch_size] -
                            test_logits_1[cur_batch_size:]).cpu().numpy()
                    else:
                        ano_score = - (test_logits_1[:cur_batch_size] -
                            torch.mean(test_logits_1[cur_batch_size:].view(
                                cur_batch_size, self.negsamp_ratio_context),
                                dim=1)).cpu().numpy()  # context
                elif self.alpha == 0.0:
                    if self.negsamp_ratio_patch == 1:
                        ano_score = - (test_logits_2[:cur_batch_size] -
                            test_logits_2[cur_batch_size:]).cpu().numpy()
                    else:
                        ano_score = - (test_logits_2[:cur_batch_size] -
                            torch.mean(test_logits_2[cur_batch_size:].view(
                                cur_batch_size, self.negsamp_ratio_patch),
                                dim=1)).cpu().numpy()  # patch

                multi_round_ano_score[round, idx] = ano_score

        ano_score_final = np.mean(multi_round_ano_score, axis=0)

        return ano_score_final

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
        y : torch.Tensor
            Labels of nodes.
        """
        edge_index = G.edge_index

        adj = to_dense_adj(edge_index)[0].to(self.device)

        edge_index = edge_index.to(self.device)
        adj = adj.to(self.device)
        x = G.x.to(self.device)

        if hasattr(G, 'y'):
            y = G.y
        else:
            y = None

        # return data objects needed for the network
        return x, adj, edge_index, y


class GRADATE(nn.Module):
    def __init__(self,
                 n_in,
                 n_h,
                 activation,
                 negsamp_round_patch,
                 negsamp_round_context,
                 readout,
                 batch_size,
                 alpha=0.1,
                 beta=0.1):
        super(GRADATE, self).__init__()
        self.read_mode = readout
        self.gcn_context = GCN(n_in, n_h, activation)
        self.gcn_patch = GCN(n_in, n_h, activation)
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.c_disc = Contextual_Discriminator(n_h, negsamp_round_context)
        self.p_disc = Patch_Discriminator(n_h, negsamp_round_patch)
        
        self.negsamp_ratio=1,
        self.readout='avg'
        self.weight_decay=0.0
        self.batch_size=300
        self.subgraph_size=4
        self.alpha=alpha
        self.beta=beta
        self.negsamp_ratio_patch=1
        self.negsamp_ratio_context=1
        self.auc_test_rounds=256
        self.contamination=0.1
        self.b_xent_patch = nn.BCEWithLogitsLoss(reduction='none',
                                            pos_weight=torch.tensor(
                                                [self.negsamp_ratio_patch]))
        self.b_xent_context = nn.BCEWithLogitsLoss(reduction='none',
                                              pos_weight=torch.tensor([
                                                  self.negsamp_ratio_context]))
                                                  
    
    def aug_random_edge(self, input_adj, drop_percent=0.2):
        """
        randomly delect partial edges and
        randomly add the same number of edges in the graph
        """
        percent = drop_percent / 2
        #row_idx, col_idx = input_adj.nonzero()
        row_idx, col_idx = input_adj.coalesce().indices()
        num_drop = int(len(row_idx) * percent)

        edge_index = [i for i in range(len(row_idx))]
        edges = dict(zip(edge_index, zip(row_idx, col_idx)))
        drop_idx = random.sample(edge_index, k=num_drop)

        list(map(edges.__delitem__, filter(edges.__contains__, drop_idx)))

        new_edges = list(zip(*list(edges.values())))
        new_row_idx = new_edges[0]
        new_col_idx = new_edges[1]
        data = np.ones(len(new_row_idx)).tolist()

        new_adj = sp.csr_matrix((data, (new_row_idx, new_col_idx)), shape=input_adj.shape)

        row_idx, col_idx = (new_adj.todense() - 1).nonzero()
        no_edges_cells = list(zip(row_idx, col_idx))
        add_idx = random.sample(no_edges_cells, num_drop)
        new_row_idx_1, new_col_idx_1 = list(zip(*add_idx))
        row_idx = new_row_idx + new_row_idx_1
        col_idx = new_col_idx + new_col_idx_1
        data = np.ones(len(row_idx)).tolist()

        new_adj = sp.csr_matrix((data, (row_idx, col_idx)), shape=input_adj.shape)

        return new_adj
    

    def run_model(self, bf, ba, sparse, samp_bias1, samp_bias2):
        h_1 = self.gcn_context(bf, ba, sparse)
        h_2 = self.gcn_patch(bf, ba, sparse)

        if self.read_mode != 'weighted_sum':
            c = self.read(h_1[:, :-1, :])
            h_mv = h_1[:, -1, :]
            h_unano = h_2[:, -1, :]
            h_ano = h_2[:, -2, :]
        else:
            c = self.read(h_1[:, :-1, :], h_1[:, -2:-1, :])
            h_mv = h_1[:, -1, :]
            h_unano = h_2[:, -1, :]
            h_ano = h_2[:, -2, :]

        ret1 = self.c_disc(c, h_mv, samp_bias1, samp_bias2)
        ret2 = self.p_disc(h_ano, h_unano, samp_bias1, samp_bias2)
        return ret1,ret2,c,h_mv

    # TODO: sampling
    def forward(self, G, adj, x, sparse=False, samp_bias1=None,
                samp_bias2=None):
        subgraph_size = 4 #TODO: manually set this for even subgraphs...
        self.batch_size = int(adj.shape[0]/subgraph_size)
        nb_nodes = G.number_of_nodes()
        cur_batch_size = self.batch_size*subgraph_size#nb_nodes
        ft_size = x.shape[1]
        idx = torch.arange(cur_batch_size)
        #idx = torch.arange(nb_nodes)
        adj=adj.sparse_resize_((G.num_src_nodes(),G.num_src_nodes()), adj.sparse_dim(),adj.dense_dim())
        adj_hat = self.aug_random_edge(adj, 0.2)
        adj=normalize_adj(adj.to_dense())

        adj = (adj + sp.eye(adj.shape[0])).todense()
        adj_hat=normalize_adj(adj_hat.todense())
        adj_hat = (adj_hat + sp.eye(adj_hat.shape[0])).todense()
        
        x = x.unsqueeze(0)
        adj = torch.FloatTensor(adj[np.newaxis]).to(G.device)
        adj_hat = torch.FloatTensor(adj_hat[np.newaxis]).to(G.device)

        subgraphs = generate_rw_subgraph(G, adj, nb_nodes, self.subgraph_size)
        '''
        lbl_patch = torch.unsqueeze(torch.cat(
            (torch.ones(cur_batch_size),
                torch.zeros(cur_batch_size * self.negsamp_ratio_patch))),
            1).to(device)
        '''
        lbl_patch = torch.cat(
            (torch.ones(cur_batch_size), torch.zeros(
                cur_batch_size * self.negsamp_ratio_context))).to(G.device)
        
        lbl_context = torch.cat(
            (torch.ones(cur_batch_size), torch.zeros(
                cur_batch_size * self.negsamp_ratio_context))).to(G.device)
        ba = []
        ba_hat = []
        bf = []
        added_adj_zero_row = torch.zeros(
            (cur_batch_size, 1, self.subgraph_size)).to(G.device)
        added_adj_zero_col = torch.zeros(
            (cur_batch_size, self.subgraph_size + 1, 1)).to(G.device)
        added_adj_zero_col[:, -1, :] = 1.
        added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(G.device)
        
        for i in idx:
            cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
            cur_feat = x[:, subgraphs[i], :]
            cur_adj_hat = adj_hat[:, subgraphs[i], :][:, :, subgraphs[i]]
            ba.append(cur_adj)
            ba_hat.append(cur_adj_hat)
            bf.append(cur_feat)
            
        #ba = ba.unsqueeze(1)
        #bf = bf.unsqueeze(1)
        ba = torch.cat(ba)
        ba_hat = torch.cat(ba_hat)

        #ba = ba.unsqueeze(1)
        ba = torch.cat((ba, added_adj_zero_row), dim=1)
        ba = torch.cat((ba, added_adj_zero_col), dim=2)
        ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
        ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)
        bf = torch.cat(bf)
        bf = torch.cat(
            (bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

        logits_1, logits_2, subgraph_embed, node_embed = self.run_model(bf, ba, sparse, samp_bias1, samp_bias2)
        logits_1_hat, logits_2_hat,  subgraph_embed_hat, node_embed_hat = self.run_model(bf, ba_hat, sparse, samp_bias1, samp_bias2)

        logits_1 = torch.sigmoid(torch.squeeze(logits_1))
        logits_2 = torch.sigmoid(torch.squeeze(logits_2))
        logits_1_hat = torch.sigmoid(torch.squeeze(logits_1_hat))
        logits_2_hat = torch.sigmoid(torch.squeeze(logits_2_hat))

        #subgraph-subgraph contrast loss
        subgraph_embed = F.normalize(subgraph_embed, dim=1, p=2)
        subgraph_embed_hat = F.normalize(subgraph_embed_hat, dim=1, p=2)
        sim_matrix_one = torch.matmul(subgraph_embed, subgraph_embed_hat.t())
        sim_matrix_two = torch.matmul(subgraph_embed, subgraph_embed.t())
        sim_matrix_three = torch.matmul(subgraph_embed_hat, subgraph_embed_hat.t())
        temperature = 1.0
        sim_matrix_one_exp = torch.exp(sim_matrix_one / temperature)
        sim_matrix_two_exp = torch.exp(sim_matrix_two / temperature)
        sim_matrix_three_exp = torch.exp(sim_matrix_three / temperature)
        nega_list = np.arange(0, cur_batch_size - 1, 1)
        nega_list = np.insert(nega_list, 0, cur_batch_size - 1)
        sim_row_sum = sim_matrix_one_exp[:, nega_list] + sim_matrix_two_exp[:, nega_list] + sim_matrix_three_exp[:, nega_list]
        sim_row_sum = torch.diagonal(sim_row_sum)
        sim_diag = torch.diagonal(sim_matrix_one)
        sim_diag_exp = torch.exp(sim_diag / temperature)
        NCE_loss = -torch.log(sim_diag_exp / (sim_row_sum))
        NCE_loss = torch.mean(NCE_loss)
        
        loss_all_1 = self.b_xent_context(logits_1, lbl_context)
        loss_all_1_hat = self.b_xent_context(logits_1_hat, lbl_context)
        loss_1 = torch.mean(loss_all_1)
        loss_1_hat = torch.mean(loss_all_1_hat)

        loss_all_2 = self.b_xent_patch(logits_2, lbl_patch)
        loss_all_2_hat = self.b_xent_patch(logits_2_hat, lbl_patch)
        loss_2 = torch.mean(loss_all_2)
        loss_2_hat = torch.mean(loss_all_2_hat)

        loss_1 = self.alpha * loss_1 + (1 - self.alpha) * loss_1_hat #node-subgraph contrast loss
        loss_2 = self.alpha * loss_2 + (1 - self.alpha) * loss_2_hat #node-node contrast loss
        loss = self.beta * loss_1 + (1 - self.beta) * loss_2 + 0.1 * NCE_loss #total loss

        with torch.no_grad():
            logits_1 = torch.sigmoid(torch.squeeze(logits_1))
            logits_2 = torch.sigmoid(torch.squeeze(logits_2))
            logits_1_hat = torch.sigmoid(torch.squeeze(logits_1_hat))
            logits_2_hat = torch.sigmoid(torch.squeeze(logits_2_hat))


            ano_score_1 = - (logits_1[:cur_batch_size] - torch.mean(logits_1[cur_batch_size:].view(
                cur_batch_size, self.negsamp_ratio_context), dim=1)).cpu().numpy()
            ano_score_1_hat = - (
                        logits_1_hat[:cur_batch_size] - torch.mean(logits_1_hat[cur_batch_size:].view(
                    cur_batch_size, self.negsamp_ratio_context), dim=1)).cpu().numpy()
            ano_score_2 = - (logits_2[:cur_batch_size] - torch.mean(logits_2[cur_batch_size:].view(
                cur_batch_size, self.negsamp_ratio_patch), dim=1)).cpu().numpy()
            ano_score_2_hat = - (
                        logits_2_hat[:cur_batch_size] - torch.mean(logits_2_hat[cur_batch_size:].view(
                    cur_batch_size, self.negsamp_ratio_patch), dim=1)).cpu().numpy()
            ano_score = self.beta * (self.alpha * ano_score_1 + (1 - self.alpha) * ano_score_1_hat)  + \
                        (1 - self.beta) * (self.alpha * ano_score_2 + (1 - self.alpha) * ano_score_2_hat)

        #return ret1, ret2, c, h_mv
        return loss, ano_score


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)),
                                  0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits


class Contextual_Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Contextual_Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, s_bias1=None, s_bias2=None):
        scs = []
        scs.append(self.f_k(h_pl, c))
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))
        logits = torch.cat(tuple(scs))
        return logits


class Patch_Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Patch_Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h_ano, h_unano, s_bias1=None, s_bias2=None):
        scs = []
        scs.append(self.f_k(h_unano, h_ano))
        h_mi = h_ano
        for _ in range(self.negsamp_round):
            h_mi = torch.cat((h_mi[-2:-1, :], h_mi[:-1, :]), 0)
            scs.append(self.f_k(h_unano, h_mi))
        logits = torch.cat(tuple(scs))
        return logits