import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import scipy.sparse as sp

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def generate_rwr_subgraph(dgl_graph, subgraph_size):
    #import ipdb; ipdb.set_trace()
    adj=dgl_graph.adjacency_matrix()
    adj=adj.sparse_resize_((dgl_graph.num_src_nodes(),dgl_graph.num_src_nodes()), adj.sparse_dim(),adj.dense_dim())
    src, dst = adj.coalesce().indices()
    dgl_graph = dgl.graph((src, dst)).to(dgl_graph.device)

    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.sampling.random_walk(dgl_graph, all_idx, restart_prob=0.95, length=subgraph_size)[0]#,
                                                             #max_nodes_per_seed=subgraph_size * 3)
    subv = []
    for i, trace in enumerate(traces):
        subv.append(torch.unique(trace, sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.sampling.random_walk(dgl_graph, [i], restart_prob=0.9, length = subgraph_size)#,
                                                                      #max_nodes_per_seed=subgraph_size * 5)
            subv[i] = torch.unique(cur_trace[0], sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= reduced_size) and (retry_time > 10):
                subv[i] = (subv[i] * reduced_size)

        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)
    return subv

class GCN(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
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

    

    def generate_rwr_subgraph(self, dgl_graph, subgraph_size):
        all_idx = list(range(dgl_graph.number_of_nodes()))
        reduced_size = subgraph_size - 1
        traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1.0,
                                                                max_nodes_per_seed=subgraph_size * 3)
        subv = []

        for i, trace in enumerate(traces):
            subv.append(torch.unique(torch.cat(trace), sorted=False).tolist())
            retry_time = 0
            while len(subv[i]) < reduced_size:
                cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9,
                                                                        max_nodes_per_seed=subgraph_size * 5)
                subv[i] = torch.unique(torch.cat(cur_trace[0]), sorted=False).tolist()
                retry_time += 1
                if (len(subv[i]) <= reduced_size) and (retry_time > 10):
                    subv[i] = (subv[i] * reduced_size)

            subv[i] = subv[i][:reduced_size]
            subv[i].append(i)
        return subv

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)


class AvgReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq,1).values


class MinReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0,2,1)
        sim = torch.matmul(seq,query)
        sim = F.softmax(sim,dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq,sim)
        out = torch.sum(out,1)
        return out


class Discriminator(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
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

    def forward(self, c, h_pl, s_bias1=None, s_bias2=None):
        scs = []
        scs.append(self.f_k(h_pl, c))
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-1, :].unsqueeze(0), c_mi[:-1, :]), dim=0)
            scs.append(self.f_k(h_pl, c_mi))
        logits = torch.cat(tuple(scs))
        return logits

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    import numpy as np
    rowsum = np.array(features.sum(1).detach().cpu())
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
    #return features.todense(), sparse_to_tuple(features)

class SLGad(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, subgraph_size=4):
        super(SLGad, self).__init__()
        self.read_mode = readout
        self.gcn_enc = GCN(n_in, n_h, activation)
        self.gcn_dec = GCN(n_h, n_in, activation)
        self.subgraph_size = subgraph_size

        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc1 = Discriminator(n_h, negsamp_round)
        self.disc2 = Discriminator(n_h, negsamp_round)
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, dgl_graph, adj, raw_features, sparse=False, msk=None, samp_bias1=None, samp_bias2=None):
        
        cur_batch_size = 300
        subgraphs_1 = generate_rwr_subgraph(dgl_graph, self.subgraph_size)
        subgraphs_2 = generate_rwr_subgraph(dgl_graph, self.subgraph_size)
        
        ba1 = []
        ba2 = []
        bf1 = []
        bf2 = []
        raw_bf1 = []
        raw_bf2 = []
        ft_size = raw_features.shape[-1]
        #import ipdb ; ipdb.set_trace()
        
        features = preprocess_features(raw_features)
        adj = normalize_adj(adj.to_dense())
        adj = (adj + sp.eye(adj.shape[0])).todense()
        
        added_adj_zero_row = torch.zeros((cur_batch_size, 1, self.subgraph_size)).to(dgl_graph.device)
        added_adj_zero_col = torch.zeros((cur_batch_size, self.subgraph_size + 1, 1)).to(dgl_graph.device)
        added_adj_zero_col[:, -1, :] = 1.
        added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(dgl_graph.device)
        idx = torch.arange(cur_batch_size)
        for i in idx:
            cur_adj_1 = adj[:, subgraphs_1[i], :][:, :, subgraphs_1[i]]
            cur_feat_1 = features[:, subgraphs_1[i], :]
            raw_cur_feat_1 = raw_features[:, subgraphs_1[i], :]
            cur_adj_2 = adj[:, subgraphs_2[i], :][:, :, subgraphs_2[i]]
            cur_feat_2 = features[:, subgraphs_2[i], :]
            raw_cur_feat_2 = raw_features[:, subgraphs_2[i], :]
            ba1.append(cur_adj_1)
            bf1.append(cur_feat_1)
            raw_bf1.append(raw_cur_feat_1)
            ba2.append(cur_adj_2)
            bf2.append(cur_feat_2)
            raw_bf2.append(raw_cur_feat_2)

            ba1 = torch.cat(ba1)
            ba1 = torch.cat((ba1, added_adj_zero_row), dim=1)
            ba1 = torch.cat((ba1, added_adj_zero_col), dim=2)
            ba2 = torch.cat(ba2)
            ba2 = torch.cat((ba2, added_adj_zero_row), dim=1)
            ba2 = torch.cat((ba2, added_adj_zero_col), dim=2)

            bf1 = torch.cat(bf1)
            bf1 = torch.cat((bf1[:, :-1, :], added_feat_zero_row, bf1[:, -1:, :]), dim=1)
            bf2 = torch.cat(bf2)
            bf2 = torch.cat((bf2[:, :-1, :], added_feat_zero_row, bf2[:, -1:, :]), dim=1)

            raw_bf1 = torch.cat(raw_bf1)
            raw_bf1 = torch.cat((raw_bf1[:, :-1, :], added_feat_zero_row, raw_bf1[:, -1:, :]), dim=1)
            raw_bf2 = torch.cat(raw_bf2)
            raw_bf2 = torch.cat((raw_bf2[:, :-1, :], added_feat_zero_row, raw_bf2[:, -1:, :]), dim=1)

        seq1, seq2, seq3, seq4, adj1, adj2 = bf1, bf2, raw_bf1, raw_bf2, ba1, ba2
        
        h_1 = self.gcn_enc(seq1, adj1, sparse)
        h_2 = self.gcn_enc(seq2, adj2, sparse)
        h_3 = self.gcn_enc(seq3, adj1, sparse)
        h_4 = self.gcn_enc(seq4, adj2, sparse)

        f_1 = self.gcn_dec(h_3, adj1, sparse)
        f_2 = self.gcn_dec(h_4, adj2, sparse)

        if self.read_mode != 'weighted_sum':
            h_mv_1 = h_1[:, -1, :]
            h_mv_2 = h_2[:, -1, :]
            c1 = self.read(h_1[:, :-1, :])
            c2 = self.read(h_2[:, :-1, :])
        else:
            h_mv_1 = h_1[:, -1, :]
            h_mv_2 = h_2[:, -1, :]
            c1 = self.read(h_1[:, :-1, :], h_1[:, -2:-1, :])
            c2 = self.read(h_2[:, :-1, :], h_2[:, -2:-1, :])

        ret1 = self.disc1(c1, h_mv_2, samp_bias1, samp_bias2)
        ret2 = self.disc2(c2, h_mv_1, samp_bias1, samp_bias2)
        ret = torch.cat((ret1, ret2), dim=-1).mean(dim=-1).unsqueeze(dim=-1)
        import ipdb ; ipdb.set_trace()
        # TODO?
        return ret, f_1, f_2
        #logits, f_1, f_2 

    def inference(self, seq1, seq2, seq3, seq4, adj1, adj2, sparse=False):
        h_1 = self.gcn_enc(seq1, adj1, sparse)
        h_2 = self.gcn_enc(seq2, adj2, sparse)
        h_3 = self.gcn_enc(seq3, adj1, sparse)
        h_4 = self.gcn_enc(seq4, adj2, sparse)

        f_1 = self.gcn_dec(h_3, adj1, sparse)
        f_2 = self.gcn_dec(h_4, adj2, sparse)

        dist1 = self.pdist(f_1[:, -2, :], seq3[:, -1, :])
        dist2 = self.pdist(f_2[:, -2, :], seq4[:, -1, :])
        dist = 0.5 * (dist1 + dist2)

        if self.read_mode != 'weighted_sum':
            h_mv_1 = h_1[:, -1, :]
            h_mv_2 = h_2[:, -1, :]
            c1 = self.read(h_1[:, :-1, :])
            c2 = self.read(h_2[:, :-1, :])
        else:
            h_mv_1 = h_1[:, -1, :]
            h_mv_2 = h_2[:, -1, :]
            c1 = self.read(h_1[:, :-1, :], h_1[:, -2:-1, :])
            c2 = self.read(h_2[:, :-1, :], h_2[:, -2:-1, :])

        ret1 = self.disc1(c1, h_mv_2, None, None)
        ret2 = self.disc2(c2, h_mv_1, None, None)
        ret = torch.cat((ret1, ret2), dim=-1).mean(dim=-1).unsqueeze(dim=-1)
        return ret, dist