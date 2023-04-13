from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, global_mean_pool
import torch
import torch.nn.functional as F
import torch.nn as nn


class HCL(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, feat_dim, str_dim):
        super(HCL, self).__init__()

        self.embedding_dim = hidden_dim * num_gc_layers

        self.encoder_feat = Encoder_GIN(feat_dim, hidden_dim, num_gc_layers)
        self.encoder_str = Encoder_GIN(str_dim, hidden_dim, num_gc_layers)
        self.proj_head_feat_g = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_str_g = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                            nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_feat_n = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_str_n = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                            nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_b = nn.Sequential(nn.Linear(self.embedding_dim * 2, self.embedding_dim), nn.ReLU(inplace=True),
                                            nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def init_structural_encoding(self, edge_index, rw_dim=16, dg_dim=16):
        A = to_scipy_sparse_matrix(edge_index, num_nodes=torch.unique(edge_index).shape[0])
        D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

        Dinv = sp.diags(D)
        RW = A * Dinv
        M = RW

        RWSE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(rw_dim-1):
            M_power = M_power * M
            RWSE.append(torch.from_numpy(M_power.diagonal()).float())
        RWSE = torch.stack(RWSE,dim=-1)

        g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(0, dg_dim - 1)
        DGSE = torch.zeros([g.num_nodes, dg_dim])
        for i in range(len(g_dg)):
            DGSE[i, int(g_dg[i])] = 1

        return torch.cat([RWSE, DGSE], dim=1) 

    def get_b(self, x_f, x_s, edge_index, batch, num_graphs):
        g_f, _ = self.encoder_feat(x_f, edge_index, batch)
        g_s, _ = self.encoder_str(x_s, edge_index, batch)
        b = self.proj_head_b(torch.cat((g_f, g_s), 1))
        return b

    def forward(self, x_f, edge_index, batch, num_graphs):

        x_s = self.init_structural_encoding(edge_index)

        g_f, n_f = self.encoder_feat(x_f, edge_index, batch)
        g_s, n_s = self.encoder_str(x_s, edge_index, batch)

        b = self.proj_head_b(torch.cat((g_f, g_s), 1))

        g_f = self.proj_head_feat_g(g_f)
        g_s = self.proj_head_str_g(g_s)

        n_f = self.proj_head_feat_n(n_f)
        n_s = self.proj_head_str_n(n_s)

        return b, g_f, g_s, n_f, n_s

    @staticmethod
    def scoring_b(b, cluster_result, temperature = 0.2):
        im2cluster, prototypes, density = cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density']

        batch_size, _ = b.size()
        b_abs = b.norm(dim=1)
        prototypes_abs = prototypes.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', b, prototypes) / torch.einsum('i,j->ij', b_abs, prototypes_abs)
        sim_matrix = torch.exp(sim_matrix / (temperature * density))

        v, id = torch.min(sim_matrix, 1)

        return v

    @staticmethod
    def calc_loss_b(b, index, cluster_result, temperature=0.2):
        im2cluster, prototypes, density = cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density']
        pos_proto_id = im2cluster[index].cpu().tolist()

        batch_size, _ = b.size()
        b_abs = b.norm(dim=1)
        prototypes_abs = prototypes.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', b, prototypes) / torch.einsum('i,j->ij', b_abs, prototypes_abs)
        sim_matrix = torch.exp(sim_matrix / (temperature * density))
        pos_sim = sim_matrix[range(batch_size), pos_proto_id]

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss + 1e-12)
        return loss

    @staticmethod
    def calc_loss_n(x, x_aug, batch, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        node_belonging_mask = batch.repeat(batch_size,1)
        node_belonging_mask = node_belonging_mask == node_belonging_mask.t()

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature) * node_belonging_mask
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-12)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-12)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        loss = global_mean_pool(loss, batch)

        return loss

    @staticmethod
    def calc_loss_g(x, x_aug, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        return loss


class Encoder_GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder_GIN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            self.convs.append(conv)


    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)


class Encoder_GCN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder_GCN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                conv = GCNConv(dim, dim)
            else:
                conv = GCNConv(num_features, dim)
            self.convs.append(conv)


    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        xpool = [global_mean_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)