import numpy as np
from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import scipy.sparse as sp


def create_adj_avg(adj_cur):
    '''
    create adjacency
    '''
    deg = np.sum(adj_cur, axis=1)
    deg = np.asarray(deg).reshape(-1)

    deg[deg != 1] -= 1

    deg = 1/deg
    deg_mat = np.diag(deg)
    adj_cur = adj_cur.dot(deg_mat.T).T

    return adj_cur

# embedding of a graph


def createWlEmbedding(node_features, adj_mat, h):
    graph_feat = []
    for it in range(h+1):
        if it == 0:
            graph_feat.append(node_features)
        else:
            adj_cur = adj_mat+np.identity(adj_mat.shape[0])

            adj_cur = create_adj_avg(adj_cur)

            np.fill_diagonal(adj_cur, 0)
            graph_feat_cur = 0.5 * \
                (np.dot(adj_cur, graph_feat[it-1]) + graph_feat[it-1])
            graph_feat.append(graph_feat_cur)
    return np.mean(np.concatenate(graph_feat, axis=1), axis=0)
    # return np.mean(graph_feat_cur, axis=0)


# embedding of each node
def createWlEmbedding1(node_features, adj_mat, h):
    graph_feat = []
    for it in range(h+1):
        if it == 0:
            graph_feat.append(node_features)
        else:
            adj_cur = adj_mat+np.identity(adj_mat.shape[0])

            adj_cur = create_adj_avg(adj_cur)

            np.fill_diagonal(adj_cur, 0)
            graph_feat_cur = 0.5 * \
                (np.dot(adj_cur, graph_feat[it-1]) + graph_feat[it-1])
            graph_feat.append(graph_feat_cur)
    return np.concatenate(graph_feat, axis=1)


def generate_hnodes(h_adj):
    h_adj = h_adj.tocoo()
    h_index = [[] for i in range(h_adj.shape[0])]
    for i, j in zip(h_adj.row, h_adj.col):
        h_index[i].append(j)
    return h_index


def generate_hadj(adj, h):
    adj_h = sp.eye(adj.shape[0])
    adj_tot = sp.eye(adj.shape[0])
    for i in range(h):
        adj_h = adj_h * adj
        adj_tot = adj_tot + adj_h
    return adj_tot


# Generate h_nodes and their height
def generate_h_nodes_n_dict(adj, h):
    adj_h = sp.eye(adj.shape[0])
    M = [{i: 0} for i in range(adj.shape[0])]
    h_index = [[i] for i in range(adj.shape[0])]
    for _ in range(h):
        #adj_h = sp.coo_matrix(adj_h * adj)
        adj_h = sp.coo_matrix(adj_h.todense()*adj)
        for i, j in zip(adj_h.row, adj_h.col):
            if j in M[i]:
                continue
            else:
                M[i][j] = _ + 1
                h_index[i].append(j)
    return M, h_index


def generate_subgraph_embeddings(attr, adj, subgraph_index, h):
    embedding = []
    for i in range(adj.shape[0]):
        root_feature = attr[i, :]
        feature = attr[subgraph_index[i]]
        feature = feature - np.tile(root_feature, (len(subgraph_index[i]), 1))
        adj_i = adj[subgraph_index[i], :][:, subgraph_index[i]]
        embedding.append(createWlEmbedding(feature, adj_i, h).reshape(1, -1))
    return np.concatenate(embedding, axis=0)


def subgraph_embeddings(attr, adj, h):
    M, h_index = generate_h_nodes_n_dict(adj, h)
    embedding = generate_subgraph_embeddings(attr, adj, h_index, h)
    return embedding, M

class GCAD():
    data = None
    centroid = []
    def __init__(self, psi, t,h):
        self.psi = psi
        self.t = t
        self.h = h

    def fit_transform(self, adj, attr):
        attr = attr.detach().cpu().numpy()
        embedding, M = subgraph_embeddings(attr, adj.detach().cpu().to_dense().numpy(), self.h)
        self.data = embedding
        self.centroid = []
        self.centroids_radius = []
        sn = self.data.shape[0]
        n, d = self.data.shape
        IDX = np.array([])  #column index
        V = []
        for i in range(self.t):
            subIndex = sample(range(sn), self.psi)
            self.centroid.append(subIndex)
            tdata = self.data[subIndex, :]
            tt_dis = cdist(tdata, tdata)
            radius = [] #restore centroids' radius
            for r_idx in range(self.psi):
                r = tt_dis[r_idx]
                r[r<0] = 0
                r = np.delete(r,r_idx)
                radius.append(np.min(r))
            self.centroids_radius.append(radius)
            nt_dis = cdist(tdata, self.data)
            centerIdx = np.argmin(nt_dis, axis=0)
            for j in range(n):
                V.append(int(nt_dis[centerIdx[j],j] <= radius[centerIdx[j]]))
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t) #row index
        #V = np.ones(self.t * n) #value
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        #scores = kmembeddings.dot(ndata.transpose())

        mean_embedding =np.mean(ndata, axis=0)
        scores = ndata.dot(mean_embedding.transpose())
        return scores.T
        #return ndata
    '''
    def transform(self, newdata):
        n, d = newdata.shape
        IDX = np.array([])
        V = []
        for i in range(self.t):
            subIndex = self.centroid[i]
            radius = self.centroids_radius[i]
            tdata = self.data[subIndex, :]
            dis = cdist(tdata, newdata)
            centerIdx = np.argmin(dis, axis=0)
            for j in range(n):
                V.append(int(dis[centerIdx[j], j] <= radius[centerIdx[j]]))
            IDX = np.concatenate((IDX, centerIdx + i * self.psi), axis=0)
        IDR = np.tile(range(n), self.t)
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.psi))
        return
    '''