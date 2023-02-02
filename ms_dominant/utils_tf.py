import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import networkx as nx
from igraph import Graph
import copy

def load_anomaly_detection_dataset(dataset, sc, mlp, parity, datadir='data'):
    data_mat = sio.loadmat(f'data/{dataset}.mat')
    mlp=True
    if mlp:
        if 'cora' in dataset:
            feats = [torch.FloatTensor(data_mat['Attributes'].toarray())]
        else:
            feats = [torch.FloatTensor(data_mat['Attributes'])]
    else:
        feats = [] 
    
    adj = data_mat['Network']
    truth = data_mat['Label']
    truth = truth.flatten()
    adj_no_loop = copy.deepcopy(adj)
    if dataset != 'weibo':
        sc_label = data_mat['scale_anomaly_label']
    else:
        sc_label = data_mat['scale_anomaly_label'][0]
    import ipdb ; ipdb.set_trace()
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = adj_no_loop + sp.eye(adj_no_loop.shape[0])
    if dataset == 'weibo':
        adj = adj.toarray()
    edge_list = Graph.Adjacency(adj_no_loop.toarray()).get_edgelist()
    # edge removal
    train_graph = nx.from_numpy_matrix(adj_no_loop.toarray())
    num_edges = np.count_nonzero(adj_no_loop.toarray())/2
    val_found = False
    random.shuffle(edge_list)
    num_edges_val = int(num_edges*0.2)
    edges_rem, edges_keep = edge_list[:num_edges_val],edge_list[num_edges_val:]
    nodes_val = np.unique(list(sum(edges_rem,())))
    nodes_train = np.unique(list(sum(edges_keep,())))

    # get biggest graph component
    #comps = [*max(nx.connected_components(nx_graph),key=len),]
    # validation set is the rest of graph
    #feat = feat.toarray()
    #val_graph = nx.from_edgelist(edges_rem)
    val_graph = copy.deepcopy(train_graph)
    val_graph.remove_edges_from(edges_keep)
    train_graph.remove_edges_from(edges_rem)
    val_adj = nx.adjacency_matrix(val_graph).todense()
    train_adj = nx.adjacency_matrix(train_graph).todense()
    val_adj = val_adj + np.eye(val_adj.shape[0])
    train_adj = train_adj + np.eye(train_adj.shape[0])
    train_adj = train_adj[nodes_train][:,nodes_train]
    val_adj = val_adj[nodes_val][:,nodes_val]
    val_feats = [feats[0][nodes_val]]
    train_feats = [feats[0][nodes_train]]
    
    return adj_norm, feats, truth, adj_no_loop.toarray(), sc_label, train_adj, train_feats, val_adj, val_feats
    '''    
    elif dataSet.startswith('weibo'):
            if self.args.learn_method == 'gnn':
                ds = dataSet
                graph_u2p_file = self.file_paths[ds]['graph_u2p']
                labels_file = self.file_paths[ds]['labels']
                features_bow_file = self.file_paths[ds]['features_bow']
                features_loc_file = self.file_paths[ds]['features_loc']

                graph_u2p = pickle.load(open(graph_u2p_file, 'rb'))
                labels = pickle.load(open(labels_file, 'rb'))
                feat_bow = pickle.load(open(features_bow_file, 'rb'))
                feat_loc = pickle.load(open(features_loc_file, 'rb'))

                if self.args.feature == 'all':
                    feat_data = np.concatenate((feat_loc, feat_bow), axis=1)
                elif self.args.feature == 'bow':
                    feat_data = feat_bow
                elif self.args.feature == 'loc':
                    feat_data = feat_loc

                m, n = np.shape(graph_u2p)
                adj_lists = {}
                feat_p = np.zeros((n, np.shape(feat_data)[1]))
                for i in range(m):
                    adj_lists[i] = set(m + graph_u2p[i,:].nonzero()[1])
                for j in range(n):
                    adj_lists[j+m] = set(graph_u2p[:,j].nonzero()[0])
                    feat_j = feat_data[graph_u2p[:,j].nonzero()[0]]
                    feat_j = np.mean(feat_j, 0)
                    feat_p[j] = feat_j

                feat_data = np.concatenate((feat_data, feat_p), 0)
                # feat_data = np.concatenate((feat_data, np.zeros((n, np.shape(feat_data)[1]))), 0)
                assert np.shape(feat_data)[0] == m+n

                assert len(feat_data) == len(labels)+n == len(adj_lists)
                test_indexs, val_indexs, train_indexs = self._split_data(len(labels))
                user_id_max = len(labels)
                train_indexs_cls = train_indexs
                train_indexs = np.arange(len(labels))

                user_id_max = len(labels)
                graph_simi = np.ones((10, 10))

                # get labels for anomaly losses if needed
                if self.args.a_loss != 'none':
                    self.logger.info(f'using {self.args.a_loss} anomaly loss.')
                    label_a_file = self.file_paths[ds][self.args.a_loss]["a_label"]
                    labels_a = pickle.load(open(label_a_file, 'rb')).astype(int)
                    # get clusters for anomaly losses if needed
                    if self.args.cluster_aloss:
                        clusters_file = self.file_paths[ds][self.args.a_loss]["a_cluster"]
                        clusters = pickle.load(open(clusters_file, 'rb'))
                        u2cluster = defaultdict(list)
                        for i in range(len(clusters)):
                            for u in clusters[i]:
                                u2cluster[u].append(i)
                        cluster_neighbors = defaultdict(set)
                        for u in range(np.shape(graph_u2p)[0]):
                            for clus_i in u2cluster[u]:
                                cluster_neighbors[u] |= clusters[clus_i]
                        for u in range(np.shape(graph_u2p)[0]):
                            cluster_neighbors[u] = cluster_neighbors[u] - set([u])
                            assert len(cluster_neighbors[u]) >= 1
                        u2size_of_cluster = {}
                        for u, neighbors in cluster_neighbors.items():
                            u2size_of_cluster[u] = len(neighbors)
                    else:
                        u2size_of_cluster = {}
                        for u in range(np.shape(graph_u2p)[0]):
                            u2size_of_cluster[u] = np.sum(labels_a == labels_a[u])

                setattr(self, dataSet+'_test', test_indexs)
                setattr(self, dataSet+'_val', val_indexs)
                setattr(self, dataSet+'_train', train_indexs)
                setattr(self, dataSet+'_train_cls', train_indexs_cls)
                setattr(self, dataSet+'_trainable', train_indexs)
                setattr(self, dataSet+'_useridmax', user_id_max)

                setattr(self, dataSet+'_feats', feat_data)
                setattr(self, dataSet+'_labels', labels)
                setattr(self, dataSet+'_adj_lists', adj_lists)
                setattr(self, dataSet+'_best_adj_lists', adj_lists)
                setattr(self, dataSet+'_simis', graph_simi)
                if self.args.a_loss != 'none':
                    setattr(self, dataSet+'_labels_a', labels_a)
                    setattr(self, dataSet+'_u2size_of_cluster', u2size_of_cluster)
                if self.args.cluster_aloss:
                    setattr(self, dataSet+'_cluster_neighbors', cluster_neighbors)

    '''

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
