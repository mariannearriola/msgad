from sklearn.metrics import average_precision_score, roc_auc_score   
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sknetwork.hierarchy import Paris, postprocess, LouvainHierarchy, LouvainIteration
import numpy as np
import scipy.sparse as sp
import torch
import scipy
import scipy.io as sio
import random
import networkx as nx
from igraph import Graph
import dgl
import copy

def getScaleClusts(dend,thresh):
    clust_labels = postprocess.cut_straight(dend,threshold=thresh)
    return clust_labels

def flatten_label(anoms):
    #if ...:
    
    flattened_anoms = []
    for anom_sc in anoms:
        for ind,i in enumerate(anom_sc):
            if ind == 0: ret_anom = np.expand_dims(i.flatten(),0)
            else: ret_anom = np.hstack((ret_anom,np.expand_dims(i.flatten(),0)))
        flattened_anoms.append(ret_anom[0])
    return flattened_anoms

def detect_anom(sorted_errors, anom_sc1, anom_sc2, anom_sc3, top_nodes_perc):
    '''
    Input:
        sorted_errors: normalized adjacency matrix
        label: positive edge list
        top_nodes_perc: negative edge list
    Output:
        all_costs: total loss for backpropagation
        all_struct_error: structure errors for each scale
    '''
    all_anom = np.concatenate((anom_sc1,np.concatenate((anom_sc2,anom_sc3),axis=None)),axis=None)
    true_anoms = 0
    cor_1, cor_2, cor_3 = 0,0,0
    for ind,error_ in enumerate(sorted_errors[:int(all_anom.shape[0]*top_nodes_perc)]):
        error = error_#.item()
        if error in all_anom:
            true_anoms += 1
        if error in anom_sc1:
            cor_1 += 1
        if error in anom_sc2:
            cor_2 += 1
        if error in anom_sc3:
            cor_3 += 1
    print(f'scale1: {cor_1}, scale2: {cor_2}, scale3: {cor_3}, total: {true_anoms}')
    
    return true_anoms/int(all_anom.shape[0]*top_nodes_perc), cor_1, cor_2, cor_3, true_anoms
    
def getClustScores(clust,scores):
    clust_keys = np.unique(clust)
    clust_dict, score_dict = {}, {}
    #anom_count,node_count = [],[]
    for key in clust_keys:
        clust_dict[key] = np.where(clust==key)[0]
        #anom_count.append(np.intersect1d(anom_sc_label,clust_dict[key]).shape[0])
        #node_count.append(clust_dict[key].shape[0])
        cum_clust_score = np.max(scores[clust_dict[key]])
        score_dict[key] = cum_clust_score
    return clust_dict, score_dict

def getHierClusterScores(graph,scores):
    """For 3 graph scales, get scale-wise anomaly predictions"""
    paris = LouvainIteration() 
    dend = paris.fit_predict(graph)

    clust1 = getScaleClusts(dend,1)
    clust2 = getScaleClusts(dend,2)
    clust3 = getScaleClusts(dend,3)

    clust_dict1,score_dict1 = getClustScores(clust1, scores)
    clust_dict2,score_dict2 = getClustScores(clust2, scores)
    clust_dict3,score_dict3 = getClustScores(clust3, scores)

    clust_dicts,score_dicts=[clust_dict1,clust_dict2,clust_dict3],[score_dict1,score_dict2,score_dict3]

    return clust_dicts, score_dicts

class anom_classifier():
    """Given node-wise or cluster-wise scores, perform binary anomaly classification"""
    def __init__(self, nu=0.5):
        super(anom_classifier, self).__init__()
        self.clf = SGDOneClassSVM(nu=nu)

    def classify(self, scores, labels, clust_nodes=None):
        # TODO: REDO as GNN
        X = np.expand_dims(scores,1)
        fit_clf = self.clf.fit(X)

        # get indices of anom label
        anom_preds = np.where(fit_clf.predict(X)==-1)[0]
        
        accs = []
        if clust_nodes is not None:
            for label in labels:
                sc_accs = []
                # for each cluster, check how many anomalies it contained. if it contains
                # at least > 90% labeled anomalies, and classifier classifies it as anomalous,
                # consider it to have 100% acc.
                for anom_clust_idx in anom_preds:
                    if np.intersect1d(clust_nodes[anom_clust_idx],label).shape[0]/clust_nodes[anom_clust_idx].shape[0] >= 0.9:
                        sc_accs.append(1)
                    else:
                        sc_accs.append(0)
                accs.append(np.mean(np.array(sc_accs)))
        else:
            for label in labels:
                accs.append(np.intersect1d(label,anom_preds).shape[0]/anom_preds.shape[0])
        return accs, anom_preds.shape[0]

def detect_anomalies(graph, scores, label, sc_label, dataset, sample=False, cluster=False):
    '''
    Input:
        scores: anomaly scores for all scales []
        label: node-wise anomaly label []
        sc_label: array containing scale-wise anomaly node ids
        dataset: dataset string
    '''
    clf = anom_classifier(nu=0.5)
    for sc,sc_score in enumerate(scores):
        if True:
            node_scores=np.array([np.mean(i[np.where(i!=0)]) for i in sc_score.todense()])
        else:
            node_scores=np.array([np.mean(i[np.where(i!=0)]) for i in sc_score])#.todense()])

        sorted_errors = np.argsort(-node_scores)
        rev_sorted_errors = np.argsort(node_scores)
        rankings = []
        
        for error in sorted_errors:
            rankings.append(label[error])
        rankings = np.array(rankings)

        print(f'SCALE {sc+1} loss',np.mean(node_scores))

        anom_sc1,anom_sc2,anom_sc3 = flatten_label(sc_label)

        detect_anom(sorted_errors, anom_sc1, anom_sc2, anom_sc3, 1)
        print('scores reverse sorted')
        detect_anom(rev_sorted_errors, anom_sc1, anom_sc2, anom_sc3, 1)
        print('')

        all_anom = [anom_sc1,anom_sc2,anom_sc3]
        all_anom_cat = [np.concatenate((anom_sc1,(np.concatenate((anom_sc2,anom_sc3)))))]

        # get clusters
        if cluster:
            clust_dicts, score_dicts = getHierClusterScores(graph,node_scores)
            cluster_accs = [clf.classify(np.array(list(x.values())),all_anom,np.array(list(y.values()))) for x,y in zip(score_dicts,clust_dicts)]
            print('cluster scores')
            print(cluster_accs)
        '''
        # classify anoms with linear classifier
        anom_accs = clf.classify(node_scores, all_anom)
        print('\nnode scores')
        print(anom_accs)
        '''
        
    with open('output/{}-ranking_{}.txt'.format(dataset, sc), 'w+') as f:
        for index in sorted_errors:
            f.write("%s\n" % label[index])

def sparse_matrix_to_tensor(coo,feat):
    coo = scipy.sparse.coo_matrix(coo)
    v = torch.FloatTensor(coo.data)
    i = torch.LongTensor(np.vstack((coo.row, coo.col)))
    dgl_graph = dgl.graph((i[0],i[1]))
    dgl_graph.edata['w'] = v
    dgl_graph.ndata['feature'] = feat
    return dgl_graph

def load_anomaly_detection_dataset(dataset, sc, datadir='data'):
    """Load anomaly detection graph dataset for model training & anomaly detection"""
    data_mat = sio.loadmat(f'data/{dataset}.mat')
    if 'cora' in dataset or 'yelp' in dataset:
        feats = torch.FloatTensor(data_mat['Attributes'].toarray())
    else:
        feats = torch.FloatTensor(data_mat['Attributes'])

    adj = data_mat['Network']
    truth = data_mat['Label'].flatten()

    sc_label = data_mat['scale_anomaly_label']
    
    if 'tfinance' in dataset:
        sc_label = sc_label[0]
        anom_sc1,anom_sc2,anom_sc3 = sc_label[0][0],[],sc_label[2][0]
    elif 'weibo' in dataset:
        sc_label = sc_label[0]
        anom_sc1,anom_sc2,anom_sc3 = sc_label[0][0],sc_label[1][0],sc_label[2][0]
    elif 'cora' in dataset:
        anom_sc1,anom_sc2,anom_sc3=sc_label[0]

    sc_label = [anom_sc1,anom_sc2,anom_sc3]
 
    return adj, feats, truth, sc_label

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    adj += sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def calc_lipschitz(A_hat_scales, A_hat_pert, anom_label):
    A_hat_diff = abs(A_hat_scales-A_hat_pert)
    A_hat_diff_pool = torch.norm(A_hat_diff)
    A_hat_diff_p = A_hat_diff_pool
    lipschitz = A_hat_diff_p
    return lipschitz

def perturb_adj(attribute_dense,adj,attr_scale,adj_prob):
    '''
    Input:
        attrs: input attributes
        adj: input adjacency
        attr_scale: degree to perturb attributes
        adj_prob: probability of edge in perturbed graph
    Output:
        attrs_pert: perturbed attributes
        adj_pert: perturbed adjacency
    '''
    adj_dense = adj.clone() 

    # Disturb structure
    for i,_ in enumerate(adj_dense):
        for j,_ in enumerate(adj_dense):
            if j >= i:
                break
            # flip if prob met
            adj_dense[i,j]=0
            adj_dense[j,i]=0
            if np.random.rand() < adj_prob:
                adj_dense[i,j] = 1.
                adj_dense[j,i] = 1.

    # Disturb attribute
        # Every node in each clique is anomalous; no attribute anomaly scale (can be added)
    print('Constructing attributed anomaly nodes...')
    for cur,_ in enumerate(attribute_dense):
        max_dist = 0
        # find attribute with greatest euclidian distance
        for j_,_ in enumerate(attribute_dense):
            cur_dist = euclidean(attribute_dense[cur].cpu(),attribute_dense[j_].cpu())
            if cur_dist > max_dist:
                max_dist = cur_dist
    
        for j_,_ in enumerate(attribute_dense):
            cur_dist = euclidean(attribute_dense[cur].cpu(),attribute_dense[j_].cpu())
            if cur_dist > max_dist/10:
                closest = cur_dist
                dists[cur] = cur_dist
                closest_idx = j_
                #max_idx = j_
        
        attribute_dense[cur] = attribute_dense[closest_idx]
        
    return attribute_dense,adj_dense