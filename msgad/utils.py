from sklearn.metrics import average_precision_score, roc_auc_score   
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

def loss_func(adj, A_hat, X_hat, pos_edges, neg_edges, sample=False, recons='struct', alpha=None):
    '''
    Input:
        adj: normalized adjacency matrix
        feat: feature matrix
        pos_edges: positive edge list
        neg_edges: negative edge list
        sample: if true, use sampled edges/non-edges for loss calculation. if false, use all edges/non-edges
        recons: structure reconstruction, feature reconstruction, or structure & feature reconstruction (dominant)
        alpha: if structure & feature reconstruction, scalar to weigh importance of structure vs feature reconstruction
    Output:
        all_costs: total loss for backpropagation
        all_struct_error: node-wise errors at each scale
    '''
    if not alpha: alpha = 1 if recons=='struct' else 0
    adjacency = adj.adjacency_matrix().to_dense().cuda()
    feat = adj.ndata['feature']

    all_costs, all_struct_error, all_feat_error = None, 0, 0
    struct_error,feat_error=None,None

    pos_edges = torch.vstack((pos_edges[0],pos_edges[1])).T
    neg_edges = torch.vstack((neg_edges[0],neg_edges[1])).T

    for recons_ind,preds in enumerate([A_hat, X_hat]):
        if not preds: continue
        for ind, sc_pred in enumerate(preds):
            # structure loss
            if recons_ind == 0:
                if sample:
                    # collect loss for selected positive/negative edges
                    pos_edge_error = get_losses(sc_pred,pos_edges,torch.tensor(1.,dtype=torch.float32).cuda())
                    neg_edge_error = get_losses(sc_pred,neg_edges,torch.tensor(0.,dtype=torch.float32).cuda())
                    struct_error = torch.cat((pos_edge_error,neg_edge_error))
                else:
                    # collect loss for all edges/non-edges in reconstruction
                    #struct_error = torch.nn.functional.binary_cross_entropy(sc_pred,adj.adjacency_matrix().to_dense().cuda(),reduction='none')
                    #import ipdb ; ipdb.set_trace()
                    diff_structure = torch.pow(sc_pred - adj.adjacency_matrix().to_dense().cuda(), 2)
                    struct_error = torch.sqrt(torch.sum(diff_structure, 1))

            # feature loss
            if recons_ind == 1:
                feat_error = torch.nn.functional.mse_loss(sc_pred,feat.cuda(),reduction='none')
                feat_error = torch.mean(feat_error,dim=0)

            # accumulate errors
            if all_costs is None:
                if recons_ind == 0:
                    all_struct_error = (struct_error).unsqueeze(-1)
                    all_costs = torch.mean(all_struct_error).unsqueeze(-1)*alpha
                if recons_ind == 1:
                    all_feat_error = (feat_error).unsqueeze(-1)
                    all_costs = (torch.mean(all_feat_error))*(1-alpha)
            else:
                if recons_ind == 0: all_struct_error = torch.cat((all_struct_error,(struct_error).unsqueeze(-1)))
                if recons_ind == 1: all_feat_error = torch.cat((all_feat_error,(feat_error).unsqueeze(-1)))
                all_costs = torch.cat((all_costs,torch.add(all_struct_error*alpha,all_feat_error*(1-alpha))))

    return all_costs, all_struct_error, all_feat_error

def get_losses(pred,edges,label):
    for edge_ind,edge in enumerate(edges):
        node1,node2=edge
        #recons_error=torch.nn.functional.binary_cross_entropy(pred[node1][node2],label)
        diff_structure = torch.pow(pred - label, 2)
        recons_error = torch.sqrt(torch.sum(diff_structure, 1))
        if edge_ind == 0:
            edge_errors=recons_error.unsqueeze(-1)
        else:
            edge_errors = torch.cat((edge_errors,recons_error.unsqueeze(-1)))
    return edge_errors

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

def flatten_label(anom):
    for ind,i in enumerate(anom):
        if ind == 0: ret_anom = np.expand_dims(i.flatten(),0)
        else: ret_anom = np.hstack((ret_anom,np.expand_dims(i.flatten(),0)))
    return ret_anom

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
    anom_sc1 = flatten_label(anom_sc1)
    anom_sc2 = flatten_label(anom_sc2)
    anom_sc3 = flatten_label(anom_sc3)
    all_anom = np.concatenate((anom_sc1,np.concatenate((anom_sc2,anom_sc3),axis=None)),axis=None)
    true_anoms = 0
    cor_1, cor_2, cor_3 = 0,0,0
    for ind,error_ in enumerate(sorted_errors[:int(all_anom.shape[0]*top_nodes_perc)]):
        error = error_.item()
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
    
def detect_anomalies(scores,  label, sc_label, dataset):
    '''
    Input:
        scores: anomaly scores for all scales []
        label: node-wise anomaly label []
        sc_label: array containing scale-wise anomaly node ids
        dataset: dataset string
    '''
    for sc,sc_score in enumerate(scores):
        sorted_errors = np.argsort(-sc_score.detach().cpu().numpy())
        rev_sorted_errors = np.argsort(sc_score.detach().cpu().numpy())
        rankings = []
        
        for error in sorted_errors:
            rankings.append(label[error])
        rankings = np.array(rankings)

        print(f'SCALE {sc+1} loss',torch.mean(sc_score).item())
        #print('AP',detect_anom_ap(scores_recons,label))
        if 'tfinance' in dataset:
            detect_anom(sorted_errors, sc_label[0][0][0], sc_label[0][0][0], sc_label[0][2][0], 1)
            print('reverse')
            detect_anom(rev_sorted_errors, sc_label[0][0][0], sc_label[0][0][0], sc_label[0][2][0], 1)
            print('')
        elif 'weibo' in dataset:
            import ipdb; ipdb.set_trace()
        else:
            detect_anom(sorted_errors, sc_label[0][0], sc_label[0][1], sc_label[0][2], 1)
            print('reverse')
            detect_anom(rev_sorted_errors, sc_label[0][0], sc_label[0][1], sc_label[0][2], 1)
            print('')
        
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
    '''
    '''
    data_mat = sio.loadmat(f'data/{dataset}.mat')
    if 'cora' in dataset or 'yelp' in dataset:
        feats = torch.FloatTensor(data_mat['Attributes'].toarray())
    else:
        feats = torch.FloatTensor(data_mat['Attributes'])

    adj = data_mat['Network']
    truth = data_mat['Label'].flatten()

    if dataset != 'weibo':
        sc_label = data_mat['scale_anomaly_label']
    else:
        sc_label = data_mat['scale_anomaly_label'][0]
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
