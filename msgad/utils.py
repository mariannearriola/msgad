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

def loss_func(adj, A_hat_scales, pos_edges_, neg_edges_backprop, sample=False):
    '''
    Input:
        adj: normalized adjacency matrix
        pos_edges: positive edge list
        neg_edges: negative edge list
    Output:
        all_costs: total loss for backpropagation
        all_structure_costs: node-wise errors at each scale
    '''
    adjacency = adj.adjacency_matrix().to_dense().cuda()
    #backprop_edges = torch.cat((pos_edges_,neg_edges_backprop),dim=0)
    all_costs, all_structure_costs = None, None
    
    for ind, A_hat in enumerate(A_hat_scales):
        if sample:
            for edge_ind,edge in enumerate(pos_edges_):
                node1,node2=edge
                recons_error=torch.nn.functional.binary_cross_entropy(A_hat[node1][node2],torch.tensor(1.,dtype=torch.float32).cuda())
                if edge_ind == 0:
                    structure_reconstruction_errors=recons_error.unsqueeze(-1)
                else:
                    structure_reconstruction_errors = torch.cat((structure_reconstruction_errors,recons_error.unsqueeze(-1)))
            for edge_ind,edge in enumerate(neg_edges_backprop):
                node1,node2=edge
                recons_error=torch.nn.functional.binary_cross_entropy(A_hat[node1][node2],torch.tensor(0.,dtype=torch.float32).cuda())
                structure_reconstruction_errors = torch.cat((structure_reconstruction_errors,recons_error.unsqueeze(-1)))
        
        else:
            structure_reconstruction_errors = torch.nn.functional.binary_cross_entropy(A_hat,adj.adjacency_matrix().to_dense().cuda(),reduction='none')
            structure_reconstruction_errors = torch.mean(structure_reconstruction_errors,dim=0)
        
        if all_costs is None:
            all_structure_costs = (structure_reconstruction_errors).unsqueeze(-1)
            all_costs = torch.mean(all_structure_costs).unsqueeze(-1)
        else:
            all_structure_costs = torch.cat((all_structure_costs,(structure_reconstruction_errors).unsqueeze(-1)),dim=0)
            all_costs = torch.cat((all_costs,torch.mean(structure_reconstruction_errors).unsqueeze(-1)))
            
    return all_costs, all_structure_costs

def load_batch(adj,bg,device):
    '''
    Input:
        adj: normalized adjacency matrix
        bg: batch index
        device: cuda device
    Output:
        batch_adj: batched adjacency matrix
        pos_edges: positive edge list
        neg_edges: negative edge list
        batch_dict: maps batch node numbering to full node numbering
    '''
    try: pos_edges = adj.find_edges(bg.cuda())
    except: raise('BATCH LOADING ERROR')

    pos_edge_weights = adj.edata['w'][bg.cuda()]
    neg_edges_backprop = dgl.sampling.global_uniform_negative_sampling(adj, int(bg.shape[0])*2)
    sel_nodes = torch.unique(torch.cat((pos_edges[0],pos_edges[1])))
    sel_nodes = torch.unique(torch.cat((sel_nodes,torch.cat((neg_edges_backprop[0],neg_edges_backprop[1])))))
        
    pos_edges_= torch.cat((pos_edges[0].unsqueeze(-1),pos_edges[1].unsqueeze(-1)),dim=1)
    neg_edges_backprop = torch.cat((neg_edges_backprop[0].unsqueeze(-1),neg_edges_backprop[1].unsqueeze(-1)),dim=1)

    batch_feats = adj.ndata['feature'][sel_nodes]
    batch_dict_for,batch_dict_rev = {k.item():v.item() for k,v in zip(sel_nodes,torch.arange(sel_nodes.shape[0]))},{k.item():v.item() for k,v in zip(torch.arange(sel_nodes.shape[0]),sel_nodes)}
    for ind,i in enumerate(pos_edges_):
        pos_edges_[ind][0],pos_edges_[ind][1] = batch_dict_for[i[0].item()],batch_dict_for[i[1].item()]
    for ind,i in enumerate(neg_edges_backprop):
        neg_edges_backprop[ind][0],neg_edges_backprop[ind][1] = batch_dict_for[i[0].item()],batch_dict_for[i[1].item()]

    batch_adj = adj.subgraph(sel_nodes)

    #batch_adj.edata['w'] = pos_edge_weights
    batch_adj = dgl.to_bidirected(batch_adj.cpu()).to(device)
    batch_adj.ndata['feature'] = batch_feats
    
    return batch_adj, pos_edges_, neg_edges_backprop, batch_dict_rev

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

def redo(anom):
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
        all_structure_costs: structure errors for each scale
    '''
    anom_sc1 = redo(anom_sc1)
    anom_sc2 = redo(anom_sc2)
    anom_sc3 = redo(anom_sc3)
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

def embed_sim(embeds,label):
    '''
    TODO: CLEAN UP
    '''
    import numpy as np
    anom_sc1 = label[0][0]#[0]
    anom_sc2 = label[1][0]#[0]
    anom_sc3 = label[2][0]#[0] 
    anoms_cat = np.concatenate((anom_sc1,np.concatenate((anom_sc2,anom_sc3),axis=None)),axis=None) 
    all_anom = [anom_sc1,anom_sc2,anom_sc3]
    # get max embedding diff for normalization
    '''
    max_diff = 0
    #import ipdb ; ipdb.set_trace()
    for ind,embed in enumerate(embeds):
        for ind_,embed_ in enumerate(embeds):
            if ind_>= ind:
                break
            max_diff = torch.norm(embed-embed_) if torch.norm(embed-embed_) > max_diff else max_diff
    '''
    # get anom embeds differences
    all_anom_diffs = []
    for anoms in all_anom:
        anoms_embs = embeds[anoms]
        anom_diffs = []
        for ind,embed in enumerate(anoms_embs):
            for ind_,embed_ in enumerate(anoms_embs):
                #if len(anom_diffs) == len(anoms): continue
                if ind_ >= ind: break
                #anom_diffs.append(torch.norm(embed-embed_)/max_diff)
                anom_diffs.append(embed@embed_.T)
        all_anom_diffs.append(anom_diffs)
    '''
    # get normal embeds differences

    normal_diffs = []
    for ind,embed in enumerate(embeds):
        if ind in anoms_cat: continue
        if len(normal_diffs) == len(all_anom_diffs):
            break
        for ind_,embed_ in enumerate(embeds):
            if ind_ >= ind: break
            if ind_ in anoms_cat: continue
            normal_diffs.append(torch.norm(embed-embed_)/max_diff)
    # get normal vs anom embeds differences
    all_norm_anom_diffs = []
    for anoms in all_anom:
        norm_anom_diffs=[]
        for ind, embed in enumerate(embeds):
            if ind in anoms_cat: continue
            for ind_,anom in enumerate(embeds[anoms]):
                #if len(norm_anom_diffs) == len(anoms): continue 
                norm_anom_diffs.append(torch.norm(embed-anom)/max_diff)
        all_norm_anom_diffs.append(norm_anom_diffs)
    print('normal-normal',sum(normal_diffs)/len(normal_diffs))
    print('anom-anom',sum(all_anom_diffs[0])/len(all_anom_diffs[0]),sum(all_anom_diffs[1])/len(all_anom_diffs[1]),sum(all_anom_diffs[2])/len(all_anom_diffs[2])) 
    print('anom-normal',sum(all_norm_anom_diffs[0])/len(all_norm_anom_diffs[0]),sum(all_norm_anom_diffs[1])/len(all_norm_anom_diffs[1]),sum(all_norm_anom_diffs[2])/len(all_norm_anom_diffs[2]))
    #import ipdb ; ipdb.set_trace()
    print('----')
    '''
    print((sum(all_anom_diffs[0])/len(all_anom_diffs[0])).item(),(sum(all_anom_diffs[1])/len(all_anom_diffs[1])).item(),(sum(all_anom_diffs[2])/len(all_anom_diffs[2])).item()) 

''' 
TODO: MADAN baseline implementation
    import ipdb ; ipdb.set_trace() 
    import MADAN.Plotters as Plotters
    from MADAN._cython_fast_funcs import sum_Sto, sum_Sout, compute_S, cython_nmi, cython_nvi
    from MADAN.LouvainClustering_fast import Clustering, norm_var_information
    import MADAN.Madan as md
    madan = md.Madan(adj, attributes=attrs[-1], sigma=0.08)
    time_scales = np.concatenate([np.array([0]), 10**np.linspace(0,5,500)])
    madan.scanning_relevant_context(time_scales, n_jobs=4)
    madan.scanning_relevant_context_time(time_scales)
    madan.compute_concentration(1000)
    print(madan.concentration,madan.anomalous_nodes)
    madan.compute_context_for_anomalies()
    print(madan.interp_com)
    print(' ------------')
'''
