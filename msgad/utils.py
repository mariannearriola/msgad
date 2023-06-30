from sknetwork.hierarchy import Paris, postprocess, LouvainHierarchy, LouvainIteration
from torch_geometric.nn import MessagePassing
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
import matplotlib.ticker as mtick
import pickle as pkl
from model import GraphReconstruction
from models.gcad import *
import matplotlib.pyplot as plt
import os
import yaml
import torch
import torch_geometric
import gc
from pygsp_ import *

def replace_node_ids_with_probabilities(probability_array, node_ids):
    unique_node_ids, node_id_counts = np.unique(node_ids, return_counts=True)
    total_nodes = len(node_ids)
    node_id_probabilities = node_id_counts / total_nodes
    probability_dict = dict(zip(unique_node_ids, node_id_probabilities))
    replace_func = np.vectorize(lambda x: probability_dict.get(x, 0))
    replaced_array = replace_func(node_ids)
    
    # Normalize the probabilities to sum to 1
    replaced_array /= np.sum(replaced_array)
    
    return replaced_array


def weighted_selection(edge_idx,clusts,num_edges):
    counts=(np.unique(edge_idx,return_counts=True)[1])/edge_idx.shape[0]
    # Calculate weights based on cluster sizes
    edge_probs = replace_node_ids_with_probabilities(counts,edge_idx)

    # Select nodes based on weights
    selected_edges = np.random.choice(np.arange(edge_idx.shape[0]), size=num_edges, replace=False, p=edge_probs)
    return selected_edges

def ensure_unique_pairs(tensor):
    # Sort each row of the tensor
    sorted_tensor, _ = torch.sort(tensor, dim=0)

    # Find unique rows using torch.unique
    unique_tensor = torch.unique(sorted_tensor, dim=0)

    return unique_tensor

def get_contr_edges(edge_idx,clusts,scales):
    #import ipdb ; ipdb.set_trace()
    edge_clusts=torch.stack([i[edge_idx[ind].detach().cpu()] for ind,i in enumerate(clusts)])

    # for each clustering, sample edges/nonedges for contrastive learning, balance by cluster id of edges
    for ind in range(scales):
        '''
        attract_edges = torch.where(edge_clusts[ind,0]==edge_clusts[ind,1])[0]
        repel_edges = torch.where(edge_clusts[ind,0]!=edge_clusts[ind,1])[0]
        import ipdb ; ipdb.set_trace()
        if ind == 0:
            attract_edges_sel = attract_edges.unsqueeze(0)
            repel_edges_sel = repel_edges.unsqueeze(0)
        else:
            attract_edges_sel = torch.cat((attract_edges_sel,attract_edges.unsqueeze(0)),dim=0)
            repel_edges_sel = torch.cat((repel_edges_sel,repel_edges.unsqueeze(0)),dim=0)

        continue
        '''
        sc_idx_attract = torch.where(edge_clusts[ind][0]==edge_clusts[ind][1])[0]
        if ind == 0:
            num_edges = sc_idx_attract.shape[0]
        attract_clusts = edge_clusts[ind,:,sc_idx_attract]
        if ind == 0:
            attract_edges_sel = sc_idx_attract[weighted_selection(attract_clusts[0],clusts[ind],num_edges)].unsqueeze(0)
        else:
            attract_edges_sel = torch.cat((attract_edges_sel,sc_idx_attract[weighted_selection(attract_clusts[0],clusts[ind],num_edges)].unsqueeze(0)),dim=0)

        sc_idx_repel = torch.where(edge_clusts[ind][0]!=edge_clusts[ind][1])[0]
        repel_clusts = edge_clusts[ind,:,sc_idx_repel]

        # sample from first row, 2nd row, (will be no intersection), then randomly select num_edges from both
        idx_found=weighted_selection(repel_clusts[0],clusts[ind],num_edges)
        repel_clusts_sel = sc_idx_repel[idx_found]
        sc_idx_left=torch.tensor(np.setdiff1d(sc_idx_repel,repel_clusts_sel))
        repel_clusts_flip = sc_idx_left[weighted_selection(edge_clusts[ind,1,sc_idx_left],clusts[ind],np.minimum(num_edges,sc_idx_left.shape[0]))]

        if np.intersect1d(np.array(repel_clusts_sel),np.array(repel_clusts_flip)).shape[0] > 0:
            raise('bug')

        repel_clusts_sel=np.array(torch.cat((repel_clusts_sel,repel_clusts_flip),dim=0))
        repel_clusts_sel = sc_idx_repel[weighted_selection(repel_clusts_sel,clusts[ind],num_edges)]
        if ind == 0:
            repel_edges_sel = torch.tensor(repel_clusts_sel).unsqueeze(0)
        else:
            repel_edges_sel = torch.cat((repel_edges_sel,torch.tensor(repel_clusts_sel).unsqueeze(0)))

    return attract_edges_sel,repel_edges_sel

def get_different_cluster_indices(cluster_2d, same_cluster_indices):
    different_cluster_indices = []
    for index in same_cluster_indices:
        cluster_id0,cluster_id1 = torch.where(cluster_2d==index)
        cluster_id = cluster_2d[:,cluster_id1]
        diff_idxs = cluster_id1[torch.where(cluster_id0[0]!=cluster_id1)]
        idx_picked= random.choice(diff_idxs)
        while idx_picked in different_cluster_indices:
            print('next',idx_picked)
            idx_picked= random.choice(diff_idxs)
        different_cluster_indices.append(idx_picked)
    return different_cluster_indices

def generate_cluster_colors(cluster_ids):
    unique_clusters = np.unique(cluster_ids)
    num_clusters = len(unique_clusters)
    color_palette = ['#%06x' % random.randint(0, 0xFFFFFF) for _ in range(num_clusters)]
    cluster_colors = {}

    for idx, cluster_id in enumerate(unique_clusters):
        cluster_colors[cluster_id] = color_palette[idx]
    color_list = [cluster_colors[cluster_id] for cluster_id in cluster_ids]
    return color_list


def collect_clust_loss(edge_ids,sc_idx,sc_idx_,loss):
    """Get average loss for each cluster, and assign back to edge-wise losses"""
    sc_idx = np.intersect1d(sc_idx,sc_idx_)
    cl = edge_ids[:,sc_idx][0].detach().cpu()
    average_tensor = torch.scatter_reduce(loss[sc_idx].detach().cpu(), 0, cl, reduce="mean")
    expanded_average = average_tensor[cl].to(edge_ids.device).mean()#/cl.shape[0]
    return expanded_average

class TBWriter:
    def __init__(self,tb,edge_ids, attract_edges_sel, repel_edges_sel,sc_label,clust):
        # edges connected to any anomaly
        norms,anom_sc1,anom_sc2,anom_sc3,anom_single=sc_label
        self.sc_idx_all,self.sc_idx_all,self.sc_idx_inside,self.sc_idx_outside,self.cl_all={},{},{},{},{}

        self.sc_labels = ['norm','anom_sc1','anom_sc2','anom_sc3','anom_single']
        for ind,i in enumerate(sc_label):
            lbl = self.sc_labels[ind]
            a,b = np.intersect1d(edge_ids[0].detach().cpu(),i,return_indices=True)[-2],np.intersect1d(edge_ids[1].detach().cpu(),i,return_indices=True)[-2]
            self.sc_idx_all[lbl]=np.unique(np.stack((a,b)).flatten())
            self.cl_all[lbl] = edge_ids[:,self.sc_idx_all[lbl]][0].detach().cpu()

        # only select edges inside cluster
        #self.sc_idx_inside[lbl] = np.where(clust[edge_ids.detach().cpu().numpy()][0] == clust[edge_ids.detach().cpu().numpy()][1])[0]
        self.sc_idx_inside = attract_edges_sel
        # only select edges outside cluster
        #self.sc_idx_outside[lbl] = np.where(clust[edge_ids.detach().cpu().numpy()][0] != clust[edge_ids.detach().cpu().numpy()][1])[0]
        self.sc_idx_outside = repel_edges_sel
        '''
        self.sc_idx_inside_outside_anom=np.setxor1d(a,b) # get edges outside of anom
        # get edges inside cluster
        self.sc_idx_inside_outside_anom_ = np.where(clust[edge_ids.detach().cpu().numpy()][0] == clust[edge_ids.detach().cpu().numpy()][1])[0]

        a,b = np.intersect1d(edge_ids[0].detach().cpu(),sc_label,return_indices=True)[-2],np.intersect1d(edge_ids[1].detach().cpu(),sc_label,return_indices=True)[-2]
        self.sc_idx_outside_inside_anom=np.intersect1d(a,b)
        self.sc_idx_outside_inside_anom_= np.where(clust[edge_ids.detach().cpu().numpy()][0] != clust[edge_ids.detach().cpu().numpy()][1])[0]
        '''
        self.tb = tb

    def tb_write_anom(self,tb,sc_label,edge_ids,pred,sc,epoch,regloss,clustloss,nonclustloss,clust,sc_idx_inside,sc_idx_outside,entropies,clust_entropies):
        """Log loss evolution for anomaly group"""

        for ind,i in enumerate(sc_label):
            lbl = self.sc_labels[ind]
            a,b = np.intersect1d(edge_ids[0],i,return_indices=True)[-2],np.intersect1d(edge_ids[1],i,return_indices=True)[-2]
            try:
                self.sc_idx_all[lbl]=np.unique(np.stack((a,b)).flatten())
            except Exception as e:
                import ipdb ; ipdb.set_trace()
            self.cl_all[lbl] = edge_ids[:,self.sc_idx_all[lbl]][0].detach().cpu()
        #import ipdb ; ipdb.set_trace()
        loss_dict,regloss_dict,loss_inclust_dict,loss_outclust_dict={},{},{},{}
        #import ipdb ; ipdb.set_trace()
        for ind,i in enumerate(self.sc_labels):
            # TODO: get cluster(s) associated with these anomalies for that scale
            # get avg entropy of those clusters
            # TODO: can always try to get node-level entropy as well
            anom_sc = i
            average_tensor = torch.scatter_reduce(pred[self.sc_idx_all[i]].detach().cpu(), 0, self.cl_all[i], reduce="mean")
            expanded_average = average_tensor[self.cl_all[i]].to(edge_ids.device).mean()
            loss_dict[anom_sc] = expanded_average 
            tb.add_scalar(f'Loss_{sc}/Anom{anom_sc}',expanded_average, epoch)

            # log regularization loss
            average_tensor = torch.scatter_reduce(regloss[self.sc_idx_all[i]].detach().cpu(), 0, self.cl_all[i], reduce="mean")
            expanded_average = average_tensor[self.cl_all[i]].to(edge_ids.device).mean()
            regloss_dict[anom_sc] = expanded_average 
            tb.add_scalar(f'Regloss{sc}/Anom{anom_sc}', expanded_average, epoch)

            # only select INSIDE cluster
            expanded_average = collect_clust_loss(edge_ids,self.sc_idx_all[i],sc_idx_inside[sc],clustloss)
            loss_inclust_dict[anom_sc] = expanded_average 
            tb.add_scalar(f'Loss_inclust{sc}/Anom{anom_sc}', expanded_average, epoch)

            # only select OUTSIDE cluster
            expanded_average = collect_clust_loss(edge_ids,self.sc_idx_all[i],sc_idx_outside[sc],nonclustloss)
            loss_outclust_dict[anom_sc] = expanded_average 
            tb.add_scalar(f'Loss_outclust{sc}/Anom{anom_sc}', expanded_average, epoch)
            #import ipdb ; ipdb.set_trace()
            #sc_clusters = clust[sc_label[ind]].unique()
            #sc_entropy = entropies[sc][sc_clusters].mean()

            clust_counts=torch.unique(clust,return_counts=True)[-1]
            anom_clusts,anom_counts=torch.unique(clust[sc_label[ind]],return_counts=True)
            #import ipdb ; ipdb.set_trace()
            if i == 'norm':
                clusts = anom_clusts[torch.where(anom_counts/clust_counts[anom_clusts] > 0.95)]
                group_ids = np.intersect1d(clust[sc_label[ind]],clusts,return_indices=True)[1]
            else:
                group_ids=sc_label[ind]

            # node entropy
            #import ipdb ; ipdb.set_trace()
            sc_entropy = entropies[sc][group_ids].mean()
            tb.add_scalar(f'Node_entropy{sc}/Anom{anom_sc}', sc_entropy, epoch)

            # clust entropy
            #anom_clusts = clust[sc_label[ind]].unique()
            #clust_entropy = np.vectorize(clust_entropies[sc].get)(anom_clusts)
            #clust_entropy = clust_entropy[clust_entropy.nonzero()].mean()
            clust_entropy = entropies[sc][group_ids].mean()
            tb.add_scalar(f'Clust_entropy{sc}/Anom{anom_sc}', clust_entropy, epoch)
            '''
            # only select INSIDE cluster, outside OF ANOM
            expanded_average = collect_clust_loss(edge_ids,self.sc_idx_inside_outside_anom,self.sc_idx_inside_outside_anom_,clustloss)
            tb.add_scalar(f'Inclust_outanom{sc}_Anom{anom_sc}', expanded_average, epoch)
            
            # only select edges between anom and other nodes not in anom group, in same cluster

            # only select edges OUTSIDE of cluser, INSIDE anom
            expanded_average = collect_clust_loss(edge_ids,self.sc_idx_outside_inside_anom,self.sc_idx_outside_inside_anom_,nonclustloss)
            tb.add_scalar(f'Outclust_inanom{sc}_Anom{anom_sc}', expanded_average, epoch)
            '''

def get_sc_label(sc_label):
    batch_sc_label = {}
    batch_sc_label_keys = ['anom_sc1','anom_sc2','anom_sc3','single']
    for sc_ind,sc_ in enumerate(sc_label):
        if batch_sc_label_keys[sc_ind] != 'single':
            scs_comb = []
            for sc__ in sc_:
                scs_comb.append(sc__)
            batch_sc_label[batch_sc_label_keys[sc_ind]] = scs_comb
        else:
            batch_sc_label[batch_sc_label_keys[sc_ind]]=sc_
    return batch_sc_label

def dgl_to_mat(g,device='cpu'):
    """Get sparse adjacency matrix from DGL graph"""
    src, dst = g.edges()
    block_adj = torch.sparse_coo_tensor(torch.stack((src,dst)),g.edata['w'].squeeze(-1),size=(g.number_of_nodes(),g.number_of_nodes()))
    return block_adj

def get_spectrum(mat,lapl=None,tag='',load=False,get_lapl=False,save_spectrum=True):
    """Eigendecompose matrix for visualization"""
    device = mat.device
    if tag != '' and get_lapl is False and save_spectrum is False:
        #fpath = self.generate_fpath('spectrum')
        try:
            e,U = np.array(sio.loadmat(f'{tag}.mat')['e'].todense())[0],sio.loadmat(f'{tag}.mat')['U'].todense()
            e,U = torch.tensor(e).to(device),torch.tensor(U).to(device)
            return e,U
        except Exception as e:
            print(e)
            pass
    
    if tag != '' and get_lapl is True:
        try:
            L = sio.loadmat(f'{tag}.mat')['L'].to_dense()
            L = torch.tensor(L).to(device).to_sparse()
            return L
        except Exception as e:
            print(e)
            pass
    
    try:
        mat = mat.to_dense().detach().cpu().numpy()
    except Exception as e:
        print(e)
        pass
    if lapl is None:
        py_g = graphs.MultiScale(mat)
        py_g.compute_laplacian('normalized')
        if get_lapl is True:
            sio.savemat(f'{tag}.mat',{'L':scipy.sparse.csr_matrix(py_g.L)})
            return py_g.L

        py_g.compute_fourier_basis()
        sio.savemat(f'{tag}.mat',{'e':scipy.sparse.csr_matrix(py_g.e),'U':scipy.sparse.csr_matrix(py_g.U)})
        U,e = torch.tensor(py_g.U).to(device),torch.tensor(py_g.e).to(device)
    else:
        e,U = scipy.linalg.eigh(np.array(lapl.to_dense(),order='F'),overwrite_a=True)
        e,U = torch.tensor(e).to(device),torch.tensor(U).to(device)
        
    return e, U

def process_graph(graph):
    """Obtain graph information from input TODO: MOVE?"""
    feats = graph.ndata['feature']
    return torch.vstack((graph.edges()[0],graph.edges()[1])), feats, graph
    mat_sparse=graph.adjacency_matrix()
    #L = torch_geometric.utils.get_laplacian(mat_sparse.coalesce().indices(),normalization='sym')
    L = get_spectrum(mat_sparse,lapl=None,tag='',load=False,get_lapl=True,save_spectrum=True)
    '''
    row,col = L[0]
    values = L[1]
    shape = mat_sparse.size()
    adj_matrix = sp.coo_matrix((values, (row, col)), shape=shape)
    graph = dgl.from_scipy(adj_matrix,eweight_name='w').to(graph.device)
    '''
    graph = dgl.from_scipy(L,eweight_name='w').to(graph.device)
    edges = torch.vstack((graph.edges()[0],graph.edges()[1]))
    graph.ndata['feature'] = feats
    #if 'edge' == self.batch_type:
    #    feats = feats['_N']
    return edges, feats, graph

def check_gpu_usage(tag):
    return
    allocated_bytes = torch.cuda.memory_allocated(torch.device('cuda'))
    cached_bytes = torch.cuda.memory_cached(torch.device('cuda'))

    allocated_gb = allocated_bytes / 1e9
    cached_gb = cached_bytes / 1e9
    print(f"{tag} -> GPU Memory - Allocated: {allocated_gb:.2f} GB, Cached: {cached_gb:.2f} GB")

def prep_args(args):
    """Retrieve arguments from config file"""
    with open(f'configs/{args.config}.yaml') as file:
        yaml_list = yaml.load(file,Loader=yaml.FullLoader)
    # args.epoch will be populated if datasaving
    if args.epoch is not None: yaml_list['MODEL']['EPOCH'] = args.epoch
    yaml_list['DATASET']['DATASAVE'] = args.datasave ; yaml_list['DATASET']['DATALOAD'] = args.dataload
    return yaml_list

def init_recons_agg(n,nfeats,exp_params):
    """A"""
    edge_anom_mats,node_anom_mats,recons_a,res_a_all = [],[],[],[]
    scales = exp_params['SCALES']
    for i in range(scales):
        am = np.zeros((n,n))
        #am = np.zeros((n,nfeats))
        edge_anom_mats.append(am)
        node_anom_mats.append(np.full((n,nfeats),-1.))
        recons_a.append(am)
        res_a_all.append(np.full((n,exp_params['MODEL']['HIDDEN_DIM']),-1.))
        #res_a_all.append(np.full((n,n),-1.))
    return edge_anom_mats,node_anom_mats,recons_a,res_a_all

def agg_recons(A_hat,res_a,struct_loss,feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats,recons_a,res_a_all,exp_params):
    """Collect batched reconstruction into graph-level reconstrution for anomaly detection"""
    edge_ids_ = edge_ids_.to('cpu').numpy()

    for sc in range(struct_loss.shape[0]):
        if exp_params['MODEL']['SAMPLE_TEST']:
            if exp_params['DATASET']['BATCH_TYPE'] == 'node' or exp_params['MODEL']['NAME'] in ['gcad']:
                node_anom_mats[sc][node_ids_.detach().cpu().numpy()[:feat_cost[sc].shape[0]]] = feat_cost[sc].detach().cpu().numpy()
                edge_anom_mats[sc][node_ids_.detach().cpu().numpy()[:feat_cost[sc].shape[0]]] = struct_loss[sc].detach().cpu().numpy()
            else:
                #edge_anom_mats[sc] = struct_loss[sc].detach().cpu().numpy()
                edge_anom_mats[sc][tuple(edge_ids_[sc,:,:])] = struct_loss[sc].detach().cpu().numpy()
                edge_anom_mats[sc] = np.maximum(edge_anom_mats[sc],edge_anom_mats[sc].T)

                #recons_a[sc] = A_hat[sc].detach().cpu().numpy()
                recons_a[sc][tuple(edge_ids_[sc,:,:])] = A_hat[sc].detach().cpu().numpy()#[edge_ids[:,0],edge_ids[:,1]].detach().cpu().numpy()
                recons_a[sc] = np.maximum(recons_a[sc],recons_a[sc].T)
                #if res_a is not None:
                #    res_a_all[sc][node_ids_.detach().cpu().numpy()] = res_a[sc].detach().cpu().numpy()
        else:
            if exp_params['DATASET']['BATCH_TYPE'] == 'node':
                node_anom_mats[sc][node_ids_.detach().cpu().numpy()[:feat_cost[sc].shape[0]]] = feat_cost[sc].detach().cpu().numpy()
                if struct_loss is not None:
                    edge_anom_mats[sc][node_ids_.detach().cpu().numpy()[:feat_cost[sc].shape[0]]] = struct_loss[sc].detach().cpu().numpy()
    return node_anom_mats,edge_anom_mats,recons_a,res_a_all

def dgl_to_nx(g):
    """Convert DGL graph to NetworkX graph"""
    nx_graph = nx.to_undirected(dgl.to_networkx(g.cpu(),edge_attrs='w'))
    node_ids = np.arange(g.num_nodes())
    return nx_graph,node_ids

def collect_recons_label(lbl,device):
    lbl_ = []
    for l in lbl:
        lbl_.append(l.to(device))
        del l ; torch.cuda.empty_cache()
    return lbl_


def seed_everything(seed=1234):
    """Set random seeds for run"""
    random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
        
def init_model(feat_size,exp_params,args):
    """Intialize model with configuration parameters"""
    struct_model,feat_model,params=None,None,None
    loaded=False
    try:
        exp_name = exp_params['EXP']
        if 'weibo' not in exp_name:
            struct_model = torch.load(f'{exp_name}.pt')
            loaded=True
    except Exception as e:
        print(e)
        pass
    
    if exp_params['MODEL']['NAME'] == 'gcad':
        gcad_model = GCAD(2,100,1)
    elif exp_params['MODEL']['NAME'] == 'madan':
        pass
    elif struct_model is None:
        struct_model = GraphReconstruction(feat_size, exp_params)
  
    device = torch.device(exp_params['DEVICE'])
    if struct_model:
        struct_model = struct_model.to(exp_params['DEVICE']) ; struct_model.requires_grad_(True) ; struct_model.train() ; params = struct_model.parameters()
    if feat_model:
        feat_model = feat_model.to(exp_params['DEVICE']) ; feat_model.train() ; params = feat_model.parameters()
    
    if exp_params['MODEL']['NAME'] == 'gcad':
        gcad = GCAD(2,100,4)
    elif exp_params['MODEL']['NAME'] == 'madan':
        pass

    return struct_model,params,loaded

def getScaleClusts(dend,thresh):
    clust_labels = postprocess.cut_straight(dend,threshold=thresh)
    return clust_labels

def flatten_label(anoms):
    flattened_anoms = []
    for anom_sc in anoms:
        for ind,i in enumerate(anom_sc):
            if ind == 0: ret_anom = np.expand_dims(i.flatten(),0)
            else: ret_anom = np.hstack((ret_anom,np.expand_dims(i.flatten(),0)))
        flattened_anoms.append(ret_anom[0])
    return flattened_anoms
    
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
                
def sparse_matrix_to_tensor(coo,feat):
    coo = scipy.sparse.coo_matrix(coo)
    v = torch.DoubleTensor(coo.data)
    i = torch.LongTensor(np.vstack((coo.row, coo.col)))
    dgl_graph = dgl.graph((i[0],i[1]),num_nodes=feat.shape[0])

    dgl_graph.edata['w'] = v
    dgl_graph.ndata['feature'] = feat
    return dgl_graph

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    adj += sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()