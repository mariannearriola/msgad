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

def process_graph(graph):
    """Obtain graph information from input TODO: MOVE?"""
    edges = torch.vstack((graph.edges()[0],graph.edges()[1]))
    feats = graph.ndata['feature']
    #if 'edge' == self.batch_type:
    #    feats = feats['_N']
    return edges, feats, graph

def check_gpu_usage(tag):
    allocated_bytes = torch.cuda.memory_allocated(torch.device('cuda'))
    cached_bytes = torch.cuda.memory_cached(torch.device('cuda'))

    allocated_gb = allocated_bytes / 1e9
    cached_gb = cached_bytes / 1e9
    print(f"{tag} -> GPU Memory - Allocated: {allocated_gb:.2f} GB, Cached: {cached_gb:.2f} GB")

def prep_args(args):
    with open(f'configs/{args.config}.yaml') as file:
        yaml_list = yaml.load(file,Loader=yaml.FullLoader)
    # args.epoch will be populated if datasaving
    if args.epoch is not None: yaml_list['MODEL']['EPOCH'] = args.epoch
    yaml_list['DATASET']['DATASAVE'] = args.datasave ; yaml_list['DATASET']['DATALOAD'] = args.dataload
    return yaml_list

def init_recons_agg(n,nfeats,exp_params):
    edge_anom_mats,node_anom_mats,recons_a,res_a_all = [],[],[],[]
    scales = 3 if 'multi-scale' in exp_params['MODEL']['NAME'] else 1
    for i in range(scales):
        am = np.zeros((n,n))
        edge_anom_mats.append(am)
        node_anom_mats.append(np.full((n,nfeats),-1.))
        recons_a.append(am)
        res_a_all.append(np.full((n,exp_params['MODEL']['HIDDEN_DIM']),-1.))
    return edge_anom_mats,node_anom_mats,recons_a,res_a_all

def agg_recons(A_hat,res_a,struct_loss,feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats,recons_a,res_a_all,exp_params):

    for sc in range(struct_loss.shape[0]):
        if exp_params['MODEL']['SAMPLE_TEST']:
            if exp_params['DATASET']['BATCH_TYPE'] == 'node' or exp_params['MODEL']['NAME'] in ['gcad']:
                node_anom_mats[sc][node_ids_.detach().cpu().numpy()[:feat_cost[sc].shape[0]]] = feat_cost[sc].detach().cpu().numpy()
                edge_anom_mats[sc][node_ids_.detach().cpu().numpy()[:feat_cost[sc].shape[0]]] = struct_loss[sc].detach().cpu().numpy()
            else:
                #edge_anom_mats[sc][tuple(edge_ids_[sc])] = struct_loss[sc].detach().cpu().numpy()
                #edge_anom_mats[sc][tuple(np.flip(edge_ids_[sc],axis=0))] = edge_anom_mats[sc][tuple(edge_ids_[sc])]
                edge_anom_mats[sc][tuple(edge_ids_)] = struct_loss[sc].detach().cpu().numpy()
                edge_anom_mats[sc][tuple(np.flip(edge_ids_,axis=0))] = edge_anom_mats[sc][tuple(edge_ids_)]
                recons_a[sc][tuple(edge_ids_)] = A_hat[sc].detach().cpu().numpy()#[edge_ids[:,0],edge_ids[:,1]].detach().cpu().numpy()
                recons_a[sc][tuple(np.flip(edge_ids_,axis=0))] = recons_a[sc][tuple(edge_ids_)]
                #recons_a[sc][tuple(edge_ids_[sc])] = A_hat[sc][edge_ids[:,0],edge_ids[:,1]].detach().cpu().numpy()
                #recons_a[sc][tuple(np.flip(edge_ids_[sc],axis=0))] = recons_a[sc][tuple(edge_ids_[sc])]
                if res_a is not None:
                    #res_a_all[sc] = res_a[sc].detach().cpu().numpy()
                    res_a_all[sc][node_ids_.detach().cpu().numpy()] = res_a[sc].detach().cpu().numpy()
        else:
            if exp_params['DATASET']['BATCH_TYPE'] == 'node':
                node_anom_mats[sc][node_ids_.detach().cpu().numpy()[:feat_cost[sc].shape[0]]] = feat_cost[sc].detach().cpu().numpy()
                if struct_loss is not None:
                    edge_anom_mats[sc][node_ids_.detach().cpu().numpy()[:feat_cost[sc].shape[0]]] = struct_loss[sc].detach().cpu().numpy()
    return node_anom_mats,edge_anom_mats,recons_a,res_a_all

def dgl_to_nx(g):
    nx_graph = nx.to_undirected(dgl.to_networkx(g.cpu()))
    node_ids = np.arange(g.num_nodes())
    return nx_graph,node_ids

def collect_recons_label(lbl,device):
    lbl_ = []
    for l in lbl:
        lbl_.append(l.to(device))
        del l ; torch.cuda.empty_cache()
    return lbl_


def seed_everything(seed=1234):
    random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
        
def init_model(feat_size,exp_params):
    struct_model,feat_model,params=None,None,None
    if exp_params['MODEL']['NAME'] == 'gcad':
        gcad_model = GCAD(2,100,1)
    elif exp_params['MODEL']['NAME'] == 'madan':
        pass
    else:
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

    return struct_model,params

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
    v = torch.FloatTensor(coo.data)
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