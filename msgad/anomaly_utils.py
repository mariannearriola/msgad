from scipy.sparse import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.io
from sklearn.metrics import roc_auc_score
from datetime import datetime
import argparse
import scipy
import networkx as nx
import torch_geometric
from torch.utils.data import DataLoader
from scipy import stats
from model_cora import EGCN
from utils import * 
from anomaly_utils import *
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KernelDensity
import random 


def plot_embed_diff(attrs,train_adj,nodes,state,model,sc,anom_sc=None,d=None):
    import matplotlib.pyplot as plt
    plt.figure()
    #weights = [model.conv.linear.weight.data]
    weights = []
    corrs = []
    for node in nodes:
        l_scores = []
        adj_diffs =[]
        for prob in range(0,100,5):
            x_pert,a_pert = perturb_adj(attrs[0],train_adj,prob/100,prob/100,node)
            
            a_pert = a_pert.cuda().type(torch.float32)
            x_pert_full = attrs[0].clone()
            x_pert_full[node]=x_pert
            x_pert_full = x_pert_full.type(torch.float32)
            #import ipdb ; ipdb.set_trace()
            #x_pert_full = attrs[0]
            #a_pert = adj
            emb_pert,_ = model([x_pert_full],a_pert,sc)
            emb, _ = model(attrs,a_pert,sc)
            #import ipdb ; ipdb.set_trace()
            emb_pert = emb_pert[sc]
            emb = emb[sc]
            l_score,bound_met=calc_lipschitz(train_adj, a_pert, attrs[0], x_pert_full, emb, emb_pert, weights, node)
            l_scores.append(l_score)
            adj_diffs.append(torch.linalg.norm(attrs[0][node]-x_pert_full[node],2).detach().cpu().numpy())#+torch.linalg.norm(train_adj[node]-a_pert[node],2).detach().cpu().numpy())
        from scipy.stats import pearsonr
        corr,_=pearsonr(adj_diffs,l_scores)
        #adj_diffs = range(0,100,5)
        plt.scatter(adj_diffs,l_scores)
        corrs.append(round(corr,3))
    plt.legend(corrs)
    plt.xlabel('perturbation')
    plt.ylabel('embedding difference')
    if anom_sc:
        plt.title(f'{str(sc)} embedding analysis of {state} sc {anom_sc} mean corr {round(np.mean(np.array(corrs)),3)}')
        plt.savefig(f'figs/{d}/{state}_{anom_sc}_recons_{str(sc)}.png')

    else:
        plt.title(f'{str(sc)} embedding analysis of {state} mean corr {round(np.mean(np.array(corrs)),3)}')
        plt.savefig(f'figs/{d}/{state}_recons_{str(sc)}.png')

def calc_lipschitz(a, a_pert, x, x_pert, embed, embed_pert, weights, node):
    norm_weights = [0]
    '''
    for weight in weights:
        #norm_weights.append(spectral_norm(weight))
        norm_weights.append(torch.linalg.norm(weight,2))
    '''
    #import ipdb ; ipdb.set_trace() 
    #A_hat_diff = torch.linalg.norm((a@x)[node]-(a_pert@x_pert)[node],2)
    embed_diff = torch.linalg.norm(embed[node]-embed_pert[node],2)
    #upper_bound = 0
    for ind,norm_weight in enumerate(norm_weights):
        if ind == 0:
            upper_bound = norm_weight
        else:
            upper_bound *= norm_weight
        #upper_bound *= norm_weight*A_hat_diff
    #import ipdb ; ipdb.set_trace()
    #print(embed_diff.item(),upper_bound.item())
    return embed_diff.item(),embed_diff <= upper_bound

def perturb_adj(attribute_dense,adj,x_prob,adj_prob,node):
    # feature perturbation
    adj_dense = adj.clone()
    #import ipdb ; ipdb.set_trace()
    r_x = torch.distributions.binomial.Binomial(1,torch.tensor([x_prob])).sample(attribute_dense[node].shape)[:,0]
    noise = torch.mul(r_x,torch.normal(torch.full(r_x.shape,0.),torch.full(r_x.shape,1.)))
    x_pert = attribute_dense[node]+noise.cuda()
     
    # structure perturbation
    half_a = torch.triu(adj)
    node_edges = torch.nonzero(half_a[node]).flatten()
    r = torch.distributions.binomial.Binomial(1,torch.tensor([1-adj_prob])).sample(node_edges.shape)[:,0]
    broken_edges = torch.where(r==0)[0]
    #import ipdb ; ipdb.set_trace()
    for idx in broken_edges:
        try:
            adj_dense[node][node_edges[idx]] = 0
            adj_dense[node_edges[idx]][node] = 0
        except:
            import ipdb ; ipdb.set_trace()
            print('oh no')
    adj_norm = normalize_adj(adj_dense.detach().cpu().numpy() + sp.eye(adj_dense.shape[0]))
    return x_pert,torch.tensor(adj_norm.todense()).cuda()

def loss_func(adj, A_hat_scales, attrs, X_hat, alpha, recons, weight=None):
    attribute_cost = torch.tensor(0., dtype=torch.float32).cuda()

    # structure reconstruction loss
    all_costs, all_structure_costs = None, None
    for ind, A_hat in enumerate(A_hat_scales):
        '''
        structure_reconstruction_errors = torch.nn.functional.binary_cross_entropy(A_hat,adj.cuda(),reduction='none')
        structure_reconstruction_errors = torch.mean(structure_reconstruction_errors,dim=1)
        '''
        for row_ind,row in enumerate(adj): 
            edge_count = torch.where(row==1)[0].shape[0]
            non_edges = row[torch.where(row==0)]
            non_edges_score = torch.randint(row.shape[0],(edge_count,))
            all_edges_score = torch.cat((torch.where(row==1)[0],non_edges_score.cuda()))
            
            recons_error=torch.nn.functional.binary_cross_entropy(A_hat[row_ind][all_edges_score],row[all_edges_score].cuda())
            recons_error+=torch.nn.functional.cosine_similarity(A_hat[row_ind][all_edges_score],torch.full(all_edges_score.shape,.5).cuda(),dim=0)*.5
            if row_ind == 0:
                structure_reconstruction_errors=recons_error.unsqueeze(-1)
            else:
                structure_reconstruction_errors = torch.cat((structure_reconstruction_errors,recons_error.unsqueeze(-1)))
        
        if all_costs is None:
            all_costs = torch.mean(structure_reconstruction_errors).unsqueeze(-1)
            all_structure_costs = (structure_reconstruction_errors).unsqueeze(-1)
        else:
            all_costs = torch.cat((all_costs,torch.mean(structure_reconstruction_errors).unsqueeze(-1)))
            all_structure_costs = torch.cat((all_structure_costs,(structure_reconstruction_errors).unsqueeze(-1)),dim=-1)

    #cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors
    return all_costs, all_structure_costs.detach().cpu(), attribute_cost

from sklearn.metrics import average_precision_score

def detect_anom_ap(errors,label):
    return average_precision_score(label,errors)

def redo(anom):
    for ind,i in enumerate(anom):
        if ind == 0:
            ret_anom = np.array(i)
        else:
            ret_anom = np.concatenate((ret_anom,np.array(i)))
    return ret_anom

def detect_anom(sorted_errors, label, top_nodes_perc,scores,thresh):
    anom_sc1 = label[0][0]#[0]
    anom_sc2 = label[0][1]#[0]
    anom_sc3 = label[0][2]#[0]
    true_anoms = 0
    def redo(anom):
        for ind,i in enumerate(anom):
            if ind == 0:
                ret_anom = np.array(i)
            else:
                ret_anom = np.concatenate((ret_anom,np.array(i)))
        return ret_anom
    anom_sc1 = redo(anom_sc1)
    anom_sc2 = redo(anom_sc2)
    anom_sc3 = redo(anom_sc3)
    all_anom = np.concatenate((anom_sc1,np.concatenate((anom_sc2,anom_sc3),axis=None)),axis=None)
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
    print(cor_1/anom_sc1.shape[0],cor_2/anom_sc2.shape[0],cor_3/anom_sc3.shape[0])
    return true_anoms/int(all_anom.shape[0]*top_nodes_perc), cor_1, cor_2, cor_3, true_anoms
     
    def find_top_anom(scores,thresh,skew=False):
        if skew:
            z=stats.zscore(scores)
            #z=z[np.where(z<0)]
            thresh=-thresh
            top_anom=sorted_errors[np.where(z<thresh)]
        else:
            z=stats.zscore(scores)
            #z=z[np.where(z>0)]
            top_anom=sorted_errors[np.where(z>thresh)]
        #import ipdb ; ipdb.set_trace()
        return top_anom
    
    if False:
        import ipdb ; ipdb.set_trace()
        import matplotlib.pyplot as plt
        plt.figure()
        #skew1=round(scipy.stats.skew(anom_inds1),.5)
        #skew2=round(scipy.stats.skew(anom_inds2),.75)
        skew3=round(scipy.stats.skew(anom_inds3),1)
        plt.hist(anom_inds1,color='r',alpha=1,range=(0,200),bins=200)
        plt.hist(anom_inds2,color='g',alpha=1,range=(0,200),bins=200)
        plt.hist(anom_inds3,color='b',alpha=1,range=(0,200),bins=200)
        plt.title(f'{skew1},{skew2},{skew3}')
        plt.savefig(f'dists_{start}_{end}')

    return true_anoms/int(all_anom.shape[0]*top_nodes_perc), cor_1, cor_2, cor_3, true_anoms# cor_4, true_anoms
