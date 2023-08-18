import torch
import numpy as np
import dgl
from utils import *
import torch_scatter
import random


class loss_func:
    def __init__(self,graph,feat,exp_params,sample=False, recons='struct', alpha=None, clusts=None, regularize=True):
        self.graph = graph
        self.feat = feat
        self.sample = sample
        self.recons = recons
        self.alpha = alpha
        self.clusts = clusts
        self.regularize=regularize
        self.model_ind= exp_params['MODEL']['IND']
        
    def calc_loss(self,lbl,A_hat,edge_ids,clusts):
        """
        Calculate reconstruction error given a graph reconstruction and its corresponding reconstruction
        label.

        
        """
        if not self.alpha: self.alpha = 1 if self.recons=='struct' else 0

        total_struct_error=torch.tensor(-1.)

        clusts = torch.tensor(clusts)
        for ind, sc_pred in enumerate(A_hat):
            samp_ind = self.model_ind if self.model_ind != 'None' else ind
            sampled_pred = sc_pred
            lbl_edges = torch.zeros(sampled_pred.shape).to(sampled_pred.device).to(torch.float64)
            check_gpu_usage('before edge idx')
            edge_idx=lbl[samp_ind].has_edges_between(edge_ids[samp_ind][:,0],edge_ids[samp_ind][:,1]).nonzero().T[0]

            lbl_edges[edge_idx] = lbl[samp_ind].edata['w'][lbl[samp_ind].edge_ids(edge_ids[samp_ind][:,0][edge_idx],edge_ids[samp_ind][:,1][edge_idx])].to(torch.float64)

            total_struct_error, edge_struct_errors,regloss,clustloss,nonclustloss,sc_idx_inside,sc_idx_outside = self.get_sampled_losses(lbl[samp_ind],sampled_pred,edge_ids[samp_ind],lbl_edges,ind,clusts[samp_ind])
            
            del sampled_pred, lbl_edges, edge_idx
            torch.cuda.empty_cache()
            check_gpu_usage('after edge idx')
                
            all_struct_error = (edge_struct_errors).unsqueeze(0) if ind == 0 else torch.cat((all_struct_error,(edge_struct_errors).unsqueeze(0)))
            all_reg_error = (regloss).unsqueeze(0) if ind == 0 else torch.cat((all_reg_error,(regloss).unsqueeze(0)))
            all_clust_error = (clustloss).unsqueeze(0) if ind == 0 else torch.cat((all_clust_error,(clustloss).unsqueeze(0)))
            all_nonclust_error = (nonclustloss).unsqueeze(0) if ind == 0 else torch.cat((all_nonclust_error,(nonclustloss).unsqueeze(0)))
            all_costs = total_struct_error.unsqueeze(0) if ind == 0 else torch.cat((all_costs,total_struct_error.unsqueeze(0)))

    
        del total_struct_error ; torch.cuda.empty_cache()
        return all_costs, all_struct_error, all_reg_error, all_clust_error, all_nonclust_error

    def get_group_idx(self,edge_ids,clust,i,anom_wise=True):
        """Get all edges associated with an anomaly group OR of the cluster(s) of the anomaly group"""
        dgl_g = dgl.graph((edge_ids[:,0],edge_ids[:,1]))
        if anom_wise:
            return dgl_g.out_edges(i,form='eid')
        else:
            anom_clusts = clust[i].unique()
            return dgl_g.out_edges(np.intersect1d(clust,anom_clusts,return_indices=True)[-2],form='eid')

    def get_sampled_losses(self,lbl,pred,edges,label,ind,clusts=None):
        """description

        Parameters
        ----------
            lbl: DGL graph
                DGL graph (label)
            pred: array-like, shape=[]
                normalized adjacency matrix
            edges: array-like, shape=[]
                feature matrix
            label: array=like, shape=[]
                positive edge list
        
        Returns
        ----------
            total_error : array-like, shape=[]
                total loss for backpropagation
            edge_errors : array-like, shape=[]
                edge-wise errors
        """
        
        # gets even # of edge/nonedge for each node in backprop
        # perform gather based on node id
        edge_errors=torch.nn.functional.mse_loss(pred.to(torch.float64), label, reduction='none')
    
        # if neg nodes doesn't cover, this means that there are disconnected nodes in label
        pos=(clusts[edges[:,0]]==clusts[edges[:,1]]).nonzero()
        pos_nodes1 = edges[pos.T[0]][:,1] ; pos_nodes2 = edges[pos.T[0]][:,0]
        pos_losses = edge_errors[pos.T[0]]
        pos_losses = torch_scatter.scatter_add(pos_losses,pos_nodes1)
        neg=(clusts[edges[:,0]]!=clusts[edges[:,1]]).nonzero()
        neg_nodes1 = edges[neg.T[0]][:,1]
        neg_losses = edge_errors[neg.T[0]]
        neg_losses = torch_scatter.scatter_add(neg_losses,neg_nodes1)
        pos_losses_tot = torch.zeros(clusts.shape[0]).to(float).to(pred.device) ; neg_losses_tot = torch.zeros(clusts.shape[0]).to(float).to(pred.device)

        edge_errors_nodewise = torch_scatter.scatter_add(edge_errors,edges[torch.cat((pos,neg)).T[0]][:,1].to(edge_errors.device))

        pos_tots,neg_tots = torch.unique(pos_nodes1,return_counts=True),torch.unique(neg_nodes1,return_counts=True)
        pos_tots_ = torch.arange(pos_losses.shape[0]).to(pos_losses.device) ; pos_tots_[pos_tots[0]] = pos_tots[1] ; pos_tots = pos_tots_
        neg_tots_ = torch.arange(neg_losses.shape[0]).to(neg_losses.device) ; neg_tots_[neg_tots[0]] = neg_tots[1] ; neg_tots = neg_tots_

        pos_losses_tot[torch.arange(pos_losses.shape[0])] = pos_losses
        neg_losses_tot[torch.arange(neg_losses.shape[0])] = neg_losses
        # NOTE: if there are more edges in the graph, assign a lower weight for backprop
        #import ipdb ; ipdb.set_trace()
        tot_error = edge_errors_nodewise
        attract_edges = pos ; repel_edges = neg
        return tot_error.mean(),tot_error, tot_error,pos_losses_tot,neg_losses_tot,attract_edges,repel_edges