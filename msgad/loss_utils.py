import torch
import numpy as np
from utils import *
import torch_scatter


class loss_func:
    def __init__(self,graph,feat,exp_params,sample=False, recons='struct', clusts=None, regularize=True):
        self.graph = graph
        self.feat = feat
        self.sample = sample
        self.recons = recons
        self.clusts = clusts
        self.regularize=regularize
        
    def calc_loss(self,lbl,A_hat,edge_ids,clusts, batch_nodes):
        """
        Calculate reconstruction error given a graph reconstruction and its corresponding reconstruction label
        Input:
            lbl: DGL graph

            A_hat: array-like, shape=[k, ...]
            edge_ids:
            clusts:
            batch_nodes:

        Output:
            all_costs: array-like, shape=[k, ]
                Cumulative scale-wise losses
            all_struct_error: array-like, shape=[k, n]
                Scale-wise total losses for each node
            all_clust_error: array-like, shape=[k, n]
                Scale-wise intra-cluster losses for each node
            all_nonclust_error: array-like, shape=[k, n]
                Scale-wise inter-cluster losses for each node
        """
        total_struct_error=torch.tensor(-1.)
        
        clusts = torch.tensor(clusts)
        for ind, sc_pred in enumerate(A_hat):
            lbl_edges = torch.zeros(edge_ids[ind].shape[0]).to(sc_pred.device).to(torch.float64)
            edge_idx=lbl.has_edges_between(edge_ids[ind][:,0],edge_ids[ind][:,1]).nonzero().T[0]
            lbl_edges[edge_idx] = lbl.edata['w'][lbl.edge_ids(edge_ids[ind][:,0][edge_idx],edge_ids[ind][:,1][edge_idx])].to(torch.float64)
            total_struct_error, edge_struct_errors,clustloss,nonclustloss = self.get_sampled_losses(sc_pred,edge_ids[ind],lbl_edges,clusts[ind],batch_nodes)
            
            del lbl_edges, edge_idx
            torch.cuda.empty_cache()
                
            all_struct_error = (edge_struct_errors).unsqueeze(0) if ind == 0 else torch.cat((all_struct_error,(edge_struct_errors).unsqueeze(0)))
            all_clust_error = (clustloss).unsqueeze(0) if ind == 0 else torch.cat((all_clust_error,(clustloss).unsqueeze(0)))
            all_nonclust_error = (nonclustloss).unsqueeze(0) if ind == 0 else torch.cat((all_nonclust_error,(nonclustloss).unsqueeze(0)))
            all_costs = total_struct_error.unsqueeze(0) if ind == 0 else torch.cat((all_costs,total_struct_error.unsqueeze(0)))
    
        del total_struct_error ; torch.cuda.empty_cache()
        return all_costs, all_struct_error, all_clust_error, all_nonclust_error

    def get_sampled_losses(self,pred,edges,label,clusts,batch_nodes):
        """Collects intra-cluster and inter-cluster loss evenly for each node

        Input:
            pred: array-like, shape=[e, ]
                normalized adjacency matrix
            edges: array-like, shape=[e, 2]
                feature matrix
            label: array=like, shape=[e, ]
                positive edge list
            clusts: array-like, shape=[k, n]
        
        Output:
            tot_error_sum : float
                total loss for backpropagation
            tot_error: array-like, shape=[3, n]
                Node-wise losses (sum of intra/inter-cluster losses)
            intra_losses_tot : array-like, shape=[3, n]
                Node-wise intra cluster losses
            inter_losses_tot : array-like, shape=[3, n]
                Node-wise inter cluster losses
        """
        # calculate edge-wise loss
        edge_errors=torch.nn.functional.mse_loss(pred.to(torch.float64), label, reduction='none')

        # decompose edge-wise loss into intra-cluster losses and inter-cluster losses (roughly same size from edge sampling)
        intra_edges=(clusts[edges[:,0]]==clusts[edges[:,1]]).nonzero()
        intra_nodes = edges[intra_edges.T[0]][:,1]
        intra_losses = edge_errors[intra_edges.T[0]]
        
        intra_losses = torch_scatter.scatter_add(intra_losses,intra_nodes)
        inter_edges=(clusts[edges[:,0]]!=clusts[edges[:,1]]).nonzero()
        inter_nodes = edges[inter_edges.T[0]][:,1]

        inter_losses = edge_errors[inter_edges.T[0]]
        inter_losses = torch_scatter.scatter_add(inter_losses,inter_nodes)

        intra_losses_tot = torch.zeros(clusts.shape[0]).to(float).to(pred.device) ; inter_losses_tot = torch.zeros(clusts.shape[0]).to(float).to(pred.device)
        
        # node-wise loss based on combination of intra/inter-cluster losses 
        all_nodes = edges[torch.cat((intra_edges,inter_edges)).T[0]][:,1]
        node_errors = torch_scatter.scatter_add(edge_errors,all_nodes.to(edge_errors.device))

        intra_nodes_all,inter_nodes_all = torch.unique(intra_nodes,return_counts=True),torch.unique(inter_nodes,return_counts=True)
        intra_nodes_all_ = torch.arange(intra_losses.shape[0]).to(intra_losses.device) ; intra_nodes_all_[intra_nodes_all[0]] = intra_nodes_all[1] ; intra_nodes_all = intra_nodes_all_
        inter_nodes_all_ = torch.arange(inter_losses.shape[0]).to(inter_losses.device) ; inter_nodes_all_[inter_nodes_all[0]] = inter_nodes_all[1] ; inter_nodes_all = inter_nodes_all_
        if len(intra_edges) > 0: intra_losses_tot[torch.arange(intra_nodes.max()+1)] = intra_losses
        if len(inter_edges) > 0: inter_losses_tot[torch.arange(inter_nodes.max()+1)] = inter_losses

        tot_error = torch.zeros(clusts.shape[0]).to(float).to(pred.device)
        tot_error[torch.arange(all_nodes.max()+1).detach().cpu()] = node_errors

        return tot_error.sum(), tot_error, intra_losses_tot, inter_losses_tot