from sknetwork.hierarchy import Paris, postprocess, LouvainHierarchy, LouvainIteration
from torch_geometric.nn import MessagePassing
import numpy as np
import scipy.sparse as sp
import torch
import torch_scatter
import scipy
import scipy.io as sio
import random
import networkx as nx
from igraph import Graph
import dgl
import copy
import matplotlib.ticker as mtick
import pickle as pkl
import model
#from models import *
#from models.gcad import *
import matplotlib.pyplot as plt
import os
import yaml
import torch
import torch_geometric
import gc
from pygspcopy import *
import sklearn
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
from anom_detector import *
from sklearn.ensemble import IsolationForest

def get_counts(pred,clust,edge_ids):
    pred_opt = torch.zeros(pred.shape)
    a1 = torch.where(clust[edge_ids[:,0]]==clust[edge_ids[:,1]])[0]
    pred_opt[a1] = 1.
    pos_nodes = edge_ids[:,1][a1].unique().detach().cpu() ; neg_nodes = edge_ids[:,1][torch.where(clust[edge_ids[:,0]]!=clust[edge_ids[:,1]])[0]].unique().detach().cpu()
    unique_elements, counts = torch.unique(edge_ids[:,1][a1], return_counts=True)
    #pos_clust_counts = torch.zeros(clust.shape[0],dtype=int).scatter_add_(0,counts.detach().cpu(),unique_elements.detach().cpu())# ; pos_clust_counts = pos_clust_counts[clust]
    pos_clust_counts = torch.zeros(clust.shape[0]) ; pos_clust_counts[unique_elements.detach().cpu()] = counts.to(torch.float32).detach().cpu()
    # Create a dictionary to store the counts of unique elements
    pos_counts = dict(zip(unique_elements.tolist(), counts.tolist()))
    unique_elements, counts = torch.unique(edge_ids[:,1][torch.where(clust[edge_ids[:,0]]!=clust[edge_ids[:,1]])[0]], return_counts=True)
    # Create a dictionary to store the counts of unique elements
    neg_counts = dict(zip(unique_elements.tolist(), counts.tolist()))
    #neg_clust_counts = torch.zeros(clust.shape[0],dtype=int).scatter_add_(0,counts.detach().cpu(),unique_elements.detach().cpu())# ; neg_clust_counts = neg_clust_counts[clust]
    neg_clust_counts = torch.zeros(clust.shape[0]) ; neg_clust_counts[unique_elements.detach().cpu()] = counts.to(torch.float32).detach().cpu()

    return a1, pred_opt, pos_nodes, pos_counts, neg_nodes, neg_counts,pos_clust_counts,neg_clust_counts

def score_multiscale_anoms(clustloss,nonclustloss, clusts, pred, edge_ids, res):

    for sc,sc_clustloss in enumerate(clustloss):
        #cl1,cl2,cl3=clustloss ; cl1 = cl1.detach().cpu() ; cl2 = cl2.detach().cpu() ; cl3 = cl3.detach().cpu()
        #nc1,nc2,nc3=nonclustloss ; nc1 = nc1.detach().cpu() ; nc2 = nc2.detach().cpu(); nc3 = nc3.detach().cpu()
        
        #cl1,cl2,cl3=gather_clust_info(cl1,clusts[0],'std'),gather_clust_info(cl2,clusts[1],'std'),gather_clust_info(cl3,clusts[2],'std')
        #nc1,nc2,nc3=gather_clust_info(nc1,clusts[0],'mean'),gather_clust_info(nc2,clusts[1],'mean'),gather_clust_info(nc3,clusts[2],'mean')
        
        score = gather_clust_info(sc_clustloss.detach().cpu(),clusts[sc],'std')
        scores_all = score if sc == 0 else torch.vstack((scores_all,score))
        #l1 = cl1
        #l2 = cl2
        #l3 = cl3

        #inter_res = res[0][0].sigmoid().detach().cpu()
        #l1 = torch.tensor(get_clust_score(res[0][0].sigmoid().detach().cpu(),np.arange(res[1][0].shape[0]),clusts,1,0))#cl1
        #l2 = torch.tensor(get_clust_score(res[1][0].sigmoid().detach().cpu(),np.arange(res[0][0].shape[0]),clusts,2,1))#nc2*(nc1-nc2)#cl2
        #l3 = torch.tensor(get_clust_score(res[0][0].sigmoid().detach().cpu(),np.arange(res[0][0].shape[0]),clusts,2,0))#nc3*(nc2-nc3)#cl3

        #l1,l2,l3=gather_clust_info(l1,clusts[0],'mean'),gather_clust_info(l2,clusts[1],'mean'),gather_clust_info(l3,clusts[2],'mean')
    #l_all = torch.vstack((torch.vstack((l1,l2)),l3))#.softmax(0)
    return scores_all, clustloss, nonclustloss

def torch_overlap(a,b):
    return a[(a.view(1, -1) == b.view(-1, 1)).any(dim=0)]

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

def generate_adjacency_matrix(cluster_ids):
    mask = cluster_ids.unsqueeze(0) == cluster_ids.unsqueeze(1)
    clust_edges = mask.nonzero().T
    gr = dgl.graph((clust_edges[0],clust_edges[1]))
    nonclust_edges = torch.stack(dgl.sampling.global_uniform_negative_sampling(gr,clust_edges.shape[-1]))
    return torch.cat((clust_edges,nonclust_edges),dim=1)
    unique_clusters = (cluster_ids).unique()
    num_clusters = len(unique_clusters)
    num_nodes = len(cluster_ids)

    # Create an empty adjacency matrix
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))

    # Create a dictionary to map cluster IDs to node indices
    cluster_indices = {cluster: torch.where(cluster_ids == cluster)[0] for cluster in unique_clusters}

    # Iterate over each cluster and connect nodes within the same cluster
    for cluster in unique_clusters:
        nodes_in_cluster = cluster_indices[cluster]
        adjacency_matrix[nodes_in_cluster[:, None], nodes_in_cluster] = 1

    return adjacency_matrix


class TBWriter:
    def __init__(self,tb, label, sc_label,clust,anoms,norms,exp_params):
        '''
        # edges connected to any anomaly
        self.sc_idx_all,self.sc_idx_all,self.sc_idx_inside,self.sc_idx_outside,self.cl_all={},{},{},{},{}
        #self.opt_entropies,self.avg_opt_entropies=[],[]
        for ind,i in enumerate(sc_label):
            a,b = np.intersect1d(edge_ids[0].detach().cpu(),i,return_indices=True)[-2],np.intersect1d(edge_ids[1].detach().cpu(),i,return_indices=True)[-2]
            self.sc_idx_all[ind]=np.unique(np.concatenate((a,b)).flatten())
            self.cl_all[ind] = edge_ids[:,self.sc_idx_all[ind]][0].detach().cpu()
        self.sc_labels = sc_label
        # only select edges inside cluster
        #self.sc_idx_inside[lbl] = np.where(clust[edge_ids.detach().cpu().numpy()][0] == clust[edge_ids.detach().cpu().numpy()][1])[0]
        self.sc_idx_inside = attract_edges_sel
        # only select edges outside cluster
        #self.sc_idx_outside[lbl] = np.where(clust[edge_ids.detach().cpu().numpy()][0] != clust[edge_ids.detach().cpu().numpy()][1])[0]
        self.sc_idx_outside = repel_edges_sel
        '''
        self.tb = tb
        
        self.clust = clust
        self.sc_labels = sc_label
        self.anoms = anoms
        self.norms = norms
        #self.anoms_cuda = torch.tensor(anoms).cuda()
        self.model_ind = exp_params['MODEL']['IND']
        self.a_clf = anom_classifier(exp_params,exp_params['DATASET']['SCALES'])
        self.truth = label
        self.score_clf=IsolationForest(n_estimators=2, warm_start=True)

    def collect_clust_loss(self,edge_ids,sc_idx,loss):
        """Get average loss for each cluster, and assign back to edge-wise losses"""
        #sc_idx = np.intersect1d(sc_idx,sc_idx_)
        cl = edge_ids[:,sc_idx][0].detach().cpu()
        average_tensor = torch.scatter_reduce(loss[sc_idx].detach().cpu(), 0, cl, reduce="mean")
        expanded_average = average_tensor[cl].to(edge_ids.device).mean()#/cl.shape[0]
        return expanded_average

    def get_group_idx(self,edge_ids,clust,i,anom_wise=True):
        """Get all edges associated with an anomaly group OR of the cluster(s) of the anomaly group"""
        dgl_g = dgl.graph((edge_ids[:,0],edge_ids[:,1]))
        if anom_wise:
            return dgl_g.in_edges(i,form='eid')
        else:
            anom_clusts = clust[i].unique()
            return dgl_g.in_edges(np.intersect1d(clust,anom_clusts,return_indices=True)[-2],form='eid')

    def plot_sep_scores(self,scores,avg_score,cluster_labels,anom_sc):
        """https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py"""
        n_clusters = cluster_labels.unique().shape[0]
        fig, ax1= plt.subplots(1, 1)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(scores) + (n_clusters + 1) * 10])

        
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = scores[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=avg_score, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.savefig(f'clust_figs/clusts_{anom_sc}.png')

    def plot_sep_scores_group(self,scores,avg_score,anom_group,anom_sc):
        """https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py"""
        fig, ax1= plt.subplots(1, 1)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(anom_group) + (len(anom_group) + 1) * 10])

        
        y_lower = 10
        for i in range(len(scores)):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = scores[i][anom_group]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / len(scores))
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        #ax1.axvline(x=np.array(avg_score).mean(), color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.savefig(f'clust_figs/clusts_{anom_sc}.png')

    def sampled_edges(self,tensor):
        """For sampling an even number of in-clust/out-clust edges"""
        ones_indices = torch.nonzero(tensor == 1).flatten()
        zeros_indices = torch.nonzero(tensor == 0).flatten()

        # Randomly sample an equal number of indices from each group
        num_samples = min(len(ones_indices), len(zeros_indices))
        sampled_ones_indices = random.sample(ones_indices.tolist(), num_samples)
        sampled_zeros_indices = random.sample(zeros_indices.tolist(), num_samples)

        # Concatenate the sampled indices together
        sampled_indices = sampled_ones_indices + sampled_zeros_indices
        return sampled_indices
    def get_anom(self,scores,anom):
        """
        Filtering mechanism using multi-sccale scoring. Prioritize scores from higher scales
        """
        scores_left = np.arange(scores[0].shape[0]).tolist()
        anoms_found,ps,final_scores=[],[],[np.zeros(scores[0].shape[0]) for i in scores]
        #import ipdb ; ipdb.set_trace()
        for sc,score in enumerate(reversed(scores)):
            #centroids1,_ = scipy.cluster.vq.kmeans(score[np.array(scores_left)],2)
            #groups1, _ = scipy.cluster.vq.vq(l1, centroids1)
            #anoms_found.append(np.array(scores_left)[(groups1 == 0).nonzero()])
            preds = self.score_clf.fit_predict(score[np.array(scores_left)].reshape(-1,1))
            anoms_found.append(np.array(scores_left)[((preds==-1).nonzero()[0])].tolist())
            ps.append(np.intersect1d(anoms_found[-1],anom).shape[0]/anom.shape[0])
            scores_left = list(set(scores_left) - set(anoms_found[-1]))
            final_scores[sc][np.array(anoms_found[-1])] = 1
        anoms_found.reverse() ; ps.reverse()
        return anoms_found,ps,final_scores
    
    def update_dict(self,gr_dict,k,v):
        if k not in gr_dict.keys():
            gr_dict[k] = [v]
        else:
            gr_dict[k].append(v)
        return gr_dict

    def tb_write_anom(self,adj,sc_label,edge_ids,pred,res,loss,sc,epoch,regloss,clustloss,nonclustloss,clusts,anom_wise=True,fracs=None):
        """Log loss evolution for anomaly group"""
        # sc idx all: contains all edge idx associated with each anomaly
        # removing edges if there is no cluster between them, does not add edges
        if self.model_ind != 'None':
            clust = clusts[self.model_ind]
        else:
            clust = clusts[sc]
        a1, pred_opt, pos_nodes, pos_counts_dict, neg_nodes, neg_counts_dict,_,_ = get_counts(pred,clust,edge_ids[sc])
    

        anom_clusts = copy.deepcopy(clust)
        it = clust.max()+1
        mean_intras,mean_inters=[],[]
        all_dicts = []
        sc_labels = np.unique(sc_label) ; sc_labels = np.append(sc_labels,-1)

        scores_all,clustloss,nonclustloss=score_multiscale_anoms(clustloss,nonclustloss, clusts, pred, edge_ids, res)
        
        self.tb.add_histogram(f'Loss_inclust_{sc}',clustloss[sc].detach().cpu().mean(), epoch)
        self.tb.add_histogram(f'Loss_outclust_{sc}',nonclustloss[sc].detach().cpu().mean(), epoch)
        for ind,(nc,cl) in enumerate(zip(nonclustloss,clustloss)):
            _, _, _, _, _, _, pos_counts,neg_counts = get_counts(pred,clusts[ind],edge_ids[ind])
            sil = torch.nan_to_num((1-torch.nan_to_num(nc.detach().cpu()/neg_counts,posinf=0,neginf=0)-(torch.nan_to_num(cl.detach().cpu()/pos_counts,posinf=0,neginf=0)))/torch.max(1-torch.nan_to_num(nc.detach().cpu()/neg_counts,posinf=0,neginf=0),(torch.nan_to_num(cl.detach().cpu()/pos_counts,posinf=0,neginf=0))),posinf=0,neginf=0)
            self.tb.add_histogram(f'Sil{ind+1}',sil.mean(), epoch)
            sils = sil if ind == 0 else torch.vstack((sils,sil))
        
        for ind,i in enumerate(sc_labels):
            anom = torch.tensor(self.anoms[np.where(np.array(sc_label)==i)]) if i != -1 else torch.tensor(self.norms)
            #anom = anom[clusts[(i-1)][anom]==clusts[(i-1)][anom].unique()[0]]
            #if i > 0: import ipdb ; ipdb.set_trace()
            sc_truth = np.zeros(clust.shape).astype(float) ; sc_truth[anom] = 1.
            #sc_label_i = np.zeros(clust.shape).astype(float) ; sc_label_i[(np.array(sc_label)==i).nonzero()] = 1.
            #import ipdb; ipdb.set_trace()
            #anom_neighbors=torch.stack(adj.out_edges(anom)).T
            #anom_neighbors=anom_neighbors[(~torch.isin(clust[anom_neighbors].unique(),clust[anom])).nonzero()]

            # gets all edges related to a group
            gr_dict = {}
            gr_dict = self.update_dict(gr_dict,f'Pred_fraction',(pred>=.5).nonzero().shape[0]/pred.shape[0])
            gr_dict = self.update_dict(gr_dict,f'True_fraction',a1.shape[0]/pred.shape[0])     
            for ind,score in enumerate(scores_all):       
                gr_dict = self.update_dict(gr_dict,f'L{ind+1}_{anom_wise}',(score[anom].detach().cpu()))
            anom_sc = ind if ind != len(self.sc_labels)-1 else 'norm'
            try:
                anom_pos_counts = np.array([pos_counts_dict[uid] if uid in pos_counts_dict.keys() else 1 for uid in anom])
                anom_neg_counts = np.array([neg_counts_dict[uid] if uid in neg_counts_dict.keys() else 1 for uid in anom])
            except Exception as e:
                print(e)
                import ipdb ; ipdb.set_trace()
            gr_dict = self.update_dict(gr_dict,f'Loss',gather_clust_info(loss[sc].detach().cpu(),clusts[sc])[anom])
            #mean_intra = gather_clust_info(clustloss[sc].detach().cpu(),clusts[sc])[anom]
            mean_intra = clustloss[sc].detach().cpu()[anom]
            gr_dict = self.update_dict(gr_dict,f'Loss_inclust',mean_intra)
            gr_dict = self.update_dict(gr_dict,f'Loss_inclust_mean',gather_clust_info(clustloss[sc][anom].detach().cpu(),clusts[sc][anom],'mean'))

            gr_dict = self.update_dict(gr_dict,f'Loss_inclust_std',gather_clust_info(clustloss[sc][anom].detach().cpu(),clusts[sc][anom],'std'))
            for ind,sil in enumerate(sils):
                gr_dict = self.update_dict(gr_dict,f'Sil{ind+1}',gather_clust_info(sil,clusts[sc])[anom])
            mean_inter = nonclustloss[sc][anom].detach().cpu()
            #mean_inter = gather_clust_info(nonclustloss[sc].detach().cpu(),clusts[sc])[anom]
            #mean_inter = (nonclustloss[gr_anom].detach().cpu()/anom_neg_counts).mean()
            if (anom_neg_counts > 10).nonzero()[0].shape[0] > 0 or (anom_pos_counts > 10).nonzero()[0].shape[0] > 0:
                raise('counted more than sampled')
            gr_dict = self.update_dict(gr_dict,f'Loss_outclust',mean_inter)
            gr_dict = self.update_dict(gr_dict,f'Loss_outclust_mean',gather_clust_info(nonclustloss[sc][anom].detach().cpu(),clusts[sc][anom],'mean'))
            #gr_dict = self.update_dict(gr_dict,f'Loss_outclust_mean',mean_inter.mean())
            #gr_dict = self.update_dict(gr_dict,f'Loss_outclust_std',mean_inter.std())
            gr_dict = self.update_dict(gr_dict,f'Loss_outclust_std',gather_clust_info(nonclustloss[sc][anom].detach().cpu(),clusts[sc][anom],'std'))
            #inter_res = res[sc][0][anom][:,anom].sigmoid().detach().cpu()
            #gr_dict = self.update_dict(gr_dict,f'Outclust_consistency_og1',get_clust_score(inter_res,anom,clusts,0,sc))
            #gr_dict = self.update_dict(gr_dict,f'Outclust_consistency_og2',get_clust_score(inter_res,anom,clusts,1,sc))
            #gr_dict = self.update_dict(gr_dict,f'Outclust_consistency_og3',get_clust_score(inter_res,anom,clusts,2,sc))

            for k,v in gr_dict.items():
                kname = k + f'_{sc}/Anom{anom_sc}_mean'
                self.tb.add_scalar(kname,np.array(v[0]).mean(), epoch)
                
                kname = k + f'_{sc}/Anom{anom_sc}_hist'
                self.tb.add_histogram(kname,np.array(v[0])[~np.isnan(np.array(v[0]))], epoch)
            mean_intras.append(gr_dict[f'Loss_inclust'][0].mean())
            mean_inters.append(gr_dict[f'Loss_outclust'][0].mean())
        print('done')
        return torch.tensor(mean_intras),torch.tensor(mean_inters),scores_all

def get_clust_score(inter_res,anom,clusts,clust_sc,sc):
    """
    Look at each clusters in sc that are disconnected in clust_sc. Get the avg embedding similarity for each node
    High: bad
    """
    
    idx = torch.triu(torch.minimum(clusts[sc][anom][...,np.newaxis]!=clusts[sc][anom],clusts[clust_sc][anom][...,np.newaxis]==clusts[clust_sc][anom])).nonzero()
    if idx.shape[0] == 0: return np.zeros(anom.shape)
    all_clusts = np.zeros(anom.shape)
    all_clusts[idx[:,0].unique()] += np.unique(idx[:,0],return_counts=True)[-1]
    all_clusts[idx[:,1].unique()] += np.unique(idx[:,1],return_counts=True)[-1]
    try:
        sum_scores = np.zeros(anom.shape)
        sum_scores[idx[:,0].unique()] += torch_scatter.scatter_add(inter_res[idx[:,0],idx[:,1]],idx[:,0])[idx[:,0].unique()].numpy()
        sum_scores[idx[:,1].unique()] += torch_scatter.scatter_add(inter_res[idx[:,0],idx[:,1]],idx[:,1])[idx[:,1].unique()].numpy()
        inter_res_val=(sum_scores/all_clusts)
    except Exception as e:
        import ipdb ; ipdb.set_trace()
        print(e)
    
    return np.nan_to_num(inter_res_val)

def gather_clust_info(mat,clust,reduce="mean"):
    if reduce=='std':
        mat = torch_scatter.scatter_std(mat,clust.to(mat.device))
    else:
        mat = torch.scatter_reduce(mat, 0, clust, reduce=reduce)
    return mat[clust]

def dgl_to_mat(g,device='cpu'):
    """Get sparse adjacency matrix from DGL graph"""
    src, dst = g.edges()
    block_adj = torch.sparse_coo_tensor(torch.stack((src,dst)),g.edata['w'].squeeze(-1),size=(g.number_of_nodes(),g.number_of_nodes()))
    return block_adj

def get_spectrum(mat,lapl=None,tag='',load=False,get_lapl=False,save_spectrum=True,n_eig=None):
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


    
        py_g.compute_fourier_basis(n_eigenvectors=n_eig)
        sio.savemat(f'{tag}.mat',{'e':scipy.sparse.csr_matrix(py_g.e),'U':scipy.sparse.csr_matrix(py_g.U)})
        U,e = torch.tensor(py_g.U).to(device),torch.tensor(py_g.e).to(device)
    else:
        e,U = scipy.linalg.eigh(np.array(lapl.to_dense(),order='F'),overwrite_a=True)
        e,U = torch.tensor(e).to(device),torch.tensor(U).to(device)
        
    return e, U

def process_graph(graph):
    """Obtain graph information from input TODO: MOVE?"""
    feats = graph[0].ndata['feature']
    return [torch.vstack((i.edges()[0],i.edges()[1])) for i in graph], feats
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
    scales = exp_params['MODEL']['SCALES']
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
    exp_name = exp_params['EXP']
    try:
        if 'weibo' not in exp_name or 'elliptic' not in exp_name:
            struct_model = torch.load(f'{exp_name}.pt')
            loaded=True
    except:
        struct_model = model.GraphReconstruction(feat_size, exp_params)
  
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