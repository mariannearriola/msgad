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
import sklearn
import matplotlib.cm as cm
import matplotlib.pyplot as plt

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
    def __init__(self,tb,edge_ids, attract_edges_sel, repel_edges_sel,sc_label,clust,anoms,exp_params):
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

        self.tb = tb
        '''
        self.clust = clust
        self.sc_labels = sc_label
        self.anoms = anoms
        self.anoms_cuda = torch.tensor(anoms).cuda()
        self.model_ind = exp_params['MODEL']['IND']

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
            return dgl_g.out_edges(i,form='eid')
        else:
            anom_clusts = clust[i].unique()
            return dgl_g.out_edges(np.intersect1d(clust,anom_clusts,return_indices=True)[-2],form='eid')

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

    def tb_write_anom(self,tb,adj,sc_label,edge_ids,pred,loss,sc,epoch,regloss,clustloss,nonclustloss,clusts,sc_idx_inside,sc_idx_outside,entropies):
        """Log loss evolution for anomaly group"""
        # sc idx all: contains all edge idx associated with each anomaly
        # removing edges if there is no cluster between them, does not add edges
        if self.model_ind != None:
            clust = clusts[self.model_ind]
        else:
            clust = clusts[sc]
        pred_opt = torch.zeros(pred.shape)
    
        #a1 = torch_overlap(torch.where(clust[edge_ids[:,0]]==clust[edge_ids[:,1]])[0],torch.where(clust[edge_ids[:,1]]==clust[edge_ids[:,0]])[0])
        a1 = torch.where(clust[edge_ids[:,0]]==clust[edge_ids[:,1]])[0]
        pred_opt[a1] = 1.

        anom_clusts = copy.deepcopy(clust)
        it = clust.max()+1
        mean_intras,mean_inters=[],[]
        all_dicts = []
        sc_labels = np.unique(sc_label)
        for ind,i in enumerate(sc_labels):
            anom = torch.tensor(self.anoms[np.where(np.array(sc_label)==i)])#.to(edge_ids.device)

            
            #import ipdb; ipdb.set_trace()
            #anom_neighbors=torch.stack(adj.out_edges(anom)).T
            #anom_neighbors=anom_neighbors[(~torch.isin(clust[anom_neighbors].unique(),clust[anom])).nonzero()]

            # gets all edges related to a group
            
            gr_clusts = np.unique(clust[anom])
            # GET CLUSTER-WISE VALUES, then report mean across clusters
            gr_dict = {'Loss':[],'RegLoss':[],'Opt_conn':[],'Loss_inclust':[],'Loss_inclust_opt':[],'Loss_outclust':[],'Loss_outclust_opt':[],'Silhouette':[],'Inclust_edges':[],'Outclust_edges':[]}
            gr_anom_sizes,gr_sizes= [],[]
            for clust_ind,gr_clust in enumerate(gr_clusts):
                #import ipdb ; ipdb.set_trace()
                #gr_anom = np.intersect1d(np.where(clust==gr_clust)[0],anom)
                gr_anom = np.where(clust==gr_clust)[0]
                gr_anom_sizes.append(gr_anom.shape[0])
                gr_sizes.append(np.where(clust==gr_clust)[0].shape[0])

            
                #sc_idx_inside_ov = self.get_group_idx(edge_ids,clust,gr_anom,anom_wise=True)
                '''
                sc_idx_aa = sc_idx_inside_ov[torch.isin(adj.out_edges(gr_anom)[1],self.anoms_cuda).nonzero()[:,0]]
                sc_idx_aa = sc_idx_aa[torch.isin(sc_idx_aa,torch.cat((sc_idx_inside[sc],sc_idx_outside[sc])).cuda())]

                # anom-anom
                mean_intra = clustloss[sc_idx_aa][pred_opt[sc_idx_aa].nonzero()].mean().detach().cpu()
                gr_dict['Loss_inclust_aa'].append(mean_intra)
                
                sc_idx_an = sc_idx_inside_ov[torch.isin(adj.out_edges(gr_anom)[1],self.anoms_cuda)==0]
                sc_idx_an = sc_idx_an[torch.isin(sc_idx_an,torch.cat((sc_idx_inside[sc],sc_idx_outside[sc])).cuda())]
                # anom-normal
                mean_intra = clustloss[sc_idx_an][pred_opt[sc_idx_an].nonzero()].mean().detach().cpu()
                gr_dict['Loss_inclust_an'].append(mean_intra)
                
                '''
                # won't be perfectly balanced, depends on anomaly group
                #sc_idx_inside_ov = sc_idx_inside_ov[torch.isin(sc_idx_inside_ov,torch.cat((sc_idx_inside[sc],sc_idx_outside[sc])).cuda())]
                
                #samp_edges = self.sampled_edges(pred_opt[sc_idx_inside_ov])
                anom_sc = ind if ind != len(self.sc_labels)-1 else 'norm'
                '''
                if sc == 2:
                    print('clusters',clust.unique().shape[0])
                    self.plot_sep_scores_group(self.opt_entropies,self.avg_opt_entropies,i,anom_sc)
                    #self.plot_sep_scores(self.opt_entropies,self.avg_opt_entropies,clust,anom_sc)
                    #import ipdb  ; ipdb.set_trace()
                '''

                #gr_dict['Loss'].append(loss[sc_idx_inside_ov].mean().detach().cpu())
                gr_dict['Loss'].append(loss[gr_anom].mean().detach().cpu())
                #expanded_average = loss[sc_idx_inside_ov][samp_edges].mean()

                # log regularization loss
                #expanded_average=regloss[sc_idx_inside_ov][samp_edges].mean()
                #gr_dict['RegLoss'].append(regloss[sc_idx_inside_ov].mean().detach().cpu())
                gr_dict['RegLoss'].append(regloss[gr_anom].mean().detach().cpu())

                # optimal ; best 
                #gr_dict['Opt_conn'].append(pred_opt[sc_idx_inside_ov].mean())

                # only select INSIDE cluster
                #expanded_average = clustloss[sc_idx_inside_ov][samp_edges][pred_opt[sc_idx_inside_ov][samp_edges].nonzero()].mean()
                #mean_intra = clustloss[sc_idx_inside_ov][pred_opt[sc_idx_inside_ov].nonzero()].mean().detach().cpu()
                mean_intra = clustloss[gr_anom].mean().detach().cpu()
                gr_dict['Loss_inclust'].append(mean_intra)
                
                #gr_dict['Inclust_edges'].append(clustloss[sc_idx_inside_ov][pred_opt[sc_idx_inside_ov].nonzero()].shape[0])
                #import ipdb ; ipdb.set_trace()

                #gr_dict['Loss_inclust_opt'].append(pred_opt[sc_idx_inside_ov][pred_opt[sc_idx_inside_ov].nonzero()].mean().detach().cpu())
                #expanded_average = pred_opt[sc_idx_inside_ov][samp_edges][pred_opt[sc_idx_inside_ov][samp_edges].nonzero()].mean()

                # only select OUTSIDE cluster
                #expanded_average = nonclustloss[sc_idx_inside_ov][samp_edges][torch.where(pred_opt[sc_idx_inside_ov][samp_edges]==0)[0]].mean()
                
                #mean_inter = nonclustloss[sc_idx_inside_ov][torch.where(pred_opt[sc_idx_inside_ov]==0)[0]].mean().detach().cpu()
                mean_inter = nonclustloss[gr_anom].mean().detach().cpu()
                gr_dict['Loss_outclust'].append(mean_inter)
                #gr_dict['Outclust_edges'].append(nonclustloss[sc_idx_inside_ov][torch.where(pred_opt[sc_idx_inside_ov]==0)[0]].shape[0])

                gr_dict['Silhouette'].append((mean_inter-mean_intra)/torch.max(mean_intra,mean_inter))

                #gr_dict['Loss_outclust_opt'].append(pred_opt[sc_idx_inside_ov][torch.where(pred_opt[sc_idx_inside_ov]==0)[0]].mean())
                #expanded_average = pred_opt[sc_idx_inside_ov][samp_edges][torch.where(pred_opt[sc_idx_inside_ov][samp_edges]==0)[0]].mean()
            #print('avg size of anom group',ind,(np.array(gr_anom_sizes)).mean(),len(gr_anom_sizes))

            for k,v in gr_dict.items():
                kname = k + f'_{sc}/Anom{anom_sc}'
                '''
                if 'Loss' in k and 'clust' in k:
                    if 'inclust' in k:
                        arr =np.array(gr_dict['Loss_inclust'])*np.array(gr_dict['Inclust_edges'])
                    elif 'outclust' in k:
                        arr =np.array(gr_dict['Loss_outclust'])*np.array(gr_dict['Outclust_edges'])
                    gr_dict[k] = arr[~np.isnan(arr)].mean()
                else:
                '''
                gr_dict[k] = np.array(v)[~np.isnan(np.array(v))].mean()
                tb.add_scalar(kname,np.array(v)[~np.isnan(np.array(v))].mean(), epoch)
            mean_intras.append(gr_dict['Loss_inclust'])
            mean_inters.append(gr_dict['Loss_outclust'])
            '''
            clust_counts=torch.unique(clust,return_counts=True)[-1]
            anom_clusts,anom_counts=torch.unique(clust[sc_label[ind]],return_counts=True)
            if i == 'norm':
                # > 95% normal nodes
                clusts = anom_clusts[torch.where(anom_counts/clust_counts[anom_clusts] > 0.95)]
                group_ids = np.intersect1d(clust[sc_label[ind]],clusts,return_indices=True)[1]
            else:
                group_ids=sc_label[ind]
            '''
            # node entropy
            #sc_entropy = entropies[sc][i].mean()
            #tb.add_scalar(f'Node_entropy{sc}/Anom{anom_sc}', sc_entropy, epoch)
            
            #sc_entropy = opt_entropy[group_ids].mean()
            #tb.add_scalar(f'Node_opt_entropy{sc}/Anom{anom_sc}', sc_entropy, epoch)

            #sc_entropy = nx.density(nx.from_numpy_matrix(adjacency_matrix[i][:,i]))
            #tb.add_scalar(f'Node_opt_density{sc}/Anom{anom_sc}', sc_entropy, epoch)


            # clust entropy
            #anom_clusts = clust[sc_label[ind]].unique()
            #clust_entropy = np.vectorize(clust_entropies[sc].get)(anom_clusts)
            #clust_entropy = clust_entropy[clust_entropy.nonzero()].mean()
            #clust_entropy = entropies[sc][group_ids].mean()
            #tb.add_scalar(f'Clust_entropy{sc}/Anom{anom_sc}', clust_entropy, epoch)
            '''
            # only select INSIDE cluster, outside OF ANOM
            expanded_average = collect_clust_loss(edge_ids,self.sc_idx_inside_outside_anom,self.sc_idx_inside_outside_anom_,clustloss)
            tb.add_scalar(f'Inclust_outanom{sc}_Anom{anom_sc}', expanded_average, epoch)
            
            # only select edges between anom and other nodes not in anom group, in same cluster

            # only select edges OUTSIDE of cluser, INSIDE anom
            expanded_average = collect_clust_loss(edge_ids,self.sc_idx_outside_inside_anom,self.sc_idx_outside_inside_anom_,nonclustloss)
            tb.add_scalar(f'Outclust_inanom{sc}_Anom{anom_sc}', expanded_average, epoch)
            '''
        print('done')
        return torch.tensor(mean_intras),torch.tensor(mean_inters)


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