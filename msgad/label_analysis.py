import sknetwork
from sknetwork.hierarchy import postprocess, LouvainIteration
import numpy as np
import torch
from utils import *
import dgl
import networkx as nx
import sklearn
import os
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.interpolate import pchip
import scipy.io as sio
import functools
import matplotlib.patches as mpatches
import sys


class LabelAnalysis:
    def __init__(self,dataset,all_anoms,norms,exp):
        self.exp=exp
        self.visualize=False if dataset not in [] else True
        self.thresh = 0.8
        self.dataset = dataset
        self.anoms_combo = all_anoms
        self.fname = f'batch_data/labels'
        self.min_anom_size=0
        self.min_sil = 0.
        self.norms = norms
        if not os.path.exists(self.fname): os.makedirs(self.fname)

    def assign_nodes_to_clusters(self, node_scores, cluster_ids):
        """
        Assign nodes to contexts based on context-specific clusters with the highest
        average scores. Prioritizes larger clusters

        Input:
            node_scores : {array-like}, shape=[scales, n]
                Computed node rankings for each context
            cluster_ids : {array-like}, shape=[scales,n]
                Cluster IDs
        Output:
            assigned_contexts: {array-like, torch tensor}, shape=[scales,n]
                Context IDs for each node
        """
        # Step 1: Calculate average scores for each node across all contexts
        #average_scores = np.mean(node_scores, axis=1)
        cluster_scores = np.zeros((np.max(cluster_ids), node_scores.shape[1]))
        for context in range(node_scores.shape[1]):
            for cluster in range(np.max(cluster_ids)):
                mask = (cluster_ids[:, context] == cluster+1)
                cluster_scores[cluster, context] = np.mean(node_scores[mask, context])

        # Step 2: Find the cluster with the highest average score for each context
        #highest_scoring_clusters = np.argmax(cluster_scores, axis=0)

        # Step 3: Assign nodes to context-specific clusters
        num_nodes, num_contexts = node_scores.shape[0], node_scores.shape[1]
        assigned_contexts = np.full(cluster_ids.shape[0],-1)
        
        # prioritize larger scales
        #for context in range(num_contexts-1,-1,-1):
        for context in range(num_contexts):
            # Sort nodes by average scores in the current context
            #sorted_nodes = np.argsort(node_scores[:, context])[::-1]
            sorted_nodes = np.argsort(node_scores[:, context])
            # Assign nodes to the highest-scoring cluster, considering constraints
            for node in sorted_nodes:
                cluster = cluster_ids[node, context]

                # assigned to a different cluster, not unassigned
                if assigned_contexts[node] != context and assigned_contexts[node] != -1:
                    # Check if any other node in the same context-specific cluster is assigned
                    
                    # assigned to a different cluster ; skip
                    prev_context =  assigned_contexts[node]
                
                    prev_cluster = cluster_ids[node,prev_context]
                    prev_score = node_scores[node,prev_context]
                    # which context has the higher node score?
                    if node_scores[node,context] > prev_score:
                        #import ipdb ; ipdb.set_trace()
                        # all nodes in the previous cluster: set scores to -1
                        #node_scores[np.where(cluster_ids[:,prev_context]==prev_cluster),prev_context] = -1
                        assigned_contexts[np.where(cluster_ids[:,prev_context]==prev_cluster)] = -1
                    else:
                        continue
                
                if node_scores[node,context] != 0:
                #if node_scores[node,context] != 2:
                    assigned_contexts[np.where(cluster_ids[:,context]==cluster)] = context
                #assigned_clusters[node, context] = cluster

        return assigned_contexts

    def rank_anom_clustering(self,cluster,clust_ind,max_clust):
        """Score anomaly group with a given clustering"""
        rankings = np.full(self.anoms_combo.shape[0],0.)

        anom_clusters = np.unique(cluster[self.anoms_combo])
        # silhouette scores, normalized between 0 and 1
        sil_samps_norm = (self.sil_samps[clust_ind] - self.sil_samps[clust_ind].min()) / (self.sil_samps[clust_ind].max() - self.sil_samps[clust_ind].min())
        
        for anom_clust in anom_clusters:
            # avg silhouette score in cluster
            rankings[np.where(cluster[self.anoms_combo] == anom_clust)[0]] = sil_samps_norm[np.where(cluster[self.anoms_combo] == anom_clust)[0]].mean()
            # percentage of anoms in cluster
            rankings[np.where(cluster[self.anoms_combo] == anom_clust)[0]] += np.where(cluster[self.anoms_combo] == anom_clust)[0].shape[0]/np.where(cluster == anom_clust)[0].shape[0]
            # size
            #rankings[np.where(cluster[self.anoms_combo] == anom_clust)[0]] += (np.where(cluster[self.anoms_combo] == anom_clust)[0].shape[0])/max_clust
            rankings[np.where(cluster[self.anoms_combo] == anom_clust)[0]] *=  (clust_ind+1)/(self.scales+1)

            # threshold on min anoms and min % of anoms
            if np.where(cluster[self.anoms_combo] == anom_clust)[0].shape[0] < self.min_anom_size or np.where(cluster[self.anoms_combo] == anom_clust)[0].shape[0]/np.where(cluster == anom_clust)[0].shape[0] < self.thresh:
                rankings[np.where(cluster[self.anoms_combo] == anom_clust)[0]] = 0

        return rankings

    def postprocess_anoms(self,clusts):
        """For an anomaly group, find cluster it is best associated with. Prioritizes larger clusters """

        anom_clusts,best_rankings = np.full(self.anoms_combo.shape,-1),np.full(self.anoms_combo.shape,-1)
        max_clust = np.array([np.unique(i,return_counts=True)[-1].max() for i in clusts])
        for clust_ind,clust in enumerate(clusts):
            clust_rankings = self.rank_anom_clustering(clust,clust_ind,max_clust[clust_ind])
            all_rankings = clust_rankings[np.newaxis,...] if clust_ind == 0 else np.vstack((all_rankings,clust_rankings[np.newaxis,...]))
            anom_clusts[np.where(clust_rankings>best_rankings)] = clust_ind
            best_rankings = np.maximum(clust_rankings,best_rankings)

        # Assign labels using group_ids and max_context_indices
        anom_clusts = self.assign_nodes_to_clusters(all_rankings.T,np.stack(clusts).T[self.anoms_combo])
        #print('before thresholding',np.unique(anom_clusts,return_counts=True)[-1])
        anom_clusts += 1
        anom_clusts = anom_clusts.astype(int)
        #anom_clusts[np.where(best_rankings < self.ranking_thresh)] = 0
                
        return anom_clusts,all_rankings

    def plot_dend(self,dend,colors,legend_colors,**kwargs):
        ax = plt.figure(figsize=(15, 10))
        ax.set_facecolor('white')
        dname = self.dataset.capitalize()
        plt.title(f'Hierarchical clustering for {dname}',fontsize='xx-large')
        with plt.rc_context({'lines.linewidth': 0.5}):
            rdict = hierarchy.dendrogram(dend,link_color_func=lambda k: colors[k],no_labels=True)
        
        legend_patches = []
        for i,col in enumerate(np.array(legend_colors[:(np.unique(colors).shape[0]-1)])):
            if i == 0: legend_patches.append(mpatches.Patch(color=col, label=f'Single node anom.'))
            else: legend_patches.append(mpatches.Patch(color=col, label=f'Scale {i} anom.'))

        plt.legend(handles=legend_patches,fontsize='x-large')
        
        #plt.legend([f'Scale {i}' for i,col in enumerate(np.flip(np.array(legend_colors[:(np.unique(colors).shape[0]-1)])))],fontsize='x-large')
        fpath = self.generate_fpath(f'preprocess_vis/{self.dataset}/{self.exp}')
        import ipdb ; ipdb.set_trace()
        plt.savefig(f'{fpath}/dend.png')
        
    def get_clusts(self,graph,scales):
        if self.dataset in 'weibo': resolutions = [.8]
        elif self.dataset in ['yelpchi_rtr']: resolutions = [.7]
        elif self.dataset in ['tfinance','elliptic']: resolutions = [.6]
        else:
            resolutions = np.arange(.5,1.1,.1)
        adj = np.array(nx.adjacency_matrix(graph,weight='weight').todense())
        for res in resolutions:
            hierarchy = LouvainIteration(resolution=res,depth=scales)
            dend = hierarchy.fit_predict(adj)
            if len(resolutions) > 1:
                print('dasgupta',res,sknetwork.hierarchy.dasgupta_score(adj, dend))
                print('tree',res,sknetwork.hierarchy.tree_sampling_divergence(adj, dend))
            clusts = [postprocess.cut_straight(dend,threshold=scale) for scale in range(scales+1)]
        return clusts, dend

    def get_group_stats(self,graph,clusts,group,norms,sc,norm=False):
        unique_clusts = np.unique((clusts[group]))
        group_dict = {'avg_sil':[]}
        for ind,j in enumerate(unique_clusts):
            if ind == 0:
                group_dict['avg_density'] = []
                group_dict['avg_size'] = []
                group_dict['avg_sp'] = []
                group_dict['anom_feat_dist'] = []
                group_dict['norm_feat_dist'] = []
            clust_anoms = np.where(clusts == j)[0]
            anom_subgraph = graph.subgraph(clust_anoms)
            group_dict['avg_density'].append(nx.density(anom_subgraph))
            group_dict['avg_size'].append(anom_subgraph.number_of_nodes())
            try:
                group_dict['avg_sp'].append(nx.average_shortest_path_length(anom_subgraph))
            except Exception as e:
                pass
            group_dict['avg_sil'].append(self.sil_samps[sc][clust_anoms].mean())
            # avg euclidean distance between features
            group_dict['anom_feat_dist'].append(sklearn.metrics.pairwise.euclidean_distances(graph.nodes[0]['feats'][clust_anoms]).mean().detach().cpu())
            if not norm:
                group_dict['norm_feat_dist'].append(sklearn.metrics.pairwise.euclidean_distances(graph.nodes[0]['feats'][clust_anoms],graph.nodes[0]['feats'][norms]).mean().detach().cpu())
        for key in list(group_dict.keys()):
            group_dict[key] = group_dict[key].mean()
        return group_dict

    def find_node_id(self, dendrogram, original_n_nodes, target_original_node_id):
        """Get dendrogram node ids corresponding to node id; used for coloring dendrogram"""
        # Initialize a mapping to keep track of node IDs
        node_id_mapping = {i: i for i in range(original_n_nodes)}
        max_val = original_n_nodes-1
        # Traverse the dendrogram and update the node ID mapping
        max_lvls=[original_n_nodes]
        for merge in dendrogram:
            # Get the IDs of the merged clusters
            left_node_id, right_node_id = int(merge[0]), int(merge[1])
    
            if left_node_id > max_lvls[-1] and right_node_id > max_lvls[-1]:
                max_lvls.append(new_node_id)
                
            new_node_id = max_val+1
            # Update the node ID mapping for the merged clusters
            
            node_id_mapping[left_node_id] = new_node_id
            node_id_mapping[right_node_id] = new_node_id
            max_val = np.maximum(max_val,new_node_id)
        #return node_id_mapping[target_original_node_id]

        dend_ids = [target_original_node_id]
        prev_dend = dend_ids[-1]
        
        try:
            cur_lvl = 0
            while True:
                if dend_ids[-1] == node_id_mapping[prev_dend]:
                    print('found equal')
                    return dend_ids[:-1]
                if node_id_mapping[prev_dend] >= max_lvls[cur_lvl]:
                    dend_ids.append(node_id_mapping[prev_dend])
                    cur_lvl += 1
                prev_dend = node_id_mapping[prev_dend]                
        except:
            return dend_ids[:-1]

    def check_number_of_labels(self,n_labels, n_samples):
        """Check that number of labels are valid.

        Parameters
        ----------
        n_labels : int
            Number of labels.

        n_samples : int
            Number of samples.
        """
        if not 1 < n_labels < n_samples:
            raise ValueError(
                "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)"
                % n_labels
            )
        
    def _silhouette_reduce(self,D_chunk, start, labels, label_freqs):
        """Accumulate silhouette statistics for vertical chunk of X.

        Parameters
        ----------
        D_chunk : {array-like, sparse matrix} of shape (n_chunk_samples, n_samples)
            Precomputed distances for a chunk. If a sparse matrix is provided,
            only CSR format is accepted.
        start : int
            First index in the chunk.
        labels : array-like of shape (n_samples,)
            Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.
        label_freqs : array-like
            Distribution of cluster labels in ``labels``.
        """
        n_chunk_samples = D_chunk.shape[0]
        # accumulate distances from each sample to each cluster
        cluster_distances = np.zeros(
            (n_chunk_samples, len(label_freqs)), dtype=D_chunk.dtype
        )

        if scipy.sparse.issparse(D_chunk):
            if D_chunk.format != "csr":
                raise TypeError(
                    "Expected CSR matrix. Please pass sparse matrix in CSR format."
                )
            for i in range(n_chunk_samples):
                indptr = D_chunk.indptr
                indices = D_chunk.indices[indptr[i] : indptr[i + 1]]
                sample_weights = D_chunk.data[indptr[i] : indptr[i + 1]]
                sample_labels = np.take(labels, indices)
                cluster_distances[i] += np.bincount(
                    sample_labels, weights=sample_weights, minlength=len(label_freqs)
                )
        else:
            for i in range(n_chunk_samples):
                sample_weights = D_chunk[i]
                sample_labels = labels
                cluster_distances[i] += np.bincount(
                    sample_labels, weights=sample_weights, minlength=len(label_freqs)
                )

        # intra_index selects intra-cluster distances within cluster_distances
        end = start + n_chunk_samples
        intra_index = (np.arange(n_chunk_samples), labels[start:end])
        # intra_cluster_distances are averaged over cluster size outside this function
        intra_cluster_distances = cluster_distances[intra_index]
        # of the remaining distances we normalise and extract the minimum
        cluster_distances[intra_index] = np.inf
        cluster_distances /= label_freqs
        inter_cluster_distances = cluster_distances.min(axis=1)
        return intra_cluster_distances, inter_cluster_distances

    def sil_samples(self,X,labels,metric,**kwds):
        X, labels = sklearn.utils.check_X_y(X, labels, accept_sparse=["csr"])
        le = sklearn.preprocessing.LabelEncoder()
        labels = le.fit_transform(labels)
        n_samples = len(labels)
        label_freqs = np.bincount(labels)
        self.check_number_of_labels(len(le.classes_), n_samples)

        kwds["metric"] = metric
        reduce_func = functools.partial(
            self._silhouette_reduce, labels=labels, label_freqs=label_freqs
        )
        results = zip(*sklearn.metrics.pairwise_distances_chunked(X, reduce_func=reduce_func, **kwds))
        intra_clust_dists, inter_clust_dists = results
        intra_clust_dists = np.concatenate(intra_clust_dists)
        inter_clust_dists = np.concatenate(inter_clust_dists)

        denom = (label_freqs - 1).take(labels, mode="clip")
        with np.errstate(divide="ignore", invalid="ignore"):
            intra_clust_dists /= denom

        sil_samples = inter_clust_dists - intra_clust_dists
        with np.errstate(divide="ignore", invalid="ignore"):
            sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
        # nan values are for clusters of size 1, and should be 0
        return np.nan_to_num(sil_samples)

    def run_dend(self,graph,scales,load=False):
        """Partition the graph into multi-scale clusters and anomalies & visualize multi-scale spatial/spectral behavior"""
        sys.setrecursionlimit(1000000) 
        self.graph = graph
        self.scales = scales
        dend = None
        if load or os.path.exists(f'{self.fname}/{self.dataset}_labels_{scales}_exp{self.exp}.mat'):
            with open(f'{self.fname}/{self.dataset}_labels_{scales}_exp{self.exp}.mat','rb') as fin:
                mat = pkl.load(fin)
            sc_all,clusts = mat['labels'],mat['clusts']
        else:
            #if True:
            clusts,dend = self.get_clusts(graph,self.scales+1)
            if 'yelpchi' not in self.dataset:
                clusts_ = clusts[1:-1]
            else:
                clusts_ = clusts[2:]
            dist = nx.adjacency_matrix(graph,weight='weight').astype(np.float).todense() ; dist = 1-dist
            np.fill_diagonal(dist,0)

            self.sil_samps = [self.sil_samples(dist,clust,metric='precomputed') for clust in clusts_]
            sc_all,all_rankings = self.postprocess_anoms(clusts_)
            
            # prints connectivity info of every anomaly and all clusters associated with it across scales
            for clust_ind,clust in enumerate(clusts_):
                for ind,sc in enumerate(np.unique(sc_all)):
                    anom_clusts = clust[self.anoms_combo[np.where(sc_all==sc)]]
                    if len(anom_clusts) == 0: continue
                    group_clusts = np.unique(anom_clusts)
                    
                    all_ = []
                    for gc in group_clusts:
                        all_.append(round(np.where(anom_clusts==gc)[0].shape[0]/np.where(clust==gc)[0].shape[0],3))
                    print('avg. anom composition for clust',clust_ind,'anom',ind,np.array(all_).mean())
            
            print('anomalous clusters found',[np.unique((clusts_[i-1][self.anoms_combo[np.where(sc_all==i)]])).shape[0] for i in range(np.unique(sc_all).shape[0])])
            if 'yelpchi' not in self.dataset:
                clusts = clusts[1:-1]
            else:
                clusts = clusts[2:]

        if self.visualize:
            sc_all_unique = np.unique(sc_all)
            clusts,dend = self.get_clusts(graph,self.scales+1)
            if dend is not None:
                colors = np.full(int(dend.max()+2),'tab:gray',dtype='object')
                sc_colors=['tab:red','tab:orange','tab:pink','tab:purple','tab:green','tab:brown','tab:cyan','tab:olive']
                for i in sc_all_unique:
                    for sc_node in self.anoms_combo[(sc_all==i).nonzero()]:
                        node_ids = np.array(self.find_node_id(dend[:,:2],dend.shape[0]+1,sc_node))
                        colors[node_ids.astype(int)] = sc_colors[i]

                self.plot_dend(dend,colors,sc_colors)
            # add opacity based on silhouette score/overral ranking
            sc_colors=['tab:gray','tab:red','tab:orange','tab:pink','tab:purple','tab:green','tab:brown','tab:cyan','tab:olive']
            
            # plot spectrum
            legend_arr = ['No anom. signal','Single node anom.']
            for i in range(1,len(sc_all_unique)):
                legend_arr.append(f'Anom. scale {i}')
            plt.figure()
            self.plot_spectrum_graph(graph,[],'norm',graph.nodes[0]['feats'],legend_arr,color=sc_colors[0])
            for ind,i in enumerate(sc_all_unique):
                group = self.anoms_combo[(sc_all==i).nonzero()]
                self.plot_spectrum_graph(graph,group,i,graph.nodes[0]['feats'],legend_arr,color=sc_colors[ind+1])

            clust_dicts = []
            sc_all_clusts = []
            for j in np.unique(sc_all):
                sc_all_clusts.append(self.anoms_combo[np.where(sc_all==j)])
            sc_all_clusts.append(self.norms)

            for i in range(len(clusts)):
                print('scale',i)
                clust_dicts.append({})
                norm_clusts,norm_clust_counts = np.unique(clusts[i][sc_all_clusts[-1]],return_counts=True)
                for jind,j in enumerate(sc_all_clusts):
                    if jind-1 != i and jind != len(sc_all_clusts)-1: continue

                    if jind == len(sc_all_clusts)-1:
                        gr_dict = self.get_group_stats(graph,clusts[i],sc_all_clusts[-1][np.where(np.in1d(clusts[i][sc_all_clusts[-1]],np.random.choice(norm_clusts[np.where(norm_clust_counts>self.min_anom_size)],size=np.unique(clusts[i][np.array(j)]).shape[0])))[0]],j,i,norm=True)
                    gr_dict = self.get_group_stats(graph,clusts[i],j,sc_all_clusts[-1][np.where(np.in1d(clusts[i][sc_all_clusts[-1]],np.random.choice(norm_clusts[np.where(norm_clust_counts>self.min_anom_size)],size=np.unique(clusts[i][np.array(j)]).shape[0])))[0]],i)
                    
                    print('group',jind,gr_dict)
                    if jind == 0:
                        clust_dicts[i] = gr_dict
            sc_all = list(sc_all)
        with open(f'{self.fname}/{self.dataset}_labels_{scales}_exp{self.exp}.mat','wb') as fout:
            pkl.dump({'labels':sc_all,'clusts':clusts},fout)
        return list(sc_all),torch.tensor(clusts)

    def plot_spectrum(self,e,U,signal,color=None):
        c = U.T@signal
        M = torch.zeros((25+1,c.shape[1])).to(torch.tensor(U).dtype)#.to(e.device).to(U.dtype)
        for j in range(c.shape[0]):
            idx = max(min(int(e[j] / 0.05), 25-1),0)
            M[idx] += c[j]**2
        M=M/sum(M)
        M[torch.where(torch.isnan(M))]=0
        y = torch.mean(M,axis=1)*100
        x = np.arange(y.shape[0])
        spline = pchip(x, y)
        X_ = np.linspace(x.min(), x.max(), 801)
        Y_ = spline(X_)
        if color:
            plt.plot(X_,Y_,color=color)
        else:
            plt.plot(X_,Y_)
        return X_,y
    
    def plot_spectrum_graph(self,graph,anoms,img_lbl,feats,legend_arr,color=None):
        lbl = nx.adjacency_matrix(graph).astype(np.float64)
        
        py_g = graphs.MultiScale(lbl)
        if os.path.exists(f'{self.dataset}_eig.mat'):
            mat = sio.loadmat(f'{self.dataset}_eig.mat')
            e,U = mat['e'][0],mat['U']
        else:
            try:
                py_g.compute_laplacian('normalized')
                py_g.compute_fourier_basis()
                
                e,U = py_g.e,py_g.U
                mat = {} ; mat['e'],mat['U'] = e.to_sparse(),U.to_sparse()
                sio.savemat(f'{self.dataset}_eig.mat',mat)
            except Exception as e_:
                print(e_)
                import ipdb ; ipdb.set_trace()
        
        signal = np.random.randn(feats.shape[0],feats.shape[0])+1

        signal[anoms]=(np.random.randn(U.shape[0])*400*self.anoms_combo.shape[0]/len(anoms))+1#*anom_tot/anom.shape[0])+1# NOTE: tried 10#*(anom_tot/anom.shape[0]))
        x,y=self.plot_spectrum(e,U,signal,color)
        plt.legend(legend_arr,fontsize='x-small')
        plt.xticks(np.arange(0,y.shape[0],step=5),np.round(np.arange(y.shape[0],step=5)*0.05,2))
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        plt.xlabel(r'$\lambda$')
        d_name = self.dataset.split("_")[0].capitalize()
        plt.title(f'Spectral energy distribution of {d_name.capitalize()}')#, label {img_lbl}')
        fpath = self.generate_fpath(f'preprocess_vis/{self.dataset}/{self.exp}')
        plt.savefig(f'{fpath}/spectra.png')

    def generate_fpath(self,fpath):
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        return fpath
