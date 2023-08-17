import sknetwork
from sknetwork.hierarchy import postprocess, LouvainIteration
from sknetwork.visualization import svg_dendrogram
from IPython.display import SVG
import numpy as np
import torch
from utils import *
import dgl
import networkx as nx
from itertools import chain
import sklearn
import os
import matplotlib.pyplot as plt
from cairosvg import svg2png
from scipy.cluster import hierarchy
from scipy.interpolate import make_interp_spline
from scipy.interpolate import pchip
import scipy.io as sio

class LabelAnalysis:
    def __init__(self,dataset,all_anoms,norms,exp):
        self.exp=exp
        self.visualize=True
        self.thresh = 0.8
        self.dataset = dataset
        self.anoms_combo = all_anoms
        self.fname = f'batch_data/labels'
        self.ranking_thresh = 0.8
        self.min_anom_size=3
        self.min_sil = 0.
        self.norms = norms
        if not os.path.exists(self.fname): os.makedirs(self.fname)

    def check_conn(self,sc_label):
        """Filter out disconnected anomaly clusters"""
        conn_check=[]
        for i in sc_label:
            try:
                conn_check.append(nx.is_connected(self.graph.subgraph(i)))
            except:
                conn_check.append(False)
        return conn_check

    def remove_anom_overlap(self,clusts):
        """For an anomaly group, find cluster it is best associated with. Prioritizes larger clusters """

        anom_clusts,best_rankings = np.full(self.anoms_combo.shape,-1),np.full(self.anoms_combo.shape,-1)
        #import ipdb ; ipdb.set_trace()
        max_clust = np.array([np.unique(i,return_counts=True)[-1].max() for i in clusts]).max()
        for clust_ind,clust in enumerate(clusts):
            clust_rankings = self.rank_anom_clustering(clust,clust_ind,max_clust)
            all_rankings = clust_rankings[np.newaxis,...] if clust_ind == 0 else np.vstack((all_rankings,clust_rankings[np.newaxis,...]))
            anom_clusts[np.where(clust_rankings>best_rankings)] = clust_ind
            best_rankings = np.maximum(clust_rankings,best_rankings)
        
       

        # Assign labels using group_ids and max_context_indices
        anom_clusts = self.assign_nodes_to_clusters(all_rankings.T,np.stack(clusts).T[self.anoms_combo])
        
        #anom_clusts = np.argmax(all_rankings,axis=0)

        
        #anom_clusts = self.fclust_scales[self.anoms_combo]-1
        print('before thresholding',np.unique(anom_clusts,return_counts=True)[-1])
        
        anom_clusts += 1
        """
        # NOTE: only get largest cluster from each anomaly (debugging)
        anom_clusters =  np.stack(clusts)[:,self.anoms_combo][anom_clusts-1,np.arange(anom_clusts.shape[0])]
        idx_keep = []
        #import ipdb ; ipdb.set_trace()
        for i in range(self.scales-1):
            sc_cluster = anom_clusters[(anom_clusts==(i+1)).nonzero()[0]]
            #els,cts = np.unique(sc_cluster,return_counts=True)
            if sc_cluster.shape[0] == 0: continue
            sc_rankings = all_rankings[i][anom_clusts==(i+1)]
            els = np.unique(sc_rankings)
            try:
                idx_keep.append((all_rankings[i]==els.max()).nonzero()[0])
            except Exception as e:
                import ipdb ; ipdb.set_trace()
                print(e)
            #idx_keep.append((anom_clusters==els[np.argmax(cts)]).nonzero()[0])
            #idx_keep.append((anom_clusters==els[np.argmax(cts)]).nonzero()[0])
            #idx_drop = (anom_clusters!=els[np.argmax(cts)]).nonzero()[0] if idx_drop is None else np.concatenate((idx_drop,(anom_clusters!=els[np.argmax(cts)]).nonzero()[0]))
        anom_clusts = np.zeros(anom_clusts.shape)
        for id,idx in enumerate(idx_keep):
            anom_clusts[idx] = id+1
        
        #import ipdb ; ipdb.set_trace()
        #els,cts = np.unique(idx_drop,return_counts=True)
        #anom_clusts_nz[els[cts==(self.scales-1)]]=0
        #anom_clusts[els[cts==(self.scales-1)]]=0
        clusts_kept = anom_clusters[anom_clusts.nonzero()]
        """
        anom_clusts = anom_clusts.astype(int)
        
        #anom_clusts[all_rankings[(anom_clusts),np.arange(all_rankings.shape[1])] == 0] = -1
        #anom_clusts += 1
        #anom_clusts[np.where(best_rankings < self.ranking_thresh)] = 0
        
        print('after thresholding',np.unique(anom_clusts,return_counts=True)[-1])
        
        return anom_clusts,all_rankings

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

    def getAnomCount(self,clust,all_anoms):
        """Retrieve all cluster information associated with anomalies"""
        clust_keys = np.unique(clust)
        clust_dict = {}
        anom_count = []
        node_count = []
        graph_nodes = list(self.graph.nodes())
        for key in clust_keys:
            #clust_dict[key] = np.where(clust==key)[0]
            clust_dict[key] = np.array([graph_nodes[i] for i in np.where(clust==key)[0]])
            anom_count.append(np.intersect1d(all_anoms,clust_dict[key]).shape[0])
            node_count.append(clust_dict[key].shape[0])
        return clust_dict,np.array(anom_count),np.array(node_count)
    
    def rank_anom_clustering(self,cluster,clust_ind,max_clust):
        """Score anomaly group with a given clustering"""
        rankings = np.full(self.anoms_combo.shape[0],0.)

        anom_clusters = np.unique(cluster[self.anoms_combo])
        # silhouette scores, normalized between 0 and 1
        sil_samps_norm = (self.sil_samps[clust_ind] - self.sil_samps[clust_ind].min()) / (self.sil_samps[clust_ind].max() - self.sil_samps[clust_ind].min())
        
        for anom_clust in anom_clusters:
            # avg silhouette score in cluster
            rankings[np.where(cluster[self.anoms_combo] == anom_clust)[0]] = 1 * sil_samps_norm[np.where(cluster[self.anoms_combo] == anom_clust)[0]].mean()
            # percentage of anoms in cluster
            rankings[np.where(cluster[self.anoms_combo] == anom_clust)[0]] += np.where(cluster[self.anoms_combo] == anom_clust)[0].shape[0]/np.where(cluster == anom_clust)[0].shape[0]
            #rankings[np.where(cluster[self.anoms_combo] == anom_clust)[0]] = 1 if np.where(cluster[self.anoms_combo] == anom_clust)[0].shape[0]/np.where(cluster == anom_clust)[0].shape[0]>0.8 else 0
            # size
            #rankings[np.where(cluster[self.anoms_combo] == anom_clust)[0]] += 0.5*(np.where(cluster[self.anoms_combo] == anom_clust)[0].shape[0])/max_clust
            #if np.unique(self.incons_scs[clust_ind][np.where(cluster[self.anoms_combo] == anom_clust)[0]]).shape[0] != 1:
            #    print('mult incons')
            #    import ipdb ; ipdb.set_trace()
            # inconsistency
            #rankings[np.where(cluster[self.anoms_combo] == anom_clust)[0]] += self.incons_scs[clust_ind][np.where(cluster[self.anoms_combo] == anom_clust)[0]]

            # threshold on min anoms and min % of anoms
            if np.where(cluster[self.anoms_combo] == anom_clust)[0].shape[0] < self.min_anom_size or np.where(cluster[self.anoms_combo] == anom_clust)[0].shape[0]/np.where(cluster == anom_clust)[0].shape[0] < self.thresh:
                rankings[np.where(cluster[self.anoms_combo] == anom_clust)[0]] = 0

        return rankings

    def filter_anom_crit(self,clust1_dict,anom,anoms1,nodes1,sil_samps):
        """Filter anoms: currently no filtering, done in removing overlap"""
        return [np.intersect1d(clust1_dict[x],anom) for x in clust1_dict.keys() if x in np.where(anoms1/nodes1 > 0)[0]]
        
    def get_sc_label(self,graph,clust,anom):
        """Given a clustering, find clusters that are anomaly-dominated"""
        
        dist = 1-np.array(nx.adjacency_matrix(graph,weight='weight').todense()).astype(np.float64)
        np.fill_diagonal(dist,0)
        sil_samps = sklearn.metrics.silhouette_samples(dist,clust,metric='precomputed')
        clust1_dict,anoms1,nodes1 = self.getAnomCount(clust,anom)
        #min_anom_size = int(np.unique(clust,return_counts=True)[-1].mean())
        anom_nodes1=self.filter_anom_crit(clust1_dict,anom,anoms1,nodes1,sil_samps)
        return anom_nodes1

    def postprocess_anoms(self,clusts):
        '''Remove overlap from detected anomalies in a top-down manner (prioritize LARGER SCALE anomalies)'''
        sc_label,all_rankings = self.remove_anom_overlap(clusts)
        return sc_label,all_rankings

    
    def plot_dend(self,dend,colors,**kwargs):
        ax = plt.figure(figsize=(20, 15))
        ax.set_facecolor('white')
        dname = self.dataset.capitalize()
        plt.title(f'Hierarchical clustering for {dname}')
        #plt.legend(np.unique(colors))
        with plt.rc_context({'lines.linewidth': 0.5}):
            rdict = hierarchy.dendrogram(dend,link_color_func=lambda k: colors[k],no_labels=True)
        
        fpath = self.generate_fpath(f'preprocess_vis/{self.dataset}/{self.exp}')
        import ipdb ; ipdb.set_trace()
        plt.savefig(f'{fpath}/dend.png')
        
    def get_clusts(self,graph,scales,resolution=1.0):
        if self.dataset in 'weibo':
            resolutions = [.8]
        elif self.dataset in ['tfinance','elliptic']:
            resolutions = [.6]
        else:
            resolutions = np.arange(.5,1.1,.1)
        adj = np.array(nx.adjacency_matrix(graph,weight='weight').todense())
        for res in resolutions:
            #if 'yelpchi' in self.dataset:# or 'elliptic' in self.dataset:
            #    hierarchy = Paris()
            hierarchy = LouvainIteration(resolution=res,depth=scales)
   
            dend = hierarchy.fit_predict(adj)
            #incons = scipy.cluster.hierarchy.inconsistent(dend,scales+1)
            if len(resolutions) > 1:
                print('dasgupta',res,sknetwork.hierarchy.dasgupta_score(adj, dend))
                print('tree',res,sknetwork.hierarchy.tree_sampling_divergence(adj, dend))
            
            clusts = [postprocess.cut_straight(dend,threshold=scale) for scale in range(scales+1)]

        fclust_scales=None
        return clusts, dend, fclust_scales

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
            group_dict['anom_feat_dist'].append(sklearn.metrics.pairwise.euclidean_distances(graph.nodes[0]['feats'][clust_anoms]).mean())
            if not norm:
                group_dict['norm_feat_dist'].append(sklearn.metrics.pairwise.euclidean_distances(graph.nodes[0]['feats'][clust_anoms],graph.nodes[0]['feats'][norms]).mean())
        for key in list(group_dict.keys()):
            group_dict[key] = np.array(group_dict[key]).mean()
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

    def run_dend(self,graph,scales,return_clusts=False,return_all=False,load=False):
        """Partition the graph into multi-scale cluster & """
        #if load or os.path.exists(f'{self.fname}/{self.dataset}_labels.mat'):
        #    mat = sio.loadmat(f'{self.fname}/{self.dataset}_labels.mat')
        #    sc_all,clusts = mat['labels'][0],mat['clusts']
        #return list(sc_all),torch.tensor(clusts) 
        self.graph = graph
        anom = self.anoms_combo

        self.scales = scales
        
        clusts,dend,self.fclust_scales = self.get_clusts(graph,scales,1.1)
        #import ipdb ; ipdb.set_trace()
        #NOTE: BEFORE clusts = clusts[1:-1] #; self.incons_scs = self.incons_scs[1:-1]
        #clusts = clusts[2:]
        clusts_ = clusts[1:-1]
        #import ipdb ; ipdb.set_trace()
        dist = 1-np.array(nx.adjacency_matrix(graph,weight='weight').todense()).astype(np.float64)
        np.fill_diagonal(dist,0)
        print('calculating silhouette samples')
        self.sil_samps = [sklearn.metrics.silhouette_samples(dist,clust,metric='precomputed') for clust in clusts_]
        print('done')
        #anom_nodes = [self.get_sc_label(graph,clust,anom) for clust in clusts]
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

                print('clust',clust_ind,'anom',ind,all_)
        print('anomalous clusters found',[np.unique((clusts_[i-1][self.anoms_combo[np.where(sc_all==i)]])).shape[0] for i in range(np.unique(sc_all).shape[0])])
  
        clusts = clusts[2:]
        if True:
            # plot dendrogram
            #new_dend = sknetwork.hierarchy.aggregate_dendrogram(dend,n_clusters=50)#no_labels=True
            
            sc_all_unique = np.unique(sc_all)
            colors = np.full(int(dend.max()+2),'tab:gray',dtype='object')
            # add opacity based on silhouette score/overral ranking
            #sc_colors=[(103, 242, 209),(255,0,0),(0, 0, 255),(90, 34, 139)]
            sc_colors=['tab:red','tab:orange','tab:pink','tab:purple']
            for i in sc_all_unique:
                #all_node_ids = []
                #if i != 0:
                #    sil_samps_norm = (self.sil_samps[i-1] - self.sil_samps[i-1].min()) / (self.sil_samps[i-1].max() - self.sil_samps[i-1].min())
                for sc_node in self.anoms_combo[(sc_all==i).nonzero()]:
                    node_ids = np.array(self.find_node_id(dend[:,:2],dend.shape[0]+1,sc_node))
                    #all_node_ids = np.concatenate((all_node_ids,node_ids))
                    colors[node_ids.astype(int)] = sc_colors[i]
                    '''
                    if i != 0:
                        for j in node_ids:
                            colors[j.astype(int)] = sc_colors[i] + (sil_samps_norm[sc_node],)
                    else:
                        for j in node_ids:
                            colors[j.astype(int)] = sc_colors[i]
                    '''
            import ipdb ; ipdb.set_trace()
            if self.visualize:
                #self.plot_dend(dend,colors)
                # add opacity based on silhouette score/overral ranking
                #sc_colors=[(103, 242, 209),(255,0,0),(0, 0, 255),(90, 34, 139)]
                sc_colors=['tab:gray','tab:red','tab:orange','tab:pink','tab:purple']
                # plot spectrum
                #self.plot_spectrum()
                plt.figure()
                self.plot_spectrum_graph(graph,[],'norm',graph.nodes[0]['feats'],color=sc_colors[0])
                for ind,i in enumerate(sc_all_unique):
                    group = self.anoms_combo[(sc_all==i).nonzero()]
                    self.plot_spectrum_graph(graph,group,i,graph.nodes[0]['feats'],color=sc_colors[ind+1])

    
                clust_dicts = []
                sc_all_clusts = []
                for j in np.unique(sc_all):
                    sc_all_clusts.append(self.anoms_combo[np.where(sc_all==j)])
                sc_all_clusts.append(self.norms)
                import ipdb ; ipdb.set_trace()
                for i in range(len(clusts)):
                    print('scale',i)
                    clust_dicts.append({})
                    norm_clusts,norm_clust_counts = np.unique(clusts[i][sc_all_clusts[-1]],return_counts=True)
                    for jind,j in enumerate(sc_all_clusts):
                        if jind-1 != i and jind != len(sc_all_clusts)-1: continue

                        if jind == len(sc_all_clusts)-1:
                            gr_dict = self.get_group_stats(graph,clusts[i],sc_all_clusts[-1][np.where(np.in1d(clusts[i][sc_all_clusts[-1]],np.random.choice(norm_clusts[np.where(norm_clust_counts>self.min_anom_size)],size=np.unique(clusts[i][np.array(j)]).shape[0])))[0]],j,i,norm=True)
                        #gr_dict = self.get_group_stats(graph,clusts[i],j,sc_all_clusts[-1],i)
                        gr_dict = self.get_group_stats(graph,clusts[i],j,sc_all_clusts[-1][np.where(np.in1d(clusts[i][sc_all_clusts[-1]],np.random.choice(norm_clusts[np.where(norm_clust_counts>self.min_anom_size)],size=np.unique(clusts[i][np.array(j)]).shape[0])))[0]],i)
                        
                        print('group',jind,gr_dict)
                        if jind == 0:
                            clust_dicts[i] = gr_dict
        sio.savemat(f'{self.fname}/{self.dataset}_labels.mat',{'labels':sc_all,'clusts':clusts})
        
        return sc_all,torch.tensor(clusts)

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
        #if 'weibo' in self.dataset:
        #    x = x[15:25] ; y = y[15:25]
        spline = pchip(x, y)
        #spline = make_interp_spline(x, y, k=21)
        X_ = np.linspace(x.min(), x.max(), 801)
        Y_ = spline(X_)
        if color:
            plt.plot(X_,Y_,color=color)
        else:
            plt.plot(X_,Y_)
        #plt.axis('equal')
        return X_,y
    def plot_spectrum_graph(self,graph,anoms,img_lbl,feats,color=None):
        #from utils import get_spectrum
        #lbl = graph.adjacency_matrix().to(torch.float64)
        lbl = nx.adjacency_matrix(graph).todense().astype(np.float64) # TODO: CHECK FOR SELF LOOPS
        #e,U = get_spectrum(lbl,tag=f'anom_vis{self.dataset}',save_spectrum=self.save_spectrum)
        py_g = graphs.MultiScale(lbl)
        try:
            mat = sio.loadmat(f'{self.dataset}_eig.mat')
            e,U = mat['e'][0],mat['U']
        except:
            py_g.compute_laplacian('normalized')
            py_g.compute_fourier_basis()
            e,U = py_g.e,py_g.U
            mat = {} ; mat['e'],mat['U'] = e,U 
            sio.savemat(f'{self.dataset}_eig.mat',mat)
        legend_arr = ['No anom. signal','Single node anom.','Anom. scale 1','Anom. scale 2','Anom. scale 3']
        signal = np.random.randn(feats.shape[0],feats.shape[0])+1

        signal[anoms]=(np.random.randn(U.shape[0])*400*self.anoms_combo.shape[0]/len(anoms))+1#*anom_tot/anom.shape[0])+1# NOTE: tried 10#*(anom_tot/anom.shape[0]))
        x,y=self.plot_spectrum(e,U,signal,color)
        plt.legend(legend_arr)
        #fpath = self.generate_fpath('filter_anom_vis')
        
        plt.xticks(x[np.arange(0,y.shape[0],step=5)*20],np.round(np.arange(y.shape[0],step=5)*0.05,2))
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        plt.xlabel(r'$\lambda$')
        d_name = self.dataset.split("_")[0].capitalize()
        plt.title(f'Spectral energy distribution of {d_name}')#, label {img_lbl}')
        fpath = self.generate_fpath(f'preprocess_vis/{self.dataset}/{self.exp}')
        plt.savefig(f'{fpath}/spectra.png')

    def generate_fpath(self,fpath):
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        return fpath