from sknetwork.hierarchy import postprocess, LouvainIteration
import numpy as np
import torch
from utils import *
import dgl
import networkx as nx
from itertools import chain

class LabelAnalysis:
    def __init__(self,anoms,dataset):
        self.thresh = 0.8
        self.dataset = dataset
        self.anoms,self.anoms_combo = self.processAnoms(anoms)

    def flatten_label(self,anoms):
        if len(anoms) == 0: return anoms
        if 'elliptic' in self.dataset:
            anoms = anoms[0]
        anom_flat = anoms[0]
        if 'elliptic' in self.dataset:
            anom_flat = anom_flat[0]
        if len(anoms) > 1:
            for i in anoms[1:]:
                if 'elliptic' in self.dataset:
                    anom_flat=np.concatenate((anom_flat,i[0]))
                else:
                    anom_flat=np.concatenate((anom_flat,i))
        return anom_flat

    def processAnoms(self,anoms):
        new_dict = {}
        all_anom = None
        for anom_ind,anom in enumerate(anoms.values()):
            if list(anoms.keys())[anom_ind] == 'single':
                anom_f = anom
            else:
                anom_f = self.flatten_label(anom)
            new_dict[list(anoms.keys())[anom_ind]] = anom_f
            if all_anom is None: all_anom = anom_f
            else: all_anom = np.append(all_anom,anom_f)
        return new_dict,all_anom

    def getScaleClusts(self,dend,thresh):
        clust_labels = postprocess.cut_straight(dend,threshold=thresh)
        return clust_labels

    def check_conn(self,sc_label):
        conn_check=[]
        for i in sc_label:
            try:
                conn_check.append(nx.is_connected(self.graph.subgraph(i)))
            except:
                conn_check.append(False)
        return conn_check

    def remove_anom_overlap(self,anom,anom_next):
        return np.setdiff1d(self.flatten_label(anom),np.unique(self.flatten_label(self.flatten_label(anom_next))))

    def getAnomCount(self,clust,anom_sc_label):
        clust_keys = np.unique(clust)
        clust_dict = {}
        anom_count = []
        node_count = []
        graph_nodes = list(self.graph.nodes())
        for key in clust_keys:
            #clust_dict[key] = np.where(clust==key)[0]
            clust_dict[key] = np.array([graph_nodes[i] for i in np.where(clust==key)[0]])
            anom_count.append(np.intersect1d(anom_sc_label,clust_dict[key]).shape[0])
            node_count.append(clust_dict[key].shape[0])
        return clust_dict,np.array(anom_count),np.array(node_count)

    def get_sc_label(self,clust,anom):
        """
        Given a clustering, find clusters that are anomaly-dominated
        """
        clust1_dict,anoms1,nodes1 = self.getAnomCount(clust,anom)
        min_anom_size = int(np.unique(clust,return_counts=True)[-1].mean())
        anom_nodes1=[np.intersect1d(clust1_dict[x],anom) for x in clust1_dict.keys() if (x in np.where(anoms1/nodes1 > self.thresh)[0] and clust1_dict[x].shape[0]>=min_anom_size and False not in self.check_conn(clust1_dict[x]))]
        return anom_nodes1

    def postprocess_anoms(self,anom_nodes_tot):
        '''Remove overlap from detected anomalies'''

        sc_label = []
        for ind,anom in enumerate(anom_nodes_tot):
            sc_label.append(self.remove_anom_overlap(anom,anom_nodes_tot[ind+1:]))
        return sc_label


    def run_dend(self,graph,scales,return_clusts=False,return_all=False):
        """Partition the graph into multi-scale cluster & """
        self.graph = graph
        anom = self.anoms_combo
        if 'yelpchi' in self.dataset or 'elliptic' in self.dataset:
            hierarchy = Paris()
        else:
            hierarchy = LouvainIteration(resolution=1.1,depth=scales)

        dend = hierarchy.fit_predict(np.array(nx.adjacency_matrix(graph,weight='weight').todense()))

        clusts = [postprocess.cut_straight(dend,threshold=scale) for scale in range(scales)]
        self.thresh = 0.8

        anom_nodes = [self.get_sc_label(clust,anom) for clust in clusts]
        sc_all = self.postprocess_anoms(anom_nodes)
        
        print('CLUSTERS')
        print([i.shape[0] for i in sc_all])
        print([np.unique(clusts[i][sc_all[i]]).shape[0] for i in range(scales)])
        return sc_all,clusts
    

    def getAnomCount(self,clust,anom_sc_label):
        clust_keys = np.unique(clust)
        clust_dict = {}
        anom_count = []
        node_count = []
        for key in clust_keys:
            clust_dict[key] = np.where(clust==key)[0]
            anom_count.append(np.intersect1d(anom_sc_label,clust_dict[key]).shape[0])
            node_count.append(clust_dict[key].shape[0])
        return clust_dict,np.array(anom_count),np.array(node_count)

    def postprocess_scales(self,sc_label):
        conns = np.array(self.check_conn(sc_label)).nonzero()[0]
        # only take connected anomalies
        sc_label_ = np.array(sc_label)[conns] if len(conns) > 0 else []
        print([i.shape[0] for i in sc_label_])
        return sc_label_

    def cluster(self,adj,label_id):
        '''
        '''
        self.adj = adj.detach().cpu().numpy()
        self.graph = nx.from_numpy_matrix(self.adj)
        # TODO: need to make sure that anomalies map here
        self.label_id = label_id
        print('clustering for',self.label_id-1)
        hierarchy = LouvainIteration()  # changed from iteration; wasn't forming connected subgraphs
        #dend = hierarchy.fit_predict(self.adj)
        #dends
        # 3-scale representations -> for each one, how much are preserved in communities?
        sc1_label,sc2_label,sc3_label = self.run_dend(self.graph)

        if label_id == 0:
            self.sc1_og,self.sc1_og_f = sc1_label,self.flatten_label(sc1_label)
            self.sc2_og,self.sc2_og_f = sc2_label,self.flatten_label(sc2_label)
            self.sc3_og,self.sc3_og_f = sc3_label,self.flatten_label(sc3_label)
        else:
            if len(sc1_label) > 0:
                print('scale1 recovered',self.check_anom_recovered(sc1_label))
            else:
                print('scale1 recovered []')
            if len(sc2_label) > 0:
                print('scale2 recovered',self.check_anom_recovered(sc2_label))
            else:
                print('scale2 recovered []')
            if len(sc3_label) > 0:
                print('scale3 recovered',self.check_anom_recovered(sc3_label))
            else:
                print('scale3 recovered []')
            

        #ndict = self.node_ranks()
        ndict = {}
        self.graph_conn = nx.average_clustering(self.graph)
        ret_arr = [self.check_conn_anom(ndict,self.sc1_og),self.check_conn_anom(ndict,self.sc2_og),self.check_conn_anom(ndict,self.sc3_og)]
        print('sc1 og connectivity',ret_arr[0])
        print('sc2 og connectivity',ret_arr[1])
        print('sc3 og connectivity',ret_arr[2])
        return ret_arr

    def check_anom_recovered(self,sc_label):
        sc1 = np.intersect1d(self.sc1_og_f,self.flatten_label(sc_label)).shape[0]
        sc2 = np.intersect1d(self.sc2_og_f,self.flatten_label(sc_label)).shape[0]
        sc3 = np.intersect1d(self.sc3_og_f,self.flatten_label(sc_label)).shape[0]
        single = self.flatten_label(sc_label).shape[0]-sc1-sc2-sc3
        return sc1,sc2,sc3,single

    def node_ranks(self):
        connected_graphs = [g for g in nx.connected_components(self.graph)]
        ndict = {k:-1. for k in self.graph.nodes}
        for connected_graph_nodes in connected_graphs:
            subgraph = self.graph.subgraph(connected_graph_nodes)
            try:
                ndict.update({k:v for k,v in zip(list(connected_graph_nodes),list(nx.centrality.eigenvector_centrality(self.graph,max_iter=200).values()))})
            except Exception as e:
                print(e)
                import ipdb ; ipdb.set_trace()
        return ndict

    def check_conn_anom(self,ndict,scale):
        coeffs = []
        for sc in scale:
            coeffs.append(nx.average_clustering(self.graph,sc)/self.graph_conn)
        return coeffs