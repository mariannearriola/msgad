from sknetwork.hierarchy import Paris, postprocess, LouvainHierarchy, LouvainIteration
import numpy as np
import torch
from utils import *
import dgl
import networkx as nx
from itertools import chain

class LabelAnalysis:
    def __init__(self,anoms,dataset):
        self.anoms,self.anoms_combo = self.processAnoms(anoms)
        self.thresh = 0.8
        self.dataset = dataset

    def flatten_label(self,anoms):
        anom_flat = anoms[0]
        if len(anoms) > 1:
            for i in anoms[1:]:
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

    def remove_anom_overlap(self,sc1,sc2,sc3):
        #sc1,sc2,sc3=np.array(sc1),np.array(sc2),np.array(sc3)
        sc1_f = list(chain(*np.array(sc1)))
        if sc2 is not None:
            sc2_f = list(chain(*np.array(sc2)))
        if sc3 is not None:
            sc3_f = list(chain(*np.array(sc3)))
        
        sc1_ret,sc2_ret,sc3_ret=[],[],[]
        overlapped=[]
        sc_sum=0
        for sc in sc1:
            if sc2 is None and sc3 is not None:
                if len(np.intersect1d(sc,sc3_f))==0:
                    sc1_ret.append(sc)
                    sc_sum += len(sc)
                else:
                    overlapped.append(sc)
            elif sc3 is None and sc2 is not None:
                if len(np.intersect1d(sc,sc2_f))==0:
                    sc1_ret.append(sc)
                    sc_sum += len(sc)
                else:
                    overlapped.append(sc)
            elif sc2 is not None and sc3 is not None:
                if len(np.intersect1d(sc,sc2_f))==0 and len(np.intersect1d(sc,sc3_f))==0:
                    sc1_ret.append(sc)
                    sc_sum += len(sc)
                else:
                    overlapped.append(sc)
            elif sc2 is None and sc3 is None:
                sc1_ret.append(sc)
                sc_sum += len(sc)
        return sc1_ret,overlapped,sc_sum

    def run_dend(self,graph,res):
        dend = []
        for i in range(3):
            dend.append(nx.community.louvain_communities(graph,resolution=res[i]))
        #dend = [i for i in nx.community.louvain_partitions(graph,threshold=thresh)]

        dend_clusts = [[list(i) for i in d] for d in dend]
        dend_anoms = []
        shapes = []
        for clust in dend_clusts:
            dend_anoms.append([np.intersect1d(i,self.anoms_combo) for i in clust if (len(i) != 0 and np.intersect1d(i,self.anoms_combo).shape[0]/len(i) >= self.thresh and np.intersect1d(i,self.anoms_combo).shape[0] >= 3)])
            shapes.append([i.shape[0] for i in dend_anoms[-1]])
        
        anom_nodes1,anom_nodes2,anom_nodes3 = dend_anoms
        sc1_label,o1,s1=self.remove_anom_overlap(anom_nodes1,anom_nodes3,anom_nodes2)
        sc1_count = [x.shape[0] for x in sc1_label]
        sc2_label,o2,s2=self.remove_anom_overlap(anom_nodes2,anom_nodes3,None)
        sc2_count = [x.shape[0] for x in sc2_label]
        sc3_label,o3,s3=self.remove_anom_overlap(anom_nodes3,None,None)
        sc3_count = [x.shape[0] for x in sc3_label]
        return sc1_label,sc2_label,sc3_label,sc1_count,sc2_count,sc3_count

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

    def cluster(self,adj,label_id):
        self.adj = adj.detach().cpu().numpy()
        self.graph = nx.from_numpy_matrix(self.adj)
        # TODO: need to make sure that anomalies map here
        self.label_id = label_id
        print('clustering for',self.label_id)
        paris = LouvainIteration()  # changed from iteration; wasn't forming connected subgraphs
        dend = paris.fit_predict(self.adj)
        #dends
        res=[1.5,0.8,0.1]
        sc1_label,sc2_label,sc3_label,sc1_count,sc2_count,sc3_count = self.run_dend(self.graph,res)

        conns_1=np.array(self.check_conn(sc1_label)).nonzero()[0]
        conns_2=np.array(self.check_conn(sc2_label)).nonzero()[0]
        conns_3=np.array(self.check_conn(sc3_label)).nonzero()[0]
        
        sc1_label = np.array(sc1_label)[conns_1] if len(conns_1) > 0 else []
        sc2_label = np.array(sc2_label)[conns_2] if len(conns_2) > 0 else []
        sc3_label = np.array(sc3_label)[conns_3] if len(conns_3) > 0 else []
        print([i.shape[0] for i in sc1_label],[i.shape[0] for i in sc2_label],[i.shape[0] for i in sc3_label])
        if 'elliptic' in self.dataset:
            return sc3_label,sc2_label,sc1_label
        clust1 = self.getScaleClusts(dend,1)
        clust2 = self.getScaleClusts(dend,2)
        clust3 = self.getScaleClusts(dend,3)
        
        #print(np.unique(clust1).shape,np.unique(clust2).shape,np.unique(clust3).shape)
        clust1_dict,anoms1,nodes1 = self.getAnomCount(clust1,self.anoms_combo)
        thresh = self.thresh
        anoms_find1=anoms1[np.where(anoms1/nodes1 > thresh)[0]]
        nodes_find1=anoms1[np.where(anoms1/nodes1 > thresh)[0]]
        #import ipdb ; ipdb.set_trace()
        anom_nodes1=[np.intersect1d(clust1_dict[x],self.anoms_combo) for x in clust1_dict.keys() if (x in np.where(anoms1/nodes1 > thresh)[0] and clust1_dict[x].shape[0]>=3)]
        clust2_dict,anoms2,nodes2 = self.getAnomCount(clust2,self.anoms_combo)
        anom_nodes2=[np.intersect1d(clust2_dict[x],self.anoms_combo) for x in clust2_dict.keys() if (x in np.where(anoms2/nodes2 > thresh)[0] and clust2_dict[x].shape[0]>=3)]
        clust3_dict,anoms3,nodes3 = self.getAnomCount(clust3,self.anoms_combo)
        anom_nodes3=[np.intersect1d(clust3_dict[x],self.anoms_combo) for x in clust3_dict.keys() if (x in np.where(anoms3/nodes3 > thresh)[0] and clust3_dict[x].shape[0]>=3)]
        sc1_label,o1,s1=self.remove_anom_overlap(anom_nodes1,anom_nodes3,anom_nodes2)
        sc1_count = [x.shape[0] for x in sc1_label]
        sc2_label,o2,s2=self.remove_anom_overlap(anom_nodes2,anom_nodes3,None)
        sc2_count = [x.shape[0] for x in sc2_label]
        sc3_label,o3,s3=self.remove_anom_overlap(anom_nodes3,None,None)
        sc3_count = [x.shape[0] for x in sc3_label]


        conns_1=np.array(self.check_conn(sc1_label)).nonzero()[0]
        conns_2=np.array(self.check_conn(sc2_label)).nonzero()[0]
        conns_3=np.array(self.check_conn(sc3_label)).nonzero()[0]
        
        sc1_label = np.array(sc1_label)[conns_1] if len(conns_1) > 0 else []
        sc2_label = np.array(sc2_label)[conns_2] if len(conns_2) > 0 else []
        sc3_label = np.array(sc3_label)[conns_3] if len(conns_3) > 0 else []
        print([i.shape[0] for i in sc1_label],[i.shape[0] for i in sc2_label],[i.shape[0] for i in sc3_label])

        if label_id == 0:
            self.sc1_og,self.sc1_og_f = sc1_label,self.flatten_label(sc1_label)
            self.sc2_og,self.sc2_og_f = sc2_label,self.flatten_label(sc2_label)
            self.sc3_og,self.sc3_og_f = sc3_label,self.flatten_label(sc3_label)
        else:
            if len(sc1_label) > 0:
                print('scale1 recovered',self.check_anom_recovered(sc1_label))
            if len(sc2_label) > 0:
                print('scale2 recovered',self.check_anom_recovered(sc2_label))
            if len(sc3_label) > 0:
                print('scale3 recovered',self.check_anom_recovered(sc3_label))
        ndict = self.node_ranks()
        ret_arr = [self.check_conn_anom(ndict,self.sc1_og),self.check_conn_anom(ndict,self.sc2_og),self.check_conn_anom(ndict,self.sc3_og)]
        print('sc1 og connectivity',ret_arr[0])
        print('sc2 og connectivity',ret_arr[1])
        print('sc3 og connectivity',ret_arr[2])
        return ret_arr

    def check_anom_recovered(self,sc_label):
        sc1 = np.intersect1d(self.sc1_og_f,self.flatten_label(sc_label)).shape[0]
        sc2 = np.intersect1d(self.sc2_og_f,self.flatten_label(sc_label)).shape[0]
        sc3 = np.intersect1d(self.sc3_og_f,self.flatten_label(sc_label)).shape[0]
        return sc1,sc2,sc3

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
        if -1 in ndict.values():
            raise('not all nodes ranked')
        for sc in scale:
            #coeffs.append(nx.average_clustering(self.graph,sc))
            coeffs.append(np.vectorize(ndict.get)(sc).mean())
        return coeffs

