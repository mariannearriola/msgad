from sknetwork.hierarchy import postprocess, LouvainIteration
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
        if len(anoms) == 0: return anoms
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
    '''
    def remove_anom_overlap(self,anom_nodes_tot,anom,anom_ex):
        sc1 = anom_nodes_tot[anom]
        
        sc1_ret,sc2_ret,sc3_ret=[],[],[]
        overlapped=[]
        sc_sum=0
        for sc in sc1:
            if len(anom_ex) == 0:
                sc1_ret.append(sc)
                sc_sum += len(sc)
                continue
            overlap=False

            for ex in anom_ex:
                if len(np.intersect1d(sc,list(chain(*np.array(anom_nodes_tot[ex])))))!=0:
                    overlap=True

            if overlap is True:
                overlapped.append(sc)
            else:
                sc1_ret.append(sc)
                sc_sum += len(sc)
        return sc1_ret
    '''
    def remove_anom_overlap(self,sc1,sc2,sc3):
        #sc1,sc2,sc3=np.array(sc1),np.array(sc2),np.array(sc3)
        sc1_f = list(chain(*np.array(sc1)))
        if sc2 is not None:
            sc2_f = list(chain(*np.array(sc2)))
        if sc3 is not None:
            sc3_f = list(chain(*np.array(sc3)))
        #db ; ipdb.set_trace()
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
        clust1_dict,anoms1,nodes1 = self.getAnomCount(clust,anom)
        anoms_find1=anoms1[np.where(anoms1/nodes1 > self.thresh)[0]]
        nodes_find1=anoms1[np.where(anoms1/nodes1 > self.thresh)[0]]
        #import ipdb ; ipdb.set_trace()
        anom_nodes1=[np.intersect1d(clust1_dict[x],anom) for x in clust1_dict.keys() if (x in np.where(anoms1/nodes1 > self.thresh)[0] and clust1_dict[x].shape[0]>=3)]
        anom_nodes1=[np.intersect1d(clust1_dict[x],anom) for x in clust1_dict.keys() if (x in np.where(anoms1/nodes1 > self.thresh)[0] and clust1_dict[x].shape[0]>=3)]
        return anom_nodes1

    def postprocess_anoms(self,anom_nodes_tot,sc):
        anom_nodes1,anom_nodes2,anom_nodes3=anom_nodes_tot
        #import ipdb ; ipdb.set_trace()
        plt_anoms_found = [i.shape[0] for i in anom_nodes_tot[sc-1]]
        if sc == 1:
            sc_label,_,_=self.remove_anom_overlap(anom_nodes1,anom_nodes3,anom_nodes2)
        elif sc == 2:
            sc_label,_,_=self.remove_anom_overlap(anom_nodes2,anom_nodes3,None)
        elif sc == 3:
            sc_label,_,_=self.remove_anom_overlap(anom_nodes3,None,None)

        plt_anoms_found = [i.shape[0] for i in sc_label]

        conns=np.array(self.check_conn(sc_label)).nonzero()[0]
        sc_label = np.array(sc_label)[conns] if len(conns) > 0 else []
        plt_anoms_found = np.array(plt_anoms_found)[conns] if len(conns) > 0 else []
        
        return sc_label,plt_anoms_found


    def run_dend(self,graph,return_clusts=False):
        self.graph = graph
        anom = self.anoms_combo
        hierarchy = LouvainIteration(resolution=1.1)
        dend = hierarchy.fit_predict(np.array(nx.adjacency_matrix(graph,weight='weight').todense()))
        clust1,clust2,clust3 = postprocess.cut_straight(dend,threshold=0),postprocess.cut_straight(dend,threshold=2),postprocess.cut_straight(dend,threshold=3)
        clust0,clust1,clust2,clust3 = postprocess.cut_straight(dend,threshold=0),postprocess.cut_straight(dend,threshold=1),postprocess.cut_straight(dend,threshold=2),postprocess.cut_straight(dend,threshold=3)
        #import ipdb  ; ipdb.set_trace()
        self.thresh = 0.8

        anom_nodes1,anom_nodes2,anom_nodes3 = self.get_sc_label(clust1,anom),self.get_sc_label(clust2,anom),self.get_sc_label(clust3,anom)
        anom_nodes_tot = [anom_nodes1,anom_nodes2,anom_nodes3]
        #sc1_label,sc2_label,sc3_label = self.postprocess_anoms(graph,anom_nodes_tot)
        sc1_label,_=self.postprocess_anoms(anom_nodes_tot,1)
        sc2_label,_=self.postprocess_anoms(anom_nodes_tot,2)
        sc3_label,_=self.postprocess_anoms(anom_nodes_tot,3)
        print('CLUSTERS')
        print([i.shape[0] for i in sc1_label],[i.shape[0] for i in sc2_label],[i.shape[0] for i in sc3_label])
     
        if return_clusts:
            return clust1,clust2,clust3
        return sc1_label,sc2_label,sc3_label
    '''
    def check_conn(self,graph,sc_label):
        conn_check=[]
        for i in sc_label:
            try: conn_check.append(nx.is_connected(graph.subgraph(i)))
            except: conn_check.append(False)
        return conn_check
    '''
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
        #if -1 in ndict.values():
        #    raise('not all nodes ranked')
        for sc in scale:
            coeffs.append(nx.average_clustering(self.graph,sc)/self.graph_conn)
            #coeffs.append(np.vectorize(ndict.get)(sc).mean())
        return coeffs