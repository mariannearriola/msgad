import ipdb
import networkx as nx
import numpy as np
import pickle
import torch
import networkx.algorithms.community as nx_comm
from scipy.stats import entropy
from scipy.io import loadmat
from scipy.spatial.distance import euclidean
from itertools import chain, combinations
from sknetwork.hierarchy import Paris, postprocess, LouvainHierarchy, LouvainIteration
import numpy.linalg as npla
from scipy.sparse.csgraph import shortest_path
import sys
import os
import argparse
import matplotlib.pyplot as plt
import random

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='weibo', type=str)
parser.add_argument('--feat_comp', default=False, type=bool)
parser.add_argument('--struct_comp', default=False, type=bool)
parser.add_argument('--reversed', default=False, type=bool)
parser.add_argument('--thresh', default=0.9, type=float, help='threshold for percentage of anomalies in cluster to accept')
parser.add_argument('--emb_comp',default=False,type=bool,help='whether to compare normal/anom')
parser.add_argument('--print_conn', default=False,type=bool, help='whether or not to print connectivity characteristics')
args = parser.parse_args()
seed_everything(1)

def flatten_label(anoms_list):
    ret_list = np.array([])
    for ind,i in enumerate(anoms_list):
        if ind == 0:
            ret_list = i
        else:
            ret_list = np.append(ret_list,i)
    return ret_list

mat_file = loadmat(f'msgad/data/{args.dataset}.mat')
if 'Network' in mat_file.keys():
    graph = mat_file['Network']
    graph = nx.from_numpy_matrix(graph.todense())
else:
    graph = mat_file['Edge-index']
    graph = nx.from_edgelist(graph.T)
feats = mat_file['Attributes']
label = mat_file['Label']
anom = np.where(label==1)[1]
norm = np.where(label==0)[1]


def find_anoms(comms_arr,anomalies):
    all_anoms_found = []
    all_nodes_found = []
    all_anoms_found_arr = []
    accessed = False
    anoms_found = 0
    nodes_found = 0
    anoms_found_arr = []
    for l_comm_ind,l_comm in enumerate(comms_arr):
            if isinstance(l_comm,int):
                accessed = True
                if int(l_comm) in anomalies:
                    anoms_found += 1
                    nodes_found += 1
                    anoms_found_arr.append(int(l_comm))
                else:
                    nodes_found += 1
                if l_comm_ind == len(comms_arr)-1:
                    all_anoms_found.append(anoms_found)
                    all_nodes_found.append(nodes_found)
                    all_anoms_found_arr.append(anoms_found_arr)
            else:
                rec,rec_nodes,rec_arr = find_anoms(l_comm,anomalies)
                all_anoms_found.append(rec)
                all_nodes_found.append(rec_nodes)
                all_anoms_found_arr.append(rec_arr)
                
    return all_anoms_found,all_nodes_found,all_anoms_found_arr

def get_comms(graph,scale):
    ret_arr = []
    anoms_found = 0
    l_comms = nx_comm.louvain_communities(graph)
    for l_comm in l_comms:
        if scale > 1:
            if not isinstance(l_comm,int):
                if len(l_comm) == 1:
                    ret_arr.append(list(l_comm))
                    continue
                l_comms_nested = get_comms(graph.subgraph(l_comm),scale-1)
                ret_arr.append(list(l_comms_nested))
            else:
                import ipdb ; ipdb.set_trace()
                ret_arr.append(list(l_comms))
        elif scale == 1:
            ret_arr.append(list(l_comm))
    return ret_arr

def get_anom_sc_labels(label,count,node_count,scale,cluster_thresh,node_thresh):
    newlabel = list(chain(*label))
    newcount = list(chain(*count))
    nodecount = list(chain(*node_count))
    while scale != 1:
        newcount = list(chain(*newcount))
        newlabel = list(chain(*newlabel))
        nodecount = list(chain(*nodecount))
        scale -= 1
    ipdb.set_trace()
    anom_labels = []
    sc_inds = np.where((np.array(newcount)/np.array(nodecount)>cluster_thresh) & (np.array(newcount)>node_thresh) )[0]
    anom_counts = []
    for ind in sc_inds:
        anom_labels.append(newlabel[ind])
        anom_counts.append(newcount[ind])
    return anom_labels,anom_counts

def remove_anom_overlap(sc1,sc2,sc3):
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
anomaly_sc_label = []

def getScaleClusts(dend,thresh):
    clust_labels = postprocess.cut_straight(dend,threshold=thresh)
    return clust_labels

def getAnomCount(clust,anom_sc_label):
    clust_keys = np.unique(clust)
    clust_dict = {}
    anom_count = []
    node_count = []
    graph_nodes = list(graph.nodes())
    for key in clust_keys:
        clust_dict[key] = np.array([graph_nodes[i] for i in np.where(clust==key)[0]])
        anom_count.append(np.intersect1d(anom_sc_label,clust_dict[key]).shape[0])
        node_count.append(clust_dict[key].shape[0])
    return clust_dict,np.array(anom_count),np.array(node_count)

adj = nx.adjacency_matrix(graph)
norm_feats,anom_feats=feats[norm],feats[anom]

def compareMats(mat,matrix_convert=False):
    prod_adj = mat@mat.T

    anom_anom = prod_adj[anom][:,anom]
    norm_norm = prod_adj[norm[:len(anom)]][:,norm[:len(anom)]]
    anom_norm = prod_adj[anom][:,norm[:len(anom)]]
    if matrix_convert:
        anom_anom,norm_norm,anom_norm=anom_anom.toarray(),norm_norm.toarray(),anom_norm.toarray()
    
    anom_anom_scores,norm_norm_scores,anom_norm_scores = np.zeros(len(anom)),np.zeros(len(anom)),np.zeros(len(anom))
    for a_ind,a in enumerate(anom):
        a_ = np.ma.array(anom_anom[a_ind], mask=False)
        a_.mask[a_ind] = True
        anom_anom_scores[a_ind] = np.mean(a_)
    
        a_ = np.ma.array(anom_norm[a_ind], mask=False)
        a_.mask[a_ind] = True
        anom_norm_scores[a_ind] = np.mean(a_)

        a_ = np.ma.array(norm_norm[a_ind], mask=False)
        a_.mask[a_ind] = True
        norm_norm_scores[a_ind] = np.mean(a_)
    return anom_anom_scores,norm_norm_scores,anom_norm_scores

def rankDeviation(aa,nn,an,std_reduce=1):
    diff=(np.mean(nn)-an)**2
    std = np.std(nn)/std_reduce
    dev0 = np.where(diff<std)[0]
    dev1 = np.where(diff[np.where(diff>=std)[0]]<std*2)[0]
    dev2 = np.where(diff[np.where(diff>=std*2)[0]]<std*3)[0]
    dev3 = np.where(diff[np.where(diff>=std*3)[0]]<std*4)[0]
    dev4 = np.where(diff[np.where(diff>=std*4)[0]]<std*5)[0]
    return [dev0,dev1,dev2,dev3,dev4]
    
#adj_aa,adj_nn,adj_an = compareMats(adj,True)
# NOTE: VS EUCLIDEAN DISTANCE?
#feat_aa,feat_nn,feat_an = compareMats(feats,False)

#adj_ranks = rankDeviation(adj_aa,adj_nn,adj_an)
#feat_ranks = rankDeviation(feat_aa,feat_nn,feat_an,std_reduce=0.5)
#adj_anom_clusters=[list([c for c in sorted(nx.connected_components(nx.subgraph(graph,i)),key=len,reverse=True)][0]) for i in adj_ranks]
#feat_anom_clusters=[list([c for c in sorted(nx.connected_components(nx.subgraph(graph,i)),key=len,reverse=True)][0]) for i in feat_ranks]

#adj_anom_clusters = adj_anom_clusters[1:]
#sc_label = [adj_anom_clusters[:2],adj_anom_clusters[2],adj_anom_clusters[3]]
import scipy.io as sio

if args.feat_comp:
    print('comparing feats')
    #dists = np.zeros(adj.shape[0])
    dists = np.zeros(anom.shape[0])
    norm_feats = norm_feats[np.random.choice(np.arange(len(norm)),len(anom))]#.todense()
    norm_mean,norm_std = np.mean(norm_feats,axis=0),np.sum(np.std(norm_feats,axis=0))/3
    sc0,sc1,sc2,sc3,sc4=[],[],[],[],[]
    for a_ind,a in enumerate(anom):
        #dists[a_ind] = np.sum(euclidean(anom_feats[a_ind].todense(),norm_mean))
        dists[a_ind] = np.sum(euclidean(anom_feats[a_ind],norm_mean))
        if dists[a_ind] > norm_std and dists[a_ind] <= norm_std*2:
            sc1.append(a)
        elif dists[a_ind] > norm_std*2 and dists[a_ind] <= norm_std*3:
            sc2.append(a)
        elif dists[a_ind] > norm_std*3 and dists[a_ind] <= norm_std*4:
            sc3.append(a)
        elif dists[a_ind] >= norm_std*4:
            sc4.append(a)
        elif dists[a_ind] <= norm_std:
            sc0.append(a)
    sc0,sc1,sc2,sc3,sc4=np.array(sc0),np.array(sc1),np.array(sc2),np.array(sc3),np.array(sc4)

def check_lists_of_numpy_arrays_equality(list1, list2):
    if len(list1) != len(list2):
        return False

    for arr1, arr2 in zip(list1, list2):
        if not np.array_equal(arr1, arr2):
            return False

    return True

def check_lists_of_sets_equality(list1, list2):
    set1 = set(map(frozenset, list1))
    set2 = set(map(frozenset, list2))
    return set1 == set2

def postprocess_anoms(anom_nodes_tot,sc):
    anom_nodes1,anom_nodes2,anom_nodes3=anom_nodes_tot
    plt_anoms_found = [i.shape[0] for i in anom_nodes_tot[sc-1]]
    if sc == 1:
        sc_label,_,_=remove_anom_overlap(anom_nodes1,anom_nodes3,anom_nodes2)
    elif sc == 2:
        sc_label,_,_=remove_anom_overlap(anom_nodes2,anom_nodes3,None)
    elif sc == 3:
        sc_label,_,_=remove_anom_overlap(anom_nodes3,None,None)

    plt_anoms_found = [i.shape[0] for i in sc_label]

    conns=np.array(check_conn(sc_label)).nonzero()[0]
    sc_label = np.array(sc_label)[conns] if len(conns) > 0 else []
    plt_anoms_found = np.array(plt_anoms_found)[conns] if len(conns) > 0 else []
    
    return sc_label,plt_anoms_found


def run_dend(graph,res,adj,anom_only=True):
    if True:
        paris = LouvainIteration(resolution=res[0])  # changed from iteration; wasn't forming connected subgraphs
        dend = paris.fit_predict(np.array(adj.todense()))
        anom_nodes_list = []
        for i in range(1,4):
            try:
                anom_clust = postprocess.cut_straight(dend,threshold=i)
            except:
                anom_nodes_list.append([])
                continue
            clust1_dict,anoms1,nodes1 = getAnomCount(anom_clust,anom)
            anom_nodes=[np.intersect1d(clust1_dict[x],anom) for x in clust1_dict.keys() if (x in np.where(anoms1/nodes1 >= args.thresh)[0] and np.intersect1d(anom,clust1_dict[x]).shape[0]>=3)]
            anom_nodes_list.append(anom_nodes)
        anom_nodes1,anom_nodes2,anom_nodes3=anom_nodes_list
    else:
        print(res)
        dend = []
        #for i in range(3):
        #    #dend.append(nx.community.louvain_communities(graph,resolution=res[i],seed=1))
        # does cut threshold = 3
        dend = [i for i in nx.community.louvain_partitions(graph,resolution=res[0],seed=1)]#[-3:]
        print(len(dend))
        dend_clusts = [[list(i) for i in d] for d in dend]
        
        dend_anoms = []
        shapes = []
        if anom_only is False:
            return dend_clusts

        for clust in dend_clusts[-3:]:
            print(max([len(i) for i in clust]))
            dend_anoms.append([np.intersect1d(i,anom) for i in clust if (len(i) != 0 and np.intersect1d(i,anom).shape[0]/len(i) >= args.thresh and np.intersect1d(i,anom).shape[0] >= 3)])
            shapes.append([i.shape[0] for i in dend_anoms[-1]])
        import ipdb ; ipdb.set_trace()
        anom_nodes1,anom_nodes2,anom_nodes3 = dend_anoms
    anom_nodes_tot = [anom_nodes1,anom_nodes2,anom_nodes3]
    sc1_label,plt_anoms_found1=postprocess_anoms(anom_nodes_tot,1)
    sc2_label,plt_anoms_found2=postprocess_anoms(anom_nodes_tot,2)
    sc3_label,plt_anoms_found3=postprocess_anoms(anom_nodes_tot,3)

    return sc1_label,sc2_label,sc3_label,plt_anoms_found1,plt_anoms_found2,plt_anoms_found3


def check_conn(sc_label):
    conn_check=[]
    for i in sc_label:
        try:
            conn_check.append(nx.is_connected(graph.subgraph(i)))
        except:
            conn_check.append(False)
    return conn_check

def hier_cluster(graph,adj):
    paris = LouvainIteration()  # changed from iteration; wasn't forming connected subgraphs
    dend = paris.fit_predict(adj)
    if True:
        xs = np.arange(0.5,2.5,0.2)
        #xs = np.arange(2,0,-0.5)
        if args.reversed:
            xs = np.flip(xs)
        x_len = len(xs)
        print(x_len)
        res = [0,0,0]
        plt_anoms_all, anoms_found = [], np.array([])
        anoms_tot = []
        for x in range(x_len):
            res = [xs[x],xs[x],xs[x]]
            sc1_label,sc2_label,sc3_label,plt_anoms_found1,plt_anoms_found2,plt_anoms_found3 = run_dend(graph,res,adj)
            
            plt_anoms_found = [plt_anoms_found1,plt_anoms_found2,plt_anoms_found3]
            sc_labels = [sc1_label,sc2_label,sc3_label]
            anoms_tot.append(sc_labels)
            plt_anoms_all.append(plt_anoms_found)
            
            for sc,sc_label in enumerate(sc_labels):
                anoms_found = np.append(anoms_found,flatten_label(sc_label))
                continue
                plt_anoms_sc = []
                if x == 0:
                    plt_anoms_sc.append(plt_anoms_found[sc])
                    anoms_found = np.append(anoms_found,flatten_label(sc_label))
                    anoms_tot.append(sc_label)
                    print('found',x,[j for j in plt_anoms_found])
                else:
                    import ipdb ; ipdb.set_trace()
                    plt_to_add = np.array([])
                    plt_anoms_found_sc = []
                    anoms_sc = []
                    for ind,anom_found in enumerate(sc_label):
                        if len(np.intersect1d(np.array(anoms_found),anom_found)) == 0:
                            plt_anoms_found_sc.append(plt_anoms_found[sc][ind])
                            anoms_found = np.append(anoms_found,anom_found)
                            anoms_sc.append(anom_found)
                    plt_anoms_sc.append(plt_anoms_found_sc)
                    anoms_tot.append(anoms_sc)
                    print('found',xs[x],[j for j in plt_anoms_found_sc])
                plt_anoms_all.append(plt_anoms_sc)
        #import ipdb ; ipdb.set_trace()
        plt.figure()
        #max_found = max([len(i) for i in plt_anoms_all])
        #top3 = xs[np.argsort(np.array([sum(i) for i in plt_anoms_all]))[-3:]]
        plt.figure()
        colors = ['red','blue','purple']
        max_score,max_ind = 0,0
        tot_entropies_sum = []
        tot_entropies = []
        import ipdb ; ipdb.set_trace()
        for res,i in enumerate(plt_anoms_all):
            tot_clust=sum([len(i_) for i_ in i])
            tot_clust_sum = sum([sum(i_) for i_ in i])
            entropy_score = entropy([len(i_)/tot_clust for i_ in i], base=2)
            entropy_score_sum = entropy([sum(i_)/tot_clust_sum for i_ in i], base=2)


            entropy_weight = 0.3  # Weight for entropy score (adjust as desired)
            total_weight = 0.7
            entropy_normalized = entropy_score# / np.log2(len(i))
            entropy_normalized_sum = entropy_score_sum# / np.log2(len(i))

            tot_entropies_sum.append(entropy_normalized_sum) ; tot_entropies.append(entropy_normalized)

            combined_score = (entropy_weight * entropy_normalized) + (total_weight * entropy_normalized_sum)
            #score = entropy_score*tot_clust
            if combined_score > max_score:
                max_ind = res
                max_score = combined_score
            for sc,j in enumerate(i):
                plt.scatter(np.full(len(j),xs[res]),j,color=colors[sc])
    
        '''
        for i in range(max_found):
            anoms_found = []
            for j in plt_anoms_all:
                if len(j) > i:
                    anoms_found.append(j[i])
                else:
                     anoms_found.append(None)
            plt.scatter(xs,anoms_found)
        plt.vlines(top3,0,100)
        print('top xs', top3)
        '''
        plt.vlines(xs[max_ind],0,3)
        if not os.path.exists(f'vis/{args.dataset}'):
            os.makedirs(f'vis/{args.dataset}')
        plt.savefig(f'vis/{args.dataset}/test_res_{args.dataset}_rev{args.reversed}_idx3.png')
        #max_ind=5
        import ipdb ; ipdb.set_trace()
        print(max_ind)
        print([[i.shape[0] for i in j] for j in anoms_tot[max_ind]])
        return anoms_tot[max_ind]


def getDensity(nx_graph,current_nodes):
    densities = []
    for i in current_nodes:
        if i.shape[0] < 3:
            continue
        densities.append(nx.density(nx_graph.subgraph(i)))
    return np.mean(np.array(densities))

def getConnectivity(sc1_label,sc2_label,sc3_label):
    print(getDensity(graph,sc1_label))
    print(getDensity(graph,sc2_label))
    print(getDensity(graph,sc3_label))
    shortest_paths = dict(nx.shortest_path_length(graph))
    sps1 = [getShortestPaths(graph,shortest_paths,x) for x in sc1_label]
    sps2 = [getShortestPaths(graph,shortest_paths,x) for x in sc2_label]
    sps3 = [getShortestPaths(graph,shortest_paths,x) for x in sc3_label]
    #print('shortest paths',sps1,sps2,sps3)

    import ipdb ; ipdb.set_trace()
def getShortestPaths(nx_graph,shortest_paths,current_nodes):
    if current_nodes.shape[0] < 3:
        return -1
    anom_sps = []
    for ind,i in enumerate(current_nodes):
        i_sps = []
        for jind,j in enumerate(current_nodes):
            
            if i==j:continue
            #i_sps.append(shortest_paths[i][j])
            try:
                i_sps.append(nx.shortest_path_length(nx_graph,source=i,target=j))
            except Exception as e:
                import ipdb ; ipdb.set_trace()
                print(e)
                print('hi')
        anom_sps.append(sum(i_sps)/len(i_sps))
    return sum(anom_sps)/len(anom_sps)
    print('AVG SHORTEST PATHS',sum(anom_sps)/len(anom_sps))

def getCuts(nx_graph,ms_anoms):
    paris = LouvainIteration()  # changed from iteration; wasn't forming connected subgraphs
    
    dend = paris.fit_predict(np.array(nx.adjacency_matrix(nx_graph).todense()))
    
    #res=[1.5,0.8,0.1]
    clust1 = getScaleClusts(dend,1)
    clust2 = getScaleClusts(dend,2)
    clust3 = getScaleClusts(dend,3)
    norm_clusts = [clust1,clust2,clust3]
    clust1_dict,norms1,nodes1 = getAnomCount(clust1,norm)
    norm_nodes1=[clust1_dict[x] for x in clust1_dict.keys() if (x in np.where(norms1/nodes1 > args.thresh)[0] and clust1_dict[x].shape[0]>=3)]
    clust2_dict,norms2,nodes2 = getAnomCount(clust2,norm)
    norm_nodes2=[clust1_dict[x] for x in clust2_dict.keys() if (x in np.where(norms2/nodes2 > args.thresh)[0] and clust2_dict[x].shape[0]>=3)]
    clust3_dict,norms3,nodes3 = getAnomCount(clust3,norm)
    norm_nodes3=[clust1_dict[x] for x in clust3_dict.keys() if (x in np.where(norms3/nodes3 > args.thresh)[0] and clust3_dict[x].shape[0]>=3)]
    norm_clusts = [norm_nodes1,norm_nodes2,norm_nodes3]

    norm_comms = [[list(i) for i in list(j) if np.intersect1d(list(i),anom).shape[0] == 0] for j in norm_clusts]
    
    cuts = []

    for sc,norm_sc in enumerate(norm_comms):
        for comm in norm_sc:
            if nx.cut_size(nx_graph,comm) > 0:
                cuts.append(nx.normalized_cut_size(nx_graph,comm))
        print('(outside) norm cuts for sc',sc,sum(cuts)/len(cuts))

    for sc,norm_sc in enumerate(norm_comms):
        for comm in norm_sc:
            if nx.cut_size(nx_graph,comm) > 0:
                cuts.append(nx_graph.subgraph(comm).number_of_edges()/nx.cut_size(nx_graph,comm))
        print('(inside) norm cuts for sc',sc,sum(cuts)/len(cuts))
    
    for a_ind,a in enumerate(ms_anoms):
        cuts = []
        for comm in a:
            if nx.cut_size(nx_graph,comm) > 0:
                #cuts.append(nx_graph.subgraph(comm).number_of_edges()/nx.cut_size(nx_graph,comm))
                cuts.append(nx.normalized_cut_size(nx_graph,comm))
        print('(outside) cuts for',a_ind,cuts)

    for a_ind,a in enumerate(ms_anoms):
        cuts = []
        for comm in a:
            if nx.cut_size(nx_graph,comm) > 0:
                cuts.append(nx_graph.subgraph(comm).number_of_edges()/nx.cut_size(nx_graph,comm))
        print('(inside) cuts for',a_ind,cuts)

sc1_label,sc2_label,sc3_label=hier_cluster(graph,adj)


sc1_label = np.array([np.intersect1d(i,anom) for i in sc1_label if i.shape[0]>2])
sc2_label = np.array([np.intersect1d(i,anom) for i in sc2_label if i.shape[0]>2])
sc3_label = np.array([np.intersect1d(i,anom) for i in sc3_label if i.shape[0]>2])


'''
anom_idx1=[np.where(anom==i)[0][0] for i in sc1_label]
anom_idx2=[np.where(anom==i)[0][0] for i in sc2_label]
anom_idx3=[np.where(anom==i)[0][0] for i in sc3_label]
'''
'''
import matplotlib.pyplot as plt
plt.figure()
plt.hist(dists[anom_idx1],stacked='true',fc=(0, 0, 1, 0.5))
plt.hist(dists[anom_idx2],stacked='true',fc=(0, 0, 1, 0.5))
plt.legend(['scale1','scale2'])
plt.savefig('weibo_feat_diff.png')

plt.figure()
plt.hist(struct_anom_diffs[anom_idx1],stacked='true', fc=(1, 0, 0, 0.5))
plt.hist(struct_anom_diffs[anom_idx2],stacked='true', fc=(0, 0, 1, 0.5))
plt.legend(['scale1','scale2'])
plt.savefig('weibo_struct_diff.png')
'''

if args.emb_comp:
    anom_check = [sc1_label]
    feats = mat_file['Attributes'].todense()
    all_anom = [sc1_label,sc2_label,sc3_label]
    all_anom=np.append(np.append(all_anom[0],all_anom[1]),all_anom[2])

    # get anom embeds differences
    all_anom_diffs = []
    for anoms in anom_check:
        anoms_embs = feats[anoms]
        anom_diffs = []
        for ind,embed in enumerate(anoms_embs):
            for ind_,embed_ in enumerate(anoms_embs):
                #if len(anom_diffs) == len(anoms): continue
                if ind_ >= ind: break
                anom_diffs.append((embed@embed_.T)/(npla.norm(embed)*npla.norm(embed_)))
        all_anom_diffs.append(anom_diffs)

    # get normal embeds differences
    normal_diffs = []
    for ind,embed in enumerate(feats):
        if ind in all_anom: continue
        if len(normal_diffs) == len(all_anom_diffs):
            break
        for ind_,embed_ in enumerate(feats):
            if ind_ >= ind: break
            if ind_ in all_anom: continue
            try:
                normal_diffs.append((embed@embed_.T)/(npla.norm(embed)*npla.norm(embed_)))
            except:
                import ipdb ; ipdb.set_trace()

    # get normal vs anom embeds differences
    all_norm_anom_diffs = []
    for anoms in anom_check:
        norm_anom_diffs=[]
        for ind, embed in enumerate(feats):
            if ind in all_anom: continue
            for ind_,anom in enumerate(feats[anoms]):
                #if len(norm_anom_diffs) == len(anoms): continue 
                #norm_anom_diffs.append(npla.norm(embed-anom)/max_diff)
                norm_anom_diffs.append((embed@anom.T)/(npla.norm(embed)*npla.norm(anom)))
    
        all_norm_anom_diffs.append(norm_anom_diffs)
    #import ipdb ; ipdb.set_trace()
    print('normal-normal',sum(normal_diffs)/len(normal_diffs))
    print('anom-anom')
    for ind,i in enumerate(all_anom_diffs):
        print(sum(i)/len(all_anom_diffs[ind]))
        
    print('anom-normal')
    for ind,i in enumerate(all_norm_anom_diffs):
        print(sum(i)/len(all_norm_anom_diffs[ind]))
        
    print('----')

all_sc_label = [sc1_label,sc2_label,sc3_label]
for ind,sc_label in enumerate(all_sc_label):
    if len(sc_label) == 0:
        all_ms_labels = np.array([[]])
    for ind_,sc in enumerate(sc_label):
        if ind == 0 and ind_ == 0:
            all_ms_labels = sc
        else:
            all_ms_labels = np.append(all_ms_labels,sc)


single_label = np.setdiff1d(anom,all_ms_labels)

if args.print_conn:
    #getConnectivity(sc1_label,sc2_label,sc3_label)
    getCuts(graph,[sc1_label,sc2_label,sc3_label])

import ipdb; ipdb.set_trace()
mat_file['anom_sc1'] = sc1_label
mat_file['anom_sc2'] = sc2_label
mat_file['anom_sc3'] = sc3_label
mat_file['anom_single'] = single_label
#sio.savemat(f'msgad/data/{args.dataset}_test.mat', mat_file)