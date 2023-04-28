import ipdb
import networkx as nx
import numpy as np
import pickle
import torch
import networkx.algorithms.community as nx_comm
from scipy.io import loadmat
from scipy.spatial.distance import euclidean
from itertools import chain
from sknetwork.hierarchy import Paris, postprocess, LouvainHierarchy, LouvainIteration
import numpy.linalg as npla
from scipy.sparse.csgraph import shortest_path
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='weibo', type=str)
parser.add_argument('--feat_comp', default=False, type=bool)
parser.add_argument('--struct_comp', default=False, type=bool)
parser.add_argument('--thresh', default=0.9, type=float, help='threshold for percentage of anomalies in cluster to accept')
parser.add_argument('--emb_comp',default=False,type=bool,help='whether to compare normal/anom')
parser.add_argument('--print_conn', default=False,type=bool, help='whether or not to print connectivity characteristics')
args = parser.parse_args()

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
    for key in clust_keys:
        clust_dict[key] = np.where(clust==key)[0]
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

def run_dend(graph,res):
    dend = []
    for i in range(3):
        dend.append(nx.community.louvain_communities(graph,resolution=res[i]))
    #dend = [i for i in nx.community.louvain_partitions(graph,threshold=thresh)]

    dend_clusts = [[list(i) for i in d] for d in dend]
    dend_anoms = []
    shapes = []
    for clust in dend_clusts:
        dend_anoms.append([np.intersect1d(i,anom) for i in clust if (len(i) != 0 and np.intersect1d(i,anom).shape[0]/len(i) >= args.thresh and np.intersect1d(i,anom).shape[0] >= 3)])
        shapes.append([i.shape[0] for i in dend_anoms[-1]])
    
    anom_nodes1,anom_nodes2,anom_nodes3 = dend_anoms
    sc1_label,o1,s1=remove_anom_overlap(anom_nodes1,anom_nodes3,anom_nodes2)
    sc1_count = [x.shape[0] for x in sc1_label]
    sc2_label,o2,s2=remove_anom_overlap(anom_nodes2,anom_nodes3,None)
    sc2_count = [x.shape[0] for x in sc2_label]
    sc3_label,o3,s3=remove_anom_overlap(anom_nodes3,None,None)
    sc3_count = [x.shape[0] for x in sc3_label]
    return sc1_label,sc2_label,sc3_label,sc1_count,sc2_count,sc3_count

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
    #dend
    res=[1.5,0.8,0.1]
    sc1_label,sc2_label,sc3_label,sc1_count,sc2_count,sc3_count = run_dend(graph,res)

    conns_1=np.array(check_conn(sc1_label)).nonzero()[0]
    conns_2=np.array(check_conn(sc2_label)).nonzero()[0]
    conns_3=np.array(check_conn(sc3_label)).nonzero()[0]
    
    sc1_label = np.array(sc1_label)[conns_1] if len(conns_1) > 0 else []
    sc2_label = np.array(sc2_label)[conns_2] if len(conns_2) > 0 else []
    sc3_label = np.array(sc3_label)[conns_3] if len(conns_3) > 0 else []
    print([i.shape[0] for i in sc1_label],[i.shape[0] for i in sc2_label],[i.shape[0] for i in sc3_label])
    if 'elliptic' in args.dataset:
        return sc3_label,sc2_label,sc1_label
    clust1 = getScaleClusts(dend,1)
    clust2 = getScaleClusts(dend,2)
    clust3 = getScaleClusts(dend,3)
    
    #print(np.unique(clust1).shape,np.unique(clust2).shape,np.unique(clust3).shape)
    clust1_dict,anoms1,nodes1 = getAnomCount(clust1,anom)
    thresh = args.thresh
    anoms_find1=anoms1[np.where(anoms1/nodes1 > thresh)[0]]
    nodes_find1=anoms1[np.where(anoms1/nodes1 > thresh)[0]]
    #import ipdb ; ipdb.set_trace()
    anom_nodes1=[np.intersect1d(clust1_dict[x],anom) for x in clust1_dict.keys() if (x in np.where(anoms1/nodes1 > thresh)[0] and clust1_dict[x].shape[0]>=3)]
    clust2_dict,anoms2,nodes2 = getAnomCount(clust2,anom)
    anom_nodes2=[np.intersect1d(clust2_dict[x],anom) for x in clust2_dict.keys() if (x in np.where(anoms2/nodes2 > thresh)[0] and clust2_dict[x].shape[0]>=3)]
    clust3_dict,anoms3,nodes3 = getAnomCount(clust3,anom)
    anom_nodes3=[np.intersect1d(clust3_dict[x],anom) for x in clust3_dict.keys() if (x in np.where(anoms3/nodes3 > thresh)[0] and clust3_dict[x].shape[0]>=3)]
    sc1_label,o1,s1=remove_anom_overlap(anom_nodes1,anom_nodes3,anom_nodes2)
    sc1_count = [x.shape[0] for x in sc1_label]
    sc2_label,o2,s2=remove_anom_overlap(anom_nodes2,anom_nodes3,None)
    sc2_count = [x.shape[0] for x in sc2_label]
    sc3_label,o3,s3=remove_anom_overlap(anom_nodes3,None,None)
    sc3_count = [x.shape[0] for x in sc3_label]


    conns_1=np.array(check_conn(sc1_label)).nonzero()[0]
    conns_2=np.array(check_conn(sc2_label)).nonzero()[0]
    conns_3=np.array(check_conn(sc3_label)).nonzero()[0]
    
    sc1_label = np.array(sc1_label)[conns_1] if len(conns_1) > 0 else []
    sc2_label = np.array(sc2_label)[conns_2] if len(conns_2) > 0 else []
    sc3_label = np.array(sc3_label)[conns_3] if len(conns_3) > 0 else []
    print([i.shape[0] for i in sc1_label],[i.shape[0] for i in sc2_label],[i.shape[0] for i in sc3_label])
    import ipdb; ipdb.set_trace()
    return sc1_label,sc2_label,sc3_label


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
    import ipdb ; ipdb.set_trace()
    comms = nx.community.louvain_communities(nx_graph)
    norm_comms = [list(i) for i in comms if np.intersect1d(list(i),anom).shape[0] == 0]
    cuts = []
    for comm in norm_comms:
        if nx.cut_size(nx_graph,comm) > 0:
            cuts.append(nx_graph.subgraph(comm).number_of_edges()/nx.cut_size(nx_graph,comm))
    print('norm cuts',cuts)
    for a_ind,a in enumerate(ms_anoms):
        cuts = []
        for comm in a:
            if nx.cut_size(nx_graph,comm) > 0:
                cuts.append(nx_graph.subgraph(comm).number_of_edges()/nx.cut_size(nx_graph,comm))
        print('cuts for',a_ind,cuts)

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
sio.savemat(f'msgad/data/{args.dataset}.mat', mat_file)