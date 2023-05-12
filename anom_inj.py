# citations (todo: fix)
# ComGA:Community-Aware Attributed Graph Anomaly Detection.
# Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning

import numpy as np
import scipy.sparse as sp
import random
import scipy.io as sio
import argparse
import pickle as pkl
import networkx as nx
import sys
import os
import os.path as osp
from sklearn import preprocessing
from scipy.spatial.distance import euclidean
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from itertools import chain
from sknetwork.hierarchy import Paris, postprocess, LouvainHierarchy, LouvainIteration

#import networkx.algorithms.community as nx_comm

def dense_to_sparse(dense_matrix):
    shape = dense_matrix.shape
    row = []
    col = []
    data = []
    for i, r in enumerate(dense_matrix):
        for j in np.where(r > 0)[0]:
            row.append(i)
            col.append(j)
            data.append(dense_matrix[i,j])

    sparse_matrix = sp.coo_matrix((data, (row, col)), shape=shape).tocsc()
    return sparse_matrix

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_citation_dataset(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("datasets/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("datasets/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    #import ipdb ; ipdb.set_trace()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj_dense = np.array(adj.todense(), dtype=np.float64)
    attribute_dense = np.array(features.todense(), dtype=np.float64)
    cat_labels = np.array(np.argmax(labels, axis = 1).reshape(-1,1), dtype=np.uint8)
    # TODO: get normal clusters
    return attribute_dense,adj_dense,labels
    graph = nx.from_scipy_sparse_matrix(adj)

    return attribute_dense, adj_dense, cat_labels, l_comms_arr


'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')  #'BlogCatalog'  'Flickr' 'cora'  'citeseer'  'pubmed' 
parser.add_argument('--seed', type=int, default=1)  #random seed
parser.add_argument('--m', type=int, default=15)  #num of fully connected nodes
parser.add_argument('--n', type=int)  
parser.add_argument('--k', type=int, default=50)  #num of clusters
args = parser.parse_args()
'''
l_comms=[]
AD_dataset_list = ['BlogCatalog', 'Flickr']
Citation_dataset_list = ['cora', 'citeseer', 'pubmed']

# Set hyperparameters of disturbing
dataset_str = "cora"  #'BlogCatalog'  'Flickr' 'cora'  'citeseer'  'pubmed'
seed = 10
m = 15  #num of fully connected nodes  #10 15 20   5 (clique size)
k = 50
s = 9 # number of scales
scale=3
size=50
prob_connect=0.93
prob_connect=0.75
prob_connect=0.05
#prob_connects=[0.3,0.43,0.925]
prob_connect = 0.3
#prob_connect=prob_connects[scale-1]
#prob_connect=0.98
scale_sizes = np.array([5,15,75,1])
#scale_sizes = np.full((num_clust,),size)
#n = np.array([30,6,2,1])
#n= np.array([5,2,2])
#n = np.full((num_clust,),1)
n = np.array([10,3,1,100])
#attr_scales = np.array([3,1,1,1])
#attr_scales = np.array([1,1,1,1])
attr_scales = np.full((np.sum(n),),1)


# Set seed
print('Random seed: {:d}. \n'.format(seed))
np.random.seed(seed)
random.seed(seed)


# Load data
print('Loading data: {}...'.format(dataset_str))
if dataset_str in AD_dataset_list:
    data = sio.loadmat('./datasets/{}/{}.mat'.format(dataset_str, dataset_str))
    attribute_dense = np.array(data['Attributes'].todense())
    attribute_dense = preprocessing.normalize(attribute_dense, axis=0)
    adj_dense = np.array(data['Network'].todense())
    cat_labels = data['Label']
elif dataset_str in Citation_dataset_list:
    attribute_dense, adj_dense, cat_labels = load_citation_dataset(dataset_str)
nx_graph = nx.from_numpy_matrix(adj_dense)
print(nx.is_connected(nx_graph))
ori_num_edge = np.sum(adj_dense)
num_node = adj_dense.shape[0]
print('Done. \n')

# Random pick anomaly nodes
all_idx = list(range(num_node))
random.shuffle(all_idx)
#anomaly_idx = all_idx[:m*n*2]

num_anom = n*scale_sizes

# ensures clique are separated
anomaly_idx = all_idx[:np.sum(num_anom)]

# select m*n structural anomalies and m*n attribute anomalies?
    # disjoint... need to ensure that anomalies and structure anomalies are combined
#structure_anomaly_idx = anomaly_idx[:m*n]
#attribute_anomaly_idx = anomaly_idx[m*n:]
structure_anomaly_idx = anomaly_idx
attribute_anomaly_idx = anomaly_idx
print('num nodes',num_node)
print('anom index',anomaly_idx)
label = np.zeros((num_node,1),dtype=np.uint8)
label[anomaly_idx,0] = 1

str_anomaly_label = np.zeros((num_node,1),dtype=np.uint8)
str_anomaly_label[structure_anomaly_idx,0] = 1
attr_anomaly_label = np.zeros((num_node,1),dtype=np.uint8)
attr_anomaly_label[attribute_anomaly_idx,0] = 1
print(np.all(attr_anomaly_label==str_anomaly_label))


# Disturb structure (dense subgraph)
    # connects node m with m nodes in structure anomaly index
print('Constructing structured anomaly nodes...')
for ind,n_ in enumerate(n):
    for ind2,n__ in enumerate(range(n_)):
        #current_nodes = structure_anomaly_idx[n_*m:(n_+1)*m]
        try:
            start_ind = n__*scale_sizes[ind]
        except Exception as e:
            import ipdb ; ipdb.set_trace()
            print(e)
        if ind != 0:
            start_ind += np.sum(num_anom[:ind])
        end_ind = start_ind+scale_sizes[ind]
        current_nodes = structure_anomaly_idx[start_ind:end_ind]
        except_nodes = np.setdiff1d(structure_anomaly_idx,current_nodes)
        normal_nodes = np.setxor1d(current_nodes,all_idx)
        
        #import ipdb ; ipdb.set_trace()
        #print(ind)
        print('for scale',scale_sizes[ind])
        print(current_nodes)
        # connect anom nodes into cluster
        for ind_,i in enumerate(current_nodes):

            for jind,j in enumerate(current_nodes):
                if jind == ind_:
                    continue
                #pass
                adj_dense[i,j]=0.
                adj_dense[j,i]=0.
                
                if np.random.rand() > prob_connect:#prob_connects[ind]:
                    adj_dense[i, j] = 1.
                    adj_dense[j, i] = 1.


            # normal_neighbors = np.setdiff1d(np.where(adj_dense[i] != 0)[0],structure_anomaly_idx)

            # disconnect from other anomalous cliques
            for jind,j in enumerate(except_nodes):
                adj_dense[i,j]=0.
                adj_dense[j,i]=0.
                '''
                if ind == 0 and jind > (except_nodes.shape[0]-6):
                    #import ipdb ; ipdb.set_trace()
                    adj_dense[i,j]=1.
                    adj_dense[j,i]=1.
                '''
                
            '''
            for jind_,j_ in enumerate(current_nodes):
                if jind_ > ind_:
                    break
                if np.random.rand() > .6:
                    adj_dense[i, j_] = 0.
                    adj_dense[j_, i] = 0
            '''
            
            
        # removes self loops
        #adj_dense[current_nodes,current_nodes] = 0.
        
        #nx_graph = nx.from_numpy_matrix(adj_dense[current_nodes,:][:,current_nodes])
        '''
        nx_graph = nx.from_numpy_matrix(adj_dense)
        
        print(nx.is_connected(nx_graph))
        shortest_paths = dict(nx.shortest_path_length(nx_graph))
        anom_sps = []
        for ind_,i in enumerate(current_nodes):
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
            if len(i_sps) == 0: continue
            else:
                anom_sps.append(sum(i_sps)/len(i_sps))
        if len(anom_sps) != 0:
            print('AVG SHORTEST PATHS',sum(anom_sps)/len(anom_sps))
        '''
        '''
        print(nx.is_connected(nx_graph))
        shortest_paths = dict(nx.shortest_path_length(nx_graph))
        anom_sps = []
        for ind,i in enumerate(normal_nodes):
            i_sps = []
            for jind,j in enumerate(normal_nodes):
                
                if i==j:continue
                #i_sps.append(shortest_paths[i][j])
                try:
                    i_sps.append(nx.shortest_path_length(nx_graph,source=i,target=j))
                except Exception as e:
                    continue
                    #import ipdb ; ipdb.set_trace()
                    #print(e)
                    #print('hi')
            if len(i_sps) == 0:
                continue
            anom_sps.append(sum(i_sps)/len(i_sps))
        print('AVG SHORTEST PATHS',sum(anom_sps)/len(anom_sps))
        '''
        #print(anom_sps)

num_add_edge = np.sum(adj_dense) - ori_num_edge
print('Done. {:d} structured nodes are constructed. ({:.0f} edges are added) \n'.format(len(structure_anomaly_idx),num_add_edge))



# TODO: find connectivity of normal clusters
nx_graph = nx.from_numpy_matrix(adj_dense)

'''
print(nx.is_connected(nx_graph))
shortest_paths = dict(nx.shortest_path_length(nx_graph))
normal_sps = []
for normal_comm in l_comms:
    if len(np.intersect1d(normal_comm,structure_anomaly_idx))>0:
        continue
    for ind, i in enumerate(normal_comm): 
        i_sps = []
        for jind,j in enumerate(normal_comm):
            if ind==jind:continue
            #i_sps.append(shortest_paths[i][j])
            try:
                i_sps.append(nx.shortest_path_length(nx_graph,source=i,target=j))
            except Exception as e:
                import ipdb ; ipdb.set_trace()
                print(e)
        try:
            normal_sps.append(sum(i_sps)/len(i_sps))
        except Exception as e:
            import ipdb ; ipdb.set_trace()
            print(e)
print('AVG NORMAL SHORTEST PATHS',sum(normal_sps)/len(normal_sps))

'''
# Disturb attribute
    # Every node in each clique is anomalous; no attribute anomaly scale (can be added)
print('Constructing attributed anomaly nodes...')
#for ind,i_ in enumerate(attribute_anomaly_idx):
all_anom_sc = []
selected_freq = {}
for ind,n_ in enumerate(n):
    anom_sc = []
    for ind2,n__ in enumerate(range(n_)):
        start_ind = n__*scale_sizes[ind]
        if ind != 0:
            start_ind += np.sum(num_anom[:ind])
        end_ind = start_ind+scale_sizes[ind]
        current_nodes = structure_anomaly_idx[start_ind:end_ind]
        anom_sc.append(current_nodes)
        for cur in current_nodes:

            picked_list = random.sample(all_idx, k)
            max_dist = 0
            # find attribute with greatest euclidian distance
            for j_ in picked_list:
                #cur_dist = euclidean(attribute_dense[i_],attribute_dense[j_])
                cur_dist = euclidean(attribute_dense[cur],attribute_dense[j_])
                if cur_dist > max_dist:
                    max_dist = cur_dist
                    #max_idx = j_
            
            anom_attr_scale = attr_scales[ind2+np.sum(n[:ind])]
            closest,closest_idx = 0,picked_list[0]
            # max_dist/3, max_dist/2, max_dist/1
            for j_ in picked_list:
                #cur_dist = euclidean(attribute_dense[i_],attribute_dense[j_])
                cur_dist = euclidean(attribute_dense[cur],attribute_dense[j_])
                if abs(cur_dist - max_dist/anom_attr_scale) < abs(closest - max_dist/anom_attr_scale):
                    closest = cur_dist
                    closest_idx = j_
            # copies attribute from node with highest euclidian distance of randomly sampled node
            #attribute_dense[i_] = attribute_dense[max_idx]
            attribute_dense[cur] = attribute_dense[closest_idx]
            if closest_idx in selected_freq.keys():
                selected_freq[closest_idx] += 1
            else:
                selected_freq[closest_idx] = 1
    all_anom_sc.append(anom_sc)

def getCuts(nx_graph,ms_anoms):
    comms = nx.community.louvain_communities(nx_graph)
    norm_comms = [list(i) for i in comms if np.intersect1d(list(i),anomaly_idx).shape[0] == 0]
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
getCuts(nx_graph,all_anom_sc)

print('Done. {:d} attributed nodes are constructed. \n'.format(len(attribute_anomaly_idx)))
res=[1.5,0.8,0.1]

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

def run_dend(graph,res):
    def flatten_label(anoms):
        anom_flat = anoms[0]
        if len(anoms) > 1:
            for i in anoms[1:]:
                anom_flat=np.concatenate((anom_flat,i))
        return anom_flat
    all_anom = None
    for anom_ind,anom in enumerate(all_anom_sc):
        if anom_ind == 3:
            anom_f = anom
        else:
            anom_f = flatten_label(anom)
        
        if all_anom is None: all_anom = anom_f
        else: all_anom = np.append(all_anom,anom_f)

    anom = all_anom
    paris = LouvainIteration()  # changed from iteration; wasn't forming connected subgraphs
    dend = paris.fit_predict(nx.adjacency_matrix(graph))
    clust1 = postprocess.cut_straight(dend,threshold=1)
    clust2 = postprocess.cut_straight(dend,threshold=2)
    clust3 = postprocess.cut_straight(dend,threshold=3)
    
    #print(np.unique(clust1).shape,np.unique(clust2).shape,np.unique(clust3).shape)
    clust1_dict,anoms1,nodes1 = getAnomCount(clust1,anom)
    thresh = 0.8
    anoms_find1=anoms1[np.where(anoms1/nodes1 > thresh)[0]]
    nodes_find1=anoms1[np.where(anoms1/nodes1 > thresh)[0]]
    #import ipdb ; ipdb.set_trace()
    anom_nodes1=[np.intersect1d(clust1_dict[x],anom) for x in clust1_dict.keys() if (x in np.where(anoms1/nodes1 > thresh)[0] and clust1_dict[x].shape[0]>=3)]
    clust2_dict,anoms2,nodes2 = getAnomCount(clust2,anom)
    anom_nodes2=[np.intersect1d(clust2_dict[x],anom) for x in clust2_dict.keys() if (x in np.where(anoms2/nodes2 > thresh)[0] and clust2_dict[x].shape[0]>=3)]
    clust3_dict,anoms3,nodes3 = getAnomCount(clust3,anom)
    anom_nodes3=[np.intersect1d(clust3_dict[x],anom) for x in clust3_dict.keys() if (x in np.where(anoms3/nodes3 > thresh)[0] and clust3_dict[x].shape[0]>=3)]
    anom_nodes_tot = [anom_nodes1,anom_nodes2,anom_nodes3]
    sc1_label = remove_anom_overlap(anom_nodes_tot,0,[2,1])
    sc2_label = remove_anom_overlap(anom_nodes_tot,1,[2])
    sc3_label = remove_anom_overlap(anom_nodes_tot,2,[])

    conns_1=np.array(check_conn(graph,sc1_label)).nonzero()[0]
    conns_2=np.array(check_conn(graph,sc2_label)).nonzero()[0]
    conns_3=np.array(check_conn(graph,sc3_label)).nonzero()[0]
    
    sc1_label = np.array(sc1_label)[conns_1] if len(conns_1) > 0 else []
    sc2_label = np.array(sc2_label)[conns_2] if len(conns_2) > 0 else []
    sc3_label = np.array(sc3_label)[conns_3] if len(conns_3) > 0 else []
    print([i.shape[0] for i in sc1_label],[i.shape[0] for i in sc2_label],[i.shape[0] for i in sc3_label])
   
    return sc1_label,sc2_label,sc3_label

def check_conn(graph,sc_label):
    conn_check=[]
    for i in sc_label:
        try:
            conn_check.append(nx.is_connected(graph.subgraph(i)))
        except:
            conn_check.append(False)
    return conn_check

def remove_anom_overlap(anom_nodes_tot,anom,anom_ex):
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

res=[1.5,0.8,0.1]
sc1_label,sc2_label,sc3_label = run_dend(nx.from_numpy_matrix(adj_dense),res)
# Pack & save them into .matip
print('Saving mat file...')
attribute = dense_to_sparse(attribute_dense)
adj = dense_to_sparse(adj_dense)

#savedir = './pygsp-master/pygsp/data/ms_data'
savedir = './msgad/data/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
import ipdb ; ipdb.set_trace()
sio.savemat('{}/{}_outsparse.mat'.format(savedir,dataset_str,str(scale)),\
            {'Network': adj, 'Label': label, 'Attributes': attribute,\
            'Class':cat_labels, 'str_anomaly_label':str_anomaly_label, 'attr_anomaly_label':attr_anomaly_label,
            'anom_sc1':all_anom_sc[0], 'anom_sc2':all_anom_sc[1], 'anom_sc3': all_anom_sc[2], 'anom_single': all_anom_sc[3],
            'l_comms': l_comms})
print('Done. The file is save as: anom_data/{}.mat \n'.format(dataset_str))
