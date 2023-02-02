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

def load_citation_datadet(dataset_str):
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
    '''
    l_comms = nx_comm.louvain_communities(graph)
    l_comms_arr = []
    for l_comm in l_comms:
        l_comms_arr.append(list(l_comm))

    l_comms_arr_2 = []
    for l_comm in l_comms_arr:
        if len(l_comm) == 1:
            l_comms_arr_2.append(l_comm)
        else:
            comms_found=False
            to_app = []
            l_comm_to_append = nx_comm.louvain_communities(graph.subgraph(l_comm))
            for l_comm_ in l_comm_to_append:
                if len(l_comm_) > 1:
                    comms_found = True
                to_app.append(list(l_comm_))
                if comms_found:
                    l_comms_arr_2.append(to_app)
                else:
                    l_comms_arr_2.append(l_comm_)


    l_comms_arr_3 = []
    for l_comm in l_comms_arr_2:
        if len(l_comm) == 1:
            l_comms_arr_3.append(l_comm)
        else:
            comms_found=False
            to_app = []
            if isinstance(l_comm,str):
                l_comm_to_append = nx_comm.louvain_communities(graph.subgraph(l_comm))
                for l_comm_ in l_comm_to_append:
                    if len(l_comm_) > 1:
                        comms_found = True
                    to_app.append(list(l_comm_))
                    if comms_found:
                        l_comms_arr_3.append(to_app)
                    else:
                        l_comms_arr_3.append(l_comm_)
            else:
                l_comm_to_append = []
                
                for l_comm_inside in l_comm:
                    if len(l_comm_inside) == 1:
                        l_comm_to_append.append(l_comm_inside)
                        continue
                    try:
                        l_comm_inner = nx_comm.louvain_communities(graph.subgraph(l_comm_inside))
                    except:
                        import ipdb ; ipdb.set_trace()
                    for l_comm_ in l_comm_inner:
                        if len(l_comm_) > 1:
                            comms_found = True
                        to_app.append(list(l_comm_))
                        if comms_found:
                            l_comm_to_append.append(to_app)
                        else:
                            l_comm_to_append.append(l_comm_)

                l_comms_arr_3.append(l_comm_to_append)
    '''

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
seed = 5
m = 15  #num of fully connected nodes  #10 15 20   5 (clique size)
k = 50
s = 9 # number of scales
#scale_sizes = np.array([5,15,45])
#n = np.array([2,1,1])
scale=1
num_clust=5
size=10
prob_connect=0.93
prob_connect=0.75
prob_connect=0.05
prob_connects=[0.3,0.43,0.925]
prob_connect=prob_connects[scale-1]
#prob_connect=0.98
scale_sizes = np.array([10,50,150])
scale_sizes = np.full((num_clust,),size)
n = np.array([30,6,2])
n= np.array([5,2,2])
n = np.full((num_clust,),1)
#attr_scales = np.array([3,1,1,1])
#attr_scales = np.array([1,1,1])
attr_scales = np.full((num_clust,),1)
#probs = np.array([0.2,0.87,0.9335])
#probs = np.array([0.05,0.3,0.5])
#probs = np.array([0.5,0.5,0.5])
#probs = np.full((num_clust,),prob_connect)
#prob = np.array([0.1,0.1,0.1])
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
    attribute_dense, adj_dense, cat_labels = load_citation_datadet(dataset_str)
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
#anomaly_idx = all_idx[:m*n]
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
            anom_sps.append(sum(i_sps)/len(i_sps))
        print('AVG SHORTEST PATHS',sum(anom_sps)/len(anom_sps))
        
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
    all_anom_sc.append(anom_sc)
print('Done. {:d} attributed nodes are constructed. \n'.format(len(attribute_anomaly_idx)))

# Pack & save them into .matip
print('Saving mat file...')
attribute = dense_to_sparse(attribute_dense)
adj = dense_to_sparse(adj_dense)

#savedir = 'anom_data/'
#savedir = './pygod/pygod/data'
savedir = './ms_dominant/data'
#savedir = './dominant/GCN_AnomalyDetection_pytorch/data'
#savedir = './pygsp-master/pygsp/data/ms_data'
if not os.path.exists(savedir):
    os.makedirs(savedir)
#import ipdb ; ipdb.set_trace()
sio.savemat('{}/{}_triple_sc{}.mat'.format(savedir,dataset_str,str(scale)),\
            {'Network': adj, 'Label': label, 'Attributes': attribute,\
            'Class':cat_labels, 'str_anomaly_label':str_anomaly_label, 'attr_anomaly_label':attr_anomaly_label, 'scale_anomaly_label': all_anom_sc, 'l_comms': l_comms})
print('Done. The file is save as: anom_data/{}.mat \n'.format(dataset_str))
