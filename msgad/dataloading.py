
import pickle as pkl
import scipy.io as sio
import os
import dgl
import numpy as np
import torch
import networkx as nx
from utils import *

class DataLoading:
    def __init__(self,exp_params):
        self.dataset = exp_params['DATASET']['NAME']
        self.batch_type = exp_params['DATASET']['BATCH_TYPE']
        self.batch_size = int(exp_params['DATASET']['BATCH_SIZE'])
        self.epoch = exp_params['MODEL']['EPOCH']
        self.device = exp_params['DEVICE']
        self.datadir = exp_params['DATASET']['DATADIR']
        self.dataload = exp_params['DATASET']['DATALOAD']
        self.datasave = exp_params['DATASET']['DATASAVE']
        self.exp_name = exp_params['EXP']
        self.num_neighbors = int(exp_params['DATASET']['NUM_NEIGHBORS_SAMP'])

    def load_anomaly_detection_dataset(self):
        """Load anomaly detection graph dataset for model training & anomaly detection"""
        data_mat = sio.loadmat(f'data/{self.dataset}.mat')
        if 'cora' in self.dataset or 'yelp' in self.dataset:
            feats = torch.FloatTensor(data_mat['Attributes'].toarray())
        else:
            feats = torch.FloatTensor(data_mat['Attributes'])
        adj,edge_idx=None,None
        if 'Edge-index' in data_mat.keys():
            edge_idx = data_mat['Edge-index']
        elif 'Network' in data_mat.keys():
            adj = data_mat['Network']
        truth = data_mat['Label'].flatten()
        return adj, edge_idx, feats, truth

    def fetch_dataloader(self, adj, neg_adj, pos_edges_full,ind):
        """
        Prepare DGL dataloaders given DGL graph

        Input:
            adj : {DGL graph}
                Input graph
        """
        if self.dataload:
            return np.arange(len(os.listdir(f'{self.datadir}/{self.exp_name}/{self.dataset}/train')))
        if self.batch_type == 'edge':
            edge_weights = adj.edata['w'].detach().cpu() ; adj = adj.cpu() ; adj_nodes = adj.nodes().detach().cpu()
            transform = dgl.transforms.AddSelfLoop()
            dgl.distributed.initialize(f'{self.exp_name}-graph')
            part_g=dgl.distributed.partition_graph(adj.to('cpu'), f'{self.exp_name}-graph_{ind}', 1, num_hops=1, part_method='metis',out_path='output/')
            dist_g = dgl.distributed.DistGraph(f'{self.exp_name}-graph_{ind}', part_config=f'output/{self.exp_name}-graph_{ind}.json')
            neg_g=dgl.distributed.partition_graph(neg_adj.to('cpu'), f'neg-{self.exp_name}-graph{ind}', 1, num_hops=1, part_method='metis',out_path='output/')
            neg_g = dgl.distributed.DistGraph(f'neg-{self.exp_name}-graph{ind}', part_config=f'output/neg-{self.exp_name}-graph{ind}.json')
            
            def sample_(seeds):
                seeds = torch.LongTensor(np.asarray(seeds))
                #frontier = dgl.sampling.sample_neighbors(adj, adj_nodes, 10, exclude_edges=adj.edges('eid')[:adj.number_of_edges()//2])
                frontier = dgl.distributed.sample_neighbors(dist_g, seeds, self.num_neighbors)
                block = dgl.to_block(frontier, seeds)
                batch_nodes = block.ndata['_ID']['_N']
                
                #pos_edges_samp = batch_nodes[torch.stack(block.edges()).T]
                pos_edges_samp = torch.stack(block.edges()).T
                if 'elliptic' in self.dataset:
                    neg_frontier = dgl.distributed.sample_neighbors(neg_g, seeds, self.num_neighbors//2)
                else:
                    neg_frontier = dgl.distributed.sample_neighbors(neg_g, seeds, self.num_neighbors)
                neg_block = dgl.to_block(neg_frontier, seeds)
                neg_edges_samp = torch.stack(neg_block.edges()).T
                neg_batch_nodes = neg_block.ndata['_ID']['_N']
                assert(torch.where(neg_edges_samp[:,0]==neg_edges_samp[:,1])[0].shape[0]==0)

                # Create boolean masks for both edge lists
                subsampled_mask = torch.zeros(len(pos_edges_full), dtype=torch.bool)
                subsampled_indices = torch.arange(len(pos_edges_samp))
                subsampled_mask[subsampled_indices] = 1
                full_mask = torch.zeros(len(pos_edges_full), dtype=torch.bool)
                full_indices = torch.arange(len(pos_edges_samp))
                full_mask[full_indices] = 1
                indices_in_full = torch.nonzero(subsampled_mask & full_mask).squeeze()
                #neg_edges_samp = neg_edges_full[indices_in_full]
                block.edata['w'] = edge_weights[indices_in_full]
                # Find the indices of the subsampled edge list in the original edge list
                return block, batch_nodes[pos_edges_samp], neg_batch_nodes[neg_edges_samp], batch_nodes
            
            batch_size = adj.number_of_nodes() if self.batch_size == 0 else int(adj.number_of_nodes()/self.batch_size)
            dataloader = dgl.distributed.DistDataLoader(dataset=adj.nodes(), batch_size=batch_size,collate_fn=sample_, shuffle=False)
        return dataloader

    def save_batch(self,loaded_input,iter,setting):
        """
        Save batch to pickle file

        Input:
            loaded_input : {array-like}
                DGL dataloading contents
            iter: {int}
                Batch ID
            setting: {str}
                Train/test
        """
        #loaded_input[0] = loaded_input[0].to_sparse()
        dirpath = f'{self.datadir}/{self.exp_name}/{self.dataset}/{setting}'
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        with open (f'{dirpath}/{iter}.pkl','wb') as fout:
            pkl.dump({'loaded_input':loaded_input},fout)

        torch.cuda.empty_cache()

    def load_batch(self,iter,setting):
        """
        Load batch from pickle file

        Input:
            iter : {int}
                Batch ID
            setting : {str}
                Train/test
        Output:
            recons: {array-like, torch tensor}, shape=[scales,n,n]
                Multi-scale adjacency reconstructions
            h: {array-like, torch tensor}, shape=[scales,n,h']
                Multi-scale embeddings produced by model
        """
        dirpath = f'{self.datadir}/{self.exp_name}/{self.dataset}/{setting}'
        with open (f'{dirpath}/{iter}.pkl','rb') as fin:
            batch_dict = pkl.load(fin)
        loaded_input = batch_dict['loaded_input']
        return loaded_input