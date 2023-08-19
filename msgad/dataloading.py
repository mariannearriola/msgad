
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
            dgl.distributed.initialize('graph-name')
            part_g=dgl.distributed.partition_graph(adj.to('cpu'), f'graph_{ind}', 1, num_hops=1, part_method='metis',out_path='output/')
            dist_g = dgl.distributed.DistGraph(f'graph_{ind}', part_config=f'output/graph_{ind}.json')
            neg_g=dgl.distributed.partition_graph(neg_adj.to('cpu'), f'neg_graph_{ind}', 1, num_hops=1, part_method='metis',out_path='output/')
            neg_g = dgl.distributed.DistGraph(f'neg_graph_{ind}', part_config=f'output/neg_graph_{ind}.json')
            
            def sample_(seeds):
                seeds = torch.LongTensor(np.asarray(seeds))
                #frontier = dgl.sampling.sample_neighbors(adj, adj_nodes, 10, exclude_edges=adj.edges('eid')[:adj.number_of_edges()//2])
                frontier = dgl.distributed.sample_neighbors(dist_g, seeds, 10)
                block = dgl.to_block(frontier, seeds)
                batch_nodes = block.ndata['_ID']['_N']
                
                #pos_edges_samp = batch_nodes[torch.stack(block.edges()).T]
                pos_edges_samp = torch.stack(block.edges()).T

                neg_frontier = dgl.distributed.sample_neighbors(neg_g, seeds, 10)
                neg_block = dgl.to_block(neg_frontier, seeds)
                neg_edges_samp = torch.stack(neg_block.edges()).T
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
                return block, batch_nodes[pos_edges_samp], batch_nodes[neg_edges_samp], batch_nodes
            
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
'''
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

    def fetch_dataloader(self, adj, pos_edges_full):
        """
        Prepare DGL dataloader given DGL graph

        Input:
            adj : {DGL graph}
                Input graph
        """
        if self.dataload:
            return np.arange(len(os.listdir(f'{self.datadir}/{self.exp_name}/{self.dataset}/train')))
        if self.batch_type == 'edge':
            if True:
                num_neighbors = 10 # NUM CONTRASTIVE PAIRS ?!
                sampler = dgl.dataloading.NeighborSampler([num_neighbors])
                #sampler = dgl.dataloading.NeighborSampler([num_neighbors,num_neighbors,num_neighbors])
            else:
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)

            neg_sampler = dgl.dataloading.negative_sampler.GlobalUniform(1,exclude_self_loops=True)
            #reverse_eids = torch.cat([torch.arange(int(adj.number_of_edges()/2),  int(adj.number_of_edges())), torch.arange(0, int(adj.number_of_edges()/2))]).to(adj.device)
            #sampler = dgl.dataloading.as_edge_prediction_sampler(sampler,negative_sampler=neg_sampler)#,exclude='reverse_id',reverse_eids=reverse_eids)
            edges=adj.edges('eid') ; edge_weights = adj.edata['w'].detach().cpu() ; adj_nodes = adj.nodes().detach().cpu() ; adj = adj.cpu()
            batch_size = self.batch_size if self.batch_size > 0 else int(adj.number_of_nodes()*num_neighbors)#int(adj.number_of_edges())
            transform = dgl.transforms.AddSelfLoop() ; adj_loop = transform(adj)
            #if self.device == 'cuda':
            if True:
                #import ipdb ; ipdb.set_trace()
                
                dgl.distributed.initialize('graph-name')
                
                part_g=dgl.distributed.partition_graph(adj.to('cpu'), 'graph_name', 1, num_hops=1, part_method='metis',out_path='output/')
                dist_g = dgl.distributed.DistGraph('graph_name', part_config='output/graph_name.json')
                neg_sampler = dgl.dataloading.negative_sampler.Uniform(10)
                def sample_(seeds):
                    seeds = torch.LongTensor(np.asarray(seeds))
                    frontier = dgl.distributed.sample_neighbors(dist_g, adj_nodes, 10)
                    block = dgl.to_block(frontier, seeds)
                    pos_edges_samp = torch.stack(block.edges()).T
                    # NOTE: number of edges/non edges does not need to be equal
                    neg_edges_samp = torch.stack(neg_sampler(adj_loop, adj_nodes)).T
                    neg_edges_samp = neg_edges_samp[(adj_loop.has_edges_between(neg_edges_samp[:,0],neg_edges_samp[:,1])==0).nonzero()[:,0]]
                    neg_graph = dgl.graph((neg_edges_samp[:,0],neg_edges_samp[:,1]),num_nodes=adj_nodes.shape[0])
                    neg_edges_samp = torch.stack(dgl.sampling.sample_neighbors(neg_graph,adj_nodes,10).edges()).T

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
                    
                    return block, pos_edges_samp, neg_edges_samp, None
             
                dataloader = dgl.distributed.DistDataLoader(dataset=adj.nodes(), batch_size=adj.number_of_nodes(),collate_fn=sample_, shuffle=False)
                
                #dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, device=self.device)
            else:
                dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6, device=self.device)
        elif self.batch_type == 'node':
            batch_size = self.batch_size if self.batch_size > 0 else int(adj.number_of_nodes())
            #sampler = dgl.dataloading.SAINTSampler(mode='walk',budget=[int(batch_size/3),batch_size])
            sampler = dgl.dataloading.ShaDowKHopSampler([4])
            if self.device == 'cuda':
                num_workers = 0
            else:
                num_workers = 4
            dataloader = dgl.dataloading.DataLoader(adj, adj.nodes(), sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, device=self.device)

        return dataloader

    def get_edge_batch(self,loaded_input,sc_label):
        """
        Organize and prepare batched info across scales for model

        Input:
            loaded_input : {array-like}
                DGL dataloading contents
            sc_label : {array-like}, shape=[anoms]
        """
        g_batches,pos_edges_tot,neg_edges_tot = [],[],[]
        for sc,loaded_in in enumerate(loaded_input):
            in_nodes, sub_graph_pos, sub_graph_neg, block = loaded_in
            pos_edges = sub_graph_pos.edges()
            neg_edges = sub_graph_neg.edges()
            pos_edges = torch.vstack((pos_edges[0],pos_edges[1])).T
            neg_edges = torch.vstack((neg_edges[0],neg_edges[1])).T
            g_batch = block[-1]#[0]
            if self.datasave: g_batch = block[0]
            w = g_batch.edata['w']
            feat = g_batch.ndata['feature']
            
            g_adj = g_batch.adjacency_matrix().to_dense()[torch.argsort(in_nodes[g_batch.dstnodes()])][:,torch.argsort(in_nodes[g_batch.dstnodes()])]
            src,dst=g_adj.nonzero()[:,0],g_adj.nonzero()[:,1]
            g_batch = dgl.graph((src,dst),num_nodes=in_nodes.shape[0]).to(g_batch.device)
            g_batch.edata['w'] = w
            g_batch.ndata['feature']=feat['_N']#[in_nodes[g_batch.dstnodes()]]
            batch_sc_label = sc_label
            
            in_nodes_tot = in_nodes.unsqueeze(0) if sc == 0 else torch.cat((in_nodes_tot,in_nodes.unsqueeze(0)),dim=0)
            pos_edges_tot.append(in_nodes[pos_edges])
            neg_edges_tot.append(in_nodes[neg_edges])
            g_batches.append(g_batch)
            batch_sc_labels = torch.tensor(batch_sc_label).unsqueeze(0) if sc == 0 else torch.cat((batch_sc_labels,torch.tensor(batch_sc_label).unsqueeze(0)))
        return in_nodes_tot, pos_edges_tot, neg_edges_tot, g_batches, batch_sc_labels

    def save_batch(self,loaded_input,lbl,iter,setting):
        """
        Save batch to pickle file

        Input:
            loaded_input : {array-like}
                DGL dataloading contents
            lbl : {array-like}, shape=[scales]
                Reconstruction labels (DGL graphs)
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
            pkl.dump({'loaded_input':loaded_input,'label':np.arange(5)},fout)

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
        lbl = batch_dict['label']
        #loaded_input[0] = loaded_input[0]#.to_dense()
        lbl_ = []
        for l in lbl:
            lbl_.append(l)#.to_dense().detach().cpu())
            del l ; torch.cuda.empty_cache()
        #lbl = [l.to_dense() for l in lbl]
        del lbl ; torch.cuda.empty_cache()
        return loaded_input,lbl_
'''