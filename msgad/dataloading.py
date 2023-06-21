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

    def rearrange_anoms(self,anom):
        ret_anom = []
        for anom_ in anom:
            ret_anom.append(anom_[0])
        return ret_anom

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
        if 1 in truth:
            anom1,anom2,anom3,anom_single=data_mat['anom_sc1'],data_mat['anom_sc2'],data_mat['anom_sc3'],data_mat['anom_single']
            if 'yelpchi' in self.dataset:
                anom1=self.rearrange_anoms(anom1) ; anom2=self.rearrange_anoms(anom2) ; anom3=self.rearrange_anoms(anom3) ; anom_single = anom_single[0]
            if 'weibo' in self.dataset:
                anom1=self.rearrange_anoms(anom1[0]) ; anom2=self.rearrange_anoms(anom2[0]) ; anom3=self.rearrange_anoms(anom3[0]) ; anom_single = anom_single[0]
            if 'tfinance' in self.dataset:
                anom3 = self.rearrange_anoms(anom3[0]) ; anom_single = anom_single[0]
            #import ipdb ; ipdb.set_trace()
            sc_label=[anom1,anom2,anom3,anom_single]
        else:
            sc_label = []
            
        return adj, edge_idx, feats, truth, sc_label

    def fetch_dataloader(self, adj, edges):
        if self.dataload:
            return np.arange(len(os.listdir(f'{self.datadir}/{self.exp_name}/{self.dataset}/train')))
        if self.batch_type == 'edge':
            if 'tfinance' in self.dataset:
                num_neighbors = 10
                sampler = dgl.dataloading.NeighborSampler([num_neighbors,num_neighbors,num_neighbors])
            else:
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)

            neg_sampler = dgl.dataloading.negative_sampler.GlobalUniform(1)
            sampler = dgl.dataloading.as_edge_prediction_sampler(sampler,negative_sampler=neg_sampler)
            edges=adj.edges('eid')
            batch_size = self.batch_size if self.batch_size > 0 else int(adj.number_of_edges())
            if self.device == 'cuda':
                dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, device=self.device)
            else:
                dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6, device=self.device)
        elif self.batch_type == 'node':
            batch_size = self.batch_size if self.batch_size > 0 else int(adj.number_of_nodes()/100)
            #sampler = dgl.dataloading.SAINTSampler(mode='walk',budget=[int(batch_size/3),batch_size])
            sampler = dgl.dataloading.ShaDowKHopSampler([4])
            if self.device == 'cuda':
                num_workers = 0
            else:
                num_workers = 4
            dataloader = dgl.dataloading.DataLoader(adj, adj.nodes(), sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, device=self.device)

        return dataloader

    def get_batch_sc_label(self,in_nodes,sc_label,g_batch):
        #dict_={k.item():v for k,v in zip(np.argsort(in_nodes),in_nodes)}
        dict_={k.item():v for k,v in zip(np.arange(in_nodes.shape[0]),in_nodes)}
        batch_sc_label = {}
        batch_sc_label_keys = ['anom_sc1','anom_sc2','anom_sc3','single']
        in_nodes_ = in_nodes.detach().cpu().numpy()
        for sc_ind,sc_ in enumerate(sc_label):
            if batch_sc_label_keys[sc_ind] != 'single':
                scs_comb = []
                for sc__ in sc_:
                    scs_comb.append(np.vectorize(dict_.get)(sc__))
                batch_sc_label[batch_sc_label_keys[sc_ind]] = scs_comb
            else:
                batch_sc_label[batch_sc_label_keys[sc_ind]]=np.vectorize(dict_.get)(sc_)

        return batch_sc_label

    def get_edge_batch(self,loaded_input,sc_label):
        in_nodes, sub_graph_pos, sub_graph_neg, block = loaded_input
        pos_edges = sub_graph_pos.edges()
        neg_edges = sub_graph_neg.edges()
        pos_edges = torch.vstack((pos_edges[0],pos_edges[1])).T
        neg_edges = torch.vstack((neg_edges[0],neg_edges[1])).T
        last_batch_node = torch.max(neg_edges)
        #last_batch_node = torch.max(pos_edges)
        g_batch = block
        if self.datasave: g_batch = block[0]
        w = g_batch.edata['w']
        feat = g_batch.ndata['feature']
        g_adj = g_batch.adjacency_matrix().to_dense()[torch.argsort(in_nodes[g_batch.dstnodes()])][:,torch.argsort(in_nodes[g_batch.dstnodes()])]
        src,dst=g_adj.nonzero()[:,0],g_adj.nonzero()[:,1]
        g_batch = dgl.graph((src,dst)).to(g_batch.device)
        g_batch.edata['w'] = w
        #import ipdb ; ipdb.set_trace()
        g_batch.ndata['feature']=feat['_N']#[in_nodes[g_batch.dstnodes()]]
        batch_sc_label = get_sc_label(sc_label)

        return in_nodes, in_nodes[pos_edges], in_nodes[neg_edges], g_batch, last_batch_node, batch_sc_label

    def save_batch(self,loaded_input,lbl,iter,setting):
        loaded_input[0] = loaded_input[0].to_sparse()
        if self.batch_type == 'edge':
            loaded_input[-1] = loaded_input[-1][0]
        dirpath = f'{self.datadir}/{self.exp_name}/{self.dataset}/{setting}'
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        with open (f'{dirpath}/{iter}.pkl','wb') as fout:
            pkl.dump({'loaded_input':loaded_input,'label':[l for l in lbl]},fout)
        for i in range(len(loaded_input)):
            del loaded_input[0]
        torch.cuda.empty_cache()

    def load_batch(self,iter,setting):
        dirpath = f'{self.datadir}/{self.exp_name}/{self.dataset}/{setting}'
        with open (f'{dirpath}/{iter}.pkl','rb') as fin:
            batch_dict = pkl.load(fin)
        loaded_input = batch_dict['loaded_input']
        lbl = batch_dict['label']
        loaded_input[0] = loaded_input[0].to_dense()
        lbl_ = []
        for l in lbl:
            lbl_.append(l)#.to_dense().detach().cpu())
            del l ; torch.cuda.empty_cache()
        #lbl = [l.to_dense() for l in lbl]
        del lbl ; torch.cuda.empty_cache()
        return loaded_input,lbl_