import torch
import numpy as np
import gc
import os
from dgl.nn import EdgeWeightNorm
import torch_geometric
from torch_geometric.nn import MLP
from utils import *
from models.dominant import *
from models.anomalydae import *
from models.gcnae import *
from models.mlpae import *
from models.msgad import *
from models.bwgnn import *
from models.amnet import *
from models.amnet_ms import *
import gc
from models.gradate import *

class GraphReconstruction(nn.Module):
    def __init__(self, in_size, exp_params, act = nn.LeakyReLU(), label_type='single'):
        super(GraphReconstruction, self).__init__()
        self.scales = int(exp_params['MODEL']['SCALES'])
        self.d = int(exp_params['MODEL']['D'])
        self.k = int(exp_params['MODEL']['K'])
        self.dataset = exp_params['DATASET']['NAME']
        self.exp_name = exp_params['EXP']
        self.model_str = exp_params['MODEL']['NAME']
        self.hidden_dim = int(exp_params['MODEL']['HIDDEN_DIM'])
        dropout = 0
        self.decode_act = nn.Sigmoid()
        seed_everything(82)
        if 'multi-scale-amnet' in self.model_str:
            self.conv = AMNet_ms(in_size, self.hidden_dim, 2, self.k, self.d) # k, num filters
            self.act_fn = nn.ReLU()
            self.linear_transform_in = nn.Sequential(nn.Linear(in_size, int(self.hidden_dim*2)),
                                                self.act_fn,
                                                nn.Linear(int(self.hidden_dim*2), self.hidden_dim),
                                                self.act_fn,
                                                nn.Linear( self.hidden_dim, in_size)
                                                )
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform(m.weight)
                    m.bias.data.fill_(0.01)

            self.linear_transform_in.apply(init_weights)
            self.module_list = nn.ModuleList()
            for i in range(3):
                self.module_list.append(nn.Linear(in_size*(self.k+1), int(self.hidden_dim)))
        elif self.model_str == 'multi-scale-dominant':
            self.module_list = nn.ModuleList()
            for i in range(3):
                self.module_list.append(DOMINANT_Base(in_size,self.hidden_dim,3,dropout,act))
        elif self.model_str == 'multi-scale-anomalydae':
            self.module_list = nn.ModuleList()
            for i in range(3):
                self.module_list.append(AnomalyDAE_Base(in_size,int(self.hidden_dim*2),self.hidden_dim,dropout,act))
        else:
            raise('model not found')

    
        
    def forward(self,adj_edges,feats,batch_edges):
        """
        Obtain learned embeddings and corresponding graph reconstructions from
        the input graph.

        Input:
            edges : {array-like, torch tensor}, shape=[s,e,2]
                Edge list of graph
            feats : {array-like, torch tensor}, shape=[n,h]
                Feature matrix of graph
        Output:
            recons_a: {array-like, torch tensor}, shape=[scales,n,n]
                Multi-scale adjacency reconstructions
            emb: {array-like, torch tensor}, shape=[scales,n,h']
                Multi-scale embeddings produced by model
        """
        
        emb,recons_a = None,None
        if self.model_str=='multi-scale-dominant' or self.model_str=='multi-scale-anomalydae':
            recons_a=[]
            for ind in range(self.scales):
                res = self.module_list[ind](feats,adj_edges[ind].T,batch_edges[ind])
                recons = self.decode_act(res)
                recons_a.append(self.collect_batch_recons(recons,batch_edges[ind]))
                del res ; torch.cuda.empty_cache()
            del recons ; torch.cuda.empty_cache()

        if 'multi-scale-amnet' in self.model_str:
            recons_a,emb = [],[]
            feats = self.linear_transform_in(feats)
            for ind in range(self.scales):
                h = self.conv(feats,adj_edges[ind],None)[:,0,:]
                h = self.module_list[ind](h)

                # collect results
                hs = h.unsqueeze(0) if ind == 0 else torch.cat((hs,h.unsqueeze(0)),dim=0)
                del h ; torch.cuda.empty_cache() ; gc.collect()
                        
            hs_t=torch.transpose(hs,1,2)
            if 'elliptic' in self.dataset:
                hs_s = hs[0].to_sparse()
                hs_t_s = hs_t[0].to_sparse()
                recons=torch.sparse.mm(hs_s,hs_t_s)
            else:
                recons = torch.bmm(hs,hs_t)
            

            del hs_t ; torch.cuda.empty_cache() ; gc.collect()
            recons_f = [torch.sigmoid(self.collect_batch_recons(recons[i],batch_edges[i])) for i in range(self.scales)]
            del recons; torch.cuda.empty_cache() ; gc.collect()
            return recons_f,hs
            
        torch.cuda.empty_cache()
        if self.model_str not in ['multi-scale','multi-scale-amnet','multi-scale-bwgnn','bwgnn','multi-scale-anomalydae','dominant','multi-scale-dominant',]:
            recons_a = [recons[0]]
        
        return recons_a,emb

    def collect_batch_recons(self,recons,batch_edges):
        n_ids_tensor = torch.arange(batch_edges.unique().shape[0],device='cuda')
        # Create an array to store the mapping of unique edge IDs to n IDs
        unique_edge_ids = torch.unique(batch_edges)
        edge_id_to_n_id = torch.zeros(torch.max(unique_edge_ids) + 1, dtype=torch.int64, device='cuda')
        edge_id_to_n_id[unique_edge_ids] = n_ids_tensor

        # Relabel the edge list using the n IDs
        relabeled_edge_list = batch_edges.clone()
        relabeled_edge_list[:,0] = edge_id_to_n_id[relabeled_edge_list[:,0]]
        relabeled_edge_list[:,1] = edge_id_to_n_id[relabeled_edge_list[:,1]]
        
        recons = recons[relabeled_edge_list[:,0],relabeled_edge_list[:,1]]

        del relabeled_edge_list ; torch.cuda.empty_cache() ; gc.collect()
        return recons
