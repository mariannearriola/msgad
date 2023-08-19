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
            self.linear_transform_in.apply(init_weights)
            self.module_list = nn.ModuleList()
            for i in range(3):
                self.module_list.append(nn.Linear(in_size*(self.k+1), int(self.hidden_dim)))
        
        elif self.model_str == 'dominant':
            self.conv = DOMINANT_Base(in_size,self.hidden_dim,3,dropout,act)
        elif self.model_str == 'multi-scale-dominant':
            self.module_list = nn.ModuleList()
            for i in range(3):
                self.module_list.append(DOMINANT_Base(in_size,self.hidden_dim,3,dropout,act))
        else:
            raise('model not found')

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        
    def forward(self,graph,edges,feats,edge_ids):
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
        
        from utils import check_gpu_usage
        emb,recons_a = None,None
        check_gpu_usage('about to run')
        if self.model_str=='multi-scale-dominant':
            recons_a=[]
            for ind in range(self.scales):
                res = self.module_list[ind](feats,edges[ind].T)
                recons_a.append(self.decode_act(res)[0,edge_ids[ind][:,0],edge_ids[ind][:,1]])#.unsqueeze(0))
        
        elif self.model_str in ['dominant']: #x, e
            recons_a,emb=[],[]
            for ind in range(self.scales):
                res = self.conv(feats,edges[ind].T)
                emb.append(res)
                recons_a.append(self.decode_act(res)[0,edge_ids[ind][:,0],edge_ids[ind][:,1]])#.unsqueeze(0))
                del res ;  torch.cuda.empty_cache()

        if 'multi-scale-amnet' in self.model_str or 'multi-scale-bwgnn' in self.model_str: # g
            recons_a,labels,emb = [],[],[]
            feats = self.linear_transform_in(feats)
            check_gpu_usage('about to run model')
            for ind in range(self.scales):
                check_gpu_usage('before conv')
                h = self.conv(feats,edges[ind],None)[:,0,:]
                check_gpu_usage('after conv')
                h = self.module_list[ind](h)

                # collect results
                hs = h.unsqueeze(0) if ind == 0 else torch.cat((hs,h.unsqueeze(0)),dim=0)
                del h ; torch.cuda.empty_cache() ; gc.collect()
            
            self.final_attn = self.attn
            
            check_gpu_usage('before bmm')
            hs_t=torch.transpose(hs,1,2)
            if 'elliptic' in self.dataset:
                hs_s = hs[0].to_sparse()
                hs_t_s = hs_t[0].to_sparse()
                prod=torch.sparse.mm(hs_s,hs_t_s)
            recons = torch.bmm(hs,hs_t)

            del hs_t ; torch.cuda.empty_cache() ; gc.collect()
            check_gpu_usage('after bmm')
            
            # TODO: check if this can be changed via list comprehension: may be an issue
            for i in range(self.scales):
                if i == 0:
                    recons_f = [torch.sigmoid(recons[i,edge_ids[i][:,0],edge_ids[i][:,1]]) for i in range(self.scales)]
                else:
                    recons_f.append(torch.sigmoid(recons[i,edge_ids[i][:,0],edge_ids[i][:,1]]))
            del recons; torch.cuda.empty_cache() ; gc.collect()
            check_gpu_usage('results collected')
            return recons_f,hs
            
        check_gpu_usage('collecting')
        torch.cuda.empty_cache()
        if self.model_str not in ['multi-scale','multi-scale-amnet','multi-scale-bwgnn','bwgnn','gradate','dominant','multi-scale-dominant']:
            recons_a = [recons[0]]
        
        return recons_a,emb
