import torch
import numpy as np
import gc
import os
from dgl.nn import EdgeWeightNorm
import torch_geometric
from label_generation import LabelGenerator
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
from models.gcad import *
from models.hogat import *
import gc
from models.gradate import *

class GraphReconstruction(nn.Module):
    def __init__(self, in_size, exp_params, act = nn.LeakyReLU(), label_type='single'):
        super(GraphReconstruction, self).__init__()
        seed_everything()
        self.in_size = in_size
        self.taus = [3,4,4]
        self.attn_weights = None
        self.b = 1
        self.norm_adj = torch_geometric.nn.conv.gcn_conv.gcn_norm
        self.norm = EdgeWeightNorm()
        self.d = int(exp_params['MODEL']['D'])
        self.dataset = exp_params['DATASET']['NAME']
        self.batch_type = exp_params['DATASET']['BATCH_TYPE']
        self.batch_size = exp_params['DATASET']['BATCH_SIZE']
        self.epoch = exp_params['MODEL']['EPOCH']
        self.exp_name = exp_params['EXP']
        self.label_type = exp_params['DATASET']['LABEL_TYPE']
        self.model_str = exp_params['MODEL']['NAME']
        self.recons = exp_params['MODEL']['RECONS']
        self.hidden_dim = int(exp_params['MODEL']['HIDDEN_DIM'])
        self.vis_filters = exp_params['VIS_FILTERS']

        self.e_adj, self.U_adj = None,None
        dropout = 0
        self.embed_act = act
        self.decode_act = torch.sigmoid
        self.weight_decay = 0.01
        self.lengths = [5,10,20]
        if 'multi-scale' in self.model_str:
            self.conv = AMNet_ms(in_size, self.hidden_dim, 1, 10, 5)
            self.conv_weight = nn.Parameter(data=torch.normal(mean=torch.full((3,5,10+1),0.),std=4).to(torch.float64)).requires_grad_(True)
            '''
            self.module_list = nn.ModuleList()
            for i in range(3):
                if 'multi-scale-amnet' == self.model_str:
                    self.module_list.append(AMNet_ms(in_size, self.hidden_dim, 1, 10, 5))
                elif 'multi-scale-bwgnn' == self.model_str:
                    self.module_list.append(MSGAD(in_size,self.hidden_dim,d=10))
            '''
        elif self.model_str == 'gradate': # NOTE: HAS A SPECIAL LOSS: OUTPUTS LOSS, NOT RECONS
            self.conv = GRADATE(in_size,self.hidden_dim,'prelu',1,1,'avg',5)
        elif self.model_str == 'bwgnn':
            self.conv = BWGNN(in_size, self.hidden_dim, d=10)
        elif self.model_str in ['anomalydae','anomaly_dae']: # x, e, batch_size (0 for no batching)
            self.conv = AnomalyDAE_Base(in_size,self.batch_size,self.hidden_dim,self.hidden_dim,dropout=dropout,act=act)
        elif self.model_str == 'dominant':
            self.conv = DOMINANT_Base(in_size,self.hidden_dim,3,dropout,act)
        elif self.model_str == 'mlpae': # x
            self.conv = MLP(in_channels=in_size,hidden_channels=self.hidden_dim,out_channels=self.batch_size,num_layers=3)
        elif self.model_str == 'amnet': # x, e
            self.conv = AMNet(in_size, self.hidden_dim, 2, 2, 2, vis_filters=self.vis_filters)
        elif self.model_str == 'ho-gat':
            self.conv = HOGAT(in_size, self.hidden_dim, dropout, alpha=0.1)
        else:
            raise('model not found')

        self.act_fn = nn.ReLU()
        self.linear_transform_in = nn.Sequential(nn.Linear(in_size, self.hidden_dim),
                                            self.act_fn,
                                            nn.Linear( self.hidden_dim, self.hidden_dim),
                                            )

        self.attn_fn = nn.Tanh()

        self.module_list = nn.ModuleList()
        for i in range(3):
            self.module_list.append(AttentionProjection(in_size,self.hidden_dim))

    def stationary_distribution(self, M, device):
        """
        Stationary distribution given the transition matrix.
        :param M: Transition matrix.
        :return: Stationary distribution.
        """

        # We solve (M^T - I) pi = 0 and 1 pi = 1. Combine them and let A = [M^T - I; 1], b = [0; 1]. We have A pi = b.
        n = M.shape[0]
        A = torch.cat([M.T - torch.eye(n).to(device),torch.ones((1,n)).to(device)],axis=0)
        b = torch.cat([torch.zeros(n),torch.tensor([1])],axis=0).to(device)

        # Solve A^T A pi = A^T pi instead (since A is not square).
        try:
            prod1= np.asarray((A.T @ A).cpu())
            prod2=np.asarray((A.T @ b).cpu())
            pi = np.linalg.solve(prod1,prod2)
        except Exception as e:
            import ipdb ; ipdb.set_trace()
            print(e)
        return pi

    def forward(self,edges,feats,vis=False,vis_name=""):
        """
        Obtain learned embeddings and corresponding graph reconstructions from
        the input graph.

        Input:
            edges : {array-like, torch tensor}, shape=[e,2]
                Edge list of graph
            feats : {array-like, torch tensor}, shape=[n,h]
                Feature matrix of graph
        Output:
            recons: {array-like, torch tensor}, shape=[3,n,n]
                Multi-scale adjacency reconstructions
            h: {array-like, torch tensor}, shape=[3,n,h']
                Multi-scale embeddings produced by model
        """
        from utils import check_gpu_usage
        res_a = None
        #edges, feats, graph_ = self.process_graph(graph)
        ''''
        if pos_edges is not None:
            all_edges = torch.vstack((pos_edges,neg_edges))
            #dst_nodes = torch.arange(last_batch_node+1)
        '''
        recons_x,recons_a=None,None
        if self.model_str in ['dominant','amnet']: #x, e
            recons = [self.conv(feats, edges, dst_nodes)]
        elif self.model_str in ['anomaly_dae','anomalydae']: #x, e, batch_size
            recons = [self.conv(feats, edges, 0,dst_nodes)]
        elif self.model_str in ['gradate']: # adjacency matrix
            loss, ano_score = self.conv(graph_, graph_.adjacency_matrix(), feats, False)
        if self.model_str == 'bwgnn':
            recons_a = [self.conv(graph_,feats,dst_nodes)]
        if 'multi-scale' in self.model_str: # g
            recons_a,labels,res_a = [],[],[]
            feats = self.linear_transform_in(feats)
            check_gpu_usage('about to run model')
            #edge_ids = torch.vstack((pos_edges,neg_edges)).to(pos_edges.device)

            for ind in range(len(self.module_list)):
                # pass input through filters
                h = self.conv(feats,edges,self.conv_weight[ind])#,dst_nodes)

                # collect attention scores
                attn_scores = self.module_list[ind](h,feats)

                if 'weibo' in self.dataset:
                    check_gpu_usage('model finished')

                # collect results
                if ind == 0:
                    score_sc = attn_scores.unsqueeze(0)
                    hs = h.unsqueeze(0)
                else:
                    score_sc = torch.cat((score_sc,attn_scores.unsqueeze(0)),dim=0)
                    hs = torch.cat((hs,h.unsqueeze(0)),dim=0)
                del attn_scores,h ; torch.cuda.empty_cache() ; gc.collect()
            
            # import ipdb ; ipdb.set_trace()
            # marginal_loss = score_sc.max(0)-score_sc.mean(0)
            self.attn_weights = F.softmax(score_sc,0).to(torch.float64) # scales x num filters x nodes
            check_gpu_usage('about to collect results')
            h_ =  (self.attn_weights.unsqueeze(-1)*hs).sum(2)
            check_gpu_usage(f'attns collected')
            #import ipdb ; ipdb.set_trace()
            #h_t = torch.transpose(h_,1,2)
            check_gpu_usage('after transpose')
            #res_a_t = torch.bmm(h_, h_t)
            recons_a=torch.sigmoid(torch.bmm(h_, torch.transpose(h_,1,2)))
            check_gpu_usage('after bmm')
            #recons_a = torch.sigmoid(h_t)
            del hs ; torch.cuda.empty_cache() ; gc.collect() 
            check_gpu_usage('results collected')
            
        
        # feature and structure reconstruction models
        torch.cuda.empty_cache()
        if self.model_str in ['anomalydae','dominant','ho-gat']:
            recons_x,recons_a = recons[0]
            if self.model_str == 'ho-gat':
                recons_ind = 0 if self.recons == 'feat' else 1
                return recons[0][recons_ind], recons[0][recons_ind+1], recons[0][recons_ind+2]
            recons_a = [recons_a]
            recons_x = [recons_x]
            
        # SAMPLE baseline reconstruction: only include batched edge reconstruction
        elif self.model_str not in ['multi-scale','multi-scale-amnet','multi-scale-bwgnn','bwgnn','gradate']:
            recons_a = [recons[0]]
        elif self.model_str == 'gradate':
            recons = [loss,ano_score]
        if 'weibo' in self.dataset:
            check_gpu_usage('returning results')

        return recons_a,recons_a,h_
        return recons_a,recons_x,res_a
