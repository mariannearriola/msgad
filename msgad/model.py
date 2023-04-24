import dgl
import torch
import numpy as np
import numpy.linalg as npla
import sympy
import math
import time
import gc
import os
from numpy import polynomial
import copy
import networkx as nx
from dgl.nn import EdgeWeightNorm
import torch_geometric
from label_generation import LabelGenerator
from torch_geometric.nn import MLP
from torch_geometric.transforms import GCNNorm
from scipy.interpolate import make_interp_spline
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
from models.gradate import *

class GraphReconstruction(nn.Module):
    def __init__(self, in_size, args, act = nn.LeakyReLU(), label_type='single'):
        super(GraphReconstruction, self).__init__()
        self.in_size = in_size
        out_size = in_size # for reconstruction
        self.taus = [3,4,4]
        self.attn_weights = None
        self.b = 1
        self.norm_adj = torch_geometric.nn.conv.gcn_conv.gcn_norm
        self.norm = EdgeWeightNorm()
        self.d = args.d
        self.e_adj, self.U_adj = None,None
        self.epoch = args.epoch
        self.label_type = args.label_type
        self.dataload = args.dataload
        self.recons = args.recons
        self.dataset = args.dataset
        self.model_str = args.model
        dropout = 0
        self.embed_act = act
        self.decode_act = torch.sigmoid
        self.batch_type = args.batch_type
        self.weight_decay = 0.01
        self.lengths = [5,10,20]
        if 'multi-scale' in args.model:
            self.module_list = nn.ModuleList()
            for i in range(3):
                if 'multi-scale-amnet' == args.model:
                    self.module_list.append(AMNet_ms(in_size, args.hidden_dim, 1, 5, 5))
                elif 'multi-scale-bwgnn' == args.model:
                    self.module_list.append(MSGAD(in_size,args.hidden_dim,d=1))

        elif args.model == 'gradate': # NOTE: HAS A SPECIAL LOSS: OUTPUTS LOSS, NOT RECONS
            self.conv = GRADATE(in_size,args.hidden_dim,'prelu',1,1,'avg',5)
        elif args.model == 'bwgnn':
            self.conv = BWGNN(in_size, args.hidden_dim, d=5)
        elif args.model in ['anomalydae','anomaly_dae']: # x, e, batch_size (0 for no batching)
            self.conv = AnomalyDAE_Base(in_size,args.batch_size,args.hidden_dim,args.hidden_dim,dropout=dropout,act=act)
        elif args.model == 'dominant':
            self.conv = DOMINANT_Base(in_size,args.hidden_dim,3,dropout,act)
        elif args.model == 'mlpae': # x
            self.conv = MLP(in_channels=in_size,hidden_channels=args.hidden_dim,out_channels=args.batch_size,num_layers=3)
        elif args.model == 'amnet': # x, e
            self.conv = AMNet(in_size, args.hidden_dim, 2, 2, 2, vis_filters=args.vis_filters)
        elif args.model == 'ho-gat':
            self.conv = HOGAT(in_size, args.hidden_dim, dropout, alpha=0.1)
        else:
            raise('model not found')

    def process_graph(self, graph):
        """Obtain graph information from input TODO: MOVE?"""
        edges = torch.vstack((graph.edges()[0],graph.edges()[1]))
        feats = graph.ndata['feature']
        if 'edge' == self.batch_type:
            feats = feats['_N']
        return edges, feats, graph

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

    def filter_anoms(self,labels,anoms,vis_name):
        #np.random.seed(seed=1)
        es,Us = [],[]
        signal = np.random.randn(labels[0].shape[0],labels[0].shape[0])
        for label in labels:
            label=torch.maximum(label, label.T)
            e,U = get_spectrum(label)
            es.append(e)
            Us.append(U)
            del e, U ; torch.cuda.empty_cache()
            
        for i in range(len(es)):
            plt.figure()
            try:
                plot_spectrum(es[i].detach().cpu(),Us[i].detach().cpu(),signal+1)
            except Exception as e:
                print(e)
                import ipdb ; ipdb.set_trace()
            for anom_ind,anom in enumerate(anoms):
                #anom = anom.flatten()
                anom = self.flatten_label(anom)
                signal_ = np.copy(signal)
                signal_[anom]*=400
                signal_ += 1
                plot_spectrum(es[i].detach().cpu(),Us[i].detach().cpu(),signal_)
                plt.legend(['no anom signal','sc1 anom signal','sc2 anom signal','sc3 anom signal'])
      
                fpath = f'vis/filter_anom_vis/{self.dataset}/{self.model_str}/{self.label_type}'
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                plt.savefig(f'{fpath}/filter_vis_{self.label_type}_{self.epoch}_{vis_name}_filter{i}.png')

        for i in range(len(es)):
            del es[0], Us[0]
        torch.cuda.empty_cache()

    def flatten_label(self,anoms):
        anom_flat = anoms[0]
        for i in anoms[1:]:
            anom_flat=np.concatenate((anom_flat,i))
        return anom_flat

    def forward(self,graph,last_batch_node,pos_edges,neg_edges,anoms,vis=False,vis_name=""):
        '''
        Input:
            graph: input dgl graph
        Output:
            recons: scale-wise adjacency reconstructions
        '''
        res_a = None
        labels = None
        #if self.model_str != 'bwgnn':
        #    graph.add_edges(graph.dstnodes(),graph.dstnodes())
        #    needs_edges = graph.has_edges_between(pos_edges[:,0],pos_edges[:,1]).int().nonzero()
        #    graph.add_edges(pos_edges[:,0][needs_edges.T[0]][0],pos_edges[:,1][needs_edges.T[0]])
        edges, feats, graph_ = self.process_graph(graph)
        if pos_edges is not None:
            all_edges = torch.vstack((pos_edges,neg_edges))
            dst_nodes = torch.arange(last_batch_node+1)
        else:
            dst_nodes = graph.nodes()
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
            
            for ind,i in enumerate(self.module_list):
                if self.model_str == 'multi-scale-amnet':
                    #print('running model')
                    #seconds = time.time()
                    #import ipdb ; ipdb.set_trace()
                    oom = False
                    mem = torch.cuda.memory_allocated()/torch.cuda.max_memory_reserved()
                    print(mem)
                    try:
                        recons,res,attn_scores = i(feats,edges,dst_nodes)
                    except RuntimeError:
                        oom = True
                    if oom:
                        print('oom')
                        import ipdb ; ipdb.set_trace()
                    if ind == 0:
                        self.attn_weights = torch.unsqueeze(attn_scores,0)
                    else:
                        self.attn_weights = torch.cat((self.attn_weights,torch.unsqueeze(attn_scores,0)))
                    try:
                        recons_a.append(recons[graph.dstnodes()][:,graph.dstnodes()])
                        res_a.append(res[graph.dstnodes()])
                    except Exception as e:
                        print(e)
                        import ipdb ; ipdb.set_trace()
                    del recons, res, i ; torch.cuda.empty_cache() ; gc.collect()
                    #print("Seconds to run model", (time.time()-seconds)/60)
                elif self.model_str == 'multi-scale-bwgnn':
                    oom = False
                    try:
                         recons,res = i(graph,feats,dst_nodes)
                    except RuntimeError:
                        oom = True
                    if oom:
                        print('out of memory')
                        import ipdb ; ipdb.set_trace()
                    recons_a.append(recons)
                    res_a.append(res)
                    del recons, res, i ; torch.cuda.empty_cache() ; gc.collect()
    
            # collect multi-scale labels
            if not self.dataload:
                lg = LabelGenerator(graph,feats,vis,vis_name,anoms)
                lg.construct_labels()
        #elif not self.dataload:
        else:
            if not self.dataload:
                lg = LabelGenerator(graph,feats,vis,vis_name,anoms)
                lg.construct_labels()
            else:
                adj_label=graph.adjacency_matrix().to_dense()[graph.dstnodes()][:,graph.dstnodes()]
                adj_label=np.maximum(adj_label, adj_label.T).to(graph.device) 
                labels = [adj_label]
        
        # feature and structure reconstruction models
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
        
        #import ipdb ; ipdb.set_trace()
        return recons_a,recons_x,labels,res_a
