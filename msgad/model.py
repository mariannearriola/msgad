import dgl
import torch
from torch_geometric.nn import MLP
from models.dominant import *
from models.anomalydae import *
from models.anemone import *
from models.cola import *
from models.conad import *
from models.done import *
from models.gaan import *
from models.gcnae import *
from models.guide import *
from models.mlpae import *
from models.ocgnn import *
from models.one import *
from models.radar import *
from models.scan import *
from models.msgad import *
from models.bwgnn import *

class GraphReconstruction(nn.Module):
    def __init__(self, in_size, hidden_size, batch_size, scales, recons, d, model_str, act = nn.LeakyReLU()):
        super(GraphReconstruction, self).__init__()
        self.in_size = in_size
        out_size = in_size # for reconstruction
        self.d = d
        self.recons = recons
        self.model_str = model_str
        dropout = 0
        self.embed_act = act
        self.decode_act = torch.sigmoid
        self.weight_decay = 0.01
        if model_str == 'multi_scale' or model_str == 'multi-scale':
            self.conv = MSGAD(in_size, hidden_size, d=self.d)
            #self.conv2 = BWGNN(in_size, hidden_size, out_size, d=self.d+1)
            #self.conv3 = BWGNN(in_size, hidden_size, out_size, d=self.d+2)
        elif model_str == 'bwgnn':
            self.conv = BWGNN(in_size, hidden_size, d=self.d)
        elif model_str == 'adone': # x, s, e
            self.conv = AdONE_Base(in_size,batch_size,hidden_size,4,dropout,act)
        elif model_str == 'anomalous': # x
            raise NotImplementedError
            w_init = torch.randn_like(torch.tensor(in_size,batch_size))
            r_init = torch.inverse((1 + self.weight_decay)
                * torch.eye(batch_size).cuda() + self.gamma * l) @ x #TODO?
            self.conv = ANOMALOUS_Base(w_init,r_init)
        elif model_str in ['anomalydae','anomaly_dae']: # x, e, batch_size (0 for no batching)
            self.conv = AnomalyDAE_Base(in_size,batch_size,hidden_size*2,hidden_size,dropout=0.2,act=F.relu)
        elif model_str == 'dominant':
            self.conv = DOMINANT_Base(in_size,hidden_size,3,dropout,act)
        elif model_str == 'done': # x, s, e
            self.conv = DONE_Base(in_size,batch_size,hidden_size,4,dropout,act)
        elif model_str == 'gaan': # x, noise, e
            self.conv = GAAN_Base(in_size,16,2,2,dropout,act)
        elif model_str == 'gcnae': # x, e
            self.conv = GCN(in_size,hidden_size,2,batch_size,dropout,act)
        elif model_str == 'guide': # x, s, e
            self.conv = GUIDE_Base(in_size,batch_size,32,4,4,dropout,act)
        elif model_str == 'mlpae': # x
            self.conv = MLP(in_channels=in_size,hidden_channels=hidden_size,out_channels=batch_size,num_layers=3)
        elif model_str == 'ogcnn': # x, s, e
            self.conv = GCN_base(in_size,hidden_size,4,dropout,act)
        elif model_str == 'radar': # x
            raise NotImplementedError
            w_init = torch.randn_like(torch.tensor(in_size,batch_size))
            r_init = torch.inverse((1 + self.weight_decay)
                * torch.eye(x.shape[0]).to(self.device) + self.gamma * l) @ x #TODO?
            self.conv = Radar_Base(w_init,r_init)
        elif model_str == 'conad': # x, e
            self.conv = CONAD_Base(in_size,hidden_size,4,dropout,act)
        else:
            raise('model not found')

    def process_graph(self, graph):
        edges = torch.vstack((graph.edges()[0],graph.edges()[1]))
        feats = graph.ndata['feature']
        feats = feats['_N']
        '''
        feats = torch.cat((feats['_N_src'],feats['_N_dst']))
        feat_ids = graph.ndata['_ID']
        feat_ids = torch.cat((feat_ids['_N_src'],feat_ids['_N_dst']))
        feats = feats[torch.argsort(feat_ids)]
        '''
        '''
        adj=graph.adjacency_matrix()
        adj=adj.sparse_resize_((graph.num_nodes(), graph.num_nodes()), adj.sparse_dim(), adj.dense_dim())
        graph_idx = adj.coalesce().indices()
        graph_ = dgl.graph((graph_idx[0],graph_idx[1])).to(graph.device)
        '''
        graph_ = graph
        return edges, feats, graph_
            
    def forward(self,graph):
        '''
        Input:
            graph: input dgl graph
        Output:
            recons: scale-wise adjacency reconstructions
        '''
        
        edges, feats, graph_ = self.process_graph(graph)
        
        if self.model_str in ['adone','done','guide','ogcnn']: # x, s, e
            recons = [self.conv(feats, adj.to_dense(), edges)]
        elif self.model_str in ['anomalous','mlpae','radar']: # x
            recons = [self.conv(feats)]
            if self.model_str == 'mlpae':
                recons = [self.decode_act(recons[0])]
        elif self.model_str in ['conad','gcnae','dominant']: #x, e
            recons = [self.conv(feats, edges)]
        elif self.model_str in ['gaan']: # x, noise, e
            gaussian_noise = torch.randn(graph.number_of_nodes(), self.noise_dim).cuda()
            recons = [self.conv(feats, gaussian_noise, edges)]
        elif self.model_str in ['anomaly_dae','anomalydae']: #x, e, batch_size
            recons = [self.conv(feats, edges, 0)]
        elif self.model_str in ['multi_scale','multi-scale','bwgnn']: # g
            #feats_ = feats[graph_.nodes()]
            #feats_src = graph.srcdata['feature']
            #feats_dst = graph.dstdata['feature']
            feats_ = feats
            recons = self.conv(graph, feats_)
            # recons must be of shape n_dst x n_dsit
        
        # feature and structure reconstruction models
        if self.model_str in ['anomalydae','dominant']:
            recons_ind = 0 if self.recons == 'feat' else 1
            recons = [recons[0][recons_ind].to_sparse()]

        if self.model_str not in ['multi-scale','multi_scale','bwgnn']:
            recons = recons[0].to_dense()
            recons = recons[graph.dstnodes()][:,graph.dstnodes()]
            recons = [recons.to_sparse()]

        return recons