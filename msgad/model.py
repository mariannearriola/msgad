import dgl
import torch
from torch_geometric.nn import MLP
from models.dominant import *
from models.anomalydae import *
from models.cola import *
from models.conad import *
from models.done import *
from models.gaan import *
from models.gcnae import *
from models.mlpae import *
from models.msgad import *
from models.bwgnn import *
from models.amnet import *
from models.gcad import *
from models.hogat import *
from models.gradate import *

class GraphReconstruction(nn.Module):
    def __init__(self, in_size, hidden_size, batch_size, scales, recons, d, model_str, batch_type, act = nn.LeakyReLU()):
        super(GraphReconstruction, self).__init__()
        self.in_size = in_size
        out_size = in_size # for reconstruction
        self.d = d
        self.recons = recons
        self.model_str = model_str
        dropout = 0
        self.embed_act = act
        self.decode_act = torch.sigmoid
        self.batch_type = batch_type
        self.weight_decay = 0.01
        if model_str == 'multi_scale' or model_str == 'multi-scale':
            #self.conv = MSGAD(in_size, hidden_size, d=self.d)
            self.module_list = nn.ModuleList()
            for i in range(3):
                self.module_list.append(MSGAD(in_size,hidden_size,d=(self.d)))#+i)))
            '''
            self.conv1 = MSGAD(in_size, hidden_size, d=self.d)
            self.conv2 = MSGAD(in_size, hidden_size, out_size, d=3)
            self.conv3 = MSGAD(in_size, hidden_size, out_size, d=4)
            '''
        elif model_str == 'gradate': # NOTE: HAS A SPECIAL LOSS: OUTPUTS LOSS, NOT RECONS
            self.conv = GRADATE(in_size,hidden_size,'prelu',1,1,'avg',5)
        elif model_str == 'bwgnn':
            self.conv = BWGNN(in_size, hidden_size, d=4)
        elif model_str in ['anomalydae','anomaly_dae']: # x, e, batch_size (0 for no batching)
            self.conv = AnomalyDAE_Base(in_size,batch_size,hidden_size*2,hidden_size,dropout=0.2,act=F.relu)
        elif model_str == 'dominant':
            self.conv = DOMINANT_Base(in_size,hidden_size,3,dropout,act)
        elif model_str == 'mlpae': # x
            self.conv = MLP(in_channels=in_size,hidden_channels=hidden_size,out_channels=batch_size,num_layers=3)
        elif model_str == 'amnet': # x, e
            self.conv = AMNet(in_size, hidden_size, 2, 2, self.d)
        elif model_str == 'ho-gat':
            self.conv = HOGAT(in_size, hidden_size, dropout, alpha=0.1)
        else:
            raise('model not found')

    def process_graph(self, graph):
        edges = torch.vstack((graph.edges()[0],graph.edges()[1]))
        # add self-loop
        #edges = torch.hstack((edges,torch.vstack((edges.unique(),edges.unique()))))
        feats = graph.ndata['feature']
        if self.batch_type == 'edge':
            feats = feats['_N']
        graph_ = graph
        return edges, feats, graph_
            
    def forward(self,graph):
        '''
        Input:
            graph: input dgl graph
        Output:
            recons: scale-wise adjacency reconstructions
        '''
        labels = None 
        edges, feats, graph_ = self.process_graph(graph)
        dst_nodes = graph.dstnodes().detach().cpu().numpy()
        if self.model_str in ['dominant','amnet']: #x, e
            recons = [self.conv(feats, edges)]
            #import ipdb ; ipdb.set_trace()
        elif self.model_str in ['anomaly_dae','anomalydae']: #x, e, batch_size
            recons = [self.conv(feats, edges, 0)]
        elif self.model_str in ['gradate']: # adjacency matrix
            loss, ano_score = self.conv(graph_, graph_.adjacency_matrix(), feats, False)
            #import ipdb ; ipdb.set_trace()
        elif self.model_str in ['ho-gat']: # x, adj
            recons = [self.conv(feats, graph_)]
        if self.model_str == 'bwgnn':
            recons = [self.conv(graph,feats)]
        elif self.model_str in ['multi_scale','multi-scale']: # g
            recons,labels = [],[]
            adj_label_ = graph.adjacency_matrix().to_dense()[dst_nodes][:,dst_nodes].cuda()#.to_sparse()
            for i in self.module_list:
                recons.append(i(graph,feats))
            adj_label = adj_label_
            for thetas in self.module_list[1].thetas[:-1]: # ignore high freq
                '''
                adj_label = adj_label_
                for ind,theta in enumerate(thetas):
                    if ind == 0:
                        final_adj_label = theta*(adj_label)
                    else:
                        final_adj_label += theta*adj_label
                    final_adj_label=torch.mm(final_adj_label,final_adj_label)
                '''
                #m,n = final_adj_label.shape
                #thr = (math.sqrt(2 * math.log(n)) / math.sqrt(n) * 1) * torch.unsqueeze(torch.tensor(thetas[-1]), dim=0).repeat(m)
                #final_adj_label = torch.mul(torch.sign(final_adj_label), (((torch.abs(final_adj_label) - thr.cuda()) + torch.abs(torch.abs(final_adj_label) - thr.cuda())) / 2))
                
                #final_adj_label[torch.where(final_adj_label<0)] = 0
                #final_adj_label[torch.where(final_adj_label>1)] = 1
                #labels.append(torch.ceil(torch.tanh(final_adj_label)))
                labels.append(torch.round(torch.tanh(adj_label)))
                adj_label = adj_label@adj_label
            
        # feature and structure reconstruction models
        if self.model_str in ['anomalydae','dominant','ho-gat']:
            recons_ind = 0 if self.recons == 'feat' else 1
            #recons = [recons[recons_ind].to_sparse()]
            recons = [recons[0][recons_ind]]
            

        # SAMPLE baseline reconstruction: only include batched edge reconstruction
        if self.model_str not in ['multi-scale','multi_scale','bwgnn','gradate']:
            recons = recons[0]
            recons = [recons[graph.dstnodes()][:,graph.dstnodes()]]
            
        elif self.model_str == 'gradate':
            recons = [loss,ano_score]
        
        return recons, labels
