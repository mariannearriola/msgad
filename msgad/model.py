import dgl
import torch
import numpy as np
import numpy.linalg as npla
import sympy
import math
from numpy import polynomial

from torch_geometric.nn import MLP
from torch_geometric.transforms import GCNNorm
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

def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas

def get_bern_coeff(degree):
    def Bernstein(de, i):
        coefficients = [0, ] * i + [math.comb(de, i)]
        first_term = polynomial.polynomial.Polynomial(coefficients)
        second_term = polynomial.polynomial.Polynomial([1, -1]) ** (de - i)
        return first_term * second_term

    out = []

    for i in range(degree + 1):
        out.append(Bernstein(degree, i).coef)

    return out


def degree_vector(G):
    """
    Degree vector for the input graph.
    :param G: Input graph.
    :return: Degree vector.
    """
    return torch.tensor([a for a in sorted(G.out_degrees()[G.dstnodes()], key=lambda a: a.item())]).cuda()

class GraphReconstruction(nn.Module):
    def __init__(self, in_size, hidden_size, batch_size, scales, recons, d, model_str, batch_type, act = nn.LeakyReLU(), label_type='single'):
        super(GraphReconstruction, self).__init__()
        self.in_size = in_size
        out_size = in_size # for reconstruction
        self.taus = [2,3,0]
        self.b = 1
        self.norm = torch_geometric.nn.conv.gcn_conv.gcn_norm
        self.d = d
        self.label_type = label_type
        self.recons = recons
        self.model_str = model_str
        dropout = 0
        self.embed_act = act
        self.decode_act = torch.sigmoid
        self.batch_type = batch_type
        self.weight_decay = 0.01
        self.lengths = [5,10,20]
        if 'multi-scale' in model_str:
            self.module_list = nn.ModuleList()
            for i in range(3):
                if 'multi-scale-amnet' == model_str:
                    self.module_list.append(AMNet_ms(in_size, hidden_size, 2, 2, 5))
                elif 'multi-scale-bwgnn' == model_str:
                    self.module_list.append(MSGAD(in_size,hidden_size,d=(self.d)))

        elif model_str == 'gradate': # NOTE: HAS A SPECIAL LOSS: OUTPUTS LOSS, NOT RECONS
            self.conv = GRADATE(in_size,hidden_size,'prelu',1,1,'avg',5)
        elif model_str == 'bwgnn':
            self.conv = BWGNN(in_size, hidden_size, d=5)
        elif model_str in ['anomalydae','anomaly_dae']: # x, e, batch_size (0 for no batching)
            self.conv = AnomalyDAE_Base(in_size,batch_size,hidden_size,hidden_size,dropout=0.2,act=F.relu)
        elif model_str == 'dominant':
            self.conv = DOMINANT_Base(in_size,hidden_size,3,dropout,act)
        elif model_str == 'mlpae': # x
            self.conv = MLP(in_channels=in_size,hidden_channels=hidden_size,out_channels=batch_size,num_layers=3)
        elif model_str == 'amnet': # x, e
            self.conv = AMNet(in_size, hidden_size, 2, 2, 5)
        elif model_str == 'ho-gat':
            self.conv = HOGAT(in_size, hidden_size, dropout, alpha=0.1)
        else:
            raise('model not found')

    def process_graph(self, graph):
        """Obtain graph information from input TODO: MOVE?"""
        edges = torch.vstack((graph.edges()[0],graph.edges()[1]))
        feats = graph.ndata['feature']
        if 'edge' == self.batch_type:
            feats = feats['_N']
        return edges, feats, graph

    def stationary_distribution(self, M):
        """
        Stationary distribution given the transition matrix.
        :param M: Transition matrix.
        :return: Stationary distribution.
        """

        # We solve (M^T - I) pi = 0 and 1 pi = 1. Combine them and let A = [M^T - I; 1], b = [0; 1]. We have A pi = b.
        n = M.shape[0]
        A = torch.cat([M.T - torch.eye(n).cuda(),torch.ones((1,n)).cuda()],axis=0)
        b = torch.cat([torch.zeros(n),torch.tensor([1])],axis=0).cuda()

        # Solve A^T A pi = A^T pi instead (since A is not square).
        try:
            prod1= np.asarray((A.T @ A).cpu())
            prod2=np.asarray((A.T @ b).cpu())
            pi = np.linalg.solve(prod1,prod2)
        except Exception as e:
            import ipdb ; ipdb.set_trace()
            print(e)
        return pi
            
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
        elif self.model_str in ['anomaly_dae','anomalydae']: #x, e, batch_size
            recons = [self.conv(feats, edges, 0)]
        elif self.model_str in ['gradate']: # adjacency matrix
            loss, ano_score = self.conv(graph_, graph_.adjacency_matrix(), feats, False)
            #import ipdb ; ipdb.set_trace()
        elif self.model_str in ['ho-gat']: # x, adj
            recons = [self.conv(feats, graph_)]
        if self.model_str == 'bwgnn':
            recons = [self.conv(graph,feats)]
        elif 'multi-scale' in self.model_str: # g
            recons,labels = [],[]
            #adj_label_ = graph.adjacency_matrix().to_dense()[dst_nodes][:,dst_nodes].cuda()#.to_sparse()

            graph = dgl.block_to_graph(graph)
            #dst_edges=graph.find_edges(dst_nodes)
            #dst_graph = dgl.graph((dst_edges[0],dst_edges[1]))
            #sp_adj = torch.sparse_coo_tensor(torch.vstack((dst_edges[0],dst_edges[1])), torch.ones(dst_edges[0].shape[0]).cuda(), (dst_graph.number_of_nodes(), dst_graph.number_of_nodes()))
            adj_label_ = graph.adjacency_matrix().to_dense()[dst_nodes].cuda()#.to_sparse()
            for i in self.module_list:
                if self.model_str == 'multi-scale-amnet':
                    recons.append(i(feats,edges))
                elif self.model_str == 'multi-scale-bwgnn':
                    recons.append(i(graph,feats))
                recons[-1] = recons[-1][graph.dstnodes()][:,graph.dstnodes()]
            adj_label = adj_label_
            num_a_edges = torch.triu(adj_label_,1).nonzero().shape[0]
            #for thetas in self.module_list[1].thetas[:-1]: # ignore high freq
            for iter_ in range(3):

                if self.label_type == 'prods':
                    #labels.append(torch.round(torch.tanh(adj_label)))
                    labels.append(adj_label)
                    adj_label = adj_label@adj_label
                elif self.label_type == 'prods_norm':
                    adj_label_norm = self.norm(adj_label.nonzero().contiguous().T)
                    adj_label = torch_geometric.utils.to_dense_adj(adj_label_norm[0])[0]
                    adj_label[adj_label_norm[0][0],adj_label_norm[0][1]]=adj_label_norm[1]
                    #labels.append(torch.round(torch.tanh(adj_label)))
                    labels.append(adj_label)
                    adj_label = adj_label@adj_label
                elif self.label_type == 'prods_norm_first':
                    if len(labels) == 0:
                        adj_label_norm = self.norm(adj_label.nonzero().contiguous().T)
                        adj_label = torch_geometric.utils.to_dense_adj(adj_label_norm[0])[0]
                        adj_label[adj_label_norm[0][0],adj_label_norm[0][1]]=adj_label_norm[1]
                    #labels.append(torch.round(torch.tanh(adj_label)))
                    labels.append(adj_label)
                    adj_label = adj_label@adj_label
                elif self.label_type == 'single':
                    labels.append(adj_label)
                elif self.label_type == 'single_norm':
                    adj_label_norm = self.norm(adj_label.nonzero().contiguous().T)
                    adj_label = torch_geometric.utils.to_dense_adj(adj_label_norm[0])[0]
                    adj_label[adj_label_norm[0][0],adj_label_norm[0][1]]=adj_label_norm[1]
                    labels.append(adj_label)
                elif self.label_type == 'single_norm_first':
                    if len(labels) == 0:
                        adj_label_norm = self.norm(adj_label.nonzero().contiguous().T)
                        adj_label = torch_geometric.utils.to_dense_adj(adj_label_norm[0])[0]
                        adj_label[adj_label_norm[0][0],adj_label_norm[0][1]]=adj_label_norm[1]
                    else:
                        adj_label = labels[-1]
                    #labels.append(adj_label)
                elif self.label_type == 'sample_prods':
                    # TODO: CONVERT TO SPARSE MM
                    if len(labels) == 0:
                        adj_label_ = adj_label
                    #labels.append(adj_label_)
                    labels.append(torch.round(torch.tanh(adj_label_)))
                    adj_label = adj_label@adj_label

                    adj_label_ = adj_label
                    top_edges = torch.argsort(adj_label)
                    upper_tri=torch.triu(adj_label,1)
                    nz = upper_tri[upper_tri.nonzero()[:,0],upper_tri.nonzero()[:,1]]
                    sorted_idx = torch.argsort(-nz)

                    # PICK TOP K EDGES FOR SAMPLING.
                    drop_idx=upper_tri.nonzero()[sorted_idx][int(num_a_edges):]
                    adj_label_[drop_idx[:,0],drop_idx[:,1]]=0
                    adj_label_[drop_idx[:,1],drop_idx[:,0]]=0
                elif self.label_type == 'sample_prods_norm':
                    adj_label_norm = self.norm(adj_label.nonzero().contiguous().T)
                    adj_label = torch_geometric.utils.to_dense_adj(adj_label_norm[0])[0]
                    adj_label[adj_label_norm[0][0],adj_label_norm[0][1]]=adj_label_norm[1]

                    labels.append(adj_label)
                    
                    adj_label = adj_label@adj_label
                    top_edges = torch.argsort(adj_label)
                    upper_tri=torch.triu(adj_label,1)
                    nz = upper_tri[upper_tri.nonzero()[:,0],upper_tri.nonzero()[:,1]]
                    sorted_idx = torch.argsort(-nz)
                    drop_idx=upper_tri.nonzero()[sorted_idx][num_a_edges:]
                    adj_label[drop_idx[:,0],drop_idx[:,1]]=0
                    adj_label[drop_idx[:,1],drop_idx[:,0]]=0
                elif self.label_type == 'sample_prods_norm_first':
                    if len(labels) == 0:
                        adj_label_norm = self.norm(adj_label.nonzero().contiguous().T)
                        adj_label = torch_geometric.utils.to_dense_adj(adj_label_norm[0])[0]
                        adj_label[adj_label_norm[0][0],adj_label_norm[0][1]]=adj_label_norm[1]

                    labels.append(adj_label)
                    
                    adj_label = adj_label@adj_label
                    top_edges = torch.argsort(adj_label)
                    upper_tri=torch.triu(adj_label,1)
                    nz = upper_tri[upper_tri.nonzero()[:,0],upper_tri.nonzero()[:,1]]
                    sorted_idx = torch.argsort(-nz)
                    drop_idx=upper_tri.nonzero()[sorted_idx][num_a_edges:]
                    adj_label[drop_idx[:,0],drop_idx[:,1]]=0
                    adj_label[drop_idx[:,1],drop_idx[:,0]]=0
                elif self.label_type == 'random_walk':
                    if len(labels) == 0:
                        labels.append(adj_label)
                    else:
                        labels.append(res)

                    D_1 = torch.diag(1 / degree_vector(graph))
                    A = adj_label
                    M = torch.matmul(D_1, A)

                    pi = self.stationary_distribution(M)
                    Pi = np.diag(pi)
                    M_tau = np.linalg.matrix_power(adj_label.detach().cpu().numpy(), self.taus[len(labels)-1])
    
                    R = np.log(Pi @ M_tau/self.b) - np.log(np.outer(pi, pi))
                    R = R.copy()
                    # Replace nan with 0 and negative infinity with min value in the matrix.
                    R[np.isnan(R)] = 0
                    R[np.isinf(R)] = np.inf
                    R[np.isinf(R)] = R.min()
                    res = torch.tensor(R).to(graph.device)
                '''
                elif self.label_type == 'sample_prods_norm_first':
                    if len(labels) == 0:
                        adj_label_norm = self.norm(adj_label.nonzero().contiguous().T)
                        adj_label[adj_label_norm[0][0],adj_label_norm[0][1]]=adj_label_norm[1]
                    else:
                        adj_label_norm = adj_label.nonzero().contiguous().T
                        adj_label[adj_label_norm[0][0],adj_label_norm[0][1]]=adj_label_norm[1]

                    labels.append(adj_label)
                    
                    adj_label = adj_label@adj_label
                    top_edges = torch.argsort(adj_label)
                    upper_tri=torch.triu(adj_label,1)
                    nz = upper_tri[upper_tri.nonzero()[:,0],upper_tri.nonzero()[:,1]]
                    sorted_idx = torch.argsort(-nz)
                    drop_idx=upper_tri.nonzero()[sorted_idx][num_a_edges:]
                    adj_label[drop_idx[:,0],drop_idx[:,1]]=0
                    adj_label[drop_idx[:,1],drop_idx[:,0]]=0
                '''
        if self.label_type == 'amnet_filter':
            
            adj_label = graph.adjacency_matrix().to_dense()[dst_nodes].cuda()
            num_a_edges = torch.triu(adj_label,1).nonzero().shape[0]
            K = 5
            bern_coeff =  get_bern_coeff(K)
            labels = []
            for k in range(0, K+1):
                adj_label_norm = self.norm(adj_label.nonzero().contiguous().T)
                adj_label_ = torch_geometric.utils.to_dense_adj(adj_label_norm[0])[0]
                adj_label_[adj_label_norm[0][0],adj_label_norm[0][1]]=adj_label_norm[1]
                coeff = bern_coeff[k]
                basis = adj_label_ * coeff[0]
                for i in range(1, K+1):
                    adj_label_ = adj_label_@adj_label_
                    basis += adj_label_ * coeff[i]

                top_edges = torch.argsort(basis)
                upper_tri=torch.triu(basis,1)
                nz = upper_tri[upper_tri.nonzero()[:,0],upper_tri.nonzero()[:,1]]
                sorted_idx = torch.argsort(-nz)
                drop_idx=upper_tri.nonzero()[sorted_idx][num_a_edges:]
                basis[drop_idx[:,0],drop_idx[:,1]]=0
                basis[drop_idx[:,1],drop_idx[:,0]]=0
                basis[torch.where(basis!=0)[0],torch.where(basis!=0)[1]]=1
                labels.append(basis)
                #labels.append(torch.sigmoid(basis))
            #import ipdb ; ipdb.set_trace()
            #labels = [adj_label,labels[0],labels[1]]
            #labels = [adj_label,labels[1],labels[3]]
            #labels=[labels[0],labels[2],labels[4]]
            #labels=[adj_label,labels[0],labels[2]]
            labels = [labels[2]]
            #labels = [labels[1],labels[3],labels[5]]
        elif self.label_type == 'bwgnn_filter':
            d=5
            thetas = calculate_theta2(d)
            #labels.append(adj_label)
            for k in range(d+1):
                adj_label_norm = self.norm(adj_label.nonzero().contiguous().T)
                adj_label_ = torch_geometric.utils.to_dense_adj(adj_label_norm[0])[0]
                adj_label_[adj_label_norm[0][0],adj_label_norm[0][1]]=adj_label_norm[1]
                #import ipdb ; ipdb.set_trace()
                coeff=thetas[k]
                basis = adj_label_ * coeff[0]
                for i in range(d+1):
                    adj_label_ = adj_label_@adj_label_
                    basis += adj_label_*coeff[i]

                # getting top edges from sparse adj:
                # just sort idx by vals, select top k
                top_edges = torch.argsort(basis)
                upper_tri=torch.triu(basis,1)
                nz = upper_tri[upper_tri.nonzero()[:,0],upper_tri.nonzero()[:,1]]
                sorted_idx = torch.argsort(-nz)
                drop_idx=upper_tri.nonzero()[sorted_idx][num_a_edges:]
                basis[drop_idx[:,0],drop_idx[:,1]]=0
                basis[drop_idx[:,1],drop_idx[:,0]]=0
                basis[torch.where(basis!=0)[0],torch.where(basis!=0)[1]]=1
                labels.append(basis)

                #labels.append(torch.sigmoid(basis))
            #import ipdb ; ipdb.set_trace()
            #labels = [adj_label,labels[0],labels[1]]
            #labels = [labels[3],labels[4],labels[5]]
            labels = [labels[0],labels[1],labels[2]]
            #labels = [adj_label,labels[1],labels[3]]
            #labels = [labels[1],labels[3],labels[5]]
            #labels = [adj_label,labels[1],labels[3]]
            #import ipdb ; ipdb.set_trace()
                
        # feature and structure reconstruction models
        if self.model_str in ['anomalydae','dominant','ho-gat']:
            recons_ind = 0 if self.recons == 'feat' else 1
            if self.model_str == 'ho-gat':
                return recons[0][recons_ind], recons[0][recons_ind+1], recons[0][recons_ind+2]
            recons = [recons[0][recons_ind]]
        
        # SAMPLE baseline reconstruction: only include batched edge reconstruction
        if self.model_str not in ['multi-scale','multi-scale-amnet','multi-scale-bwgnn','bwgnn','gradate']:
            recons = recons[0]
            recons = [recons[graph.dstnodes()][:,graph.dstnodes()]]
        elif self.model_str == 'gradate':
            recons = [loss,ano_score]
        return recons, labels
