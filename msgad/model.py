import dgl
import torch
import numpy as np
import numpy.linalg as npla
import sympy
import math
import copy
import os
from numpy import polynomial
import networkx as nx
from dgl.nn import EdgeWeightNorm
import torch_geometric
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

def get_spectrum(mat):
    d = np.zeros(mat.shape[0])
    degree_in = np.ravel(mat.sum(axis=0))
    degree_out = np.ravel(mat.sum(axis=1))
    dw = (degree_in + degree_out) / 2
    disconnected = (dw == 0)
    np.power(dw, -0.5, where=~disconnected, out=d)
    D = scipy.sparse.diags(d)
    L = scipy.sparse.identity(mat.shape[0]) - D * mat * D
    L[disconnected, disconnected] = 0
    e, U = scipy.linalg.eigh(np.array(L), overwrite_a=True)
    return e, U

def plot_spectrum(e,U,signal):
    e[0] = 0.
    c = np.dot(U.transpose(), signal)
    M = np.zeros((15,c.shape[1]))
    c = np.array(c)
    for j in range(signal.shape[0]):
        idx = min(int(e[j] / 0.1), 15-1)
        M[idx] += c[j]**2
    M=M/sum(M)
    y = M[:,0]
    x = np.arange(y.shape[0])
    spline = make_interp_spline(x, y)
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = spline(X_)
    plt.plot(X_,Y_)

def edge_index_select(t, query_row, query_col):
    t_idx = t.indices()
    row, col, val = t_idx[0],t_idx[1],t.values()
    row_mask = row == query_row.view(-1, 1)
    col_mask = col == query_col.view(-1, 1)
    mask = torch.max(torch.logical_and(row_mask, col_mask), dim=0).values
    return val[mask],mask.nonzero().flatten()

def neg_edge_index_select(t, query_row, query_col):
    t = t.coalesce()
    row, col, val = t.indices()[0],t.indices()[1],t.values()
    row_mask = row == query_row.view(-1, 1)
    col_mask = col == query_col.view(-1, 1)
    mask = torch.max(torch.logical_and(row_mask, col_mask), dim=0).values
    return val[mask]

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
    return torch.tensor([a for a in sorted(G.out_degrees()[G.dstnodes()], key=lambda a: a.item())]).to(G.device)

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
                    self.module_list.append(AMNet_ms(in_size, args.hidden_dim, 2, 2, 3))
                elif 'multi-scale-bwgnn' == args.model:
                    self.module_list.append(MSGAD(in_size,args.hidden_dim,d=5))

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

    def construct_labels(self,graph,feats,vis,vis_name):
        """
        Stationary distribution given the transition matrix.
        :param M: Transition matrix.
        :return: Stationary distribution.
        """
        g = dgl.graph(graph.edges()).cpu()
        g.edata['_ID'] = graph.edata['_ID'].cpu()
        g.edata['w'] = torch.full(g.edata['_ID'].shape,1.)
        num_a_edges = g.num_edges()
        if 'norm' in self.label_type:
            epsilon = 1e-8
            g.edata['w'] = self.norm(graph,(graph.edata['w']+epsilon)).cpu()
        if 'prods' in self.label_type:
            labels = []
            #graph_.add_edges(graph_.dstnodes(),graph_.dstnodes())
            adj_label = graph.adjacency_matrix()
            adj_label.sparse_resize_((adj_label.size(0), adj_label.size(0)), adj_label.sparse_dim(), adj_label.dense_dim())
            adj_label = adj_label.to_dense().to(graph.device)
            adj_label=np.maximum(adj_label, adj_label.T) #?
            for k in range(5):
                adj_label_ = adj_label
                upper_tri=torch.triu(adj_label_,1)
                nz = upper_tri[upper_tri.nonzero()[:,0],upper_tri.nonzero()[:,1]]
                sorted_idx = torch.argsort(-nz)
                drop_idx=upper_tri.nonzero()[sorted_idx][num_a_edges:]
                adj_label_[drop_idx[:,0],drop_idx[:,1]]=0
                adj_label_[drop_idx[:,1],drop_idx[:,0]]=0
                adj_label_[torch.where(adj_label_>0)[0],torch.where(adj_label_>0)[1]]=1
                adj_label_[torch.where(adj_label_<0)[0],torch.where(adj_label_<0)[1]]=0
                labels.append(adj_label_)
                adj_label = adj_label@adj_label
            if vis == True:
                legend = []
                plt.figure()
                #import ipdb ; ipdb.set_trace()
                for label_ind,label in enumerate(labels):
                    e, U = get_spectrum(label)
                    e[0] = 0
                    try:
                        assert -1e-5 < e[0] < 1e-5
                    except:
                        print('Eigenvalues out of bounds')
                        import ipdb ; ipdb.set_trace()
                    plot_spectrum(e,U,feats.detach().cpu().numpy())
                
                    legend.append(str(label_ind))
                plt.legend(legend)
                fpath = f'label_vis/{self.dataset}'
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                plt.savefig(f'{fpath}/labels_{self.label_type}_{self.epoch}_{vis_name}.png')
        elif self.label_type == 'single':
            labels = []
            '''
            edges = graph.has_edges_between(all_edges[:,0],all_edges[:,1]).float()
            labels_pos_eids=graph.edge_ids(all_edges[:,0][edges.nonzero()].flatten(),all_edges[:,1][edges.nonzero()].flatten())
            edges[torch.where(edges!=0)[0]] = graph.edata['w'][labels_pos_eids]
            '''
            for k in range(3):
                adj = graph.adjacency_matrix().to_dense().to(graph.device)
                adj=np.maximum(adj, adj.T)
                labels.append(adj)
        if 'random-walk' in self.label_type:
            nx_graph = nx.to_undirected(dgl.to_networkx(g.cpu()))
            #node_ids = graph.ndata['_ID']['_N']
            node_ids = np.arange(g.num_nodes())
            connected_graphs = [g for g in nx.connected_components(nx_graph)]
            node_dict = {k:v.item() for k,v in zip(list(nx_graph.nodes),node_ids)}
            labels = []
            full_labels = [torch.zeros((g.num_nodes(),g.num_nodes())).to(graph.device),torch.zeros((g.num_nodes(),g.num_nodes())).to(graph.device),torch.zeros((g.num_nodes(),g.num_nodes())).to(graph.device)]
            adj = graph.adjacency_matrix().to_dense().to(graph.device)
            adj=np.maximum(adj, adj.T)
            labels.append(adj)
            for connected_graph_nodes in connected_graphs:
                subgraph = dgl.from_networkx(nx_graph.subgraph(connected_graph_nodes))
                nodes_sel = [node_dict[j] for j in list(connected_graph_nodes)]
                for i in range(3):
                    D_1 = torch.diag(1 / degree_vector(subgraph))
                    A = subgraph.adjacency_matrix().to_dense()
                    A=np.maximum(A, A.T)
                    M = torch.matmul(D_1, A)

                    pi = self.stationary_distribution(M.to(graph.device),graph.device)
                    Pi = np.diag(pi)
                    M_tau = np.linalg.matrix_power(A.detach().cpu().numpy(), self.taus[i])

                    R = np.log(Pi @ M_tau/self.b) - np.log(np.outer(pi, pi))
                    R = R.copy()
                    # Replace nan with 0 and negative infinity with min value in the matrix.
                    R[np.isnan(R)] = 0
                    R[np.isinf(R)] = np.inf
                    R[np.isinf(R)] = R.min()
                    res = torch.tensor(R).to(graph.device)
                    lbl_idx = torch.tensor(nodes_sel).to(torch.long)
                    full_labels[i][lbl_idx.reshape(-1,1),lbl_idx]=res
            # post-cleaning label
            for i in range(3):
                upper_tri=torch.triu(full_labels[i],1)
                nz = upper_tri[upper_tri.nonzero()[:,0],upper_tri.nonzero()[:,1]]
                sorted_idx = torch.argsort(-nz)
                drop_idx=upper_tri.nonzero()[sorted_idx][num_a_edges:]
                full_labels[i][drop_idx[:,0],drop_idx[:,1]]=0
                full_labels[i][drop_idx[:,1],drop_idx[:,0]]=0
                
                full_labels[i][torch.where(full_labels[i]>0)[0],torch.where(full_labels[i]>0)[1]]=1
                full_labels[i][torch.where(full_labels[i]<0)[0],torch.where(full_labels[i]<0)[1]]=0
                labels.append(full_labels[i].to(graph.device))
    
        if 'filter' in self.label_type:   
            K = 5
            if 'amnet' in self.label_type:
                coeffs =  get_bern_coeff(K)
                label_idx=[0,2,3]
            elif 'bwgnn' in self.label_type:
                coeffs = calculate_theta2(K)
                label_idx=[0,2,4]
            labels = []
            adj_label = graph.adjacency_matrix()
            adj_label.sparse_resize_((adj_label.size(0), adj_label.size(0)), adj_label.sparse_dim(), adj_label.dense_dim())
            adj_label = adj_label.to_dense().to(graph.device)
            adj_label=np.maximum(adj_label, adj_label.T) #?
            num_a_edges = torch.triu(adj_label,1).nonzero().shape[0]
            labels = [adj_label]
            for k in range(0, K+1):
                adj_label_norm = self.norm_adj(adj_label.nonzero().contiguous().T)
                adj_label_ = torch_geometric.utils.to_dense_adj(adj_label_norm[0])[0]
                #adj_label_ = adj_label
                adj_label_[adj_label_norm[0][0],adj_label_norm[0][1]]=adj_label_norm[1]
                coeff = coeffs[k]
                basis = adj_label_ * coeff[0]
                for i in range(1, K+1):
                    adj_label_ = adj_label_@adj_label_
                    basis += adj_label_ * coeff[i]
                
                upper_tri=torch.triu(basis,1)
                nz = upper_tri[upper_tri.nonzero()[:,0],upper_tri.nonzero()[:,1]]
                sorted_idx = torch.argsort(-nz)
                drop_idx=upper_tri.nonzero()[sorted_idx][num_a_edges:]
                basis[drop_idx[:,0],drop_idx[:,1]]=0
                basis[drop_idx[:,1],drop_idx[:,0]]=0
                
                basis[torch.where(basis>0)[0],torch.where(basis>0)[1]]=1
                basis[torch.where(basis<0)[0],torch.where(basis<0)[1]]=0
                
                labels.append(basis)
            #labels=[adj_label,labels[0],labels[3]]
            #labels=[labels[1],labels[3],labels[-1]]
            if vis == True and 'test' not in vis_name:
                plt.figure()
                legend = []
                for label_ind,label in enumerate(labels):
                    e,U = get_spectrum(label)
                    try:
                        assert -1e-5 < e[0] < 1e-5
                    except:
                        print('Eigenvalues out of bounds')
                        import ipdb ; ipdb.set_trace()
                    e[0] = 0
                    plot_spectrum(e,U,feats.detach().cpu().numpy())
                    #plt.plot(y)
                    legend.append(str(label_ind))
                plt.legend(legend)
                fpath = f'label_vis/{self.dataset}'
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                plt.savefig(f'{fpath}/labels_{self.label_type}_{self.epoch}_{vis_name}.png')
            labels = [labels[1],labels[2],labels[3]]
        else:
            labels = None
        '''
        label_edges = []
        for label_id in label_idx:
            label_edge=labels[label_id].has_edges_between(all_edges[:,0].cpu(),all_edges[:,1].cpu()).float()
            labels_pos_eids=labels[label_id].edge_ids(all_edges[:,0].cpu()[label_edge.nonzero()].flatten(),all_edges[:,1].cpu()[label_edge.nonzero()].flatten())
            label_edge[torch.where(label_edge!=0)[0]] = labels[label_id].edata['w'][labels_pos_eids]
            label_edges.append(label_edge.to(graph.device))

        labels = label_edges
        '''
        return labels
            
    def forward(self,graph,last_batch_node,pos_edges,neg_edges,vis=False,vis_name=""):
        '''
        Input:
            graph: input dgl graph
        Output:
            recons: scale-wise adjacency reconstructions
        '''
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
            recons = [self.conv(feats, edges,dst_nodes)]
        elif self.model_str in ['anomaly_dae','anomalydae']: #x, e, batch_size
            recons = [self.conv(feats, edges, 0,dst_nodes)]
        elif self.model_str in ['gradate']: # adjacency matrix
            loss, ano_score = self.conv(graph_, graph_.adjacency_matrix(), feats, False)
        if self.model_str == 'bwgnn':
            recons_a = [self.conv(graph_,feats,dst_nodes)]
        if 'multi-scale' in self.model_str: # g
            recons_a,labels = [],[]
            
            for ind,i in enumerate(self.module_list):
                if self.model_str == 'multi-scale-amnet':
                    recons,res,attn_scores = i(feats,edges,dst_nodes)
                    if ind == 0:
                        self.attn_weights = torch.unsqueeze(attn_scores,0)
                    else:
                        self.attn_weights = torch.cat((self.attn_weights,torch.unsqueeze(attn_scores,0)))
                    recons_a.append(recons)
                    
                elif self.model_str == 'multi-scale-bwgnn':
                    recons,res = i(graph,feats,dst_nodes)
                    recons_a.append(recons)
                
                if ind == 0 and vis == True:
                    plt.figure()
                    adj_label = graph.adjacency_matrix()
                    adj_label.sparse_resize_((adj_label.size(0), adj_label.size(0)), adj_label.sparse_dim(), adj_label.dense_dim())
                    adj_label = adj_label.to_dense().to(graph.device)
                    e,U= get_spectrum(adj_label)
                    e[0] = 0
                    try:
                        assert -1e-5 < e[0] < 1e-5
                    except:
                        print('Eigenvalues out of bounds')
                        import ipdb ; ipdb.set_trace()
                    plot_spectrum(e,U,feats.detach().cpu().numpy())
                if vis == True:
                    try:
                        plot_spectrum(e,U,res.detach().cpu().numpy())
                    except Exception as e:
                        import ipdb ; ipdb.set_trace()
                        print(e)
            
            if vis == True:
                plt.legend(['og','sc1','sc2','sc3'])
                fpath = f'filter_vis/{self.dataset}'
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                plt.savefig(f'{fpath}/filter_vis_{self.label_type}_{self.epoch}_{vis_name}.png')
            # collect multi-scale labels
            if not self.dataload:
                labels = self.construct_labels(graph,feats,vis,vis_name)
        elif not self.dataload:
            adj_label=graph.adjacency_matrix().to_dense().to(graph.device)
            adj_label=np.maximum(adj_label, adj_label.T)
            labels = [adj_label]
        # single scale reconstruction label
        
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
        return recons_a,recons_x, labels
