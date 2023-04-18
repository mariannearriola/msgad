import dgl
import torch
import numpy as np
import numpy.linalg as npla
import sympy
import math
import copy
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
        self.b = 1
        self.norm_adj = torch_geometric.nn.conv.gcn_conv.gcn_norm
        self.norm = EdgeWeightNorm()
        self.d = args.d
        self.label_type = args.label_type
        self.dataload = args.dataload
        self.recons = args.recons
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
                    self.module_list.append(AMNet_ms(in_size, args.hidden_dim, 2, 5, 5))
                elif 'multi-scale-bwgnn' == args.model:
                    self.module_list.append(MSGAD(in_size,args.hidden_dim,d=(self.d)))

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
            
    def forward(self,graph,last_batch_node,pos_edges,neg_edges,vis=False):
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
            #import ipdb ; ipdb.set_trace()
        if self.model_str == 'bwgnn':
            recons_a = [self.conv(graph_,feats,dst_nodes)]
        if 'multi-scale' in self.model_str: # g
            recons_a,labels = [],[]
            #import ipdb ; ipdb.set_trace()
            for ind,i in enumerate(self.module_list):
                if self.model_str == 'multi-scale-amnet':
                    recons,res = i(feats,edges,dst_nodes)
                    recons_a.append(recons)
                if ind == 0 and vis == True:
                    plt.figure()
                    L = i.L
                    e,U = np.linalg.eigh(L)
                    e[0] = 0
                    c = np.dot(U.transpose(), feats.detach().cpu().numpy())
                    M = np.zeros((15,c.shape[1]))
                    
                    for j in range(feats.shape[0]):
                        #import ipdb ; ipdb.set_trace()
                        idx = min(int(e[j] / 0.1), 15-1)
                        M[idx] += c[j]**2
                    #for j in range(50):
                    #    plt.plot(M[:,j])
                    M=M/sum(M)
                    plt.plot(M[:,0])
                elif self.model_str == 'multi-scale-bwgnn':
                    recons_a.append(i(graph,feats,dst_nodes))
                if vis == True:
                    c = np.dot(U.transpose(), res.detach().cpu().numpy())
                    M = np.zeros((15,c.shape[1]))
                    
                    for j in range(feats.shape[0]):
                        #import ipdb ; ipdb.set_trace()
                        idx = min(int(e[j] / 0.1), 15-1)
                        M[idx] += c[j]**2
                    #for j in range(50):
                    #    plt.plot(M[:,j])
                    M=M/sum(M)
                    plt.plot(M[:,0])
                    #plt.plot(np.mean(M,axis=1))
            if vis == True:
                plt.legend(['og','sc1','sc2','sc3'])
                plt.savefig(f'filter_vis_{self.label_type}.png')
            # collect multi-scale labels
            g = dgl.graph(graph.edges()).cpu()
            g.edata['_ID'] = graph.edata['_ID'].cpu()
            g.edata['w'] = torch.full(g.edata['_ID'].shape,1.)
            num_a_edges = g.num_edges()

            if not self.dataload:
                if 'norm' in self.label_type:
                    epsilon = 1e-8
                    g.edata['w'] = self.norm(graph,(graph.edata['w']+epsilon)).cpu()
                if 'prods' in self.label_type:
                    labels = []
                    adj_label = graph.adjacency_matrix().to_dense().to(graph.device)
                    for k in range(3):
                        labels.append(adj_label)
                        adj_label = adj_label@adj_label
                elif self.label_type == 'single':
                    labels = []
                    '''
                    edges = graph.has_edges_between(all_edges[:,0],all_edges[:,1]).float()
                    labels_pos_eids=graph.edge_ids(all_edges[:,0][edges.nonzero()].flatten(),all_edges[:,1][edges.nonzero()].flatten())
                    edges[torch.where(edges!=0)[0]] = graph.edata['w'][labels_pos_eids]
                    '''
                    for k in range(3):
                        labels.append(graph.adjacency_matrix().to_dense().to(graph.device))
                if 'random-walk' in self.label_type:
                    nx_graph = nx.to_undirected(dgl.to_networkx(g.cpu()))
                    #node_ids = graph.ndata['_ID']['_N']
                    node_ids = np.arange(g.num_nodes())
                    connected_graphs = [g for g in nx.connected_components(nx_graph)]
                    node_dict = {k:v.item() for k,v in zip(list(nx_graph.nodes),node_ids)}
                    labels = []
                    full_labels = [torch.zeros((g.num_nodes(),g.num_nodes())).to(graph.device),torch.zeros((g.num_nodes(),g.num_nodes())).to(graph.device),torch.zeros((g.num_nodes(),g.num_nodes())).to(graph.device)]
             
                    labels.append(graph.adjacency_matrix().to_dense().to(graph.device))
                    for connected_graph_nodes in connected_graphs:
                        subgraph = dgl.from_networkx(nx_graph.subgraph(connected_graph_nodes))
                        nodes_sel = [node_dict[j] for j in list(connected_graph_nodes)]
                        for i in range(3):
                            D_1 = torch.diag(1 / degree_vector(subgraph))
                            A = subgraph.adjacency_matrix().to_dense()
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
                    
                    adj_label = graph.adjacency_matrix().to_dense().to(graph.device)#[dst_nodes].cuda()
                    num_a_edges = torch.triu(adj_label,1).nonzero().shape[0]
                    labels = [adj_label]
                    for k in range(0, K+1):
                        adj_label_norm = self.norm_adj(adj_label.nonzero().contiguous().T)
                        adj_label_ = torch_geometric.utils.to_dense_adj(adj_label_norm[0])[0]
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
   
                    if vis == True:
                        plt.figure()
                        legend = []
                        for label_ind,label in enumerate(labels):
                            # remove self loop
                            label.fill_diagonal_(0)
                            edge_index = label.nonzero().T
                            edge_weight = torch.ones(edge_index.shape[-1]).to(label.device)
                            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
                            edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                                    'sym', feats.dtype,
                                                                    label.shape[0])
                            edge_weight.masked_fill_(edge_weight == float('inf'), 0)
                            edge_index = edge_index.detach().cpu().numpy() ; edge_weight = edge_weight.detach().cpu().numpy()
                            edge_weight = edge_weight / 2
                            L = np.zeros((label.shape[0],label.shape[0]))
                            L[edge_index[0],edge_index[1]]=edge_weight
                            e,U = np.linalg.eigh(L)
                            e[0] = 0
                            c = np.dot(U.transpose(), feats.detach().cpu().numpy())
                            M = np.zeros((15,c.shape[1]))
                            
                            for j in range(feats.shape[0]):
                                #import ipdb ; ipdb.set_trace()
                                idx = min(int(e[j] / 0.1), 15-1)
                                M[idx] += c[j]**2
                            #for j in range(50):
                            #    plt.plot(M[:,j])
                            M=M/sum(M)
                            y = M[:,0]
                            
                            x = np.arange(y.shape[0])
                            spline = make_interp_spline(x, y)
 
                            # Returns evenly spaced numbers
                            # over a specified interval.
                            X_ = np.linspace(x.min(), x.max(), 500)
                            Y_ = spline(X_)
                            plt.plot(X_,Y_)
                            
                            #plt.plot(y)
                            legend.append(str(label_ind))
                        plt.legend(legend)
                        plt.savefig(f'labels_{self.label_type}.png')
                    labels=[labels[0],labels[4],labels[-1]]
                    '''
                    g= dgl.graph(graph.edges()).cpu()
                    g.edata['w'] = torch.full(graph.edges()[0].shape,1.)
                    epsilon = 1e-8
                    g.edata['w'] = self.norm(g,(g.edata['w']+epsilon)).cpu()
                    import ipdb ; ipdb.set_trace()
                    for k in range(0,K+1):
                        prod_graph = dgl.khop_graph(g,k=k)
                        prod_graphs.append(prod_graph.subgraph(dst_nodes))
                        #prod_graph=dgl.khop_out_subgraph(g, dst_nodes, store_ids=True, k=k+1)[0]
                        #prod_graphs.append(g.edge_subgraph(prod_graph.edata['_ID'],relabel_nodes=False))
                    import ipdb ; ipdb.set_trace()
                    #full_e = torch.zeros(prod_graphs[-1].num_edges())
                    #full_eids = prod_graphs[-1].edata['_ID']
                    basis_ = prod_graphs[0]
                    #basis_ = dgl.khop_out_subgraph(g, dst_nodes, store_ids=True, k=0)[0]
                    #basis_ = g.edge_subgraph(basis_.edata['_ID'],relabel_nodes=False)
                    
                    #basis_.edata['w']=torch.ceil(basis_.edata['w'])
                    #labels_edges=basis_.has_edges_between(all_edges[:,0].cpu(),all_edges[:,1].cpu()).float()
                    #labels_pos_eids=basis_.edge_ids(all_edges[:,0].cpu()[labels_edges.nonzero()].flatten(),all_edges[:,1].cpu()[labels_edges.nonzero()].flatten())
                    #labels_edges[torch.where(labels_edges!=0)[0]] = basis_.edata['w'][labels_pos_eids]
                    #labels.append(labels_edges.to(graph.device))
                    
                    #basis_.edata['w']=torch.full(g.edges()[0].shape,1.)
                    #basis_.edata['w'] = self.norm(basis_,(basis_.edata['w']+epsilon)).cpu()
                    #labels.append(basis_.adjacency_matrix().to_dense().to(graph.device))
                    for k in range(0, K+1):
                        coeff = coeffs[k]
                        edata = (basis_.edata['w'] * coeff[0])
                        basis = copy.deepcopy(basis_)
                        basis.edata['w'] = edata
                        for i in range(1, K+1):
                            prod_graph = prod_graphs[i]
                            prod_graph.edata['w']*=coeff[i]
                            basis = dgl.adj_sum_graph([prod_graph,basis],'w')
                        if 'sample' in self.label_type:
                            drop_edges = torch.argsort(-basis.edata['w'])[num_a_edges:]
                            basis = dgl.remove_edges(basis,drop_edges)
                            basis.edata['w']=torch.sigmoid(basis.edata['w'])
                        if 'round' in self.label_type:
                            basis.edata['w'][torch.where(basis.edata['w']<0)]=0
                            basis.edata['w'][torch.where(basis.edata['w']>0)]=1
                            #basis.edata['w'][basis.edata['w'].nonzero()]=1.
                        labels_edges=basis.has_edges_between(all_edges[:,0].cpu(),all_edges[:,1].cpu()).float()
                        labels_pos_eids=basis.edge_ids(all_edges[:,0].cpu()[labels_edges.nonzero()].flatten(),all_edges[:,1].cpu()[labels_edges.nonzero()].flatten())
                        labels_edges[torch.where(labels_edges!=0)[0]] = basis.edata['w'][labels_pos_eids]
                        labels.append(labels_edges.to(graph.device))
                        #labels.append(basis.adjacency_matrix().to_dense().to(graph.device))
                    '''
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
        elif not self.dataload:
            labels = [graph.adjacency_matrix().to_dense().to(graph.device)]
        # single scale reconstruction label
        '''
        if 'multi-scale' not in self.model_str:
            edges = graph.has_edges_between(all_edges[:,0],all_edges[:,1]).float()
            #labels_pos_eids=graph.edge_ids(all_edges[:,0][edges.nonzero()].flatten(),all_edges[:,1][edges.nonzero()].flatten())
            #edges[torch.where(edges!=0)[0]] = 1.#graph.edata['w'][labels_pos_eids]
            labels = [edges]
        '''
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
            #recons = [recons[graph.dstnodes()][:,graph.dstnodes()]]
            #recons_a = [recons_a[dst_nodes][:,dst_nodes]]

        elif self.model_str == 'gradate':
            recons = [loss,ano_score]
        return recons_a,recons_x, labels
