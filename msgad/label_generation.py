import dgl
import torch
import numpy as np
import sympy
import math
import os
from numpy import polynomial
import copy
import networkx as nx
import torch_geometric
from scipy.interpolate import make_interp_spline

class LabelGenerator:
    def __init__(self,graph,feats,vis,vis_name,anoms):
        self.graph = graph
        self.feats = feats
        self.vis_name = vis_name
        self.vis = vis
        self.anoms= anoms

    def calculate_theta2(self,d):
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

    def get_bern_coeff(self,degree):
        def Bernstein(de, i):
            coefficients = [0, ] * i + [math.comb(de, i)]
            first_term = polynomial.polynomial.Polynomial(coefficients)
            second_term = polynomial.polynomial.Polynomial([1, -1]) ** (de - i)
            return first_term * second_term
        out = []
        for i in range(degree + 1):
            out.append(Bernstein(degree, i).coef)
        return out


    def degree_vector(self,G):
        """
        Degree vector for the input graph.
        :param G: Input graph.
        :return: Degree vector.
        """
        return torch.tensor([a for a in sorted(G.out_degrees()[G.dstnodes()], key=lambda a: a.item())]).to(G.device)

    def normalize_adj(self,adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def get_spectrum(self,mat):
    
        d_ = np.zeros(mat.shape[0])
        degree_in = np.ravel(mat.sum(axis=0).detach().cpu().numpy())
        degree_out = np.ravel(mat.sum(axis=1).detach().cpu().numpy())
        dw = (degree_in + degree_out) / 2
        disconnected = (dw == 0)
        np.power(dw, -0.5, where=~disconnected, out=d_)
        D = scipy.sparse.diags(d_)
        L = torch.eye(mat.shape[0])
        L -= torch.tensor(D * mat.detach().cpu().numpy() * D)
        L[disconnected, disconnected] = 0
        L = L.to(mat.device)
        print('eig')
        try:
            e,U = torch.linalg.eigh(L)
        except Exception as e:
            print(e)
            return
        del L, D
        print('eig done')
        '''
        try:
            assert -1e-5 < e[0] < 1e-5
        except:
            print('Eigenvalues out of bounds')
            import ipdb ; ipdb.set_trace()
            e[0] = 0
        '''
        return e, U.to(torch.float32)

    def plot_spectrum(self,e,U,signal):
        c = U.T@signal
        M = torch.zeros((15,c.shape[1])).to(e.device)
        for j in range(e.shape[0]):
            idx = min(int(e[j] / 0.1), 15-1)
            M[idx] += c[j]**2
        M=M/sum(M)
        y = M[:,0].detach().cpu().numpy()
        x = np.arange(y.shape[0])
        try:
            spline = make_interp_spline(x, y)
        except:
            import ipdb ; ipdb.set_trace()
        X_ = np.linspace(x.min(), x.max(), 500)
        Y_ = spline(X_)
        plt.xlabel('lambda')
        plt.plot(X_,Y_)

    def construct_labels(self):
        """
        Stationary distribution given the transition matrix.
        :param M: Transition matrix.
        :return: Stationary distribution.
        """
        #g = dgl.graph(graph.edges()).cpu()
        #g.edata['_ID'] = graph.edata['_ID'].cpu()
        #g.edata['w'] = torch.full(g.edata['_ID'].shape,1.)
        num_a_edges = self.graph.num_edges()
        if 'norm' in self.label_type:
            epsilon = 1e-8
            g.edata['w'] = self.norm(self.graph,(self.graph.edata['w']+epsilon)).cpu()
        if 'prods' in self.label_type:
            labels = []
            #graph_.add_edges(graph_.dstnodes(),graph_.dstnodes())
            adj_label = self.graph.adjacency_matrix()
            adj_label.sparse_resize_((adj_label.size(0), adj_label.size(0)), adj_label.sparse_dim(), adj_label.dense_dim())
            adj_label = adj_label.to_dense()
            adj_label=np.maximum(adj_label, adj_label.T).to(self.graph.device)  #?
            for k in range(5):
                adj_label_ = adj_label
                upper_tri=torch.triu(adj_label_,1)
                nz = upper_tri[upper_tri.nonzero()[:,0],upper_tri.nonzero()[:,1]]
                sorted_idx = torch.argsort(-nz)
                drop_idx=upper_tri.nonzero()[sorted_idx][num_a_edges:]
                keep_idx=upper_tri.nonzero()[sorted_idx][:num_a_edges]
                adj_label_[drop_idx[:,0],drop_idx[:,1]]=0.
                adj_label_[drop_idx[:,1],drop_idx[:,0]]=0.
                adj_label_[keep_idx[:,0],keep_idx[:,1]]=1.
                adj_label_[keep_idx[:,1],keep_idx[:,0]]=1.
                labels.append(adj_label_)
                adj_label = adj_label@adj_label
            

        elif self.label_type == 'single':
            labels = []
            '''
            edges = graph.has_edges_between(all_edges[:,0],all_edges[:,1]).float()
            labels_pos_eids=graph.edge_ids(all_edges[:,0][edges.nonzero()].flatten(),all_edges[:,1][edges.nonzero()].flatten())
            edges[torch.where(edges!=0)[0]] = graph.edata['w'][labels_pos_eids]
            '''
            for k in range(3):
                adj = self.graph.adjacency_matrix().to_dense()
                adj=np.maximum(adj, adj.T).to(self.graph.device) 
                labels.append(adj)
            
        if 'random-walk' in self.label_type:
            nx_graph = nx.to_undirected(dgl.to_networkx(g.cpu()))
            #node_ids = graph.ndata['_ID']['_N']
            node_ids = np.arange(g.num_nodes())
            connected_graphs = [g for g in nx.connected_components(nx_graph)]
            node_dict = {k:v.item() for k,v in zip(list(nx_graph.nodes),node_ids)}
            labels = []
            full_labels = [torch.zeros((g.num_nodes(),g.num_nodes())).to(self.graph.device),torch.zeros((g.num_nodes(),g.num_nodes())).to(graph.device),torch.zeros((g.num_nodes(),g.num_nodes())).to(graph.device)]
            adj = self.graph.adjacency_matrix().to_dense()
            adj=np.maximum(adj, adj.T).to(self.graph.device) 
            labels.append(adj)
            for connected_graph_nodes in connected_graphs:
                subgraph = dgl.from_networkx(nx_graph.subgraph(connected_graph_nodes))
                nodes_sel = [node_dict[j] for j in list(connected_graph_nodes)]
                for i in range(3):
                    D_1 = torch.diag(1 / self.degree_vector(subgraph))
                    A = subgraph.adjacency_matrix().to_dense()
                    A=np.maximum(A, A.T)
                    M = torch.matmul(D_1, A)

                    pi = self.stationary_distribution(M.to(self.graph.device),self.graph.device)
                    Pi = np.diag(pi)
                    M_tau = np.linalg.matrix_power(A.detach().cpu().numpy(), self.taus[i])

                    R = np.log(Pi @ M_tau/self.b) - np.log(np.outer(pi, pi))
                    R = R.copy()
                    # Replace nan with 0 and negative infinity with min value in the matrix.
                    R[np.isnan(R)] = 0
                    R[np.isinf(R)] = np.inf
                    R[np.isinf(R)] = R.min()
                    res = torch.tensor(R).to(self.graph.device)
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
                
                full_labels[i][torch.where(full_labels[i]<0)[0],torch.where(full_labels[i]<0)[1]]=0
                labels.append(full_labels[i].to(self.graph.device))

        if 'filter' in self.label_type:   
            K = 10
            if 'amnet' in self.label_type:
                coeffs =  self.get_bern_coeff(K)
                #label_idx = [0,1,2,3,4,5,6,7,8,9,10]
                label_idx=[1,2,3]
            elif 'bwgnn' in self.label_type:
                coeffs = self.calculate_theta2(K)
                #label_idx=[0,1,2,3,5,6,7,8,9,10]
                label_idx = [0,1,2]
            labels = []
            adj_label = self.graph.adjacency_matrix()
            adj_label.sparse_resize_((adj_label.size(0), adj_label.size(0)), adj_label.sparse_dim(), adj_label.dense_dim())
            #adj_label = dgl.block_to_graph(graph).subgraph({'_N_src':graph.dstnodes(),'_N_dst':graph.dstnodes()})
            adj_label = adj_label.to_dense()[self.graph.dstnodes()][:,self.graph.dstnodes()]
            adj_label=np.maximum(adj_label, adj_label.T)#.to(graph.device) #?
            if self.vis==True and 'test' not in self.vis_name:
                e_adj,U_adj = self.get_spectrum(torch.tensor(adj_label).to(self.graph.device))
            num_a_edges = torch.triu(adj_label,1).nonzero().shape[0]
            adj_label_norm = self.norm_adj(adj_label.nonzero().contiguous().T)
            adj_label_norm_ = torch_geometric.utils.to_dense_adj(adj_label_norm[0])[0]
            adj_label_norm_[adj_label_norm[0][0],adj_label_norm[0][1]]=adj_label_norm[1]
            adj_label_norm_=adj_label_norm_
            adj_label_ = adj_label_norm_#.to(graph.device)
            zero_rows = adj_label.shape[0] - adj_label_.shape[0]
            adj_label_ = torch.nn.functional.pad(adj_label_,(0,zero_rows,0,zero_rows))
            adj_label_ = adj_label_.to_sparse()
            g=dgl.graph(('coo',(adj_label_.indices()))).to('cpu')
            g.edata['w']=adj_label_.values()
            num_a_edges = self.graph.num_edges()
            
            g_prod = g
            adj_label_ = g_prod.adjacency_matrix().to_dense()#.to(graph.device)
            adj_label_[g_prod.edges()]=g_prod.edata['w']
            prods = [g]
            adj_label_norm_ = adj_label_norm_.cuda()
            del adj_label_norm, adj_label_norm_, adj_label ; torch.cuda.empty_cache()
            
            for i in range(1, K+1):
                g_prod = dgl.adj_product_graph(g_prod,g,'w')
                prods.append(g_prod)
                torch.cuda.empty_cache() #?
                
            del g_prod ; torch.cuda.empty_cache()
            
            for label_id in label_idx:
                coeff = coeffs[label_id]
                basis = copy.deepcopy(g)
                g_prod = copy.deepcopy(g)
                basis.edata['w'] *= coeff[0]
                for i in range(1, K+1):
                    basis_ = prods[i-1]
                    basis_.edata['w'] *= coeff[i]
                    basis = dgl.adj_sum_graph([basis,basis_],'w')

                    del basis_
                    torch.cuda.empty_cache()
                    
                del g_prod ; torch.cuda.empty_cache()
                nz = basis.edata['w'].unique()
                if nz.shape[0] < num_a_edges:
                    basis_ = basis.adjacency_matrix().to_dense()
                    basis_[basis.edges()[0]][:,basis.edges()[1]] = 1.
                    basis_ = torch.nn.functional.pad(basis_,(0,zero_rows,0,zero_rows))
                else:
                    try:
                        sorted_idx = torch.min(torch.topk(nz,num_a_edges).values)
                    except Exception as e:
                        print(e)
                        import ipdb ; ipdb.set_trace()

                    edges_rem=torch.where(basis.edata['w']<sorted_idx)[0]
                    
                    basis=dgl.remove_edges(basis,edges_rem)
                    basis_ = basis.adjacency_matrix().to_dense()
                    basis_[basis.edges()[0]][:,basis.edges()[1]] = 1.
                    basis_ = torch.nn.functional.pad(basis_,(0,zero_rows,0,zero_rows))
                    del nz, sorted_idx
                
                labels.append(basis_.to(self.graph.device))
                del basis, basis_ ; torch.cuda.empty_cache()
            for i in range(len(prods)):
                del prods[0]
            torch.cuda.empty_cache()
            #print("Seconds to get labels", (time.time()-seconds)/60)

        if self.vis == True and 'test' not in self.vis_name:
            plt.figure()
            self.plot_spectrum(e_adj, U_adj,self.feats[self.graph.dstnodes()])
            legend = ['original adj.']
            for label_ind,label in enumerate(labels):
                print(label_ind)
                if not torch.equal(label,label.T):
                    label=torch.maximum(label, label.T).to(self.graph.device) 
                try:
                    e,U = self.get_spectrum(label)
                    e = e.to(self.graph.device)
                    U = U.to(self.graph.device)
                except Exception as e:
                    print(e)
                    import ipdb ; ipdb.set_trace()
                try:
                    self.plot_spectrum(e,U,self.feats[self.graph.dstnodes()])
                except Exception as e:
                    print(e)
                    import ipdb ; ipdb.set_trace()
                legend.append(f'label {str(label_ind+1)}')
            plt.legend(legend)
            fpath = f'vis/label_vis/{self.dataset}/{self.model_str}/{self.label_type}'
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            plt.savefig(f'{fpath}/labels_{self.label_type}_{self.epoch}_{self.vis_name}.png')
            #print("Seconds to visualize labels", (time.time()-seconds)/60)
            del e,U ; torch.cuda.empty_cache()
            self.filter_anoms(labels,self.anoms,self.vis_name)
            
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
