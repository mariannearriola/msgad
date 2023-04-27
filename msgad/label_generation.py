import dgl
import torch
import numpy as np
import sympy
import math
import scipy
import gc
import scipy.sparse as sp
import os
from numpy import polynomial
import copy
import networkx as nx
import torch_geometric
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

class LabelGenerator:
    def __init__(self,graph,dataset,model_str,epoch,label_type,feats,vis,vis_name,anoms):
        self.graph = graph
        self.feats = feats
        self.dataset=dataset
        self.epoch = epoch
        self.model_str = model_str
        self.norm_adj = torch_geometric.nn.conv.gcn_conv.gcn_norm
        self.label_type = label_type
        self.vis_name = vis_name
        self.vis = vis
        self.anoms= anoms

    def flatten_label(self,anoms):
        anom_flat = anoms[0]
        for i in anoms[1:]:
            anom_flat=np.concatenate((anom_flat,i))
        return anom_flat

    def filter_anoms(self,labels,anoms,vis_name,img_num=None):
        #np.random.seed(seed=1)
        print('filter anoms')
        signal = np.random.randn(labels[0].shape[0],self.feats.shape[-1])

        #anom_tot = 0
        #for anom in anoms[:-1]:
        #    anom_tot += self.flatten_label(anom).shape[0]
        #anom_tot += anoms[-1].shape[0]
        for i,label in enumerate(labels):
            lbl=torch.maximum(label, label.T)
            try:
                e,U = self.get_spectrum(lbl)
            except Exception as e:
                print(e)
                import ipdb ; ipdb.set_trace()
            e = e.to(self.graph.device)
            U = U.to(self.graph.device)
            del lbl ; torch.cuda.empty_cache() ; gc.collect()
           
            plt.figure()
            try:
                self.plot_spectrum(e.detach().cpu(),U.detach().cpu(),signal+1)
            except Exception as e:
                print(e)
                import ipdb ; ipdb.set_trace()
                
            for anom_ind,anom in enumerate(anoms.values()):
                #anom = anom.flatten()
                if anom_ind != len(anoms)-1:
                    anom = self.flatten_label(anom)
                signal_ = np.copy(signal)
                signal_[anom]*=(400)#*(anom_tot/anom.shape[0]))
                signal_ += 1
                self.plot_spectrum(e.detach().cpu(),U.detach().cpu(),signal_)
                plt.legend(['no anom signal','sc1 anom signal','sc2 anom signal','sc3 anom signal','single anom signal'])
      
                fpath = f'vis/filter_anom_vis/{self.dataset}/{self.model_str}/{self.label_type}'
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                if img_num:
                    plt.savefig(f'{fpath}/filter_vis_{self.label_type}_{self.epoch}_{vis_name}_filter{img_num}.png')
                else:
                    plt.savefig(f'{fpath}/filter_vis_{self.label_type}_{self.epoch}_{vis_name}_filter{i}.png')
            del e,U ; torch.cuda.empty_cache() ; gc.collect()
            
        torch.cuda.empty_cache()


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
        #print('eig', torch.cuda.memory_allocated()/torch.cuda.max_memory_reserved())
        try:
            e,U = torch.linalg.eigh(L)
        except Exception as e:
            print(e)
            return
        del L, D ; torch.cuda.empty_cache() ; gc.collect()
        #print('eig done', torch.cuda.memory_allocated()/torch.cuda.max_memory_reserved())
        '''
        try:
            assert -1e-5 < e[0] < 1e-5
        except:
            print('Eigenvalues out of bounds')
            import ipdb ; ipdb.set_trace()
            e[0] = 0
        '''
        return e, U.to(torch.float32)

    def plot_spectrum(self,e,U,signal,color=None):
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
        if color:
            plt.plot(X_,Y_,color=color)
        else:
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
            
            adj_label = self.graph.adjacency_matrix()
            adj_label.sparse_resize_((adj_label.size(0), adj_label.size(0)), adj_label.sparse_dim(), adj_label.dense_dim())
            adj_label = adj_label.to_dense()[self.graph.dstnodes()][:,self.graph.dstnodes()]
            adj_label=np.maximum(adj_label, adj_label.T).to(self.graph.device) #?
            '''
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
            #g=dgl.graph(('coo',(adj_label_.indices()))).to('cpu')
            '''
            #g.edata['w']=adj_label_.values()
            #graph_.add_edges(graph_.dstnodes(),graph_.dstnodes())
            
            #adj_label = self.graph.adjacency_matrix()#[self.graph.dstnodes()]
            #adj_label.sparse_resize_((adj_label.size(0), adj_label.size(0)), adj_label.sparse_dim(), adj_label.dense_dim())
            #adj_label = adj_label.to_dense()
            #adj_label=np.maximum(adj_label, adj_label.T).to(self.graph.device)  #?
            #prods = [g]
            #adj_label_norm_ = adj_label_norm_.cuda()
            #del adj_label_norm, adj_label_norm_, adj_label ; torch.cuda.empty_cache()
            #g_prod = copy.deepcopy(g)
            #for i in range(1,5):
            #    print('prod',i)
            #    g_prod = dgl.adj_product_graph(g_prod,g,'w')
            #    labels.append(g_prod.adjacency_matrix().to_dense().to(self.graph.device))
            #    torch.cuda.empty_cache() #?
            #coeffs = self.calculate_theta2(5)
            #labels.append(adj_label)
            adj_label_ = adj_label
            for k in range(5):
                upper_tri=torch.triu(adj_label_,1)
                nz = upper_tri[upper_tri.nonzero()[:,0],upper_tri.nonzero()[:,1]]
                sorted_idx = torch.argsort(-nz)
                drop_idx=upper_tri.nonzero()[sorted_idx][num_a_edges:]
                keep_idx=upper_tri.nonzero()[sorted_idx][:num_a_edges]
                adj_label_[drop_idx[:,0],drop_idx[:,1]]=0.
                adj_label_[drop_idx[:,1],drop_idx[:,0]]=0.
                adj_label_[keep_idx[:,0],keep_idx[:,1]]=1.
                adj_label_[keep_idx[:,1],keep_idx[:,0]]=1.
                #adj_label_.fill_diagonal_(1.)
                labels.append(adj_label_)
                adj_label_ = adj_label_@adj_label
            labels = [labels[1],labels[2],labels[3]]
            

        elif self.label_type == 'single':
            labels = []
            '''
            edges = graph.has_edges_between(all_edges[:,0],all_edges[:,1]).float()
            labels_pos_eids=graph.edge_ids(all_edges[:,0][edges.nonzero()].flatten(),all_edges[:,1][edges.nonzero()].flatten())
            edges[torch.where(edges!=0)[0]] = graph.edata['w'][labels_pos_eids]
            '''
            for k in range(3):
                adj = self.graph.adjacency_matrix().to_dense()[self.graph.dstnodes()]
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
                #label_idx=[1,2,3]
                #label_idx=[0]
                label_idx=[0,1,2,3,4,5,6,7,8,9,10]
                #label_idx = [0,1]
                #label_idx = [6,8,10]
                #label_idx = [2,3,4]
            elif 'bwgnn' in self.label_type:
                coeffs = self.calculate_theta2(K)
                label_idx=[0,1,2,3,4,5]#,6,7,8,9,10]
                #label_idx = [4,5,6]#,2]
                #label_idx=[2,4]
            labels = []
            adj_label = self.graph.adjacency_matrix()
            adj_label.sparse_resize_((adj_label.size(0), adj_label.size(0)), adj_label.sparse_dim(), adj_label.dense_dim())
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
            #import ipdb ; ipdb.set_trace()
            num_a_edges = self.graph.num_edges()
            print('num a edges',num_a_edges)
        
            # NOTE : remove if og wanted
            #labels = [adj_label.to(self.graph.device)]

            del adj_label, adj_label_norm_, adj_label_ ; torch.cuda.empty_cache()
            
            prods = [g]
            g_prod = copy.deepcopy(g)
            for i in range(1, K+1):
                print('prod',i)
                g_prod = dgl.adj_product_graph(g_prod,g,'w')
                prods.append(g_prod)

            del g_prod ; torch.cuda.empty_cache()
            print('generating labels')

            #labels=[]
            for label_id in label_idx:
                print(f'label_id {label_id}')
                coeff = coeffs[label_id]
                basis = copy.deepcopy(g)
                basis.edata['w'] *= coeff[0]
                #basis = prods[0] * coeff[0]
                for i in range(1, K+1):
                    basis_ = prods[i]
                    basis_.edata['w'] *= coeff[i]
                    basis = dgl.adj_sum_graph([basis,basis_],'w')
                    #basis += prods[i] * coeff[i]
                    del basis_
                    torch.cuda.empty_cache()
                    
                nz = basis.edata['w']#.unique()
                if nz.shape[0] < num_a_edges:
                    print('hi')
                    basis_ = basis.adjacency_matrix().to_dense()
                    basis_[basis.edges()[0]][:,basis.edges()[1]] = 1.
                    basis_ = torch.nn.functional.pad(basis_,(0,zero_rows,0,zero_rows))
                else:
                    #print(basis.number_of_edges())
                    basis_ = dgl.sampling.select_topk(basis, int(num_a_edges/basis.number_of_nodes()), 'w')
                    print(basis_.number_of_edges())
                    basis_ = basis_.adjacency_matrix().to_dense()
                    '''
                    sorted_idx = torch.topk(nz,num_a_edges).indices
                    basis = dgl.edge_subgraph(basis,sorted_idx,relabel_nodes=False)
                    basis_ = basis.adjacency_matrix().to_dense()
                    #basis_[basis.edges()[0]][:,basis.edges()[1]] = 1.
                    basis_ = torch.nn.functional.pad(basis_,(0,zero_rows,0,zero_rows))
                    # TO ADD:  basis_.fill_diagonal_(1.)
                    del nz, sorted_idx
                    '''
                
                if self.vis == True and 'test' not in self.vis_name:
                    self.filter_anoms([basis_],self.anoms,self.vis_name,label_id)
                else:
                    labels.append(basis_.to(self.graph.device))
                del basis, basis_ ; torch.cuda.empty_cache()
            
            for i in range(len(prods)):
                del prods[0]
            del g
            torch.cuda.empty_cache()
            print('done')
            #print("Seconds to get labels", (time.time()-seconds)/60)
            
        if self.vis == True and 'test' not in self.vis_name:
            print('visualizing labels')
            
            plt.figure()
            #import ipdb ; ipdb.set_trace()
            adj_label = self.graph.adjacency_matrix()
            adj_label.sparse_resize_((adj_label.size(0), adj_label.size(0)), adj_label.sparse_dim(), adj_label.dense_dim())
            adj_label = adj_label.to_dense()[self.graph.dstnodes()][:,self.graph.dstnodes()]
            adj_label=np.maximum(adj_label, adj_label.T).to(self.graph.device) #?
            
            legend = []
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
                del e,U,label ; torch.cuda.empty_cache() ; gc.collect()
                legend.append(f'label {str(label_ind+1)}')

            legend.append('original adj.')
            e_adj,U_adj = self.get_spectrum(torch.tensor(adj_label).to(self.graph.device))
            self.plot_spectrum(e_adj, U_adj,self.feats[self.graph.dstnodes()],color="cyan")
            del e_adj, U_adj ; torch.cuda.empty_cache() ; gc.collect()

            plt.legend(legend)
            fpath = f'vis/label_vis/{self.dataset}/{self.model_str}/{self.label_type}'
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            plt.savefig(f'{fpath}/labels_{self.label_type}_{self.epoch}_{self.vis_name}.png')
            #print("Seconds to visualize labels", (time.time()-seconds)/60)
            
            self.filter_anoms(labels,self.anoms,self.vis_name)

        return labels
