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
from dgl.nn.pytorch.conv import EdgeWeightNorm
import matplotlib.pyplot as plt

class LabelGenerator:
    def __init__(self,graph,dataset,model_str,epoch,label_type,feats,vis,vis_name,anoms,batch_size):
        self.graph = graph
        self.feats = feats
        self.dataset=dataset
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_str = model_str
        #self.norm_adj = torch_geometric.nn.conv.gcn_conv.gcn_norm
        self.norm = EdgeWeightNorm(norm='both')
        self.label_type = label_type
        self.vis_name = vis_name
        self.vis = vis
        self.anoms= anoms

    def flatten_label(self,anoms):
        anom_flat = anoms[0]
        for i in anoms[1:]:
            anom_flat=np.concatenate((anom_flat,i))
        return anom_flat

    def anom_response(self,adj,labels,anoms,vis_name,img_num=None):
        signal = np.random.randn(self.feats.shape[0],self.feats.shape[0])

        anom_tot = 0
        for anom in list(anoms.values())[:-1]:
            anom_tot += self.flatten_label(anom).shape[0]
        anom_tot += list(anoms.values())[-1].shape[0]
        all_nodes = torch.arange(self.feats.shape[0])
        e,U = self.get_spectrum(adj.to(torch.float64))
        es,us=[e],[U]
        del e, U ; torch.cuda.empty_cache() ; gc.collect()
        for i,label in enumerate(labels):
            lbl=torch.maximum(label, label.T)
            e,U = self.get_spectrum(lbl.to(torch.float64))
            es.append(e) ; us.append(U)
            del lbl, e, U ; torch.cuda.empty_cache() ; gc.collect()

        for anom_ind,anom in enumerate(anoms.values()):
            plt.figure()
            legend = []
            for i,label in enumerate(range(len(es))):
                e,U = es[i],us[i]
                e = e.to(self.graph.device) ; U = U.to(self.graph.device)

                #anom = anom.flatten()
                if len(anom) == 0:
                    continue
                if anom_ind != len(anoms)-1:
                    anom_f = self.flatten_label(anom)
                else:
                    anom_f = anom
                #anom_mask=np.setdiff1d(all_nodes,anom_f)
                signal_ = copy.deepcopy(signal)+1
                signal_[anom_f]=(np.random.randn(U.shape[0])*400*anom_tot/anom.shape[0])+1# NOTE: tried 10#*(anom_tot/anom.shape[0]))
                
                self.plot_spectrum(e.detach().cpu(),U.detach().cpu(),signal_)
                if i == 0:
                    legend.append('original adj')
                else:
                    legend.append(f'{i} label')
            plt.legend(legend)

            fpath = f'vis/filter_anom_ev/{self.dataset}/{self.model_str}/{self.label_type}/{self.epoch}'
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            plt.savefig(f'{fpath}/filter_vis_{self.label_type}_{vis_name}_filter{list(anoms.keys())[anom_ind]}.png')
            del e,U ; torch.cuda.empty_cache() ; gc.collect()
                
            torch.cuda.empty_cache()

    def filter_anoms(self,labels,anoms,vis_name,img_num=None):
        np.random.seed(123)
        #print('filter anoms',torch.cuda.memory_allocated()/torch.cuda.max_memory_reserved())
        signal = np.random.randn(self.feats.shape[0],self.feats.shape[0])

        #signal = np.ones((labels[0].shape[0],self.feats.shape[-1]))

        anom_tot = 0
        for anom in list(anoms.values())[:-1]:
            anom_tot += self.flatten_label(anom).shape[0]
        anom_tot += list(anoms.values())[-1].shape[0]
        all_nodes = torch.arange(self.feats.shape[0])

        for i,label in enumerate(labels):
            lbl=torch.maximum(label, label.T)
            try:
                e,U = self.get_spectrum(lbl.to(torch.float64))
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
            legend_arr = ['no anom signal','sc1 anom signal','sc2 anom signal','sc3 anom signal','single anom signal']
            legend =['no anom signal']
            for anom_ind,anom in enumerate(anoms.values()):
                #anom = anom.flatten()
                if len(anom) == 0:
                    continue
                if anom_ind != len(anoms)-1:
                    anom_f = self.flatten_label(anom)
                else:
                    anom_f = anom
                #anom_mask=np.setdiff1d(all_nodes,anom_f)
                signal_ = copy.deepcopy(signal)+1
                try:
                    signal_[anom_f]=(np.random.randn(U.shape[0])*400*anom_tot/anom.shape[0])+1# NOTE: tried 10#*(anom_tot/anom.shape[0]))
                except Exception as e:
                    print(e)
                    import ipdb ; ipdb.set_trace()
                
                self.plot_spectrum(e.detach().cpu(),U.detach().cpu(),signal_)
                legend.append(legend_arr[anom_ind+1])
                plt.legend(legend)
      
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
        #L = torch.eye(mat.shape[0])
        mat_sparse = scipy.sparse.csr_matrix(mat.detach().cpu().numpy())
        L = scipy.sparse.identity(mat.shape[0]) - D * mat_sparse * D
        #L -= torch.tensor(D * mat.detach().cpu().numpy() * D)
        L[disconnected, disconnected] = 0
        L.eliminate_zeros()
        #L = L.to(mat.device)
        '''
        edges = torch.vstack((mat.edges()[0],mat.edges()[1])).T
        L = torch_geometric.utils.get_laplacian(edges.T,torch.sigmoid(mat.edata['w']),normalization='sym',num_nodes=self.feats.shape[0])
        L = torch_geometric.utils.to_dense_adj(L[0],edge_attr=L[1]).to(mat.device)[0]
        #print('eig', torch.cuda.memory_allocated()/torch.cuda.max_memory_reserved())
        '''
        try:
            e,U = torch.linalg.eigh(torch.tensor(L.toarray()).to(mat.device))
            #e,U=scipy.linalg.eigh(np.asfortranarray(L.toarray()), overwrite_a=True)
            #e,U=scipy.linalg.eigh(np.asfortranarray(L.detach().cpu().numpy()), overwrite_a=True)
            assert -1e-5 < e[0] < 1e-5
            e[0] = 0
        except Exception as e:
            print(e)
            return
        e,U = torch.tensor(e).to(mat.device),torch.tensor(U).to(mat.device)
        #del L, edges ; torch.cuda.empty_cache() ; gc.collect()
        #import ipdb ; ipdb.set_trace()
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
        return e, U#.to(torch.float32)

    def plot_spectrum(self,e,U,signal,color=None):
        #import ipdb ; ipdb.set_trace()
        c = U.T@signal
        M = torch.zeros((10,c.shape[1])).to(e.device).to(U.dtype)
        for j in range(c.shape[0]):
            idx = min(int(e[j] / 0.1), 10-1)
            M[idx] += c[j]**2
        M=M/sum(M)
        #y = M[:,0].detach().cpu().numpy()
        #print('nans',torch.where(torch.isnan(M))[0].shape)
        M[torch.where(torch.isnan(M))]=0
        y = torch.mean(M,axis=1).detach().cpu().numpy()
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
            if 'cora' in self.dataset:
                K = 5
            else:
                K = 5
                
            if 'amnet' in self.label_type:
                coeffs =  self.get_bern_coeff(K)
                label_idx=np.arange(K+1)
                if 'cora' in self.dataset:
                    label_idx = [0,2,5]
                #    label_idx=[0,2,4]
                #    label_idx = [1,2,10]
                #    label_idx = [1,3]
                #    label_idx = [0,9]
                #    label_idx = [2,5]
                #    label_idx = np.add(label_idx,1)
                if 'elliptic' in self.dataset and self.batch_size != 0:
                    label_idx = [6,8,10]
                    label_idx = np.add(label_idx,1)
                if 'weibo' in self.dataset: label_idx = [0,1,5] # try 0,1,5 (remove original if original label bad)
            elif 'bwgnn' in self.label_type:
                coeffs = self.calculate_theta2(K)
                label_idx=np.arange(K+1)
                #if 'cora' in self.dataset:
                #    label_idx = [1,5]
            labels = []
            adj_label = None

            dgl.seed(123)
            g = self.graph.to('cpu')#self.graph.device)#.to('cpu')
            if 'cora' in self.dataset: g = g.to(self.graph.device)
            g = dgl.to_homogeneous(g).subgraph(g.srcnodes())
            num_a_edges = g.num_edges()
            if 'cora' in self.dataset:
                g.edata['w'] = self.norm(g,torch.ones(num_a_edges).to(self.graph.device)).to(self.graph.device)
            else:
                g.edata['w'] = self.norm(g,torch.ones(num_a_edges).to('cpu')).to('cpu')#(self.graph.device)).to(self.graph.device)
            adj_label = g.adjacency_matrix().to_dense()
            #if self.vis == True and 'test' not in self.vis_name:
            self.filter_anoms([adj_label.to(self.graph.device)],self.anoms,self.vis_name,'og')
            
            if 'cora' not in self.dataset: print('num a edges',num_a_edges)
        
            # NOTE : remove if og wanted
            #if 'cora' in self.dataset:# or 'weibo' in self.dataset:
            #    labels = [adj_label.to(self.graph.device)]
            if adj_label is not None: del adj_label ; torch.cuda.empty_cache()
            
            prods = [g]
            g_prod = copy.deepcopy(g)
            for i in range(1, K+1):
                #if self.graph.device == 'cpu':
                if 'cora' not in self.dataset: print('prod',i)
                g_prod = dgl.adj_product_graph(g_prod,g,'w')
                prods.append(g_prod)

            del g_prod ; torch.cuda.empty_cache()
            #print('generating labels')
            if 'cora' not in self.dataset:
                g = g.to('cpu')#(self.graph.device)
            
            for label_id in label_idx:
                #if self.graph.device == 'cpu':
                if 'cora' not in self.dataset: print(f'label_id {label_id}')
                coeff = coeffs[label_id]
                basis = copy.deepcopy(g)
                basis.edata['w'] *= coeff[0]
                
                #basis = prods[0] * coeff[0]
                for i in range(1, K+1):
                    basis_ = copy.deepcopy(prods[i])
                    basis_.edata['w'] *= coeff[i]
                    basis = dgl.adj_sum_graph([basis,basis_],'w')
                    #basis += prods[i] * coeff[i]
                    del basis_
                    torch.cuda.empty_cache()
                    
                nz = basis.edata['w']#.unique()
                if False:#nz.shape[0] < num_a_edges:
                    print('hi')
                    basis_ = basis.adjacency_matrix().to_dense()
                    basis_[basis.edges()[0]][:,basis.edges()[1]] = 1.
                    basis_ = torch.nn.functional.pad(basis_,(0,zero_rows,0,zero_rows))
                else:
                    #print(basis.number_of_edges())
                    #basis_ = dgl.sampling.select_topk(basis, int(num_a_edges/basis.number_of_nodes()), 'w')
                    #print(basis_.number_of_edges())
                    #basis_ = basis_.adjacency_matrix().to_dense().to(self.graph.device)
                    if 'weibo' in self.dataset:
                        #sorted_idx = torch.topk(nz,int(num_a_edges*3)).indices
                        sorted_idx = torch.topk(nz,int(num_a_edges)).indices
                        basis = dgl.edge_subgraph(basis,sorted_idx,relabel_nodes=False)
                        #basis_ = dgl.sampling.select_topk(basis, int(num_a_edges/(basis.number_of_nodes()*2)), 'w')
                    else:
                        sorted_idx = torch.topk(nz,int(num_a_edges)).indices
                        basis = dgl.edge_subgraph(basis,sorted_idx,relabel_nodes=False)
                    #if 'cora' not in self.dataset: print(basis.number_of_edges())
                    basis_ = basis.adjacency_matrix().to_dense().to(self.graph.device)
                    #basis_[basis.edges()] = basis.edata['w'].to(self.graph.device)
                    
                    #basis_ = torch.nn.functional.pad(basis_,(0,zero_rows,0,zero_rows))
                    # TO ADD:  basis_.fill_diagonal_(1.)
                    #del nz, sorted_idx
                    
                
                if self.vis == True and 'test' not in self.vis_name:
                    if 'tfinance' in self.dataset:
                        import ipdb ; ipdb.set_trace()
                    print(label_idx)
                    try:
                        self.filter_anoms([basis_],self.anoms,self.vis_name,label_id)
                    except Exception as e:
                        print(e)
                        import ipdb ; ipdb.set_trace()
                        print(e)
                labels.append(basis_)
                del basis, basis_ ; torch.cuda.empty_cache()
            
            for i in range(len(prods)):
                del prods[0]
            del g ; torch.cuda.empty_cache()
            #print('done')
            #print("Seconds to get labels", (time.time()-seconds)/60)

        if self.vis == True and 'test' not in self.vis_name:
            if self.graph.device == 'cpu': 
                print('visualizing labels')
            
            plt.figure()
            adj_label = dgl.to_homogeneous(self.graph).subgraph(self.graph.srcnodes()).adjacency_matrix().to_dense()
            adj_label=np.maximum(adj_label, adj_label.T).to(self.graph.device) #?
            
            e_adj,U_adj = self.get_spectrum(torch.tensor(adj_label).to(self.graph.device).to(torch.float64))
            self.plot_spectrum(e_adj,U_adj,self.feats.to(U_adj.dtype),color="cyan")
            fpath = f'vis/label_vis/{self.dataset}/{self.model_str}/{self.label_type}'
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            plt.savefig(f'{fpath}/labels_{self.label_type}_{self.epoch}_{self.vis_name}_og.png')
            del e_adj, U_adj ; torch.cuda.empty_cache() ; gc.collect()
            
            legend = ['og']
            for label_ind,label in enumerate(labels):
                #print(label_ind)
                label = label
                if not torch.equal(label,label.T):
                    label=torch.maximum(label, label.T).to(self.graph.device) 
                try:
                    e,U = self.get_spectrum(label.to(torch.float64).to(self.graph.device))
                    e = e.to(self.graph.device)
                    U = U.to(self.graph.device)
                except Exception as e:
                    print(e)
                    import ipdb ; ipdb.set_trace()
                try:
                    self.plot_spectrum(e,U,self.feats.to(U.dtype))
                except Exception as e:
                    print(e)
                    import ipdb ; ipdb.set_trace()
                del e,U,label ; torch.cuda.empty_cache() ; gc.collect()
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                #plt.savefig(f'{fpath}/labels_{self.label_type}_{self.epoch}_{self.vis_name}_{label_ind+1}.png')
                legend.append(f'label {str(label_ind+1)}')
            plt.legend(legend)
            plt.savefig(f'{fpath}/labels_{self.label_type}_{self.epoch}_{self.vis_name}_{label_ind+1}.png')

            #plt.legend(legend)
            #plt.savefig(f'{fpath}/labels_{self.label_type}_{self.epoch}_{self.vis_name}.png')
            #print("Seconds to visualize labels", (time.time()-seconds)/60)
            self.anom_response(adj_label.to(self.graph.device),labels,self.anoms,self.vis_name)
            del adj_label ; torch.cuda.empty_cache() ; gc.collect()
            #self.filter_anoms(labels,self.anoms,self.vis_name)
        return labels
