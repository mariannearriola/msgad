import dgl
import torch
import numpy as np
import sympy
import math
import scipy
import gc
from utils import *
import scipy.sparse as sp
import os
from numpy import polynomial
import copy
import networkx as nx
import torch_geometric
from scipy.interpolate import make_interp_spline
from dgl.nn.pytorch.conv import EdgeWeightNorm
import matplotlib.pyplot as plt
from label_analysis import LabelAnalysis
from visualization import Visualizer

class LabelGenerator:
    def __init__(self,graph,feats,vis,vis_name,anoms,exp_params,visualizer):
        self.graph = graph
        self.feats = feats
        self.exp_name = exp_params['EXP']
        self.dataset=exp_params['DATASET']['NAME']
        self.epoch = exp_params['MODEL']['EPOCH']
        self.model_str = exp_params['MODEL']['NAME']
        #self.norm_adj = torch_geometric.nn.conv.gcn_conv.gcn_norm
        self.norm = EdgeWeightNorm(norm='both')
        self.label_type = exp_params['DATASET']['LABEL_TYPE']
        self.vis_name = vis_name
        self.vis = vis
        self.anoms= anoms
        self.coeffs = None
        self.visualizer = visualizer

    def flatten_label(self,anoms):
        anom_flat = anoms[0]
        for i in anoms[1:]:
            anom_flat=np.concatenate((anom_flat,i))
        return anom_flat

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
        return X_,Y_

    def prep_filters(self,K):
        if 'amnet' in self.label_type:
            self.coeffs =  self.get_bern_coeff(K)
            label_idx=np.arange(0,K+1,1)
            #if 'cora' in self.dataset:
            #    label_idx = np.array([1,7,10]
        elif 'bwgnn' in self.label_type:
            self.coeffs = self.calculate_theta2(K)
            label_idx=np.arange(0,K+1,1)
            if 'multi' in self.label_type:
                label_idx = np.array([7,8,9])
            elif 'full' not in self.label_type:
                label_idx=np.array([5,5,5])
            #label_idx=np.array([5,8,10])
        elif 'single' in self.label_type:
            label_idx = np.full((3,),0)
            #label_idx = np.arange(3)
        return label_idx.tolist()

    def make_label(self,g,ind,prods):
        if 'filter' in self.label_type:
            coeff = self.coeffs[ind]
            basis = copy.deepcopy(g)
            basis.edata['w'] *= coeff[0]
            for i in range(1, len(prods)):
                #print('mem',torch.cuda.memory_allocated()/torch.cuda.max_memory_reserved())
                basis_ = copy.deepcopy(prods[i])
                basis_.edata['w'] *= coeff[i]
                #basis_ = dgl.remove_self_loop(basis_)
                #sorted_idx = torch.topk(torch.abs(basis_.edata['w']),int(self.num_a_edges)).indices#/(ind+1))).indices
                #basis_ = dgl.edge_subgraph(basis_,sorted_idx,relabel_nodes=False)
                basis = dgl.adj_sum_graph([basis,basis_],'w')
                del basis_
                #torch.cuda.empty_cache()
            weight_threshold = 0.
            mask = torch.sigmoid(basis.edata['w']) > weight_threshold
            basis = dgl.edge_subgraph(basis,mask.nonzero().T[0],relabel_nodes=False)

            return basis
            z = torch.abs(basis.edata['w'])
            nz = basis.edata['w']
            sorted_idx = torch.topk(nz,int(self.num_a_edges)).indices
            #sorted_idx = torch.topk(nz,int(num_a_edges*(label_ind+1))).indices
            return dgl.edge_subgraph(basis,sorted_idx,relabel_nodes=False)

        elif 'random-walk' in self.label_type:
            raise(NotImplementedError)
        else:
            return prods[ind]
            
    def prep_input(self,g):
        prods = [g]
        if 'filter' not in self.label_type and 'single' not in self.label_type:
            return prods
        g_prod = copy.deepcopy(g)
        
        for i in range(1, self.K+1):
            if 'cora' not in self.dataset: print('prod',i)
            g_prod = dgl.adj_product_graph(g_prod,g,'w')
            #sorted_idx = torch.topk(g_prod.edata['w'],int(self.num_a_edges)).indices
            #g_prod = dgl.edge_subgraph(g_prod,sorted_idx,relabel_nodes=False)
            prods.append(g_prod)
            #print('mem',torch.cuda.memory_allocated()/torch.cuda.max_memory_reserved())
        del g_prod# ; torch.cuda.empty_cache()
        return prods

    def construct_labels(self):
        """
        Stationary distribution given the transition matrix.
        :param M: Transition matrix.
        :return: Stationary distribution.
        """
        # initialize visualizer
        # prep
        self.num_a_edges = self.graph.num_edges()
        self.K = 10 if 'filter' in self.label_type else 3
        label_idx = self.prep_filters(self.K)
        labels = []
        g = self.graph.to('cpu')#self.graph.device)#.to('cpu')
        if 'weibo' in self.dataset:
            print('before norm',torch.cuda.memory_allocated()/torch.cuda.memory_reserved())
        # normalize graph if needed
        if 'norm' in self.label_type:
            if 'cora' in self.dataset: g = g.to(self.graph.device)
            g = dgl.to_homogeneous(g).subgraph(g.srcnodes())
            num_a_edges = g.num_edges()
            if 'cora' in self.dataset:
                g.edata['w'] = self.norm(g,torch.ones(num_a_edges).to(self.graph.device)).to(self.graph.device)
            else:
                g.edata['w'] = self.norm(g,torch.ones(num_a_edges).to('cpu')).to('cpu')#(self.graph.device)).to(self.graph.device)
            g = dgl.remove_self_loop(g)
        else:
            g.edata['w'] = torch.ones(self.num_a_edges)
        if 'weibo' in self.dataset: print('getting dense adj')
        adj_label = g.adjacency_matrix().to_dense()
        if 'weibo' in self.dataset: print('visualizing')
        if self.visualizer is not None: self.visualizer.filter_anoms(self.graph,adj_label.to(self.graph.device),self.anoms,self.vis_name,'og')
        labels = [g]
        if 'weibo' in self.dataset:
            print('after norm',torch.cuda.memory_allocated()/torch.cuda.memory_reserved())
        
        # visualize input
        if self.visualizer is not None:
            adj_label=adj_label.to(self.graph.device).to(torch.float64)
            e_adj,U_adj = self.get_spectrum(torch.maximum(adj_label,adj_label.T))
            xs_labelvis,ys_labelvis = self.plot_spectrum(e_adj,U_adj,self.feats.to(U_adj.dtype),color='cyan')
            
            self.label_analysis = LabelAnalysis(self.anoms,self.dataset)
            sc1_cons,sc2_cons,sc3_cons=self.label_analysis.cluster(adj_label.to(self.graph.device).to(torch.float64),0)
            sc1s = [np.array(sc1_cons)[...,np.newaxis][np.array(sc1_cons)[...,np.newaxis].nonzero()].mean()]
            sc2s = [np.array(sc2_cons)[...,np.newaxis][np.array(sc2_cons)[...,np.newaxis].nonzero()].mean()]
            sc3s = [np.array(sc3_cons)[...,np.newaxis][np.array(sc3_cons)[...,np.newaxis].nonzero()].mean()]
        
        del adj_label
        #torch.cuda.empty_cache()
        #if adj_label is not None: del adj_label ; torch.cuda.empty_cache()
        if 'weibo' in self.dataset:
            print('before prods',torch.cuda.memory_allocated()/torch.cuda.memory_reserved())
        # make labels
        prods = self.prep_input(g)
        if 'weibo' in self.dataset:
            print('after prods',torch.cuda.memory_allocated()/torch.cuda.memory_reserved())
        for label_id in label_idx:
            label = self.make_label(g,label_id,prods)
            print('label id',label_id,'label edges',label.number_of_edges())
            if 'weibo' in self.dataset:
                print(torch.cuda.memory_allocated()/torch.cuda.memory_reserved())

            labels.append(label)
            #del label ; torch.cuda.empty_cache()
            # visualize labels TODO move 
            if self.vis == True and 'test' not in self.vis_name:
                basis_ = label.adjacency_matrix().to_dense()
                if 'tfinance' in self.dataset:
                    import ipdb ; ipdb.set_trace()
                self.visualizer.filter_anoms(self.graph,basis_,self.anoms,self.vis_name,label_id)

                e,U = self.get_spectrum(torch.maximum(basis_, basis_.T).to(torch.float64).to(self.graph.device))
                e = e.to(self.graph.device) ; U = U.to(self.graph.device)
                x_labelvis,y_labelvis=self.plot_spectrum(e,U,self.feats.to(U.dtype))
                del e,U ; torch.cuda.empty_cache() ; gc.collect()

                xs_labelvis = np.hstack((xs_labelvis,x_labelvis)) ; ys_labelvis = np.hstack((ys_labelvis,y_labelvis))

                sc1_cons,sc2_cons,sc3_cons=self.label_analysis.cluster(basis_,label_id+1)
                sc1s.append(np.array(sc1_cons)[...,np.newaxis][np.array(sc1_cons)[...,np.newaxis].nonzero()].mean())
                sc2s.append(np.array(sc2_cons)[...,np.newaxis][np.array(sc2_cons)[...,np.newaxis].nonzero()].mean())
                sc3s.append(np.array(sc3_cons)[...,np.newaxis][np.array(sc3_cons)[...,np.newaxis].nonzero()].mean())
        #import ipdb ; ipdb.set_trace()
        del g ; torch.cuda.empty_cache()

        if 'random-walk' in self.label_type:
            nx_graph,node_ids = dgl_to_nx(g)
            connected_graphs = [g for g in nx.connected_components(nx_graph)]
            node_dict = {k:v.item() for k,v in zip(list(nx_graph.nodes),node_ids)}
            labels = []
            full_labels = [torch.zeros((g.num_nodes(),g.num_nodes())).to(self.graph.device),torch.zeros((g.num_nodes(),g.num_nodes())).to(graph.device),torch.zeros((g.num_nodes(),g.num_nodes())).to(graph.device)]
            adj = self.graph.adjacency_matrix().to_dense()
            #adj=np.maximum(adj, adj.T).to(self.graph.device) 
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
                labels.append(dgl.from_numpy_matrix(full_labels[i]).to(self.graph.device))
        if 'weibo' in self.dataset:
            print(torch.cuda.memory_allocated()/torch.cuda.memory_reserved())

        if self.visualizer is not None and 'test' not in self.vis_name:
            print('visualizing labels')
            scs_tot = [sc1s,sc2s,sc3s] ; self.visualizer.visualize_label_conns(label_idx,scs_tot)
            self.visualizer.visualize_labels(x_labelvis,y_labelvis,self.vis_name)
            self.visualizer.anom_response(self.graph,labels,self.anoms,self.vis_name)

            torch.cuda.empty_cache() ; gc.collect()
            #self.filter_anoms(labels,self.anoms,self.vis_name)
        #return labels
        return labels[1:]
