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

def dgl_to_nx(g):
    nx_graph = nx.to_undirected(dgl.to_networkx(g.cpu()))
    node_ids = np.arange(g.num_nodes())
    return nx_graph,node_ids

class LabelGenerator:
    def __init__(self,graph,feats,vis,vis_name,anoms,norms,exp_params,visualizer):
        self.graph = graph
        self.feats = feats
        self.exp_name = exp_params['EXP']
        self.device = exp_params['DEVICE']
        self.dataset=exp_params['DATASET']['NAME']
        self.epoch = exp_params['MODEL']['EPOCH']
        self.label_idx = exp_params['DATASET']['LABEL_IDX']
        self.model_str = exp_params['MODEL']['NAME']
        #self.norm_adj = torch_geometric.nn.conv.gcn_conv.gcn_norm
        self.norm = EdgeWeightNorm(norm='both')
        self.label_type = exp_params['DATASET']['LABEL_TYPE']
        self.K = exp_params['DATASET']['K']
        self.vis_name = vis_name
        self.vis = vis
        self.anoms= anoms
        self.norms = norms
        self.coeffs = None
        self.visualizer = visualizer
        self.save_spectrum = exp_params['VIS']['SAVE_SPECTRUM']

        self.taus = [2,3,4] ; self.b = 1

    def flatten_label(self,anoms):
        anom_flat = anoms[0]
        for i in anoms[1:]:
            anom_flat=np.concatenate((anom_flat,i))
        return anom_flat

    def calculate_theta2(self,d):
        """
        ...
        """
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
        """
        ...
        """
        def Bernstein(de, i):
            coefficients = [0, ] * i + [math.comb(de, i)]
            first_term = polynomial.polynomial.Polynomial(coefficients)
            second_term = polynomial.polynomial.Polynomial([1, -1]) ** (de - i)
            return first_term * second_term
        out = []
        for i in range(degree + 1):
            out.append(Bernstein(degree, i).coef)
        return out

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

    def prep_filters(self,K):
        """
        Initialize parameters for filters needed to generate reconstruction labels.

        Input:
            K : {int}
                Filter parameter for label generation
        Output:
            label_idx: {array-like, numpy array}, shape=[3,]
                Coefficients to use for reconstruction labels after generation
        """
        if 'amnet' in self.label_type:
            self.coeffs =  self.get_bern_coeff(K)
            label_idx=np.arange(0,K+1,1)
            if self.label_idx != None and self.label_idx != 'None':
                label_idx = np.array(self.label_idx)
            if 'multi' in self.label_type:
                label_idx = np.array([7,8,9])
            elif 'full' not in self.label_type:
                label_idx=np.array([5,5,5])
            #label_idx=np.array([5,8,10])

        elif 'bwgnn' in self.label_type:
            self.coeffs = self.calculate_theta2(K)
            label_idx=np.arange(0,K+1,1)
            if self.label_idx != None and self.label_idx != 'None':
                label_idx = np.array(self.label_idx)
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
        """
        For a given filter coefficient, generate the corresponding reconstruction label.

        Input:
            g : {DGL graph}
                Filter parameter for label generation
            ind: {int}
                Subset of K to use as reconstruction labels
            prods : {array-like, shape=[k,]}
                Array containing pre-calculated product matrices
        Output:
            label: {DGL graph}
                Reconstruction label of k-th filter
        """
        if 'filter' in self.label_type:
            coeff = self.coeffs[ind]
            basis = copy.deepcopy(g)
            basis.edata['w'] *= coeff[0]
            #basis_ = copy.deepcopy(basis)
            for i in range(1, len(prods)):
                basis_ = copy.deepcopy(prods[i])
                #basis_n = prods[i-1]; basis_n.edata['w'] *= -1
                #basis_ = dgl.adj_sum_graph([basis_n,dgl.adj_product_graph(basis_,basis,'w')],'w')
                #basis_ = dgl.adj_sum_graph([basis_n,basis_],'w')
                #basis_ = dgl.adj_product_graph(g,basis_,'w')
                basis_.edata['w'] *= coeff[i]
                basis_ = dgl.remove_self_loop(basis_)
                '''
                if 'before' in self.exp_name:
                    nz = torch.abs(basis_.edata['w'])
                    if 'div' in self.exp_name:
                        sorted_idx = torch.topk(nz,int(self.num_a_edges/(ind+1))).indices
                    elif 'mult' in self.exp_name:
                        if nz.shape[0] < int(self.num_a_edges*(ind+1)):
                            sorted_idx = np.arange(nz.shape[0])
                        else:
                            sorted_idx = torch.topk(nz,int(self.num_a_edges*(ind+1))).indices
                    elif 'constant' in self.exp_name:
                        sorted_idx = torch.topk(nz,int(self.num_a_edges)).indices
                    elif 'nosample' in self.exp_name:
                        pass
                    basis_ = dgl.edge_subgraph(basis_,sorted_idx,relabel_nodes=False)
                '''
                #sorted_idx = torch.topk(torch.abs(basis_.edata['w']),int(self.num_a_edges)).indices#/(ind+1))).indices
                #basis_ = dgl.edge_subgraph(basis_,sorted_idx,relabel_nodes=False)
                basis = dgl.adj_sum_graph([basis,basis_],'w')
                if torch.isnan(basis.edata['w']).any():
                    print('nan found')
                    import ipdb ; ipdb.set_trace()
                

                #import ipdb ; ipdb.set_trace()
                #del basis_
                #del basis_n, basis_f
                #del basis_f
                #torch.cuda.empty_cache()
            #del basis_
            #import ipdb ; ipdb.set_trace()
            if self.visualizer is not None:
                self.visualizer.plot_filtered_nodes(basis,self.anoms,ind)

            #weight_threshold = 0.
            #mask = torch.sigmoid(basis.edata['w']) > weight_threshold
            #basis = dgl.edge_subgraph(basis,mask.nonzero().T[0],relabel_nodes=False)
            
            # for node, get mean...? edge weight?
            #if 'noeig' not in self.exp_name:
            #    plt.figure()
            #   plt.savefig(f'{self.exp_name}_{ind}.png')


            #del basis_
            #return basiss
            
            if 'after' in self.exp_name:
                #nz = torch.abs(basis.edata['w'])
                nz = -basis.edata['w']
                #sorted_idx = torch.topk(nz,int(self.num_a_edges)).indices
                if 'div' in self.exp_name:
                    sorted_idx = torch.topk(nz,int(self.num_a_edges/(ind+1))).indices
                elif 'mult' in self.exp_name:
                    sorted_idx = torch.topk(nz,int(self.num_a_edges*(ind+1))).indices
                elif 'constant' in self.exp_name:
                    #sorted_idx = torch.topk(nz,int(self.num_a_edges)+self.graph.number_of_nodes()).indices
                    sorted_idx = torch.topk(nz,int(self.num_a_edges)*2).indices
                elif 'nosample' in self.exp_name:
                    weight_threshold = 0.
                    mask = torch.sigmoid(basis.edata['w']) > weight_threshold
                    return dgl.edge_subgraph(basis,mask.nonzero().T[0],relabel_nodes=False)
                if 'ones' in self.exp_name:
                    basis.edata['w'][basis.edata['w'].nonzero()] = 1.
                basis_f = dgl.edge_subgraph(basis,sorted_idx,relabel_nodes=False)
            
            if self.visualizer is not None:
                if ind == 0:
                    self.connects_plot = np.array(self.visualizer.plot_sampled_filtered_nodes(basis,basis_f,self.anoms,ind))[np.newaxis, ...]
                else:
                    self.connects_plot = np.concatenate((self.connects_plot,np.array(self.visualizer.plot_sampled_filtered_nodes(basis,basis_f,self.anoms,ind))[np.newaxis, ...]),axis=0)
            
            legend_dict = {'normal':'green','anom_sc1':'red','anom_sc2': 'blue', 'anom_sc3': 'purple', 'single': 'yellow'}
            if ind == self.K:
                plt.figure()
                for group in range(self.connects_plot.shape[1]):
                    plt.plot(np.arange(self.connects_plot.shape[0]),self.connects_plot[:,group],color=list(legend_dict.values())[group])
                import ipdb ; ipdb.set_trace()
                plt.savefig(f'conns_{self.exp_name}.png')
            return basis_f
        elif 'random-walk' in self.label_type:
            raise(NotImplementedError)
        else:
            return prods[ind]
            
    def prep_input(self,g):
        prods = [g]
        if 'filter' not in self.label_type and 'single' not in self.label_type and 'prods' not in self.label_type:
            return prods
        g_prod = copy.deepcopy(g)
        
        for i in range(1, self.K+1):
            if 'cora' not in self.dataset: print('prod',i)
            g_prod = dgl.adj_product_graph(g_prod,g,'w')
            #sorted_idx = torch.topk(g_prod.edata['w'],int(self.num_a_edges)).indices
            #g_prod = dgl.edge_subgraph(g_prod,sorted_idx,relabel_nodes=False)
            prods.append(g_prod)
        del g_prod# ; torch.cuda.empty_cache()
        return prods

    def construct_labels(self):
        """
        Stationary distribution given the transition matrix.
        :param M: Transition matrix.
        :return: Stationary distribution.
        """
        from utils import get_spectrum, dgl_to_mat
        # initialize visualizer
        # prep
        print('constructing label')
        self.num_a_edges = self.graph.num_edges()
        #self.K = 5 if 'filter' in self.label_type else 3
        if 'filter' in self.label_type:
            label_idx = self.prep_filters(self.K)
        labels = []
        g = self.graph.to('cpu')#self.graph.device)#.to('cpu')
        # normalize graph if needed
        #labels = [g]
        adj_label = dgl_to_mat(g).to_dense()

        if 'norm' in self.label_type:
            if 'cora' in self.dataset: g = g.to(self.graph.device)
            g = dgl.to_homogeneous(g).subgraph(g.srcnodes())
            num_a_edges = g.num_edges()
            if 'cora' in self.dataset:
                g.edata['w'] = self.norm(g,torch.ones(num_a_edges).to(self.graph.device)).to(self.graph.device)
            else:
                g.edata['w'] = self.norm(g,torch.ones(num_a_edges).to('cpu')).to('cpu')#(self.graph.device)).to(self.graph.device)
            g = dgl.remove_self_loop(g)
        elif 'lapl' in self.label_type:
            print('getting lapl')
            mat_sparse=dgl_to_mat(g).to(g.device).to(torch.float64)
            if not os.path.exists(f'vis/spectrum/{self.exp_name}'):
                os.makedirs(f'vis/spectrum/{self.exp_name}')
            L = get_spectrum(mat_sparse,tag=f'vis/spectrum/{self.exp_name}/og_model',get_lapl=True,save_spectrum=self.save_spectrum)
            #L = torch_geometric.utils.get_laplacian(mat_sparse.coalesce().indices(),normalization='sym')
            
            g = dgl.from_scipy(L,eweight_name='w')
            print('got lapl')
        else:
            g.edata['w'] = torch.ones(self.num_a_edges)
        if 'elliptic' in self.dataset: print('getting dense adj')

        # NOTE: do not get the spectrum of cora for now
        if self.visualizer is not None and 'noeig' not in self.exp_name and 'cora' not in self.dataset:
           self.visualizer.filter_anoms(self.graph,adj_label,self.anoms,self.vis_name,'original')
        
        # visualize input
        if self.visualizer is not None:
            adj_label=adj_label.to(self.graph.device).to(torch.float64)
            #if 'noeig' not in self.exp_name:
            #    e_adj,U_adj = get_spectrum(adj_label.to_sparse())
            #    xs_labelvis,ys_labelvis = self.visualizer.plot_spectrum(e_adj,U_adj,self.feats.to(U_adj.dtype),color='cyan')
            if 'nocluster' not in self.exp_name:
                self.label_analysis = LabelAnalysis(self.anoms,self.dataset)
                sc1_cons,sc2_cons,sc3_cons=self.label_analysis.cluster(adj_label.to(self.graph.device).to(torch.float64),0)
                sc1s = [np.array(sc1_cons)[...,np.newaxis][np.array(sc1_cons)[...,np.newaxis].nonzero()].mean()]
                sc2s = [np.array(sc2_cons)[...,np.newaxis][np.array(sc2_cons)[...,np.newaxis].nonzero()].mean()]
                sc3s = [np.array(sc3_cons)[...,np.newaxis][np.array(sc3_cons)[...,np.newaxis].nonzero()].mean()]
        
        del adj_label
        #torch.cuda.empty_cache()
        #if adj_label is not None: del adj_label ; torch.cuda.empty_cache()
        # make labels
        print('prepping input')
        prods = self.prep_input(g)
        #prods = np.arange(self.K)
        print('prepped')
        if 'single' in self.label_type:
            g = g.to(self.device)
            return [g,g,g,g]
        if 'prods' in self.label_type:
            labels.append(g)
            for ind,basis in enumerate(prods):
                nz = torch.abs(basis.edata['w'])
                sorted_idx = np.arange(nz.shape[0])
                #nz = basis.edata['w']
                #sorted_idx = torch.topk(nz,int(self.num_a_edges)).indices
                if 'div' in self.exp_name:
                    sorted_idx = torch.topk(nz,int(self.num_a_edges/(ind+1))).indices
                elif 'mult' in self.exp_name:
                    sorted_idx = torch.topk(nz,int(self.num_a_edges*(ind+1))).indices
                elif 'constant' in self.exp_name:
                    if nz.shape[0] > int(self.num_a_edges)+self.graph.number_of_nodes():
                        sorted_idx = torch.topk(nz,int(self.num_a_edges)+self.graph.number_of_nodes()).indices
                elif 'nosample' in self.exp_name:
                    labels.append(basis)
                    continue
                if 'ones' in self.exp_name:
                    basis.edata['w'] /= basis.edata['w']
                basis = dgl.edge_subgraph(basis,sorted_idx,relabel_nodes=False)
                labels.append(basis)
            if self.vis == True:
              
                for label_id,basis_ in enumerate(labels):
                    self.visualizer.filter_anoms(self.graph,dgl_to_mat(basis_).to_dense(),self.anoms,self.vis_name,label_id)
            
            return [i.to(self.device) for i in labels]
        if 'filter' in self.label_type:
            #self.visualizer.filter_anoms(self.graph,adj_label,self.anoms,self.vis_name,-1)
            
            for label_id in label_idx:
                print('making label')
                label = self.make_label(g,label_id,prods)
                print('label id',label_id,'label edges',label.number_of_edges())
           
                labels.append(label)
                #del label ; torch.cuda.empty_cache()
                # visualize labels TODO move 
                if self.vis == True and 'test' not in self.vis_name:
                    nx_graph = label.to_networkx(node_attrs=None, edge_attrs=['w'])
                    adj_matrix = nx.adjacency_matrix(nx_graph, weight='w')
                    basis_ = torch.tensor(adj_matrix.toarray())
                    
                    #basis_ = label.adjacency_matrix().to_dense()
                    if 'noeig' not in self.exp_name:
                        self.visualizer.filter_anoms(self.graph,basis_,self.anoms,self.vis_name,label_id)

                    if 'nocluster' not in self.exp_name:
                        e,U = get_spectrum(basis_.to(torch.float64).to(self.graph.device).to_sparse(),lapl=basis_.to(torch.float64).to(self.graph.device).to_sparse(),save_spectrum=self.save_spectrum)
                        e = e.to(self.graph.device) ; U = U.to(self.graph.device)
                        x_labelvis,y_labelvis=self.visualizer.plot_spectrum(e,U,self.feats.to(U.dtype))
                        xs_labelvis = np.hstack((xs_labelvis,x_labelvis)) ; ys_labelvis = np.hstack((ys_labelvis,y_labelvis))
                        sc1_cons,sc2_cons,sc3_cons=self.label_analysis.cluster(basis_,label_id+1)
                        sc1s.append(np.array(sc1_cons)[...,np.newaxis][np.array(sc1_cons)[...,np.newaxis].nonzero()].mean())
                        sc2s.append(np.array(sc2_cons)[...,np.newaxis][np.array(sc2_cons)[...,np.newaxis].nonzero()].mean())
                        sc3s.append(np.array(sc3_cons)[...,np.newaxis][np.array(sc3_cons)[...,np.newaxis].nonzero()].mean())
            del g ; torch.cuda.empty_cache()

        if 'random-walk' in self.label_type:
            nx_graph,node_ids = dgl_to_nx(g)
            connected_graphs = [g for g in nx.connected_components(nx_graph)]
            node_dict = {k:v.item() for k,v in zip(list(nx_graph.nodes),node_ids)}
            labels = []
            full_labels = [torch.zeros((g.num_nodes(),g.num_nodes())).to(self.graph.device),torch.zeros((g.num_nodes(),g.num_nodes())).to(self.graph.device),torch.zeros((g.num_nodes(),g.num_nodes())).to(self.graph.device)]
            adj = self.graph.adjacency_matrix().to_dense()
            #adj=np.maximum(adj, adj.T).to(self.graph.device) 
            labels.append(g)
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
                drop_idx=upper_tri.nonzero()[sorted_idx][self.num_a_edges:]
                full_labels[i][drop_idx[:,0],drop_idx[:,1]]=0
                full_labels[i][drop_idx[:,1],drop_idx[:,0]]=0
                
                full_labels[i][torch.where(full_labels[i]<0)[0],torch.where(full_labels[i]<0)[1]]=0
                labels.append(dgl.from_networkx(nx.from_numpy_matrix(full_labels[i].detach().cpu().numpy())).to(self.graph.device))


        if self.visualizer is not None and 'test' not in self.vis_name:
            print('visualizing labels')
            if 'noeig' not in self.exp_name:
                self.visualizer.visualize_labels(x_labelvis,y_labelvis,self.vis_name)
            if 'nocluster' not in self.exp_name:
                scs_tot = [sc1s,sc2s,sc3s] ; self.visualizer.visualize_label_conns(label_idx,scs_tot)
                self.visualizer.anom_response(self.graph,labels,self.anoms,self.vis_name)

            torch.cuda.empty_cache() ; gc.collect()
            #self.filter_anoms(labels,self.anoms,self.vis_name)
        import ipdb ; ipdb.set_trace()
        return [i.to(self.device) for i in labels]
