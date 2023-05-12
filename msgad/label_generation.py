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

class LabelGenerator:
    def __init__(self,graph,dataset,model_str,epoch,label_type,feats,vis,vis_name,anoms,batch_size,exp_name):
        self.graph = graph
        self.feats = feats
        self.exp_name = exp_name
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
        self.coeffs = None

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
                
                x,y=self.plot_spectrum(e.detach().cpu(),U.detach().cpu(),signal_)
                if i == 0:
                    legend.append('original adj')
                else:
                    legend.append(f'{i} label')
            plt.legend(legend)

            fpath = f'vis/filter_anom_ev/{self.dataset}/{self.model_str}/{self.label_type}/{self.epoch}/{self.exp_name}'
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            plt.savefig(f'{fpath}/filter_vis_{self.label_type}_{vis_name}_filter{list(anoms.keys())[anom_ind]}.png')
            del e,U ; torch.cuda.empty_cache() ; gc.collect()
                
            torch.cuda.empty_cache()

    def filter_anoms(self,label,anoms,vis_name,img_num=None):
        np.random.seed(123)
        #print('filter anoms',torch.cuda.memory_allocated()/torch.cuda.max_memory_reserved())
        signal = np.random.randn(self.feats.shape[0],self.feats.shape[0])

        #signal = np.ones((labels[0].shape[0],self.feats.shape[-1]))

        anom_tot = 0
        for anom in list(anoms.values())[:-1]:
            anom_tot += self.flatten_label(anom).shape[0]
        anom_tot += list(anoms.values())[-1].shape[0]
        all_nodes = torch.arange(self.feats.shape[0])

        #for i,label in enumerate(labels):
        lbl=torch.maximum(label, label.T)
        e,U = self.get_spectrum(lbl.to(torch.float64))
        e = e.to(self.graph.device) ; U = U.to(self.graph.device)
        del lbl ; torch.cuda.empty_cache() ; gc.collect()
        
        plt.figure()
        x,y=self.plot_spectrum(e.detach().cpu(),U.detach().cpu(),signal+1)

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
            
            x,y=self.plot_spectrum(e.detach().cpu(),U.detach().cpu(),signal_)
            legend.append(legend_arr[anom_ind+1])
            plt.legend(legend)
    
            fpath = f'vis/filter_anom_vis/{self.dataset}/{self.model_str}/{self.label_type}/{self.exp_name}'
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            plt.savefig(f'{fpath}/filter_vis_{self.label_type}_{self.epoch}_{vis_name}_filter{img_num}.png')

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
            #label_idx=np.array([5,8])
        elif 'single' in self.label_type:
            label_idx = np.full((3,),1)
            label_idx = np.arange(3)
        return label_idx.tolist()

    def make_label(self,g,ind,prods):
        if 'filter' in self.label_type:
            coeff = self.coeffs[ind]
            basis = copy.deepcopy(g)
            basis.edata['w'] *= coeff[0]
            for i in range(1, self.K+1):
                basis_ = copy.deepcopy(prods[i])
                basis_.edata['w'] *= coeff[i]
                sorted_idx = torch.topk(basis_.edata['w'],int(self.num_a_edges)).indices
                basis_ = dgl.edge_subgraph(basis_,sorted_idx,relabel_nodes=False)
                basis = dgl.adj_sum_graph([basis,basis_],'w')
                #basis += prods[i] * coeff[i]
                del basis_
                torch.cuda.empty_cache()
            return basis
            #nz = basis.edata['w']#.unique()
            #sorted_idx = torch.topk(nz,int(self.num_a_edges)).indices
            #sorted_idx = torch.topk(nz,int(num_a_edges*(label_ind+1))).indices
            #return dgl.edge_subgraph(basis,sorted_idx,relabel_nodes=False)
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
            prods.append(g_prod)
            #print('mem',torch.cuda.memory_allocated()/torch.cuda.max_memory_reserved())
        del g_prod ; torch.cuda.empty_cache()
        return prods

    def construct_labels(self):
        """
        Stationary distribution given the transition matrix.
        :param M: Transition matrix.
        :return: Stationary distribution.
        """
        # prep
        self.num_a_edges = self.graph.num_edges()
        self.K = 10 if 'filter' in self.label_type else 3
        label_idx = self.prep_filters(self.K)
        labels = []
        g = self.graph.to('cpu')#self.graph.device)#.to('cpu')

        # normalize graph if needed
        if 'norm' in self.label_type:
            if 'cora' in self.dataset: g = g.to(self.graph.device)
            g = dgl.to_homogeneous(g).subgraph(g.srcnodes())
            num_a_edges = g.num_edges()
            if 'cora' in self.dataset:
                g.edata['w'] = self.norm(g,torch.ones(num_a_edges).to(self.graph.device)).to(self.graph.device)
            else:
                g.edata['w'] = self.norm(g,torch.ones(num_a_edges).to('cpu')).to('cpu')#(self.graph.device)).to(self.graph.device)
        else:
            g.edata['w'] = torch.ones(self.num_a_edges)
        adj_label = g.adjacency_matrix().to_dense()
        if self.vis == True and 'test' not in self.vis_name: self.filter_anoms(adj_label.to(self.graph.device),self.anoms,self.vis_name,'og')
        labels = [g]

        # visualize input
        if self.vis == True and 'test' not in self.vis_name:
            adj_label=adj_label.to(self.graph.device).to(torch.float64)
            e_adj,U_adj = self.get_spectrum(torch.maximum(adj_label,adj_label.T))
            xs_labelvis,ys_labelvis = self.plot_spectrum(e_adj,U_adj,self.feats.to(U_adj.dtype),color='cyan')
            
            self.label_analysis = LabelAnalysis(self.anoms,self.dataset)
            sc1_cons,sc2_cons,sc3_cons=self.label_analysis.cluster(adj_label.to(self.graph.device).to(torch.float64),0)
            sc1s = [np.array(sc1_cons)[...,np.newaxis][np.array(sc1_cons)[...,np.newaxis].nonzero()].mean()]
            sc2s = [np.array(sc2_cons)[...,np.newaxis][np.array(sc2_cons)[...,np.newaxis].nonzero()].mean()]
            sc3s = [np.array(sc3_cons)[...,np.newaxis][np.array(sc3_cons)[...,np.newaxis].nonzero()].mean()]
        

        if adj_label is not None: del adj_label ; torch.cuda.empty_cache()

        # make labels
        prods = self.prep_input(g)
        #import ipdb ; ipdb.set_trace()
        for label_id in label_idx:
            label = self.make_label(g,label_id,prods)

            labels.append(label)
            # visualize labels TODO move 
            if self.vis == True and 'test' not in self.vis_name:
                basis_ = label.adjacency_matrix().to_dense()
                if 'tfinance' in self.dataset:
                    import ipdb ; ipdb.set_trace()
                self.filter_anoms(basis_,self.anoms,self.vis_name,label_id)

                e,U = self.get_spectrum(torch.maximum(basis_, basis_.T).to(torch.float64).to(self.graph.device))
                e = e.to(self.graph.device) ; U = U.to(self.graph.device)
                x_labelvis,y_labelvis=self.plot_spectrum(e,U,self.feats.to(U.dtype))
                del e,U ; torch.cuda.empty_cache() ; gc.collect()

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

        if self.vis == True and 'test' not in self.vis_name and 'filter' in self.label_type:
            if 'cora' in self.dataset:
                plt.figure()
                label_idx.insert(0,'og')
                # some loop..
                plt.plot(label_idx,[0 if math.isnan(i) else i for i in sc1s],color='r')
                plt.plot(label_idx,[0 if math.isnan(i) else i for i in sc2s],color='g')
                plt.plot(label_idx,[0 if math.isnan(i) else i for i in sc3s],color='b')
                plt.savefig(f'label_vis_{self.label_type}_{self.exp_name}_{self.K}_{self.dataset}.png')
                if self.graph.device == 'cpu': 
                    print('visualizing labels')
            
            fpath = f'vis/label_vis/{self.dataset}/{self.model_str}/{self.label_type}/{self.exp_name}'
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            
            legend = ['og']
            plt.figure()
            for label_ind,(x_label,y_label) in enumerate(zip(x_labelvis,y_labelvis)):
                plt.plot(x_label,y_label)
                legend.append(f'label {str(label_ind+1)}')
            plt.legend(legend)
            plt.savefig(f'{fpath}/labels_{self.label_type}_{self.epoch}_{self.vis_name}_{label_ind+1}.png')

            #print("Seconds to visualize labels", (time.time()-seconds)/60)
            # TODO
            #self.anom_response(adj_label.to(self.graph.device),labels[1:],self.anoms,self.vis_name)
            fpath = f'vis/filter_anom_ev/{self.dataset}/{self.model_str}/{self.label_type}/{self.epoch}/{self.exp_name}'
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            torch.cuda.empty_cache() ; gc.collect()
            #self.filter_anoms(labels,self.anoms,self.vis_name)
        return labels
        #return labels[1:]