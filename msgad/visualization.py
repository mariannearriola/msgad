
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from model import *

class Visualizer:
    def __init__(self,adj,feats,args,sc_label,norms,anoms,recons_labels):
        self.device = args.device
        self.dataset = args.dataset
        self.model = args.model
        self.adj = adj
        self.feats = feats
        self.label_type = args.label_type
        self.epoch = args.epoch
        self.norms=norms
        self.anom=anoms
        self.sc_label=sc_label
        self.recons_labels=recons_labels
        pass

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
            import ipdb ; ipdb.set_trace()
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

    def flatten_label(self,anoms):
        anom_flat = anoms[0][0]
        for i in anoms[1:]:
            anom_flat=np.concatenate((anom_flat,i[0]))
        return anom_flat

    def plot_recons(self,recons_a):
        #import ipdb ; ipdb.set_trace()
        #recons_a = recons_a.detach().cpu().numpy()
        #plt.figure()
        #legend = ['og']
        #e_adj,U_adj= self.get_spectrum(self.adj.adjacency_matrix().to_dense().to(self.device).to(torch.float64))
        #e_adj,U_adj = e_adj.detach().cpu(),U_adj.detach().cpu()
        #self.plot_spectrum(e_adj,U_adj,self.feats[self.adj.dstnodes()].to(U_adj.dtype))
        for r_ind,r_ in enumerate(recons_a):
            # plot label
            plt.figure()
            e_adj,U_adj= self.get_spectrum(self.recons_labels[r_ind].to(self.device).to(torch.float64))
            #e_adj,U_adj= self.get_spectrum(self.adj.adjacency_matrix().to_dense().to(self.device).to(torch.float64))
            e_adj,U_adj = e_adj.detach().cpu(),U_adj.detach().cpu()
            self.plot_spectrum(e_adj,U_adj,self.feats[self.adj.dstnodes()].to(U_adj.dtype))


            #r_ = torch.tensor(np.ceil(r_)).detach().to(self.device)
            #r_ = torch.sigmoid(torch.tensor(r_)).to(self.device)
            r_ = torch.tensor(r_).to(self.device)
            nz = r_.flatten().unique()
            if nz.shape[0] > self.adj.number_of_edges():
                sorted_idx = torch.min(torch.topk(nz,self.adj.number_of_edges()).values)
                r_ = torch.gt(r_,sorted_idx).float()

            r_symm = torch.maximum(r_, r_.T)

            if len(torch.nonzero(r_symm))==0:
                import ipdb ; ipdb.set_trace()
            e,U= self.get_spectrum(r_symm.to(torch.float64))
            e,U = e.detach().cpu(),U.detach().cpu()
            self.plot_spectrum(e,U,self.feats[self.adj.dstnodes()].to(U.dtype))
            #legend.append(f'{r_ind}')
            legend = ['label','recons']
            fpath = f'vis/recons_vis/{self.dataset}/{self.model}/{self.label_type}/{self.epoch}'
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            plt.legend(['original graph','sc1 recons.','sc2 recons.','sc3 recons.'])
            plt.savefig(f'{fpath}/recons_vis_{r_ind}_test.png')

    def plot_filters(self,res_a_all):
        plt.figure()
        adj_label = self.adj.adjacency_matrix()
        adj_label.sparse_resize_((adj_label.size(0), adj_label.size(0)), adj_label.sparse_dim(), adj_label.dense_dim())
        adj_label = adj_label.to_dense()[self.adj.dstnodes()][:,self.adj.dstnodes()]
        adj_label=torch.maximum(adj_label, adj_label.T).to(self.adj.device) 
        adj_label += torch.eye(adj_label.shape[0]).to(self.adj.device)
        try:
            e_adj,U_adj = self.get_spectrum(adj_label.to(torch.float64))
            self.plot_spectrum(e_adj.detach().cpu(),U_adj.detach().cpu(),self.feats[self.adj.dstnodes()].to(U_adj.dtype))
        except Exception as e:
            print(e)
            import ipdb ; ipdb.set_trace()
        e_adj = e_adj
        U_adj = U_adj
        for res in res_a_all:
            res = torch.tensor(res).to(self.device).to(torch.float32)
            try:
                self.plot_spectrum(e_adj,U_adj,res[self.adj.dstnodes()].to(U_adj.dtype))
            except Exception as e:
                print(e)
                import ipdb ; ipdb.set_trace()
                print(e)
        #print("Seconds to get spectrum", (time.time()-seconds)/60)
        plt.legend(['original feats','sc1 embed.','sc2 embed.','sc3 embed.'])
        fpath = f'vis/filter_vis/{self.dataset}/{self.model}/{self.label_type}'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        plt.savefig(f'{fpath}/filter_vis_{self.label_type}_{self.epoch}_test.png')

    def plot_attn_scores(self,attn_weights,lams):
        def flatten_label_attn(sc_label):
            anom_flat = sc_label[0]#[0]
            for i in sc_label[1:]:
                anom_flat=np.concatenate((anom_flat,i))#[0]))
            return anom_flat
            
        # epoch x 3 x num filters x nodes
        attn_weights_arr = [attn_weights[:,:,:,self.norms]]
        for anom_ind,anom in enumerate(self.sc_label):
            if 'cora'in self.dataset:
                attn_weights_arr.append(attn_weights[:,:,:,anom.flatten()])
            else:
                try:
                    if anom_ind == 0:
                        anom_tot = flatten_label_attn(anom)
                        attn_weights_arr.append(attn_weights[:,:,:,flatten_label_attn(anom)])
                    else:
                        anom_f = flatten_label_attn(anom)
                        if anom_f.ndim == 2: anom_f = anom_f[0]
                        anom_tot = np.append(anom_tot,anom_f)
                        attn_weights_arr.append(attn_weights[:,:,:,anom_f])
                        del anom_f
                except Exception as e:
                    attn_weights_arr.append(attn_weights[:,:,:,anom])

        legend = []
        colors=['green','red','blue','purple','yellow']
        legend=['norm','anom sc1','anom sc2','anom sc3','single']
        # scales x groups x filters x seq
        scale_com_atts = np.zeros((3,len(legend),attn_weights.shape[2],attn_weights.shape[0]))
        # for each scale
        for scale in range(attn_weights.shape[1]):
            p_min,p_max=np.inf,-np.inf
            model_lams = lams[scale]
            # for each filter
            for filter in range(attn_weights.shape[2]):
                plt.figure()
                # for each group
                for ind,attn_weight in enumerate(attn_weights_arr):
                    data=attn_weight[:,scale,filter,:]
                    scale_attn = data.mean(axis=1)
                    if np.min(scale_attn) < p_min:
                        p_min = np.min(scale_attn)
                    if np.max(scale_attn) > p_max:
                        p_max = np.max(scale_attn)
                    
                    plt.plot(scale_attn,color=colors[ind])
                    scale_com_atts[scale,ind,filter]=scale_attn*model_lams[filter]
                    #try:
                    #    plt.fill_between(np.arange(data.shape[0]), data.mean(axis=1)[0] - data.std(axis=1)[0], data.mean(axis=1)[0] + data.std(axis=1)[0], color=colors[ind], alpha=0.1, label='_nolegend_')
                    #except Exception as e:
                    #    print(e)
                        #import ipdb ; ipdb.set_trace()
                plt.legend(legend)
                plt.xlabel('epochs')
                if self.sc_label is None:
                    plt.ylabel(f'mean attention value for normal nodes')
                else:
                    plt.ylabel(f'mean attention value for anomaly scale {ind+1}')
                fpath = f'vis/attn_vis/{self.dataset}/{self.model}/{self.label_type}/{self.epoch}'
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                plt.ylim((p_min,p_max))
                #plt.xlim((args.epoch-3,args.epoch-1))
                plt.savefig(f'{fpath}/model_sc{scale}_fil{filter}.png')
        for sc,com_att in enumerate(scale_com_atts):
            plt.figure()
            # group : filters x seq
            for ind,group in enumerate(com_att):
                plt.plot(group.sum(axis=0),color=colors[ind])
            plt.legend(legend)
            plt.savefig(f'{fpath}/combined_filters_scale{sc}.png')
            