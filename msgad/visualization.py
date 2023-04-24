
import matplotlib.pyplot as plt
from model import *

class Visualizer:
    def __init__(self,adj,feats,args,sc_label,norms,anoms):
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
        pass

    def flatten_label(self,anoms):
        anom_flat = anoms[0][0]
        for i in anoms[1:]:
            anom_flat=np.concatenate((anom_flat,i[0]))
        return anom_flat

    def plot_recons(self,recons_a):
        #import ipdb ; ipdb.set_trace()
        #recons_a = recons_a.detach().cpu().numpy()
        plt.figure()
        legend = ['og']
        try:
            e_adj,U_adj= get_spectrum(self.adj.adjacency_matrix().to_dense().to(self.device))
        except Exception as e:
            print(e)
            import ipdb ; ipdb.set_trace()
        e_adj,U_adj = e_adj.detach().cpu(),U_adj.detach().cpu()
        plot_spectrum(e_adj,U_adj,self.feats[self.adj.dstnodes()])
        for r_ind,r_ in enumerate(recons_a):
            #r_ = torch.tensor(np.ceil(r_)).detach().to(self.device)
            #r_ = torch.sigmoid(torch.tensor(r_)).to(self.device)
            r_ = torch.tensor(r_).to(self.device)
            nz = r_.flatten().unique()
            sorted_idx = torch.min(torch.topk(nz,self.adj.number_of_edges()).values)
            r_ = torch.gt(r_,sorted_idx).float()

            r_symm = torch.maximum(r_, r_.T)+torch.eye(r_.shape[0]).to(self.adj.device) 

            if len(torch.nonzero(r_symm))==0:
                import ipdb ; ipdb.set_trace()
            try:
                e,U= get_spectrum(r_symm)
            except Exception as e:
                print(e)
                import ipdb ; ipdb.set_trace()
            e,U = e.detach().cpu(),U.detach().cpu()
            plot_spectrum(e,U,self.feats[self.adj.dstnodes()])
            legend.append(f'{r_ind}')
        import ipdb ; ipdb.set_trace()
        fpath = f'vis/recons_vis/{self.dataset}/{self.model}/{self.label_type}'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        plt.legend(['original graph','sc1 recons.','sc2 recons.','sc3 recons.'])
        plt.savefig(f'{fpath}/recons_vis_{self.epoch}_test.png')

    def plot_filters(self,res_a_all):
        plt.figure()
        adj_label = self.adj.adjacency_matrix()
        adj_label.sparse_resize_((adj_label.size(0), adj_label.size(0)), adj_label.sparse_dim(), adj_label.dense_dim())
        adj_label = adj_label.to_dense()[self.adj.dstnodes()][:,self.adj.dstnodes()]
        adj_label=torch.maximum(adj_label, adj_label.T).to(self.adj.device) 
        adj_label += torch.eye(adj_label.shape[0]).to(self.adj.device)
        try:
            e_adj,U_adj= get_spectrum(adj_label)
            plot_spectrum(e_adj.detach().cpu(),U_adj.detach().cpu(),self.feats[self.adj.dstnodes()])
        except Exception as e:
            print(e)
            import ipdb ; ipdb.set_trace()
        e_adj = e_adj
        U_adj = U_adj
        for res in res_a_all:
            res = torch.tensor(res).to(self.device).to(torch.float32)
            try:
                plot_spectrum(e_adj,U_adj,res[self.adj.dstnodes()])
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

    def plot_attn_scores(self,attn_weights):
        def flatten_label_attn(sc_label):
            anom_flat = sc_label[0][0]
            for i in sc_label[1:]:
                anom_flat=np.concatenate((anom_flat,i[0]))
            return anom_flat

        # epoch x 3 x num filters x nodes
        attn_weights_arr = [attn_weights[:,:,:,self.norms]]
        for anom in self.sc_label:
            if 'cora'in self.dataset:
                attn_weights_arr.append(attn_weights[:,:,:,self.anom.flatten()])
            else:
                attn_weights_arr.append(attn_weights[:,:,:,flatten_label_attn(self.anom)])

        #import ipdb ; ipdb.set_trace()
        legend = []
        colors=['green','red','purple','blue']
        legend=['norm','anom sc1','anom sc2','anom sc3']
        for scale in range(attn_weights.shape[1]):
            p_min,p_max=np.inf,-np.inf
            plt.figure()
            for ind,attn_weight in enumerate(attn_weights_arr):
                data=attn_weight[:,scale,0,:].detach().cpu().numpy()
                scale_attn = data.mean(axis=1)
                if np.min(scale_attn) < p_min:
                    p_min = np.min(scale_attn)
                if np.max(scale_attn) > p_max:
                    p_max = np.max(scale_attn)
                plt.plot(scale_attn,color=colors[ind])
                plt.fill_between(np.arange(attn_weight.shape[0]), data.mean(axis=1) - data.std(axis=1), data.mean(axis=1) + data.std(axis=1), color=colors[ind], alpha=0.1, label='_nolegend_')
            
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
            plt.savefig(f'{fpath}/model_sc{scale}.png')