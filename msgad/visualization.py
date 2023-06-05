
import matplotlib.pyplot as plt
import scipy
import torch
import numpy as np
from scipy.interpolate import make_interp_spline
import scipy.io as sio
from model import *
import copy
import math
from utils import *

class Visualizer:
    def __init__(self,adj,feats,exp_params,sc_label,norms,anoms):
        self.dataset = exp_params['DATASET']['NAME']
        self.epoch = exp_params['MODEL']['EPOCH']
        self.device = exp_params['DEVICE']
        self.exp_name = exp_params['EXP']
        self.label_type = exp_params['DATASET']['LABEL_TYPE']
        self.model = exp_params['MODEL']['NAME']
        self.adj = adj
        self.feats = feats
        self.norms=norms
        self.anom=anoms
        self.sc_label=sc_label

    def generate_fpath(self,folder):
        fpath = f'vis/{folder}/{self.dataset}/{self.model}/{self.label_type}/{self.exp_name}/{self.epoch}'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        return fpath

    def plot_spectrum(self,e,U,signal,color=None):
        c = U.T@signal
        M = torch.zeros((40+1,c.shape[1])).to(e.device).to(U.dtype)
        for j in range(c.shape[0]):
            idx = max(min(int(e[j] / 0.05), 40-1),0)
            M[idx] += c[j]**2
        M=M/sum(M)
        #y = M[:,0].detach().cpu().numpy()
        #print('nans',torch.where(torch.isnan(M))[0].shape)
        M[torch.where(torch.isnan(M))]=0
        y = torch.mean(M,axis=1).detach().cpu().numpy()
        x = np.arange(y.shape[0])
        #if 'weibo' in self.dataset:
        #    x = x[15:25] ; y = y[15:25]
        spline = make_interp_spline(x, y)
        X_ = np.linspace(x.min(), x.max(), 401)
        Y_ = spline(X_)
        #plt.xticks(np.arange(y.shape[0])/bar_scale)
        if color:
            plt.plot(X_,Y_,color=color)
        else:
            plt.plot(X_,Y_)
        #plt.axis('equal')
        return X_,y

    def plot_loss_curve(self,losses):
        plt.figure()
        for loss in losses:
            plt.plot(loss)
        fpath = self.generate_fpath('loss')
        plt.legend(['sc1 model','sc2 model','sc3 model'])
        plt.savefig(f'{fpath}/loss.png')

    def flatten_label(self,anoms):
        anom_flat = anoms[0]#[0]
        for i in anoms[1:]:
            anom_flat=np.concatenate((anom_flat,i))#[0]))
        return anom_flat

    def plot_recons(self,recons_a,recons_labels):
        from utils import get_spectrum
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
            e_adj,U_adj= get_spectrum(dgl_to_mat(recons_labels[r_ind].to(self.device)).to(torch.float64).coalesce())

            #e_adj,U_adj= get_spectrum(self.adj.adjacency_matrix().to_dense().to(self.device).to(torch.float64))
            e_adj,U_adj = e_adj.detach().cpu(),U_adj.detach().cpu()
            self.plot_spectrum(e_adj,U_adj,self.feats[self.adj.dstnodes()].to(U_adj.dtype))

            #r_ = torch.tensor(np.ceil(r_)).detach().to(self.device)
            #r_ = torch.sigmoid(torch.tensor(r_)).to(self.device)
            r_ = torch.tensor(r_).to(self.device)
            nz = r_.flatten().unique()
            if nz.shape[0] > self.adj.number_of_edges():
                sorted_idx = torch.min(torch.topk(nz,self.adj.number_of_edges()).values)
                r_ = torch.gt(r_,sorted_idx).float()


            if len(torch.nonzero(r_))==0:
                print(r_ind,'failed')
                import ipdb ; ipdb.set_trace()
            e,U= get_spectrum(r_.to_sparse().to(torch.float64))
            
            e,U = e.detach().cpu(),U.detach().cpu()
            self.plot_spectrum(e,U,self.feats[self.adj.dstnodes()].to(U.dtype))
            #legend.append(f'{r_ind}')
            legend = ['label','recons']
            fpath = self.generate_fpath('recons_vis')
            plt.xlabel(r'$\lambda$')
            plt.legend(['original graph','sc1 recons.','sc2 recons.','sc3 recons.'])
            plt.savefig(f'{fpath}/recons_vis_{r_ind}_test.png')

    def plot_final_filters(self,filters):
        for sc in range(len(filters)):
            sc_filter = filters[sc]
            plt.figure()
            for filter in sc_filter:
                y = filter
                x = np.arange(y.shape[0])
                spline = make_interp_spline(x, y)
                X_ = np.linspace(x.min(), x.max(), 500)
                Y_ = spline(X_)
                plt.plot(X_,Y_)
            fpath = self.generate_fpath('final_filter_vis')
            plt.legend(np.arange(len(sc_filter)))
            plt.savefig(f'{fpath}/{sc}.png')

    def plot_filters(self,res_a_all):
        from utils import get_spectrum
        plt.figure()
        adj_label = dgl_to_mat(self.adj)
        adj_label.sparse_resize_((adj_label.size(0), adj_label.size(0)), adj_label.sparse_dim(), adj_label.dense_dim())
        adj_label = adj_label.to_dense()[self.adj.dstnodes()][:,self.adj.dstnodes()].to(self.adj.device)
        adj_label += torch.eye(adj_label.shape[0]).to(self.adj.device)
        try:
            e_adj,U_adj = get_spectrum(adj_label.to_sparse().to(torch.float64))
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
        fpath = self.generate_fpath('filter_vis')
        plt.savefig(f'{fpath}/filter_vis_{self.label_type}_{self.epoch}_test.png')

    def plot_attn_scores(self,attn_weights,edge_anom_mats):
        def flatten_label_attn(sc_label):
            anom_flat = sc_label[0]#[0]
            for i in sc_label[1:]:
                anom_flat=np.concatenate((anom_flat,i))#[0]))
            return anom_flat
            
        # epoch x 3 x num filters x nodes
        attn_weights_arr = [attn_weights[:,:,self.norms,:]]
        anom_groups = [self.norms]
        for anom_ind,anom in enumerate(self.sc_label):
            if False:#'cora'in self.dataset:
                attn_weights_arr.append(attn_weights[:,:,:,anom.flatten()])
            else:
                try:
                    if anom_ind == 0:
                        anom_tot = flatten_label_attn(anom)
                        anom_groups.append(flatten_label_attn(anom))
                        attn_weights_arr.append(attn_weights[:,:,flatten_label_attn(anom),:])
                    elif anom_ind == len(self.sc_label)-1:
                        anom_groups.append(anom.T[0])
                        attn_weights_arr.append(attn_weights[:,:,anom.T[0],:])
                    else:
                        anom_f = flatten_label_attn(anom)
                        if anom_f.ndim == 2: anom_f = anom_f[0]
                        anom_tot = np.append(anom_tot,anom_f)
                        anom_groups.append(anom_f)
                        attn_weights_arr.append(attn_weights[:,:,anom_f,:])
                        del anom_f
                except Exception as e:
                    attn_weights_arr.append(attn_weights[:,:,anom,:])

        legend = []
        colors=['green','red','blue','purple','yellow']
        legend=['norm','anom sc1','anom sc2','anom sc3','single']
        # scales x groups x filters x seq
        #scale_com_atts = np.zeros((3,len(legend),attn_weights.shape[0],attn_weights.shape[2]))
        # for each scale
        for scale in range(attn_weights.shape[1]):
            p_min,p_max=np.inf,-np.inf
            # for each filter
            for filter in range(attn_weights.shape[3]):
                plt.figure()
                # for each group
                for ind,attn_weight in enumerate(attn_weights_arr):
                    try:
                        data=attn_weight[:,scale,:,filter]
                    except Exception as e:
                        print(e)
                        import ipdb ; ipdb.set_trace()
                    scale_attn = data.mean(axis=1)
                    if np.min(scale_attn) < p_min:
                        p_min = np.min(scale_attn)
                    if np.max(scale_attn) > p_max:
                        p_max = np.max(scale_attn)
                    
                    plt.plot(scale_attn,color=colors[ind])
                    #scale_com_atts[scale,ind,:,filter]=scale_attn*model_lams[filter]
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
                fpath = self.generate_fpath(f'attn_vis/sc{scale}')
                plt.ylim((p_min,p_max))
                plt.savefig(f'{fpath}/fil{filter}.png')
        
        # plot range across scales for each filter
        #import ipdb ; ipdb.set_trace()      
        legend=['Normal','Anom. scale 1','Anom. scale 2','Anom. scale 3','Single-node anom.']
        for scale in range(attn_weights.shape[1]):
            new_legend = []
            plt.figure()
            for ind,attn_weight in enumerate(attn_weights_arr):
                group = attn_weight[:,scale,:,:].mean(1)
                #ranges = group.max(axis=1)-group.min(axis=1)
                ranges = group.sum(1)
                plt.plot(ranges,color=colors[ind])
                attn_lbl = legend[ind] + f'_model{np.argmax(group[-1])}'
                new_legend.append(attn_lbl)
            #plt.legend(new_legend)
            plt.legend(legend)
            plt.xlabel('Epoch')
            plt.ylabel('Attention score')
            plt.title(f'Avg. attentions for model {scale+1}')
            fpath = self.generate_fpath(f'attn_vis/scale_filters')
            plt.savefig(f'{fpath}/combined_filters_scale{scale}.png')
        plt.figure()
        for sc in range(edge_anom_mats.shape[1]):
            plt.figure()
            for ind,anom_group in enumerate(anom_groups):
                group_score=edge_anom_mats[:,sc,anom_group]
                group_loss=torch.stack([torch.stack([torch.mean(j[j.nonzero()],axis=0) for j in i]) for i in group_score]).squeeze(-1)
                group_loss[torch.where(torch.isnan(group_loss))] = 0.
                group_loss = group_loss.mean(1)
                plt.plot(group_loss,color=colors[ind])
                #group = edge_anom_mat[anom_group].mean()
                #sc_losses.append(group)
            fpath = self.generate_fpath(f'loss_{sc}')
            plt.legend(legend)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Loss curve for model {sc+1}')
            plt.savefig(f'{fpath}/combined_loss.png')
            #plt.bar(np.arange(len(sc_losses))+offset,np.array(sc_losses),width=width)
        #plt.xticks(np.arange(len(anom_groups)), legend)
        plt.legend(['scale1 model','scale2 model', 'scale3 model'])

        fpath = self.generate_fpath('loss')
        plt.savefig(f'{fpath}/combined_loss.png')

    def plot_filter_weights(self,filter_weights):
        plt.figure()
        legend = []
        width,offset = 0.25,0
        for ind,i in enumerate(filter_weights):
            plt.bar(np.arange(len(i))+offset,i,width=width)
            offset += width
            legend.append(ind)
        plt.legend(legend)
        fpath = self.generate_fpath('lam_vis')
        plt.savefig(f'{fpath}/{self.epoch}_{self.exp_name}_epoch.png')
            
    def visualize_labels(self,x_labelvis,y_labelvis,vis_name):        
        legend = ['og']
        plt.figure()
        for label_ind,(x_label,y_label) in enumerate(zip(x_labelvis,y_labelvis)):
            plt.plot(x_label,y_label)
            legend.append(f'label {str(label_ind+1)}')
        plt.legend(legend)
        fpath = self.generate_fpath('label_vis')
        plt.savefig(f'{fpath}/labels_{self.label_type}_{self.epoch}_{vis_name}_{label_ind+1}.png')

    def visualize_label_conns(self,label_idx,scs_tot):
        sc1s,sc2s,sc3s=scs_tot
        plt.figure()
        if label_idx[0] != 'og':
            label_idx.insert(0,'og')
        plt.plot(label_idx,[0 if math.isnan(i) else i for i in sc1s],color='r')
        plt.plot(label_idx,[0 if math.isnan(i) else i for i in sc2s],color='g')
        plt.plot(label_idx,[0 if math.isnan(i) else i for i in sc3s],color='b')
        fpath = self.generate_fpath('label_vis_conn')
        plt.savefig(f'{fpath}/label_vis_{self.label_type}_{self.exp_name}_{self.dataset}.png')

    def anom_response(self,graph,labels,anoms,vis_name,img_num=None):
        from utils import get_spectrum
    
        signal = np.random.randn(self.feats.shape[0],self.feats.shape[0])

        anom_tot = 0
        for anom in list(anoms.values())[:-1]:
            anom_tot += self.flatten_label(anom).shape[0]
        anom_tot += list(anoms.values())[-1].shape[0]
        es,us=[],[]
        for i,label in enumerate(labels):
            lbl = dgl_to_mat(label)
            e,U = get_spectrum(lbl.to(torch.float64))
            es.append(e) ; us.append(U)
            del lbl, e, U ; torch.cuda.empty_cache() ; gc.collect()

        for anom_ind,anom in enumerate(anoms.values()):
            plt.figure()
            legend = []
            for i,label in enumerate(range(len(es))):
                e,U = es[i],us[i]
                e = e.to(graph.device) ; U = U.to(graph.device)

                #anom = anom.flatten()
                if len(anom) == 0:
                    continue
                if anom_ind != len(anoms)-1:
                    anom_f = self.flatten_label(anom)
                else:
                    anom_f = anom
                #anom_mask=np.setdiff1d(all_nodes,anom_f)
                signal_ = copy.deepcopy(signal)+1
                signal_[anom_f]=(np.random.randn(U.shape[0])*400)+1#*anom_tot/anom.shape[0])+1# NOTE: tried 10#*(anom_tot/anom.shape[0]))
                
                x,y=self.plot_spectrum(e.detach().cpu(),U.detach().cpu(),signal_)
                if i == 0:
                    legend.append('original adj')
                else:
                    legend.append(f'{i} label')
            plt.legend(legend)
            fpath = self.generate_fpath('filter_anom_env')
            plt.savefig(f'{fpath}/filter_vis_{self.label_type}_{vis_name}_filter{list(anoms.keys())[anom_ind]}.png')
            del e,U ; torch.cuda.empty_cache() ; gc.collect()
                
            torch.cuda.empty_cache()

    def filter_anoms(self,graph,label,anoms,vis_name,img_num=None):

        signal = np.random.randn(self.feats.shape[0],self.feats.shape[0])
        #signal = np.ones((labels[0].shape[0],self.feats.shape[-1]))
        from utils import get_spectrum

        anom_tot = 0
        for anom in list(anoms.values())[:-1]:
            anom_tot += self.flatten_label(anom).shape[0]
        anom_tot += list(anoms.values())[-1].shape[0]
        all_nodes = torch.arange(self.feats.shape[0])
        
        #for i,label in enumerate(labels):
        lbl = label.to_sparse().to(torch.float64)
        fpath = self.generate_fpath('spectrum')
        try:
            e,U = get_spectrum(lbl,f'{fpath}/anom_vis{img_num}')
        except Exception as e_:
            import ipdb ; ipdb.set_trace()
            print(e_)
        #e = e.to(graph.device) ; U = U.to(graph.device)
        del lbl ; torch.cuda.empty_cache() ; gc.collect()
        
        plt.figure()
        #x,y=self.plot_spectrum(e.detach().cpu(),U.detach().cpu(),signal+1)
        try:
            x,y=self.plot_spectrum(e,U,signal+1)
        except Exception as e:
            print(e)
            import ipdb ; ipdb.set_trace()
        legend_arr = ['No anom. signal','Anom. scale 1','Anom. scale 2','Anom. scale 3','Single node anom.']
        legend = [legend_arr[0]]
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
                signal_[anom_f]=(np.random.randn(U.shape[0])*400*anom_tot/anom_f.shape[0])+1# NOTE: tried 10#*(anom_tot/anom.shape[0]))
            except Exception as e:
                print(e)
                import ipdb ; ipdb.set_trace()
            
            x,y=self.plot_spectrum(e.detach().cpu(),U.detach().cpu(),signal_)
            legend.append(legend_arr[anom_ind+1])
            
            plt.legend(legend)
            fpath = self.generate_fpath('filter_anom_vis')
            
            plt.xticks(x[np.arange(0,y.shape[0],step=5)*10],np.round(np.arange(y.shape[0],step=5)*0.05,2))
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
            plt.xlabel(r'$\lambda$')
            d_name = self.dataset.split("_")[0]
            plt.title(f'Spectrum for {d_name}, label {img_num}')
            plt.savefig(f'{fpath}/filter_vis_{self.label_type}_{self.epoch}_{vis_name}_filter{img_num}.png')

        del e,U ; torch.cuda.empty_cache() ; gc.collect()
            
        torch.cuda.empty_cache()