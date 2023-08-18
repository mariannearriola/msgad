from sklearn.metrics import average_precision_score, roc_auc_score   
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
#from utils import *
from torcheval.metrics.functional.aggregation.auc import auc
import pickle as pkl

class anom_classifier():
    """Given node-wise anomaly scores and known labels, evaluate anomaly detection accuracy"""
    def __init__(self, exp_params, scales, dataset=None, epoch=None, exp_name=None, model=None):
        super(anom_classifier, self).__init__()
        self.dataset = exp_params['DATASET']['NAME'] if exp_params is not None else dataset
        self.epoch = exp_params['MODEL']['EPOCH'] if exp_params is not None else epoch
        self.exp_name = exp_params['EXP'] if exp_params is not None else exp_name
        self.title = ""
        self.scales = scales

    def plot_anom_sc(self,sorted_errors,anom,ms_anoms_num,color,scale_name):
        rankings_sc = np.zeros(sorted_errors.shape[0])
        rankings_sc[np.intersect1d(sorted_errors,anom,return_indices=True)[1]] = 1
        rankings_sc = rankings_sc.nonzero()[0]
        rankings_sc = np.append(rankings_sc,np.full(ms_anoms_num-rankings_sc.shape[0],np.max(rankings_sc)))
        rankings_auc = auc(torch.tensor(np.arange(rankings_sc.shape[0]))/rankings_sc.shape[0],torch.tensor(rankings_sc)).item()
        plt.plot(np.arange(rankings_sc.shape[0]),rankings_sc,color=color)
        fpath = f'vis/hit_at_k_model_wise/{self.dataset}/rankings/{scale_name}'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        fname = f'{fpath}/{self.exp_name}.pkl'
        rankings_dict = {'rankings':rankings_sc,'rankings_auc':rankings_auc}
        with open(fname,'wb') as fout:
            pkl.dump(rankings_dict,fout)
        return rankings_auc

    def plot_anom_perc(self,sorted_errors,anom,color,scale_name):
        rankings_sc = np.zeros(sorted_errors.shape[0])
        rankings_sc[np.intersect1d(sorted_errors,anom,return_indices=True)[1]] = 1
        rankings_sc = rankings_sc.nonzero()[0]
        anom_tot = sorted_errors.shape[0]
        percs_sc = np.zeros(anom_tot)
        for ind,ranking_ind in enumerate(range(rankings_sc.shape[0]-1)):
            percs_sc[rankings_sc[ranking_ind]:rankings_sc[ranking_ind+1]] = ind/rankings_sc.shape[0]
        percs_sc[rankings_sc[-1]:] = 1.
        
        percs_auc = auc(torch.tensor(np.arange(percs_sc.shape[0]))/percs_sc.shape[0],torch.tensor(percs_sc)).item()
        
        plt.plot(np.arange(percs_sc.shape[0]),percs_sc,color=color)
        fpath = f'vis/perc_at_k_model_wise/{self.dataset}/rankings/{scale_name}'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        percs_dict = {'percs':percs_sc,'percs_auc':percs_auc}
        fname = f'{fpath}/{self.exp_name}.pkl'
        with open(fname,'wb') as fout:
            pkl.dump(percs_dict,fout)
        return percs_auc

    def plot_percentages(self,hit_rankings,sorted_errors,anoms,ms_anoms_num,sc):
        plt.figure()
        anom_groups = np.unique(anoms)
        fpath = f'vis/perc_at_k/{self.dataset}/{self.exp_name}/{self.epoch}/{self.exp_name}'
        legend=[]
        for i,anom in enumerate(anom_groups):
            prec = self.plot_anom_perc(sorted_errors,anoms[np.where(anoms==anom)[0]],self.colors[i],self.anoms[i])
            legend_str = f'{self.anoms[i]}, auc' + str(round(prec,2))
            legend.append(f'{self.anoms[i]}, auc')
        plt.legend(legend)
        plt.xlabel('# predictions')
        plt.ylabel('% anomaly detected')
        plt.title('Percent anomaly detected @ k')

        if not os.path.exists(fpath):
            os.makedirs(fpath)
        plt.savefig(f'{fpath}/sc{sc}_perc_at_k_{self.title}.png')

    def hit_at_k(self,hit_rankings,sorted_errors,anoms,ms_anoms_num,sc):
        """

        """
        plt.figure()
        legend,aucs = [],[]
        for ind,scale in enumerate(anoms):
            col = (np.random.random(), np.random.random(), np.random.random())
            perc_auc=self.plot_anom_sc(sorted_errors,scale,ms_anoms_num,col,ind)
            plt.plot(np.arange(ms_anoms_num),hit_rankings,'gray')
            rankings_auc = auc(torch.tensor(np.arange(hit_rankings.shape[0])),torch.tensor(hit_rankings)).item()
            legend.append(ind)
            aucs.append(perc_auc)
        plt.legend(legend)
        fpath = f'vis/hit_at_k/{self.dataset}/{self.exp_name}/{self.epoch}/{self.exp_name}'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        plt.legend(aucs)
        plt.ylabel('# predictions')
        plt.xlabel('# multi-scale anomalies detected')
        #plt.axhline(y=ms_anoms_num, color='b', linestyle='-')
        #plt.ylim(top=ms_anoms_num*2)
        plt.savefig(f'{fpath}/sc{sc}_hit_at_k.png')
    
    def detect_anom(self,errors,sorted_errors, sc_label, label, top_nodes_perc, verbose=True):
        '''
        Input:
            sorted_errors: normalized adjacency matrix
            label: positive edge list
            top_nodes_perc: negative edge list
        Output:
            all_costs: total loss for backpropagation
            all_struct_error: structure errors for each scale
        '''
        anom_groups = np.unique(sc_label)
        full_anom = np.where(label==1)[0]
        all_hits,all_precs,all_rocs = [],[],[]

        for ind,anom_id in enumerate(anom_groups):
            anom  = full_anom[(sc_label==anom_id).nonzero()]
            anom_lbl = np.zeros(label.shape) ; anom_lbl[anom] = 1
            hits = np.zeros(errors.shape) ; hits[np.intersect1d(anom,sorted_errors,return_indices=True)[-1]] = 1
            all_hits.append(hits)
            all_precs.append(hits[:int(full_anom.shape[0]*top_nodes_perc)].nonzero()[0].shape[0]/anom.shape[0])
            all_rocs.append(roc_auc_score(anom_lbl,errors))
            if verbose:
                print(f'scale {ind} precision',all_precs[-1])
                print(f'scale {ind} roc',all_rocs[-1])
        hits = np.zeros(errors.shape) ; hits[np.intersect1d(full_anom,sorted_errors,return_indices=True)[-1]] = 1
        all_hits.append(hits)
        all_precs.append(hits[:int(full_anom.shape[0]*top_nodes_perc)].nonzero()[0].shape[0]/full_anom.shape[0])
        all_rocs.append(roc_auc_score(label,errors))
        if verbose:
            print('full precision',all_precs[-1])
            print('full roc',all_rocs[-1])
        return np.array(all_hits), np.array(all_precs), np.array(all_rocs)

    def anom_histogram(self,rankings,x,label,ind):
        plt.figure()
        sort_idx = np.argsort(rankings)

        if 'anom' not in label:
            clust_colors = np.array(generate_cluster_colors(x))
            x = clust_colors
        sorted_x,sorted_y=x[sort_idx],rankings[sort_idx]
        plt.bar(np.arange(sorted_y.shape[0]),sorted_y,color=sorted_x,width=1.)
        fpath = f'vis/histogram-scores/{self.dataset}/{self.exp_name}/{self.epoch}/{label}'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        fname = f'{fpath}/{ind}.png'
        plt.savefig(fname)


    def calc_prec(self, scores, label, all_anom,verbose=True,log=False):
        """
        Input:
            scores: anomaly scores for all scales []
            label: node-wise anomaly label []
            sc_label: array containing scale-wise anomaly node ids
            input_scores: 
        """
        if log:
            if not os.path.exists(f'output/{self.dataset}'): os.makedirs('output/rocs')

        all_precs,all_rocs,all_hits=[],[],[]
        for sc,sc_score in enumerate(scores):
            
            node_scores = sc_score
            if -1 in node_scores:
                print('not all node anomaly scores calculated for node sampling!')
                     
            node_scores[np.isnan(node_scores).nonzero()] = 0.
            if sc == 0:
                all_scores = torch.tensor(node_scores).unsqueeze(0)
            else:
                all_scores = torch.cat((all_scores,torch.tensor(node_scores).unsqueeze(0)),dim=0)
            sorted_errors = np.argsort(-node_scores)
            rev_sorted_errors = np.argsort(node_scores)
            # add plots for scale-specific anomalies
            if verbose:
                print(f'SCALE {sc+1} loss',node_scores.sum(),node_scores.mean())

            hits,precs,rocs=self.detect_anom(node_scores,sorted_errors, all_anom, label, 1,verbose)
            if verbose: print('scores reverse sorted')
            rev_hits,revprec,revrocs=self.detect_anom(-node_scores,rev_sorted_errors, all_anom, label, 1,verbose)

            if verbose: print('')
            all_rocs.append(np.array(rocs)) ; all_precs.append(np.array(precs)) ; all_hits.append(np.array(hits))
                
            if log and 'multiscale' not in self.exp_name:
                with open(f'output/{self.dataset}/{self.scales}-sc{sc+1}_{self.exp_name}.pkl', 'wb') as fout:
                    pkl.dump({'rocs':rocs,'precs':precs,'hits':hits},fout)

        all_rocs,all_precs = np.stack(all_rocs),np.stack(all_precs)
        if 'multiscale' in self.exp_name: all_rocs = np.diagonal(all_rocs) ; all_precs = np.diagonal(all_precs) ; all_hits = np.diagonal(all_hits)
        if log and 'multiscale' in self.exp_name:
            with open(f'output/{self.dataset}/{self.scales}-sc{sc+1}_{self.exp_name}.pkl', 'wb') as fout:
                pkl.dump({'rocs':all_rocs,'precs':all_precs,'hits':all_hits},fout)
    
        return all_scores,all_precs,all_rocs
