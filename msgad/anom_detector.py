
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import torch
import os
import pickle as pkl

class anom_classifier():
    """Given node-wise anomaly scores and known labels, evaluate anomaly detection accuracy"""
    def __init__(self, exp_params, scales, out_path, dataset=None, epoch=None, exp_name=None, model=None):
        super(anom_classifier, self).__init__()
        self.dataset = exp_params['DATASET']['NAME'] if exp_params is not None else dataset
        self.epoch = exp_params['MODEL']['EPOCH'] if exp_params is not None else epoch
        self.exp_name = exp_params['EXP'] if exp_params is not None else exp_name
        self.title = ""
        self.scales = scales
        self.out_path = out_path
        self.reverse_scoring = False
    
    def detect_anom(self,errors,sorted_errors, sc_label, label, top_nodes_perc, verbose=True):
        """
        Input:
            sorted_errors: normalized adjacency matrix
            label: positive edge list
            top_nodes_perc: negative edge list
        Output:
            all_costs: total loss for backpropagation
            all_struct_error: structure errors for each scale
        """
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
                print(f'scale {ind} precision:',np.round(all_precs[-1],4), 'roc:', np.round(all_rocs[-1]),4)
        hits = np.zeros(errors.shape) ; hits[np.intersect1d(full_anom,sorted_errors,return_indices=True)[-1]] = 1
        all_hits.append(hits)
        all_precs.append(hits[:int(full_anom.shape[0]*top_nodes_perc)].nonzero()[0].shape[0]/full_anom.shape[0])
        all_rocs.append(roc_auc_score(label,errors))
        if verbose:
            print('full precision',all_precs[-1])
            print('full roc',all_rocs[-1])
        return np.array(all_hits), np.array(all_precs), np.array(all_rocs)

    def calc_anom_stats(self, scores, label, all_anom, verbose=True, log=True):
        """
        Input:
            scores: array-like, shape=[k, n]
                anomaly scores for all scales
            label: array-like, shape=[n, 1]
                binary node-wise anomaly label
            all_anom: array-like, shape=[a,1]
                array containing scale-wise anomaly assignments
        Output:
            all_scores: array-like,
            all_precs: array-like, shape=[k+1, 1]
                Group-wise precisions
            all_rocs: array-like, shape=[k+1, 1]
                
        """
        if log:
            if not os.path.exists(f'{self.out_path}/{self.dataset}'): os.makedirs(f'{self.out_path}/{self.dataset}')

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
                print(f'SCALE {sc+1}')
            
            hits,precs,rocs=self.detect_anom(node_scores,sorted_errors, all_anom, label, 1,verbose)
    
            if self.reverse_scoring is True:
                if verbose: print('scores reverse sorted')
                rev_hits,revprec,revrocs=self.detect_anom(-node_scores,rev_sorted_errors, all_anom, label, 1,verbose)

            if verbose: print('')
            all_rocs.append(np.array(rocs)) ; all_precs.append(np.array(precs)) ; all_hits.append(np.array(hits))

            if log:
                with open(f'{self.out_path}/{self.dataset}/{self.scales}-sc{sc+1}_{self.exp_name}.pkl', 'wb') as fout:
                    pkl.dump({'rocs':rocs,'precs':precs,'hits':hits},fout)

        return all_scores,all_precs,all_rocs