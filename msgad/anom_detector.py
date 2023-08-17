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
    """Given node-wise or cluster-wise scores, perform binary anomaly classification"""
    def __init__(self, exp_params, dataset=None, epoch=None, exp_name=None, model=None, recons=None, detection_type=None):
        super(anom_classifier, self).__init__()
        self.dataset = exp_params['DATASET']['NAME'] if exp_params is not None else dataset
        self.epoch = exp_params['MODEL']['EPOCH'] if exp_params is not None else epoch
        self.exp_name = exp_params['EXP'] if exp_params is not None else exp_name
        self.model = exp_params['MODEL']['NAME'] if exp_params is not None else model
        self.recons = exp_params['MODEL']['RECONS'] if exp_params is not None else recons
        self.detection_type = exp_params['DETECTION']['TYPE'] if exp_params is not None else detection_type
        self.title = ""
        #self.auc = AUC()
        self.colors = ['cyan','red','blue','purple']
        self.anoms=['single','scale1','scale2','scale3']

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
        fname = f'{fpath}/{self.model}.pkl'
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
        fname = f'{fpath}/{self.model}.pkl'
        with open(fname,'wb') as fout:
            pkl.dump(percs_dict,fout)
        return percs_auc

    def plot_percentages(self,hit_rankings,sorted_errors,anoms,ms_anoms_num,sc):
        plt.figure()
        anom_groups = np.unique(anoms)
        fpath = f'vis/perc_at_k/{self.dataset}/{self.model}/{self.epoch}/{self.exp_name}'
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
        fpath = f'vis/hit_at_k/{self.dataset}/{self.model}/{self.epoch}/{self.exp_name}'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        plt.legend(aucs)
        plt.ylabel('# predictions')
        plt.xlabel('# multi-scale anomalies detected')
        #plt.axhline(y=ms_anoms_num, color='b', linestyle='-')
        #plt.ylim(top=ms_anoms_num*2)
        plt.savefig(f'{fpath}/sc{sc}_hit_at_k.png')

    def detect_anom_sampled(self,all_scores,clusts,anoms_all,cutoff_label,stds):
        all_sc_anom_found = []
        self.title += '_sampled'
        for sc,tot_score in enumerate(np.flip(np.array(all_scores[:(cutoff_label)]),0)):
            lbl = (cutoff_label-1-sc)
            score_check = tot_score
            if sc > 0:
                score_check[sc_anom_found] = 0.
            num_anoms_found = np.where(tot_score>tot_score.std()*stds)[0].shape[0]
            average_tensor = torch.scatter_reduce(torch.tensor(tot_score), 0, torch.tensor(clusts[cutoff_label-1-sc]), reduce="mean")
            clust_loss = np.array(average_tensor[clusts[cutoff_label-1-sc]])
            sc_anom_found = np.argsort(-tot_score*clust_loss)[:num_anoms_found]
            all_sc_anom_found.append(sc_anom_found)
            self.plot_percentages(tot_score*clust_loss, np.argsort(-tot_score*clust_loss),anoms_all,None,lbl)
        lbl = -1
        score_check = tot_score ; score_check[sc_anom_found] = 0.
        num_anoms_found = np.where(tot_score>tot_score.std()*stds)[0].shape[0]
        average_tensor = torch.scatter_reduce(torch.tensor(tot_score), 0, torch.tensor(clusts[cutoff_label-1-sc]), reduce="mean")
        clust_loss = np.array(average_tensor[clusts[cutoff_label-1-sc]])
        sc_anom_found = np.argsort(-tot_score*clust_loss)[:num_anoms_found]
        all_sc_anom_found.append(sc_anom_found)
        self.plot_percentages(tot_score*clust_loss, np.argsort(-tot_score*clust_loss),anoms_all,None,lbl)
        return all_sc_anom_found

    def flatten_label(self,anoms):
        anom_flat = anoms[0]#[0]
        for i in anoms[1:]:
            anom_flat=np.concatenate((anom_flat,i))#[0]))
        return anom_flat
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
        #all_anom = self.flatten_label(sc_label)
        full_anom = np.where(label==1)[0]
        all_corrs,all_precs,all_rocs = [],[],[]
        for ind,anom_id in enumerate(anom_groups):
            anom  = full_anom[(sc_label==anom_id).nonzero()]
            anom_lbl = np.zeros(label.shape) ; anom_lbl[anom] = 1
            all_corrs.append(np.intersect1d(anom,sorted_errors[:int(full_anom.shape[0]*top_nodes_perc)]))
            all_precs.append(all_corrs[-1].shape[0]/anom.shape[0])
            all_rocs.append(roc_auc_score(anom_lbl,errors))
            if verbose:
                print(f'scale {ind} precision',all_corrs[-1].shape[0],all_corrs[-1].shape[0]/anom.shape[0])
                print(f'scale {ind} roc',all_rocs[-1])
            
        true_anoms = np.intersect1d(full_anom,sorted_errors[:int(full_anom.shape[0]*top_nodes_perc)])
        all_precs.append(true_anoms.shape[0]/full_anom.shape[0])
        all_rocs.append(roc_auc_score(label,errors))
        if verbose:
            print('full precision',true_anoms.shape[0],true_anoms.shape[0]/full_anom.shape[0])
            print('full roc',all_rocs[-1])
        return true_anoms, all_corrs, all_precs, all_rocs

    def get_node_score(self,score):
        if 'mean' in self.detection_type:
            node_score = np.mean(score[score.nonzero()])
        elif 'std' in self.detection_type:
            node_score = np.std(score[score.nonzero()])
        if np.isnan(node_score):
            return 0.
        return node_score

    def get_scores(self,graph,mat,clusts):
        for sc,sc_score in enumerate(mat):
            for ind,score in enumerate(sc_score):
                node_score = 0 if score.max() == 0 else np.array(score[score.nonzero()]).mean()

                if ind == 0:
                    #node_scores = np.array([score[np.array([n for n in graph.neighbors(ind)])].mean()])
                    node_scores = node_score
                else:
                    #node_scores = np.append(node_scores, np.array([score[np.array([n for n in graph.neighbors(ind)])].mean()]))
                    node_scores = np.append(node_scores, node_score)      
                
                #node_scores = np.mean(sc_score,axis=1)
            # run graph transformer with node score attributes
            # anom_preds = anom_clf.forward(node_scores,graph.edges())
            node_scores[np.isnan(node_scores).nonzero()] = 0.
            if sc == 0:
                all_scores = torch.tensor(node_scores).unsqueeze(0)
            else:
                all_scores = torch.cat((all_scores,torch.tensor(node_scores).unsqueeze(0)),dim=0)
        return all_scores

    def anom_histogram(self,rankings,x,label,ind):
        plt.figure()
        sort_idx = np.argsort(rankings)

        if 'anom' not in label:
            clust_colors = np.array(generate_cluster_colors(x))
            x = clust_colors
        sorted_x,sorted_y=x[sort_idx],rankings[sort_idx]
        plt.bar(np.arange(sorted_y.shape[0]),sorted_y,color=sorted_x,width=1.)
        fpath = f'vis/histogram-scores/{self.dataset}/{self.model}/{self.epoch}/{label}'
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        fname = f'{fpath}/{ind}.png'
        plt.savefig(fname)


    # clust = nonclust if one null
    def calc_clust_prec(self, graph, clust_scores, nonclust_scores, lbl, anoms_found_prev, anoms,norms, sc_label, attns, clusts, cluster=False, input_scores=False, clust_anom_mats=None, clust_inds=None):
        '''
        Input:
            scores: anomaly scores for all scales []
            label: node-wise anomaly label []
            sc_label: array containing scale-wise anomaly node ids
            dataset: dataset string
        '''
        import ipdb ; ipdb.set_trace()
        # anom_clf = MessagePassing(aggr='max')
        if len(sc_label[0]) > 0:
            anom_single,anom_sc1,anom_sc2,anom_sc3 = anoms[sc_label==0],anoms[sc_label==1],anoms[sc_label==2],anoms[sc_label==3]
        else:
            anom_sc1,anom_sc2,anom_sc3=[],[],[]
        color_names=[np.full(norms.shape,'green'),np.full(anom_sc1.shape,'red'),np.full(anom_sc2.shape,'blue'),np.full(anom_sc3.shape,'purple'),np.full(anom_single.shape,'cyan'),]
        anoms=[norms,anom_sc1,anom_sc2,anom_sc3,anom_single]
        color_names = np.concatenate(color_names)
        anoms = np.concatenate(anoms)

        #all_clust_scores = self.get_scores(graph,clust_scores,clusts[0])[0]
        #all_nonclust_scores = self.get_scores(graph,nonclust_scores,clusts[1])[0]
        #clust_score_norm,nonclust_score_norm = all_clust_scores/all_clust_scores.max(),all_nonclust_scores/all_nonclust_scores.max()
        #clust_score_norm[torch.where(torch.isnan(clust_score_norm))] = 0. ; nonclust_score_norm[torch.where(torch.isnan(nonclust_score_norm))] = 0.

        #cum_score = clust_score_norm+nonclust_score_norm
        cum_score = torch.scatter_add(attns,clusts)
        #self.anom_histogram(clust_score_norm,color_names,f'clustscore-anom',lbl)
        #self.anom_histogram(nonclust_score_norm,color_names,f'nonclustscore-anom',lbl)
        self.anom_histogram(cum_score,color_names,f'cumscore-anom',lbl)


        #self.anom_histogram(clust_score_norm,clusts[0],f'clustscore-clust',lbl)
        #self.anom_histogram(nonclust_score_norm,clusts[1],f'nonclustscore-clust',lbl)
        if not np.array_equal(clusts[0],clusts[1]):
            self.anom_histogram(cum_score,clusts,f'cumscore-clust',lbl)

        if anoms_found_prev is not None:
            cum_score[anoms_found_prev] = 0.
        num_anoms_found = np.where(cum_score>cum_score.std()*self.stds)[0].shape[0]
        #average_tensor = torch.scatter_reduce(torch.tensor(cum_score), 0, torch.tensor(clusts), reduce="mean")
        #clust_loss = np.array(average_tensor[clusts])
        #sc_anom_found = np.argsort(-cum_score*clust_loss)[:num_anoms_found]
        sc_anom_found = np.argsort(-cum_score)[:num_anoms_found]
        self.plot_percentages(cum_score, np.argsort(-cum_score),[anom_single,anom_sc1,anom_sc2,anom_sc3],None,lbl)
        if anoms_found_prev is not None:
            sc_anom_found = np.append(anoms_found_prev,sc_anom_found)

        return cum_score,sc_anom_found
    #    with open('output/{}-ranking_{}.txt'.format(self.dataset, sc), 'w+') as f:
    #        for index in sorted_errors:
    #            f.write("%s\n" % label[index])
    

    def calc_prec(self, scores, label, all_anom, clusts, cluster=False, input_scores=False, clust_anom_mats=None, clust_inds=None,verbose=True):
        '''
        Input:
            scores: anomaly scores for all scales []
            label: node-wise anomaly label []
            sc_label: array containing scale-wise anomaly node ids
        '''
   
        for sc,sc_score in enumerate(scores):
            if self.recons == 'both':
                node_scores = sc_score#.detach().cpu().numpy()
            elif self.recons == 'feat':
                node_scores = np.mean(sc_score,axis=1)
            elif input_scores:
                node_scores = sc_score
                if -1 in node_scores:
                    print('not all node anomaly scores calculated for node sampling!')
            else:
                for ind,score in enumerate(sc_score):
                    if ind == 0:
                        #node_scores = np.array([score[np.array([n for n in graph.neighbors(ind)])].mean()])
                        node_scores = np.array(score[score.nonzero()]).mean()
                    else:
                        #node_scores = np.append(node_scores, np.array([score[np.array([n for n in graph.neighbors(ind)])].mean()]))
                        node_scores = np.append(node_scores, np.array(score[score.nonzero()]).mean())                    
                #node_scores = np.mean(sc_score,axis=1)
            # run graph transformer with node score attributes
            # anom_preds = anom_clf.forward(node_scores,graph.edges())
            node_scores[np.isnan(node_scores).nonzero()] = 0.
            if sc == 0:
                all_scores = torch.tensor(node_scores).unsqueeze(0)
            else:
                all_scores = torch.cat((all_scores,torch.tensor(node_scores).unsqueeze(0)),dim=0)
            sorted_errors = np.argsort(-node_scores)
            rev_sorted_errors = np.argsort(node_scores)
            rankings = label[sorted_errors]

            full_anoms = label.nonzero()[0]
            ms_anoms_num = full_anoms.shape[0]
            
            hit_rankings = rankings.nonzero()[0]
            #self.hit_at_k(hit_rankings,sorted_errors,all_anom,ms_anoms_num,sc)
            
            #self.plot_percentages(hit_rankings,sorted_errors,all_anom,ms_anoms_num,sc)

            #import ipdb ; ipdb.set_trace()
            # add plots for scale-specific anomalies
            if verbose:
                print(f'SCALE {sc+1} loss',np.sum(node_scores),np.mean(node_scores))

            _,_,prec,rocs=self.detect_anom(node_scores,sorted_errors, all_anom, label, 1,verbose)
            if verbose:
                print('scores reverse sorted')
            _,_,revprec,revrocs=self.detect_anom(-node_scores,rev_sorted_errors, all_anom, label, 1,verbose)
            if verbose:
                print('')

            #all_anom_cat =self.flatten_label(all_anom)

            # get clusters
            '''
            if cluster:
                clust_dicts, score_dicts = getHierClusterScores(graph,node_scores)
                cluster_accs = [clf.classify(np.array(list(x.values())),all_anom,np.array(list(y.values()))) for x,y in zip(score_dicts,clust_dicts)]
                print('cluster scores')
                print(cluster_accs)
            '''
            '''
            # classify anoms with linear classifier
            anom_accs = clf.classify(node_scores, all_anom)
            print('\nnode scores')
            print(anom_accs)
            '''
        #cutoff_label = 2 ; stds = 3
        #all_sc_anom_found = self.detect_anom_sampled(all_scores,clusts,all_anom,cutoff_label,stds)
        return all_scores,prec,rocs
    #    with open('output/{}-ranking_{}.txt'.format(self.dataset, sc), 'w+') as f:
    #        for index in sorted_errors:
    #            f.write("%s\n" % label[index])
