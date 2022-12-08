# -*- coding: utf-8 -*-
import os
import unittest
# noinspection PyProtectedMember
from numpy.testing import assert_equal
from numpy.testing import assert_raises
import numpy as np
import torch
#from torch_geometric.seed import seed_everything

from pygod.models import GCNAE
from pygod.utils.metric import eval_roc_auc
from pygod.utils.utility import load_data
import scipy.io as sio
import scipy.sparse as sp
import scipy
from scipy import stats
from sklearn.metrics import average_precision_score
#seed_everything(42)


class TestGCNAE(unittest.TestCase):
    def setUp(self):
        # use the pre-defined fake graph with injected outliers
        # for testing purpose

        # the roc should be higher than this; it is model dependent
        self.roc_floor = 0.60

        #test_graph = torch.load(os.path.join('pygod', 'test', 'test_graph.pt'))
        dataset = 'weibo'
        data_mat = sio.loadmat(f'../data/{dataset}.mat')
        ''' 
        data = load_data('inj_cora')
        import ipdb ; ipdb.set_trace()
        '''
        feats = torch.FloatTensor(data_mat['Attributes'])#.toarray())
        #feats = []
        #for scales in range(1,sc+1):
        #    feats.append(torch.FloatTensor(sio.loadmat(f'../smoothed_graphs/{scales}_{dataset}.mat')['Attributes'].toarray()))
        #import ipdb ; ipdb.set_trace()
        self.adj = data_mat['Network']
        #feat = data_mat['Attributes']
        truth = data_mat['Label']
        truth = truth.flatten()
        label = data_mat['scale_anomaly_label']
        self.truth = truth
        #import ipdb ; ipdb.set_trace()
        adj_norm = self.normalize_adj(self.adj + sp.eye(self.adj.shape[0]))
        adj_norm = adj_norm#.toarray()
        sp_adj = self.adj + sp.eye(self.adj.shape[0])
        adj = self.adj#.toarray()
        
        #import ipdb ; ipdb.set_trace() 
        test_graph=adj
        self.graph = test_graph
        self.feat = feats
        
        start,end=1,4
        
        self.model = GCNAE(start=start,end=end,hid_dim=128,adj=sp_adj)
        self.data={'graph':self.graph,'feat':self.feat,'truth':self.truth}
        self.model.fit(self.data)
        
        outlier_scores = self.model.decision_scores_
        outlier_scores = self.model.decision_function(self.data)
        outlier_scores=[outlier_scores]
        #import ipdb ; ipdb.set_trace()
        for ind,outlier_score in enumerate(outlier_scores):
            print(f'scale {ind+1}')
            anom_rankings = np.argsort(-outlier_score)
            print('AP',self.detect_anom_ap(outlier_score,truth))
            # TODO: get anomaly labels
            
            #print(self.detect_anom(anom_rankings,label,.5,start,end,outlier_score[anom_rankings]))
            #print(self.detect_anom(anom_rankings,label,.75,start,end,outlier_score[anom_rankings]))
            print(self.detect_anom(anom_rankings,label.T,1,start,end,outlier_score[anom_rankings],ind))
            print('---------------------')
        
    from sklearn.metrics import average_precision_score   
    def detect_anom_ap(self,errors,label):
        #import ipdb ;ipdb.set_trace()
        return average_precision_score(label,errors)
    
    def detect_anom(self,sorted_errors, label, top_nodes_perc,start,end,scores,scale):
         
        anom_sc1 = label[0][0][0]
        anom_sc2 = label[2][0][0]
        anom_sc3 = label[3][0][0]
        all_anom = np.concatenate((anom_sc1,np.concatenate((anom_sc2,anom_sc3),axis=None)),axis=None)
        anom_scores = []
        norm_scores = []
        for ind,error in enumerate(sorted_errors):
            if error in all_anom:
                anom_scores.append(scores[ind])
            else:
                norm_scores.append(scores[ind])
        anom_scores = np.array(anom_scores)
        norm_scores = np.array(norm_scores)
        skew = False
        # need to redo now :)
        thresh=[2.0,2.0,2.0]
        if skew:
            z=stats.zscore(scores)
            #z=z[np.where(z<0)]
            thresh=-thresh
            top_anom=sorted_errors[np.where(z<thresh[scale])]
        else:
            z=stats.zscore(scores)
            #z=z[np.where(z>0)]
            top_anom=sorted_errors[np.where(z>thresh[scale])]
        #import ipdb ; ipdb.set_trace()
        top_sc3=np.intersect1d(top_anom,anom_sc3).shape[0]
        top_sc2=np.intersect1d(top_anom,anom_sc2).shape[0]
        top_sc1=np.intersect1d(top_anom,anom_sc1).shape[0]
        if top_nodes_perc == 1:
            print('top scales found from deviation',top_sc1,top_sc2,top_sc3)
        true_anoms = 0
        cor_1, cor_2, cor_3 = 0,0,0
        anom_inds1,anom_inds2,anom_inds3=[],[],[]
        for ind,error in enumerate(sorted_errors[:int(all_anom.shape[0])]):
            '''
            if label[ind] == 1:
                true_anoms += 1
            '''
            if error in all_anom:
                true_anoms += 1
            if error in anom_sc1:
                print(ind,error)
                #all_inds.append(ind)
                anom_inds1.append(ind)
                cor_1 += 1
            if error in anom_sc2:
                anom_inds2.append(ind)
                cor_2 += 1
            if error in anom_sc3:
                anom_inds3.append(ind)
                cor_3 += 1
            #if error in all_anom:
            #    print(ind)
        #import ipdb ; ipdb.set_trace()
        if False:
            import ipdb ; ipdb.set_trace()
            import matplotlib.pyplot as plt
            plt.figure()
            #skew1=round(scipy.stats.skew(anom_inds1),.5)
            #skew2=round(scipy.stats.skew(anom_inds2),.75)
            skew3=round(scipy.stats.skew(anom_inds3),1)
            plt.hist(anom_inds1,color='r',alpha=1,range=(0,200),bins=200)
            plt.hist(anom_inds2,color='g',alpha=1,range=(0,200),bins=200)
            plt.hist(anom_inds3,color='b',alpha=1,range=(0,200),bins=200)
            plt.title(f'{skew1},{skew2},{skew3}')
            plt.savefig(f'dists_{start}_{end}')

        return true_anoms/int(all_anom.shape[0]), cor_1, cor_2, cor_3, true_anoms
        
    def normalize_adj(self,adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        def test_parameters(self):
            assert (hasattr(self.model, 'decision_scores_') and
                    self.model.decision_scores_ is not None)
            assert (hasattr(self.model, 'labels_') and
                    self.model.labels_ is not None)
            assert (hasattr(self.model, 'threshold_') and
                    self.model.threshold_ is not None)
            assert (hasattr(self.model, '_mu') and
                    self.model._mu is not None)
            assert (hasattr(self.model, '_sigma') and
                    self.model._sigma is not None)
            assert (hasattr(self.model, 'model') and
                    self.model.model is not None)

    def test_train_scores(self):
        assert_equal(len(self.model.decision_scores_), len(self.data['truth']))

    def test_prediction_scores(self):
        pred_scores = self.model.decision_function(self.data)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.data['truth'].shape[0])

        # check performance
        assert (eval_roc_auc(self.data['truth'], pred_scores) >= self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.model.predict(self.data)
        assert_equal(pred_labels.shape[0], self.data['truth'].shape[0])

    def test_prediction_proba(self):
        pred_proba = self.model.predict_proba(self.data)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_linear(self):
        pred_proba = self.model.predict_proba(self.data, method='linear')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_unify(self):
        pred_proba = self.model.predict_proba(self.data, method='unify')
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.model.predict_proba(self.data, method='something')

    def test_prediction_labels_confidence(self):
        pred_labels, confidence = self.model.predict(self.data,
                                                     return_confidence=True)
        assert_equal(pred_labels.shape[0], self.data['truth'].shape[0])
        assert_equal(confidence.shape[0], self.data['truth'].shape[0])
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_prediction_proba_linear_confidence(self):
        pred_proba, confidence = self.model.predict_proba(self.data,
                                                          method='linear',
                                                          return_confidence=True)
        assert (pred_proba.min() >= 0)
        assert (pred_proba.max() <= 1)

        assert_equal(confidence.shape[0], self.data['truth'].shape[0])
        assert (confidence.min() >= 0)
        assert (confidence.max() <= 1)

    def test_model_clone(self):
        pass
        # clone_clf = clone(self.model)

    def tearDown(self):
        pass
        # remove the data folder
        # rmtree(self.path)


if __name__ == '__main__':
    #import ipdb ; ipdb.set_trace()
    unittest.main()
