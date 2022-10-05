# -*- coding: utf-8 -*-
import os
import unittest
# noinspection PyProtectedMember
from numpy.testing import assert_equal
from numpy.testing import assert_raises
import numpy as np
import torch
#from torch_geometric.seed import seed_everything

from pygod.models import MLPAE
from pygod.utils.metric import eval_roc_auc
import scipy.io as sio
import scipy.sparse as sp
#seed_everything(42)


class TestMLPAE(unittest.TestCase):
    def setUp(self):
        # use the pre-defined fake graph with injected outliers
        # for testing purpose

        # the roc should be higher than this; it is model dependent
        self.roc_floor = 0.60

        #test_graph = torch.load(os.path.join('pygod', 'test', 'test_graph.pt'))
        dataset = 'cora_triple_anom'
        data_mat = sio.loadmat(f'../data/{dataset}.mat')
        feats = torch.FloatTensor(data_mat['Attributes'].toarray())
        #feats = []
        #for scales in range(1,sc+1):
        #    feats.append(torch.FloatTensor(sio.loadmat(f'../smoothed_graphs/{scales}_{dataset}.mat')['Attributes'].toarray()))
        #import ipdb ; ipdb.set_trace()
        adj = data_mat['Network']
        #feat = data_mat['Attributes']
        truth = data_mat['Label']
        truth = truth.flatten()
        label = data_mat['scale_anomaly_label']
        self.truth = truth
        #import ipdb ; ipdb.set_trace()
        adj_norm = self.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_norm = adj_norm.toarray()
        adj = adj + sp.eye(adj.shape[0])
        adj = adj.toarray()
        #import ipdb ; ipdb.set_trace() 
        test_graph=adj
        self.graph = test_graph
        self.feat = feats
        self.model = MLPAE()
        self.data={'graph':self.graph,'feat':self.feat,'truth':self.truth}
        self.model.fit(self.data)
        
        
        outlier_scores = self.model.decision_scores_
        outlier_scores = self.model.decision_function(self.data)
        anom_rankings = np.argsort(-outlier_scores)
        # TODO: get anomaly labels
        print(self.detect_anom(anom_rankings,label,.5))
        print(self.detect_anom(anom_rankings,label,.75))
        print(self.detect_anom(anom_rankings,label,1))
        
    def detect_anom(self,sorted_errors, label, top_nodes_perc):
          
        anom_sc1 = label[0][0][0]
        anom_sc2 = label[1][0][0]
        anom_sc3 = label[2][0][0]
        all_anom = np.concatenate((anom_sc1,np.concatenate((anom_sc2,anom_sc3),axis=None)),axis=None)
        
        true_anoms = 0
        #import ipdb ; ipdb.set_trace()
        cor_1, cor_2, cor_3 = 0,0,0
        for ind,error in enumerate(sorted_errors[:int(all_anom.shape[0]*top_nodes_perc)]):
            '''
            if label[ind] == 1:
                true_anoms += 1
            '''
            if error in all_anom:
                true_anoms += 1
            if error in anom_sc1:
                cor_1 += 1
            if error in anom_sc2:
                cor_2 += 1
            if error in anom_sc3:
                cor_3 += 1
            if error in all_anom:
                print(ind)
        return true_anoms/int(all_anom.shape[0]*top_nodes_perc), cor_1, cor_2, cor_3, true_anoms
        
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
