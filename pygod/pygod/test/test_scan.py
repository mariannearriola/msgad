# -*- coding: utf-8 -*-
import os
import unittest
from numpy.testing import assert_equal
from numpy.testing import assert_warns
from numpy.testing import assert_raises

import torch
from torch_geometric.seed import seed_everything

from pygod.metric import eval_roc_auc
from pygod.detector import SCAN

seed_everything(717)


class TestSCAN(unittest.TestCase):
    def setUp(self):
        self.roc_floor = 0.60

        self.train_data = torch.load(os.path.join('pygod/test/train_graph.pt'))
        self.test_data = torch.load(os.path.join('pygod/test/test_graph.pt'))

    def test_full(self):
        detector = SCAN(eps=0.2, mu=2)
        detector.fit(self.train_data)

        pred, score, conf = detector.predict(return_pred=True,
                                             return_score=True,
                                             return_conf=True)

        assert_equal(pred.shape[0], self.train_data.y.shape[0])
        # TODO: assert (eval_roc_auc(self.train_data.y, score) >= self.roc_floor)
        assert_equal(conf.shape[0], self.train_data.y.shape[0])
        assert (conf.min() >= 0)
        assert (conf.max() <= 1)

        prob = detector.predict(return_pred=False,
                                return_prob=True,
                                prob_method='linear')
        assert_equal(prob.shape[0], self.train_data.y.shape[0])
        assert (prob.min() >= 0)
        assert (prob.max() <= 1)

        prob = detector.predict(return_pred=False,
                                return_prob=True,
                                prob_method='unify')
        assert_equal(prob.shape[0], self.train_data.y.shape[0])
        assert (prob.min() >= 0)
        assert (prob.max() <= 1)

        with assert_raises(ValueError):
            detector.predict(return_prob=True,
                             prob_method='something')

    def test_params(self):
        detector = SCAN(eps=0.2,
                        mu=2,
                        contamination=0.3,
                        verbose=3)
        detector.fit(self.train_data)

        with assert_warns(UserWarning):
            detector.predict(self.test_data)
