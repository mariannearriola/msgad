# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
import scipy.io as sio
import torch
from torch_geometric.utils import to_dense_adj

from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5


class MultiScale(Graph):
    r"""Minnesota road network (from MatlabBGL).

    Parameters
    ----------
    connected : bool
        If True, the adjacency matrix is adjusted so that all edge weights are
        equal to 1, and the graph is connected. Set to False to get the
        original disconnected graph.

    References
    ----------
    See :cite:`gleich`.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Minnesota()
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=0.5)
    >>> _ = G.plot(ax=axes[1])

    """
    def update_anom(self,anom_type):
        anom_sc1 = self.data['anom_sc1']
        anom_sc2 = self.data['anom_sc2']
        anom_sc3 = self.data['anom_sc3']
        anom_single = self.data['anom_single']
        if 'weibo' in self.fl:
            self.anom_combo = np.concatenate((np.concatenate((np.concatenate((self.flatten_label(anom_sc1),self.flatten_label(anom_sc2)),axis=None),self.flatten_label(anom_sc3)),axis=None),anom_single),axis=None)
        else:
            self.anom_combo = np.concatenate((np.concatenate((np.concatenate((anom_sc1,anom_sc2),axis=None),anom_sc3),axis=None),anom_single),axis=None)
        def getWeights(anom):
            return [i.shape[-1] for i in anom]
            
        if anom_type == 'none':
            anoms = None
            anom_w = None
        elif anom_type == 'single':
            anoms = anom_single[0]
            anom_w = np.ones(anoms.shape[0])
        elif anom_type == 'all':
            anoms = self.fix_label(anom_sc1)
            if len(anom_sc2) > 0:
                for i in self.fix_label(anom_sc2):
                    anoms.append(i)
            for i in self.fix_label(anom_sc3):
                anoms.append(i)
            anoms.append(anom_single[-0])
            #anoms =  np.concatenate((np.concatenate((np.concatenate((anom_sc1,anom_sc2),axis=None),anom_sc3),axis=None),anom_single))
            anom_w = None
        elif anom_type == 'sc1':
            anoms = anom_sc1
        elif anom_type == 'sc2':
            anoms = anom_sc2
        elif anom_type == 'sc3':
            anoms = anom_sc3
        elif anom_type == 'random':
            anoms = np.random.randint(0,self.A.shape[0],size=(anom_single[0].shape[0]))
            anom_w = None
            
        if 'weibo' in self.fl and anom_type not in ['single','none','all','random']:
            anoms = self.fix_label(anoms)

        #self.anom_flat = self.flatten_label(anoms) if anom_type not in ['single','none','all'] else anoms
        if anom_type in ['single','none','random']:
            self.anom_flat = anoms
        else:
            for ind,i in enumerate(anoms):
                if ind == 0:
                    self.anom_flat = i
                else:
                    self.anom_flat = np.concatenate((self.anom_flat,i))
        if anom_type not in ['single','none','random']:
            anom_w = getWeights(anoms)

        if anom_type in ['single','random']:
            anom_tot = anoms.shape[0]
        elif anom_type != 'none':
            anom_tot = sum([i.shape[0] for i in anoms])#anom_combo.shape[0]#anoms.shape[0]
        else:
            anom_tot = [0]
        anom_w = None
        return anoms,anom_tot,anom_w
    
    def fix_label(self,anoms):
        #if self.fl in ['weibo','weibo_aug']:
        #    anoms = anoms[0]
        if ('weibo' in self.fl or 'tfinance' in self.fl) and type(anoms[0][0]) != np.int64:
            anoms = anoms[0]
            anom_flat = [anoms[0][0].flatten()]
        else:
            anom_flat = [anoms[0]]
        for i in anoms[1:]:
            if 'weibo' in self.fl or 'tfinance' in self.fl:
                anom_flat.append(i[0])
            else:
                anom_flat.append(i)
                #anom_flat=np.concatenate((anom_flat,i))
        return anom_flat
        
    def flatten_label(self,anoms):
        if 'weibo' in self.fl and type(anoms[0][0]) != np.int64:
            anoms = anoms[0]
        anom_flat = anoms[0]
        if 'weibo' in self.fl:
            anom_flat = anom_flat[0]
        for i in anoms[1:]:
            if 'weibo' in self.fl:
                anom_flat=np.concatenate((anom_flat,i[0]))
            else:
                anom_flat=np.concatenate((anom_flat,i))
        return anom_flat

    def __init__(self, graph_type="cora_original", anom_type='single', drop_anoms=False, connected=True, **kwargs):

        self.connected = connected
        fl=graph_type
        self.fl = fl
        #data = sio.loadmat(f'pygsp/data/ms_data/{fl}.mat')

        data = sio.loadmat(f'../msgad/data/{fl}.mat')
        if 'Edge-index' in data.keys():
            A = to_dense_adj(torch.tensor(data['Edge-index']))[0]
        else:
            A = data['Network'].todense()
        #A = np.maximum(A,A.T)
        X = data['Attributes']
        if 'cora' in fl:
            X = X.todense()
        self.data = data
        
                    
        plotting = {"limits": np.array([-98, -89, 43, 50]),
                    "vertex_size": 40}
        import ipdb ; ipdb.set_trace()
        super(MultiScale, self).__init__(A, X, None, None, None,#coords=data['xy'],
                                        plotting=plotting, **kwargs)
        return
        

        anoms,anom_tot,anom_w = self.update_anom(anom_type)
        to_remove = []

        if drop_anoms and anoms is not None:
            to_remove = np.setdiff1d(self.anom_combo,self.anom_flat)
        elif drop_anoms and anoms is None:
            to_remove = self.anom_combo

        if len(to_remove) > 0:
            if anom_type != 'none':
                lbl_idx = np.zeros(A.shape[0])
                if anom_type != 'none':
                    np.put(lbl_idx,self.anom_flat,1.)
                
                lbl_idx = np.delete(lbl_idx,to_remove) ; new_idx = lbl_idx.nonzero()[0]
                lbl_dict = {k:v for k,v in zip(self.anom_flat,new_idx)}
                anoms = [np.vectorize(lbl_dict.get)(i) for i in anoms]
            A = np.delete(A,to_remove,0) 
            A = np.delete(A,to_remove,1)
            X = np.delete(X,to_remove,0)
        import ipdb ; ipdb.set_trace()         
        '''
        if connected:
            # Missing edges needed to connect the graph.
            A = sparse.lil_matrix(A)
            A[348, 354] = 1
            A[354, 348] = 1
            A = sparse.csc_matrix(A)

            # Binarize: 8 entries are equal to 2 instead of 1.
            A = (A > 0).astype(bool)
        '''
        super(Minnesota, self).__init__(A, X, anoms, anom_tot, anom_w,#coords=data['xy'],
                                        plotting=plotting, **kwargs)

    def _get_extra_repr(self):
        return dict(connected=self.connected)