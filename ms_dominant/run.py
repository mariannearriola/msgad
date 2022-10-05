from scipy.sparse import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse
import scipy.io
from sklearn.metrics import roc_auc_score
from datetime import datetime
import argparse
import scipy

#from model import Dominant
from model import EGCN
from utils import load_anomaly_detection_dataset



def loss_func(adj, A_hat_scales, attrs, X_hat, alpha, weight=None):
    attribute_cost = torch.tensor(0., dtype=torch.float32).cuda()

    # structure reconstruction loss
    all_costs, all_structure_costs = None, None
    for ind, A_hat in enumerate(A_hat_scales):
        if ind > 0:
            break
        #diff_structure = torch.pow(A_hat - adj, 2)
        #structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
        weight = None
        #import ipdb ; ipdb.set_trace()
        if weight is not None:
            #structure_reconstruction_errors = F.binary_cross_entropy(A_hat.flatten(), adj.flatten(), weight = weight)
            structure_reconstruction_errors = F.mse_loss(A_hat, attrs, weight = weight, reduction="none")
        else:
            #structure_reconstruction_errors = F.binary_cross_entropy(A_hat.flatten(), adj.flatten())
            structure_reconstruction_errors = F.mse_loss(A_hat, attrs, reduction="none")
        
        #structure_reconstruction_errors = F.mse_loss(A_hat, adj)
        #import ipdb ; ipdb.set_trace()
        structure_cost = torch.mean(torch.mean(structure_reconstruction_errors,axis=1))
        #structure_reconstruction_errors = F.mse_loss(A_hat,adj)
        
        #if ind == 2:
        '''
        if ind == 1:
        #if ind == len(A_hat_scales)-1:
            structure_reconstruction_errors *= 1
            structure_cost *= 1
        else:
            structure_reconstruction_errors *= 0
            structure_cost *= 0
        '''
        if all_costs is None:
            all_costs = structure_reconstruction_errors
            all_structure_costs = structure_cost
        else:
            all_costs = torch.add(all_costs,structure_reconstruction_errors)
            all_structure_costs = torch.add(all_structure_costs,structure_cost)
        
    #cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors

    #return cost, structure_cost, attribute_cost
    return all_costs, all_structure_costs, attribute_cost

def detect_anom(sorted_errors, label, top_nodes_perc):      
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

def train_dominant(args):

    adj, attrs_det, label, adj_label, sc_label = load_anomaly_detection_dataset(args.dataset, args.scales)
    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    model = EGCN(in_size = attrs_det[0].size(1), out_size = args.hidden_dim, scales = len(attrs_det))
    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        attrs = []
        for attr in attrs_det:
            attrs.append(attr.to(device))
        model = model.cuda()

    # weigh positive edges for loss calculation
    
    weight_mask = torch.where(adj_label.flatten() == 1)[0]
    weight_tensor = torch.ones(adj_label.flatten().shape).to(device)
    pos_weight = float(adj_label.shape[0] * adj_label.shape[0] - adj_label.sum()) / adj_label.sum()
    weight_tensor[weight_mask] = pos_weight
    
    #weight_tensor = 0
    #import ipdb ; ipdb.set_trace() 
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    best_loss = torch.tensor(float('inf')).cuda()
    model.train()
    for epoch in range(args.epoch):
        optimizer.zero_grad()
        #import ipdb ; ipdb.set_trace()
        #A_hat, X_hat = model(attrs, adj)
        A_hat_scales, X_hat = model(attrs[0],adj)
        loss, struct_loss, feat_loss = loss_func(adj_label, A_hat_scales, attrs[0], X_hat, args.alpha, weight_tensor)
        #import ipdb ; ipdb.set_trace()
        l = torch.mean(loss)
        '''
        if l < best_loss:
            best_loss = l
            torch.save(model,'best_model.pt')
        '''
        l.backward()
        optimizer.step() 
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))

        #if epoch%10 == 0 or epoch == args.epoch - 1:
        if epoch == args.epoch-1:
            model.eval()
            #A_hat, X_hat = model(attrs, adj)
            A_hat_scales, X_hat = model(attrs[0],adj)
            loss, struct_loss, feat_loss = loss_func(adj_label, A_hat_scales, attrs[0], X_hat, args.alpha, weight_tensor)
            score = loss.detach().cpu().numpy()
            print("Epoch:", '%04d' % (epoch))#, 'Auc', roc_auc_score(label, score))

    print('best loss:', best_loss)
    adj, attrs_det, label, adj_label, sc_label = load_anomaly_detection_dataset(args.dataset, args.scales)
    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    #attrs = torch.FloatTensor(attrs)

    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        attrs = []
        for attr in attrs_det:
            attrs.append(attr.to(device))
        
    # weigh positive edges for loss calculation
    '''
    weight_mask = torch.where(adj_label.flatten() == 1)[0]
    weight_tensor = torch.ones(adj_label.flatten().shape).to(device)
    pos_weight = float(adj_label.shape[0] * adj_label.shape[0] - adj_label.sum()) / adj_label.sum()
    weight_tensor[weight_mask] = pos_weight
    '''

    #model = torch.load('best_model.pt')
    
    #torch.save(model,'model.pt')
    #model = torch.load('model.pt')
    model.eval()
    A_hat_scales, X_hat = model(attrs[0],adj)

    loss, struct_loss, feat_loss = loss_func(adj_label, A_hat_scales, attrs[0], X_hat, args.alpha)
    scores = loss.detach().cpu().numpy()
    
    import ipdb ; ipdb.set_trace()
    # anomaly evaluation
    for sc, A_hat in enumerate(A_hat_scales):
        recons_errors = F.mse_loss(A_hat.detach().cpu(), attrs[0].detach().cpu(),reduction="none")
        #scores = scipy.stats.skew(recons_errors.numpy(),axis=0)
        scores = np.mean(recons_errors.numpy(),axis=1)
        sorted_errors = np.argsort(-scores)
        rankings = []
        for error in sorted_errors:
            rankings.append(label[error])
        rankings = np.array(rankings)

        #import ipdb ; ipdb.set_trace()
        print(f'SCALE {sc+1}')
        #prec,anom1,anom2,anom3,all_anom=detect_anom(sorted_errors, label, 1)
        print(detect_anom(sorted_errors, sc_label, 1))
        print(detect_anom(sorted_errors, sc_label, .75))
        print(detect_anom(sorted_errors, sc_label, .50))
        #print(detect_anom(sorted_errors, label, .25))
        print('')
    
        with open('output/{}-ranking_{}.txt'.format(args.dataset, sc), 'w') as f:
            for index in sorted_errors:
                f.write("%s\n" % label[index])
        
        import pandas as pd
        #df = pd.DataFrame({'AD-GCA':reconstruction_errors})
        df = pd.DataFrame({'AD-GCA':scores})
        df.to_csv('output/{}-scores_{}.csv'.format(args.dataset, sc), index=False, sep=',')
    ''' 
    import ipdb ; ipdb.set_trace() 
    import MADAN.Plotters as Plotters
    from MADAN._cython_fast_funcs import sum_Sto, sum_Sout, compute_S, cython_nmi, cython_nvi
    from MADAN.LouvainClustering_fast import Clustering, norm_var_information
    import MADAN.Madan as md
    madan = md.Madan(adj, attributes=attrs[-1], sigma=0.08)
    time_scales = np.concatenate([np.array([0]), 10**np.linspace(0,5,500)])
    madan.scanning_relevant_context(time_scales, n_jobs=4)
    madan.scanning_relevant_context_time(time_scales)
    madan.compute_concentration(1000)
    print(madan.concentration,madan.anomalous_nodes)
    madan.compute_context_for_anomalies()
    print(madan.interp_com)
    print(' ------------')
    '''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora_triple_anom', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--hidden_dim', type=int, default=1433, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=80, help='Training epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--scales', default='3', type=int, help='number of scales for multi-scale analysis')

    args = parser.parse_args()

    train_dominant(args)
