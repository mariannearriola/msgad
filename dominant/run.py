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

#from model import Dominant
from model import EGCN
from utils import load_anomaly_detection_dataset



def loss_func(adj, A_hat_scales, attrs, X_hat, alpha, weight=None):
    attribute_cost = torch.tensor(0., dtype=torch.float32).cuda()

    # structure reconstruction loss
    all_costs, all_structure_costs = None, None
    #for ind,A_hat in enumerate(A_hat_scales[-2:]):
    for ind, A_hat in enumerate(A_hat_scales):
        #diff_structure = torch.pow(A_hat - adj, 2)
        #structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
        #import ipdb ; ipdb.set_trace()
        if weight is not None:
            structure_reconstruction_errors = F.binary_cross_entropy(A_hat.flatten(), adj.flatten(), weight = weight)
        else:
            structure_reconstruction_errors = F.binary_cross_entropy(A_hat.flatten(), adj.flatten())
        #structure_reconstruction_errors = F.mse_loss(A_hat, adj)

        structure_cost = torch.mean(structure_reconstruction_errors)
        if ind == len(A_hat_scales)-1:
            structure_reconstruction_errors *= 1
            structure_cost *= 1
        else:
            structure_reconstruction_errors *= 0
            structure_cost *= 0

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
    anom_sc1 = np.array([1653,879])
    anom_sc2 = np.array([1276, 376, 804, 867, 906, 1143, 574, 1671, 2, 962, 2183, 643, 196, 636, 1446])
    anom_sc3 = np.array([2355, 1422, 2557, 1222, 788, 1526, 2143, 1895, 1405, 731, 968, 657, 2300, 783, 2424, 1547, 2399, 361, 967, 703, 402, 1423, 583, 1924, 2585, 1624, 395, 862, 294, 0, 1195, 1270, 479, 1213, 2353, 1172, 2277, 1286, 935, 2136, 2623, 665, 2468, 2132, 414])
    all_anom = np.concatenate((anom_sc1,np.concatenate((anom_sc2,anom_sc3),axis=None)),axis=None)

    true_anoms = 0
    #import ipdb ; ipdb.set_trace()
    cor_1, cor_2, cor_3 = 0,0,0
    for error in sorted_errors[:int(62*top_nodes_perc)]:
        if label[error] == 1:
            true_anoms += 1
        if error in anom_sc1:
            cor_1 += 1
        if error in anom_sc2:
            cor_2 += 1
        if error in anom_sc3:
            cor_3 += 1
    return true_anoms/int(62*top_nodes_perc), cor_1, cor_2, cor_3, true_anoms

def train_dominant(args):

    adj, attrs_det, label, adj_label = load_anomaly_detection_dataset(args.dataset, args.scales)
    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    #attrs = torch.FloatTensor(attrs)

    model = EGCN(in_size = attrs_det[0].size(1), out_size = args.hidden_dim, scales = len(attrs_det))
    #model = Dominant(feat_size = attrs.size(1), hidden_size = args.hidden_dim, dropout = args.dropout)

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
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    best_loss = torch.tensor(float('inf')).cuda()
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        #A_hat, X_hat = model(attrs, adj)
        A_hat_scales, X_hat = model(attrs)
        loss, struct_loss, feat_loss = loss_func(adj_label, A_hat_scales, attrs[-1], X_hat, args.alpha, weight_tensor)
        #import ipdb ; ipdb.set_trace()
        l = torch.mean(loss)
        if l < best_loss:
            best_loss = l
            torch.save(model,'best_model.pt')
        l.backward()
        optimizer.step() 
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))

        #if epoch%10 == 0 or epoch == args.epoch - 1:
        if epoch == args.epoch-1:
            model.eval()
            #A_hat, X_hat = model(attrs, adj)
            A_hat_scales, X_hat = model(attrs)
            loss, struct_loss, feat_loss = loss_func(adj_label, A_hat_scales, attrs[-1], X_hat, args.alpha, weight_tensor)
            score = loss.detach().cpu().numpy()
            print("Epoch:", '%04d' % (epoch))#, 'Auc', roc_auc_score(label, score))

    print('best loss:', best_loss)
    adj, attrs_det, label, adj_label = load_anomaly_detection_dataset('cora_sparse_anoms', args.scales)
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
    model = torch.load('best_model.pt')
    '''
    #torch.save(model,'model.pt')
    model = torch.load('model.pt')
    model.eval()
    A_hat, X_hat = model(attrs, adj)
    '''
    loss, struct_loss, feat_loss = loss_func(adj_label, A_hat_scales, attrs[-1], X_hat, args.alpha)
    #score = loss.detach().cpu().numpy()
    scores = loss.detach().cpu().numpy()
    import scipy
    import torch.nn.functional as F
 

    # anomaly evaluation
    for sc, A_hat in enumerate(A_hat_scales):
        recons_errors = F.mse_loss(A_hat.detach().cpu(), adj_label.detach().cpu(),reduction="none")
        scores = scipy.stats.skew(recons_errors.numpy(),axis=0)
        
        sorted_errors = np.argsort(scores)
        rankings = []
        for error in sorted_errors:
            rankings.append(label[error])
        rankings = np.array(rankings)

        #import ipdb ; ipdb.set_trace()
        print(f'SCALE {sc+1}')
        #prec,anom1,anom2,anom3,all_anom=detect_anom(sorted_errors, label, 1)
        print(detect_anom(sorted_errors, label, 1))
        print(detect_anom(sorted_errors, label, .75))
        print(detect_anom(sorted_errors, label, .50))
        #print(detect_anom(sorted_errors, label, .25))
        print('')
    
        with open('output/{}-ranking_{}.txt'.format(args.dataset, sc), 'w') as f:
            for index in sorted_errors:
                f.write("%s\n" % label[index])
        
        import pandas as pd
        #df = pd.DataFrame({'AD-GCA':reconstruction_errors})
        df = pd.DataFrame({'AD-GCA':scores})
        df.to_csv('output/{}-scores_{}.csv'.format(args.dataset, sc), index=False, sep=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='BlogCatalog', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.8, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--scales', default='1', type=int, help='number of scales for multi-scale analysis')

    args = parser.parse_args()

    train_dominant(args)
