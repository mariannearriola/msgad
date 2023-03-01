from scipy.sparse import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.io
from sklearn.metrics import roc_auc_score
from datetime import datetime
import argparse
import scipy
import networkx as nx
import torch_geometric
from torch.utils.data import DataLoader
from scipy import stats
#from model import Dominant
from model_cora import EGCN
from utils import * 
from anomaly_utils import *
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KernelDensity
import random 


def train_dominant(args):

    adj, attrs_det, label, adj_label, sc_label, adj_train, attrs_train, adj_val, attrs_val = load_anomaly_detection_dataset(args.dataset, args.scales, args.mlp, args.parity)
    adj, adj_train, adj_val = torch.FloatTensor(adj), torch.FloatTensor(adj_train), torch.FloatTensor(adj_val)
    adj_label = torch.FloatTensor(adj_label)
    #adj = adj_label
    model = EGCN(in_size = attrs_det[0].size(1), out_size = args.hidden_dim, scales = args.scales, recons = args.recons, mlp = args.mlp, d = args.d)
    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_train = adj_train.to(device)
        adj_val = adj_val.to(device)
        adj_label = adj_label.to(device)
        attrs_train[0] = attrs_train[0].to(device)
        attrs_val[0] = attrs_val[0].to(device)
        attrs = []
        for attr in attrs_det:
            attrs.append(attr.to(device))
        model = model.cuda()
    adj_ori = adj_train.clone() + torch.eye(adj_train.shape[0]).to(device)
    attrs_ori = attrs[0].clone()
    # weigh positive edges for loss calculation
    weight_mask = torch.where(adj_train.flatten() == 1)[0]
    weight_tensor = torch.ones(adj_train.flatten().shape).to(device)
    pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
    weight_tensor[weight_mask] = pos_weight
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    score = None
    #adj=torch_geometric.utils.to_dense_adj(edge_index)[0]
    nx_graph = nx.from_numpy_matrix(adj_label.detach().cpu().numpy())
    
    best_loss = torch.tensor(float('inf')).cuda()
    model.train()
    for epoch in range(args.epoch):
        
        optimizer.zero_grad()
        X_hat,A_hat_scales = model(attrs,adj,sc_label)
        loss, struct_loss, feat_loss = loss_func(adj_label, A_hat_scales, attrs[0], X_hat, args.alpha, args.recons, weight_tensor)
        l = torch.mean(loss)
        if l < best_loss:
            best_loss = l
            torch.save(model,f'best_model_{args.d}.pt')
        l.backward()
        optimizer.step()

        num_nonzeros=[]
        num_posedges = []
        for sc, A_hat in enumerate(A_hat_scales):
            num_nonzeros.append(torch.where(A_hat < 0.5)[0].shape[0])
            num_posedges.append(torch.where(A_hat >= 0.5)[0].shape[0])
        print(A_hat_scales[-1])
        print("Epoch:", '%04d' % (epoch), "train_loss=", loss, "Non-edges:",np.round(np.array(num_nonzeros)/(A_hat.shape[0]**2),decimals=2))
        '''
        #if epoch%10 == 0 or epoch == args.epoch - 1:
        if epoch == args.epoch-1:
            #import ipdb ; ipdb.set_trace()
            model.eval()
            #A_hat, X_hat = modl(attrs, adj)
            A_hat_scales, X_hat = model(attrs_train,adj_train,sc_label)
            loss, struct_loss, feat_loss = loss_func(adj_train, A_hat_scales, attrs_train, X_hat, args.alpha, args.recons, weight_tensor)
            score = loss.detach().cpu().numpy()
            num_nonzeros=[]
            for sc, A_hat in enumerate(A_hat_scales):
                num_nonzeros.append(torch.where(A_hat < 0.5)[0].shape[0])
            print("Epoch:", '%04d' % (epoch), 'Non-edges:', num_nonzeros)#, 'Auc', roc_auc_score(label, score))
            break
        '''
    print('best loss:', best_loss)
    adj, attrs_det, label, adj_label, sc_label, train_adj, train_attrs, val_adj, val_attrs = load_anomaly_detection_dataset(args.dataset, args.scales, args.mlp, args.parity)
    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    #attrs = torch.FloatTensor(attrs)
    #import ipdb ; ipdb.set_trace()
    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        train_adj = torch.FloatTensor(train_adj)#.to(device)
        #adj=adj_label
        attrs = []
        for attr in attrs_det:
            attrs.append(attr.to(device))
    anom_idx,norm_idx = np.where(label==1)[0], np.where(label==0)[0]
    # weigh positive edges for loss calculation
    '''
    weight_mask = torch.where(adj_label.flatten() == 1)[0]
    weight_tensor = torch.ones(adj_label.flatten().shape).to(device)
    pos_weight = float(adj_label.shape[0] * adj_label.shape[0] - adj_label.sum()) / adj_label.sum()
    weight_tensor[weight_mask] = pos_weight
    '''

    model = torch.load(f'best_model_{args.d}.pt')
    
    #torch.save(model,'model.pt')
    #model = torch.load('model.pt')
    model.eval()
    
    #import ipdb ; ipdb.set_trace()
    #lipschitz = calc_lipschitz(A_hat_scales, A_hat_scales_pert, factor_pert)
    #scores = lipschitz.detach().cpu().numpy()
    #scores = loss.detach().cpu().numpy()
    thresh=[1.3,2.1,2.1,2.1,2.1,2.1,2.1,2.1]
    #import ipdb ; ipdb.set_trace()
    # anomaly evaluation
    score = None
    #adj=torch_geometric.utils.to_dense_adj(edge_index)[0]
    nx_graph = nx.from_numpy_matrix(adj_label.detach().cpu().numpy())
    #A_hat_scales, X_hat = model(attrs,adj,sc_label)

    scores, l_scores = torch.zeros(adj.shape[0],args.d+1),np.zeros((adj.shape[0],args.d+1))
    
    # !!! EMBEDDINGS RETRIEVED !!!
    #X_hat,A_hat_scales = model([attrs_ori],adj_ori,sc_label)
    A_hat_scales,X_hat = model(attrs,adj,sc_label)
    #weights = [model.conv.linear.weight.data]#,model.conv.linear2.weight.data]
    #weights = [model.conv.attention_layer.state_dict()['lin_l.weight'],model.conv.attention_layer.state_dict()['lin_r.weight']]
    # TODO: for n random/normal nodes, plot how pert affects l_score
    '''
    rand_anoms = np.random.choice(np.where(label==1)[0],5)
    rand_norms = np.random.choice(np.where(label==0)[0],5)
    for rand_anom in rand_anoms:
        plot_embed_diff(attrs,adj_label,random_anom,'anom')
    for rand_norm in rand_norms:
        plot_embed_diff(attrs,adj_label,random_norm,'norm')
    weights = [] 
    for node in range(adj.shape[0]):
        #import ipdb ; ipdb.set_trace()
        x_pert,a_pert = perturb_adj(attrs[0],train_adj,0.5,0.5,node)
        
        #import ipdb ; ipdb.set_trace()
        a_pert = a_pert.to(device).cuda().type(torch.float32)
        x_pert_full = attrs[0].clone()
        x_pert_full[node]=x_pert
        x_pert_full = x_pert_full.type(torch.float32)
        #import ipdb ; ipdb.set_trace()
        #x_pert_full = attrs[0]
        #a_pert = adj
        emb_pert,_ = model([x_pert_full],a_pert,sc_label)
        #import ipdb ; ipdb.set_trace()
        l_score,bound_met=calc_lipschitz(adj_ori, a_pert, attrs_ori, x_pert_full, X_hat, emb_pert, weights, node)
        l_scores[node] = l_score
        #import ipdb ; ipdb.set_trace()
        
        if not bound_met:
            print('node',node,'does not meet bound. anomaly label is ',label[node])
        else:
            print('node',node,'meets bound! label is',label[node])
    print('done')
    '''
    #import ipdb ; ipdb.set_trace()
    rand_anoms=[]
    for s_lb in sc_label[0]:
        rand_anoms.append(np.random.choice(redo(s_lb),5))
    rand_norms = np.random.choice(np.where(label==0)[0],15)
        
    loss, struct_loss, feat_loss = loss_func(adj_label, A_hat_scales, attrs[0], X_hat, args.alpha, args.recons)
    scores = struct_loss
    for sc, A_hat in enumerate(A_hat_scales):
            '''
            weights = [model.conv.linear.weight.data]#,model.conv.linear2.weight.data]    
            for node in range(adj.shape[0]):
                #import ipdb ; ipdb.set_trace()
                x_pert,a_pert = perturb_adj(attrs[0],train_adj,0.8,0.8,node)
                
                #import ipdb ; ipdb.set_trace()
                a_pert = a_pert.to(device).cuda().type(torch.float32)
                x_pert_full = attrs[0].clone()
                x_pert_full[node]=x_pert
                x_pert_full = x_pert_full.type(torch.float32)
                #import ipdb ; ipdb.set_trace()
                #x_pert_full = attrs[0]
                #a_pert = adj
                emb_pert,_ = model([x_pert_full],a_pert,sc_label)
                #import ipdb ; ipdb.set_trace()
                bound_met,l_score=calc_lipschitz(adj_ori, a_pert, attrs_ori, x_pert_full, X_hat, emb_pert, weights, node)
                #import ipdb ; ipdb.set_trace()
                 
                if not bound_met:
                    print('node',node,'does not meet bound. anomaly label is ',label[node])
                else:
                    print('node',node,'meets bound! label is',label[node])
                
                l_scores[node][sc] = l_score
                a_pert,x_pert_full = a_pert.detach().cpu(),x_pert_full.detach().cpu()
                emb_pert,x_pert = emb_pert.detach().cpu(),x_pert.detach().cpu()
                del emb_pert
                del a_pert
                del x_pert_full
                del x_pert
                torch.cuda.empty_cache()
            print('done')
        '''
        #recons_errors = F.mse_loss(A_hat.detach().cpu(), adj.detach().cpu(),reduction="none")
        #scores=recons_errors
        #lipschitz = calc_lipschitz(A_hat_scales[sc], A_hat_scales_pert[sc], factor_pert)
        #scores = lipschitz.detach().cpu().numpy()
    kdes = []
    for sc, A_hat in enumerate(A_hat_scales):    
        #scores = np.mean(recons_errors.numpy(),axis=1)
        #import ipdb ; ipdb.set_trace()
        scores_recons = scores[:,sc]
        sorted_errors = np.argsort(-scores_recons)
        rev_sorted_errors = np.argsort(scores_recons)
        rankings = []
        for error in sorted_errors:
            rankings.append(label[error])
        rankings = np.array(rankings)

        #import ipdb ; ipdb.set_trace()
        print(f'SCALE {sc+1}, LOSS',torch.mean(scores_recons).item())
        try:
            print('AP',detect_anom_ap(scores_recons,label))
        except:
            import ipdb ; ipdb.set_trace()
        #prec,anom1,anom2,anom3,all_anom=detect_anom(sorted_errors, label, 1)
        #print(detect_anom(sorted_errors, sc_label, 1))
        #print(detect_anom(sorted_errors, sc_label, .75))
        print('RECONSTRUCTION:mean')
        
        print(detect_anom(sorted_errors, sc_label, 1,scores_recons[sorted_errors],thresh[sc]))
        #print('reverse')
        #print(detect_anom(rev_sorted_errors, sc_label, 1,scores_recons[rev_sorted_errors],thresh[sc]))
        print('')
        '''
        print('SKEW')
        scores_recons = l_scores[:,sc]
        l_sorted_errors = np.argsort(scores_recons)
        rev_l_sorted_errors = np.argsort(-scores_recons)
        print(detect_anom(l_sorted_errors,sc_label,1,scores_recons[l_sorted_errors],thresh[sc]))
        print('reverse')
        print(detect_anom(rev_l_sorted_errors,sc_label,1,scores_recons[rev_l_sorted_errors],thresh[sc]))
        print('\n\n')
        '''
        print('RECONSTRUCTION: by kernel density')
        #import ipdb ; ipdb.set_trace()
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(scores[:,sc].reshape(-1,1))
        #kde = kde.score(scores[:,sc].reshape(-1,1))
        #kdes.append(kde)
        #print(kde)
        #intervals = UniDip(scores_recons[rev_sorted_errors],mrg_dst=0.00001).run()
        #kdes.append(len(intervals)) 
        #import ipdb ; ipdb.set_trace() 
        scores_dens = kde.score_samples(scores[:,sc].reshape(-1,1))
        #scores_dens = scipy.stats.tstd(scores_dens)
        print('score:',kde.score(scores[:,sc].reshape(-1,1))) 
        kdes.append(scores_dens)
        
        sorted_dens = np.argsort(scores_dens)
        rev_sorted_dens = np.argsort(-scores_dens)
        import ipdb ; ipdb.set_trace()
        print(detect_anom(sorted_dens, sc_label, 1,scores_dens[sorted_dens],thresh[sc]))
        #print('reverse')
        #print(detect_anom(rev_sorted_dens, sc_label, 1,scores_dens[sorted_dens],thresh[sc]))
        #print('')
        print('---')
        print('')
        #import ipdb ; ipdb.set_trace()
        '''
        scores = np.zeros(adj.shape[0])
        for i in np.arange(adj.shape[0]):
            #lipschitz_og = calc_lipschitz(adj, A_pert, factor_pert.cuda(), label)
            lipschitz = calc_lipschitz(A_hat_scales[sc], A_hat_scales_pert[sc], factor_pert.cuda(),label)
            scores[i] = lipschitz.detach().cpu().numpy()
        '''
        '''
        #lipschitz /= lipschitz_og
        #scores = lipschitz.detach().cpu().numpy()
        import ipdb ; ipdb.set_trace()
        scores_lip = l_scores[:,sc]#.detach().cpu().numpy()
        sorted_errors = np.argsort(scores_lip)
        rankings = []
        for error in sorted_errors:
            rankings.append(label[error])
        rankings = np.array(rankings)
        
        #import ipdb ; ipdb.set_trace()
        print(f'SCALE {sc+1}')
        try:
            print('AP',detect_anom_ap(scores_lip,label))
        except:
            import ipdb ; ipdb.set_trace()
        #prec,anom1,anom2,anom3,all_anom=detect_anom(sorted_errors, label, 1)
        #print(detect_anom(sorted_errors, sc_label, 1))
        #print(detect_anom(sorted_errors, sc_label, .75))
        print('INSTABILITY')
        print(detect_anom(sorted_errors, sc_label, 1,scores_lip[sorted_errors],thresh[sc]))
        #print(detect_anom(sorted_errors, label, .25))
        print('')
        '''
        #import ipdb ; ipdb.set_trace()
    #print(kdes)
    print('avg scores anoms',torch.mean(scores[anom_idx,:],dim=0))
    print('avg scores norm',torch.mean(scores[norm_idx,:],dim=0))
    '''
    with open('output/{}-ranking_{}.txt'.format(args.dataset, sc), 'w') as f:
        for index in sorted_errors:
            f.write("%s\n" % label[index])
    
    import pandas as pd
    #df = pd.DataFrame({'AD-GCA':reconstruction_errors})
    df = pd.DataFrame({'AD-GCA':scores})
    df.to_csv('output/{}-scores_{}.csv'.format(args.dataset, sc), index=False, sep=',')
    '''
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
    parser.add_argument('--dataset', default='cora_triple_sc_all', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--scales', default='4', type=int, help='number of scales for multi-scale analysis')
    parser.add_argument('--recons', default='struct', type=str, help='reconstruct features or structure')
    parser.add_argument('--mlp', default=False, type=bool, help='include features for mlp or not')
    parser.add_argument('--parity', default=None, type=str, help='even, odd, or regular scales')
    parser.add_argument('--cutoff',default=50, type=float, help='validation cutoff')
    parser.add_argument('--d', default=1, type=int, help='d parameter for BWGNN filters')
    args = parser.parse_args()

    train_dominant(args)
