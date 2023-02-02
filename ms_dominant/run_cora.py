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
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KernelDensity
import random 
from unidip import UniDip

def calc_lipschitz(a, a_pert, x, x_pert, embed, embed_pert, weights, node):
    norm_weights = [0]
    '''
    for weight in weights:
        #norm_weights.append(spectral_norm(weight))
        norm_weights.append(torch.linalg.norm(weight,2))
    '''
    #import ipdb ; ipdb.set_trace() 
    #A_hat_diff = torch.linalg.norm((a@x)[node]-(a_pert@x_pert)[node],2)
    embed_diff = torch.linalg.norm(embed[node]-embed_pert[node],2)
    #upper_bound = 0
    for ind,norm_weight in enumerate(norm_weights):
        if ind == 0:
            upper_bound = norm_weight
        else:
            upper_bound *= norm_weight
        #upper_bound *= norm_weight*A_hat_diff
    #import ipdb ; ipdb.set_trace()
    #print(embed_diff.item(),upper_bound.item())
    return embed_diff.item(),embed_diff <= upper_bound
def perturb_adj(attribute_dense,adj,x_prob,adj_prob,node):
    # feature perturbation
    adj_dense = adj.clone()
    #import ipdb ; ipdb.set_trace()
    r_x = torch.distributions.binomial.Binomial(1,torch.tensor([x_prob])).sample(attribute_dense[node].shape)[:,0]
    noise = torch.mul(r_x,torch.normal(torch.full(r_x.shape,0.),torch.full(r_x.shape,1.)))
    x_pert = attribute_dense[node]+noise.cuda()
     
    # structure perturbation
    half_a = torch.triu(adj)
    node_edges = torch.nonzero(half_a[node]).flatten()
    r = torch.distributions.binomial.Binomial(1,torch.tensor([1-adj_prob])).sample(node_edges.shape)[:,0]
    broken_edges = torch.where(r==0)[0]
    #import ipdb ; ipdb.set_trace()
    for idx in broken_edges:
        try:
            adj_dense[node][node_edges[idx]] = 0
            adj_dense[node_edges[idx]][node] = 0
        except:
            import ipdb ; ipdb.set_trace()
            print('oh no')
    adj_norm = normalize_adj(adj_dense.detach().cpu().numpy() + sp.eye(adj_dense.shape[0]))
    return x_pert,torch.tensor(adj_norm.todense()).cuda()

def loss_func(adj, A_hat_scales, attrs, X_hat, alpha, recons, weight=None):
    attribute_cost = torch.tensor(0., dtype=torch.float32).cuda()

    # structure reconstruction loss
    all_errors= []
    all_costs, all_structure_costs = None, None
    #import ipdb ; ipdb.set_trace()
    for ind, A_hat in enumerate(A_hat_scales): 
        #diff_structure = torch.pow(A_hat - adj, 2)
        #structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
        weight = None
        if weight is not None:
            if recons == 'struct':
                import ipdb ; ipdb.set_trace()
                structure_reconstruction_errors = F.binary_cross_entropy(A_hat.flatten(), adj.flatten(), weight=weight)
                #import ipdb ; ipdb.set_trace()
                #structure_reconstruction_errors = F.binary_cross_entropy(A_hat[torch.where(A_hat > 0.5)].flatten(), adj[torch.where(A_hat > 0.5)].flatten(), weight=weight)
                #structure_reconstruction_errors = F.binary_cross_entropy(A_hat, adj,reduction='none', weight = weight)
            elif recons == 'feat':
                structure_reconstruction_errors = F.mse_loss(A_hat, attrs[0], reduction='none', weight= weight)
        else:
            #structure_reconstruction_errors = F.mse_loss(A_hat.flatten(), adj.flatten(), reduction="none")
            #import ipdb ; ipdb.set_trace()
            if recons == 'struct':
                #import ipdb ; ipdb.set_trace()
                #structure_reconstruction_errors = F.binary_cross_entropy(A_hat[torch.where(A_hat > 0.5)].flatten(), adj[torch.where(A_hat > 0.5)].flatten())
                structure_reconstruction_errors = F.binary_cross_entropy(A_hat.flatten(),adj.flatten())
                #structure_reconstruction_errors = F.binary_cross_entropy(A_hat, adj,reduction='none') 
                #structure_reconstruction_errors = F.binary_cross_entropy(A_hat.flatten(), adj.flatten())
            elif recons == 'feat':
                structure_reconstruction_errors = F.mse_loss(A_hat, attrs[0], reduction='none')
        
        #structure_reconstruction_errors = F.mse_loss(A_hat, adj)
        #import ipdb ; ipdb.set_trace()
        #structure_cost = torch.mean(torch.mean(structure_reconstruction_errors,axis=1))
        structure_cost = torch.mean(structure_reconstruction_errors)
        #structure_reconstruction_errors = F.mse_loss(A_hat,adj)
        #if ind == 0:
        structure_reconstruction_errors /= (ind+1)
        structure_cost /= (ind+1)
        #if ind == 2:
        '''
        if ind == 3:
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
        all_errors.append(structure_cost)
    #cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors

    #return cost, structure_cost, attribute_cost
    return all_costs, all_structure_costs, attribute_cost,all_errors

from sklearn.metrics import average_precision_score   
def detect_anom_ap(errors,label):
    #import ipdb ;ipdb.set_trace()
    return average_precision_score(label,errors)

def detect_anom(sorted_errors, label, top_nodes_perc,scores,thresh):
    anom_sc1 = label[0][0]#[0]
    anom_sc2 = label[0][1]#[0]
    anom_sc3 = label[0][2]#[0]
    def redo(anom):
        #import ipdb ; ipdb.set_trace()
        for ind,i in enumerate(anom):
            if ind == 0:
                ret_anom = np.array(i)
            else:
                ret_anom = np.concatenate((ret_anom,np.array(i)))
        return ret_anom
    anom_sc1 = redo(anom_sc1)
    anom_sc2 = redo(anom_sc2)
    anom_sc3 = redo(anom_sc3)
    all_anom = np.concatenate((anom_sc1,np.concatenate((anom_sc2,anom_sc3),axis=None)),axis=None)
    #all_anom = self.anoms
    true_anoms = 0
    #import ipdb ; ipdb.set_trace()
    cor_1, cor_2, cor_3 = 0,0,0
    for ind,error_ in enumerate(sorted_errors[:int(all_anom.shape[0]*top_nodes_perc)]):
        '''
        if label[ind] == 1:
            true_anoms += 1
        '''
        error = error_.item()
        if error in all_anom:
            true_anoms += 1
        if error in anom_sc1:
            cor_1 += 1
        if error in anom_sc2:
            cor_2 += 1
        if error in anom_sc3:
            cor_3 += 1
        #if error in all_anom:
        #    print(ind)
    #import ipdb ; ipdb.set_trace()
    print(cor_1/anom_sc1.shape[0],cor_2/anom_sc2.shape[0],cor_3/anom_sc3.shape[0])
    return true_anoms/int(all_anom.shape[0]*top_nodes_perc), cor_1, cor_2, cor_3, true_anoms
     
    anom_sc1 = label[0][0]#[0]
    anom_sc2 = label[1][0]#[0]
    anom_sc3 = label[2][0]#[0]
    #non_anom = label[3]
    #all_anom = np.concatenate((anom_sc1,np.concatenate((anom_sc2,np.concatenate((anom_sc3,non_anom),axis=None)),axis=None)),axis=None)
    all_anom = np.concatenate((anom_sc1,np.concatenate((anom_sc2,anom_sc3),axis=None)),axis=None)
    import ipdb ; ipdb.set_trace()
    anom_scores = []
    norm_scores = []
    for ind,error in enumerate(sorted_errors):
        if error in all_anom:
            anom_scores.append(scores[ind])
        else:
            norm_scores.append(scores[ind])
    import ipdb ; ipdb.set_trace()
    anom_scores = np.array(anom_scores)
    norm_scores = np.array(norm_scores)
    def find_top_anom(scores,thresh,skew=False):
        if skew:
            z=stats.zscore(scores)
            #z=z[np.where(z<0)]
            thresh=-thresh
            top_anom=sorted_errors[np.where(z<thresh)]
        else:
            z=stats.zscore(scores)
            #z=z[np.where(z>0)]
            top_anom=sorted_errors[np.where(z>thresh)]
        #import ipdb ; ipdb.set_trace()
        return top_anom
    #import ipdb ; ipdb.set_trace()
    '''
    top_anom1=find_top_anom(scores,thresh)
    top_sc3=np.intersect1d(top_anom1,anom_sc3).shape[0]
    top_anom2=find_top_anom(scores,thresh)
    top_sc2=np.intersect1d(top_anom2,anom_sc2).shape[0]
    top_anom3=find_top_anom(scores,thresh)
    top_sc1=np.intersect1d(top_anom3,anom_sc1).shape[0]
    if top_nodes_perc == 1:
        print('top scales found from deviation')
        print(top_sc1,top_sc2,top_sc3)
        print('top anom found',len(top_anom1),len(top_anom2),len(top_anom3))
    print(np.intersect1d(top_anom3,all_anom).shape[0]/top_anom3.shape[0])
    '''
    true_anoms = 0
    cor_1, cor_2, cor_3, cor_4 = 0,0,0,0
    anom_inds1,anom_inds2,anom_inds3,anom_inds_none=[],[],[],[]
    #for ind,error in enumerate(sorted_errors[:int(all_anom.shape[0]*top_nodes_perc)]):
    for ind,error in enumerate(sorted_errors[:all_anom.shape[0]]):
        '''
        if label[ind] == 1:
            true_anoms += 1
        '''
        if error in all_anom:
            true_anoms += 1
        if error in anom_sc1:
            #print(ind,error)
            #all_inds.append(ind)
            anom_inds1.append(ind)
            cor_1 += 1
        if error in anom_sc2:
            anom_inds2.append(ind)
            cor_2 += 1
        if error in anom_sc3:
            anom_inds3.append(ind)
            cor_3 += 1
        #if error in non_anom:
        #    anom_inds_none.append(ind)
        #    cor_4 += 1
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

    return true_anoms/int(all_anom.shape[0]*top_nodes_perc), cor_1, cor_2, cor_3, true_anoms# cor_4, true_anoms
'''
def detect_anom(sorted_errors, label, top_nodes_perc):
    anom_sc1 = label[0][0]
    anom_sc2 = label[1][0]
    #anom_sc3 = label[2][0]
    anom_sc3 = []
    all_anom = np.concatenate((anom_sc1,np.concatenate((anom_sc2,anom_sc3),axis=None)),axis=None)
    anom_found = []
    true_anoms = 0
    #import ipdb ; ipdb.set_trace()
    cor_1, cor_2, cor_3 = 0,0,0
    for ind,error in enumerate(sorted_errors[:int(all_anom.shape[0]*top_nodes_perc)]):
        
        #if label[ind] == 1:
        #    true_anoms += 1
        
        if error in all_anom:
            true_anoms += 1
            anom_found.append(error)
        if error in anom_sc1:
            cor_1 += 1
        if error in anom_sc2:
            cor_2 += 1
        if error in anom_sc3:
            cor_3 += 1
        #if error in anom_sc1:
        #    print(error)
    return true_anoms/int(all_anom.shape[0]*top_nodes_perc), cor_1, cor_2, cor_3, true_anoms#,anom_found
'''
def train_dominant(args):

    adj, attrs_det, label, adj_label, sc_label, adj_train, attrs_train, adj_val, attrs_val = load_anomaly_detection_dataset(args.dataset, args.scales, args.mlp, args.parity)
    #import ipdb ; ipdb.set_trace()
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
    adj_ori = adj.clone()
    attrs_ori = attrs[0].clone()
    # weigh positive edges for loss calculation
    weight_mask = torch.where(adj_train.flatten() == 1)[0]
    weight_tensor = torch.ones(adj_train.flatten().shape).to(device)
    pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
    weight_tensor[weight_mask] = pos_weight
    
    #weight_tensor = 0
    #import ipdb ; ipdb.set_trace() 
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    #import ipdb ; ipdb.set_trace()

    score = None
    #adj=torch_geometric.utils.to_dense_adj(edge_index)[0]
    nx_graph = nx.from_numpy_matrix(adj_label.detach().cpu().numpy())
    
    best_loss = torch.tensor(float('inf')).cuda()
    model.train()
    for epoch in range(args.epoch):
        
        optimizer.zero_grad()
        #A_hat, X_hat = model(attrs, adj)
        #A_hat_scales, X_hat = model(attrs_train,adj_train,sc_label)
        #import ipdb ; ipdb.set_trace()
        X_hat,A_hat_scales = model(attrs,adj,sc_label)
        #A_hat_scales_val, X_hat_val = model(attrs_val,adj_val,sc_label)
        
        #import ipdb ; ipdb.set_trace()
        loss, struct_loss, feat_loss,all_errors = loss_func(adj, A_hat_scales, attrs[0], X_hat, args.alpha, args.recons, weight_tensor)
        #loss_val, struct_loss_val, feat_loss_val, all_errors_val = loss_func(adj_val, A_hat_scales_val, attrs_val, X_hat_val, args.alpha, args.recons, weight_tensor)
        #import ipdb ; ipdb.set_trace()
        l = torch.mean(loss)
        #val_l = torch.mean(loss_val)
        #if val_l < args.cutoff:
        #    epoch = args.epoch-1
        '''   
        if l < best_loss:
            best_loss = l
            torch.save(model,'best_model.pt')
        '''
        l.backward()
        optimizer.step()

        num_nonzeros=[]
        for sc, A_hat in enumerate(A_hat_scales):
            num_nonzeros.append(torch.where(A_hat < 0.5)[0].shape[0])
        #import ipdb ; ipdb.set_trace()
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()), "validation/struct_loss=", "{:.5f}".format(l.item()), "Non-edges:",num_nonzeros)
        '''
        #if epoch%10 == 0 or epoch == args.epoch - 1:
        if epoch == args.epoch-1:
            #import ipdb ; ipdb.set_trace()
            model.eval()
            #A_hat, X_hat = modl(attrs, adj)
            A_hat_scales, X_hat = model(attrs_train,adj_train,sc_label)
            loss, struct_loss, feat_loss,all_errors = loss_func(adj_train, A_hat_scales, attrs_train, X_hat, args.alpha, args.recons, weight_tensor)
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

    #model = torch.load('best_model.pt')
    
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
    X_hat,A_hat_scales = model([attrs_ori],adj_ori,sc_label)
    #weights = [model.conv.linear.weight.data]#,model.conv.linear2.weight.data]
    #weights = [model.conv.attention_layer.state_dict()['lin_l.weight'],model.conv.attention_layer.state_dict()['lin_r.weight']]
    '''
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
    
    loss, struct_loss, feat_loss,all_errors = loss_func(adj, A_hat_scales, attrs[0], X_hat, args.alpha, args.recons)

    for sc, A_hat in enumerate(A_hat_scales):
        
        if args.recons == 'struct':
            ''' 
            try:
                recons_errors = []
                for node_ind,node in enumerate(A_hat):
                    #recons_error = scipy.stats.skew(F.binary_cross_entropy(node[torch.where(node>0.5)],adj[node_ind][torch.where(node>0.5)],reduction='none').detach().cpu())
                    recons_error = scipy.stats.skew(F.binary_cross_entropy(node[torch.where(adj[node_ind]>0.5)],adj[node_ind][torch.where(adj[node_ind]>0.5)],reduction='none').detach().cpu())
                    
                    recons_errors.append(recons_error)
            except:
                import ipdb ; ipdb.set_trace()
                print('hi')
            recons_errors = torch.tensor(recons_errors)
            '''
            recons_errors = F.binary_cross_entropy(A_hat.detach().cpu(), adj.detach().cpu(), reduction="none")
            scores_ = torch.mean(recons_errors,axis=0).detach().cpu().numpy() 
            scores_s = scipy.stats.skew(recons_errors.numpy(),axis=1)
            for ind,score in enumerate(scores_):
                scores[ind][sc] = float(score)
            for ind,score in enumerate(scores_s):
                l_scores[ind][sc] = score
            #recons_errors = F.binary_cross_entropy(A_hat.detach().cpu(), adj.detach().cpu(),reduction="none")
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
        elif args.recons == 'feat':
            import ipdb ; ipdb.set_trace()
            recons_errors = F.mse_loss(A_hat.detach().cpu(), attrs[0].detach().cpu(), reduction='none')
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
        print('reverse')
        print(detect_anom(rev_sorted_errors, sc_label, 1,scores_recons[rev_sorted_errors],thresh[sc]))
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
        #print('RECONSTRUCTION: by kernel density')
        #import ipdb ; ipdb.set_trace()
        #kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(scores[:,sc].reshape(-1,1))
        #kde = kde.score(scores[:,sc].reshape(-1,1))
        #kdes.append(kde)
        #print(kde)
        intervals = UniDip(scores_recons[rev_sorted_errors],mrg_dst=0.00001).run()
        kdes.append(len(intervals)) 
        '''
        scores_dens = kde.score_samples(scores[:,sc].reshape(-1,1))
        scores_dens = scipy.stats.tstd(scores_dens)
        kdes.append(scores_dens)
        '''
        #sorted_dens = np.argsort(scores_dens)
        #rev_sorted_dens = np.argsort(-scores_dens)
        #print(detect_anom(sorted_dens, sc_label, 1,scores_dens[sorted_dens],thresh[sc]))
        #print('reverse')
        #print(detect_anom(rev_sorted_dens, sc_label, 1,scores_dens[sorted_dens],thresh[sc]))
        
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
    print(kdes)
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
