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
from model_tf import EGCN
from utils_tf import *
from scipy.spatial.distance import euclidean
import random 

def calc_lipschitz(A_hat_scales, A_hat_pert, factor_pert, anom_label):
    #structure_reconstruction_errors = F.binary_cross_entropy(A_hat_scales, A_hat_pert, reduction='none')
    A_hat_diff = abs(A_hat_scales-A_hat_pert)
    A_hat_diff_pool = torch.norm(A_hat_diff)#,dim=1)
    A_hat_diff_p = A_hat_diff_pool
    #A_hat_diff_pool = torch.mean(A_hat_diff,axis=1)
    #import ipdb ; ipdb.set_trace()
    #A_hat_diff=(A_hat_scales@A_hat_pert.T)/(torch.norm(A_hat_scales)*torch.norm(A_hat_pert))
    #A_hat_diff_p = torch.norm(A_hat_diff)
    '''
    if torch.max(A_hat_diff_p.any() > 1 or A_hat_diff_p.any() < 0:
        import ipdb ; ipdb.set_trace()
    lipschitz = 1-A_hat_diff_p#*factor_pert
    '''
    lipschitz = A_hat_diff_p
    return lipschitz

def perturb_adj(attribute_dense,adj,attr_scale,adj_prob):
    adj_dense = adj.clone() 
    dists_adj = torch.zeros(attribute_dense.shape[0])
    #import ipdb ; ipdb.set_trace()
    for i,_ in enumerate(adj_dense):
        for j,_ in enumerate(adj_dense):
            if j > i:
                break
            if j == i:
                continue
            # flip if prob met
            try:
                adj_dense[i,j]=0
                adj_dense[j,i]=0
                if np.random.rand() < adj_prob:
                    adj_dense[i,j] = 1.
                    adj_dense[j,i] = 1.
                    '''
                    if adj_dense[i,j] == 0:
                        adj_dense[i, j] = 1.
                        adj_dense[j, i] = 1. 
                    if adj_dense[i,j] == 1:
                        adj_dense[i, j] = 0.
                        adj_dense[j, i] = 0. 
                    '''
                    dists_adj[i] += 1
                    dists_adj[j] += 1
            except:
                import ipdb ; ipdb.set_trace()
                print('hi')
    #num_add_edge = np.sum(adj_dense) - ori_num_edge
    #import ipdb ;ipdb.set_trace() 
    # Disturb attribute
        # Every node in each clique is anomalous; no attribute anomaly scale (can be added)
    print('Constructing attributed anomaly nodes...')
    #for ind,i_ in enumerate(attribute_anomaly_idx):
    all_anom_sc = []
    '''
    dists = torch.zeros(attribute_dense.shape[0])
    for cur,_ in enumerate(attribute_dense):
        #picked_list = random.sample(all_idx, k)
        max_dist = 0
        # find attribute with greatest euclidian distance
        for j_,_ in enumerate(attribute_dense):
            #cur_dist = euclidean(attribute_dense[i_],attribute_dense[j_])
            #import ipdb ; ipdb.set_trace()
            cur_dist = euclidean(attribute_dense[cur].cpu(),attribute_dense[j_].cpu())
            if cur_dist > max_dist:
                max_dist = cur_dist
    
        for j_,_ in enumerate(attribute_dense):
            cur_dist = euclidean(attribute_dense[cur].cpu(),attribute_dense[j_].cpu())
            if cur_dist > max_dist/10:
                closest = cur_dist
                dists[cur] = cur_dist
                closest_idx = j_
                #max_idx = j_
        
        attribute_dense[cur] = attribute_dense[closest_idx]
    '''
    #all_anom_sc.append(anom_sc)
    dists_adj = dists_adj/torch.full((dists_adj.shape[0],),torch.max(dists_adj))
    #dists = dists/torch.full((dists.shape[0],),torch.max(dists))
    #factor_pert = dists_adj + dists
    factor_pert = dists_adj
    return attribute_dense,adj_dense,factor_pert

def loss_func(adj, A_hat_scales, pos_edges_, neg_edges_backprop, alpha, recons, weight=None):
    attribute_cost = torch.tensor(0., dtype=torch.float32).cuda()
    all_errors= []
    all_costs, all_structure_costs = None, None
    for ind, A_hat in enumerate(A_hat_scales): 
        for row_ind,row in enumerate(adj.adjacency_matrix().to_dense().cuda()): 
            edge_count = torch.where(row==1)[0].shape[0]
            non_edges = row[torch.where(row==0)]
            non_edges_score = torch.randint(row.shape[0],(edge_count,))
            all_edges_score = torch.cat((torch.where(row==1)[0],non_edges_score.cuda()))

            recons_error=torch.nn.functional.binary_cross_entropy(A_hat[row_ind][all_edges_score],row[all_edges_score].cuda())
            if row_ind == 0:
                structure_reconstruction_errors=recons_error.unsqueeze(-1)
            else:
                structure_reconstruction_errors = torch.cat((structure_reconstruction_errors,recons_error.unsqueeze(-1)))
        if all_costs is None:
            all_costs = structure_reconstruction_errors
            all_structure_costs = (structure_reconstruction_errors).unsqueeze(-1)
        else:
            all_costs = torch.add(all_costs,structure_reconstruction_errors)
            all_structure_costs = torch.cat((all_structure_costs,(structure_reconstruction_errors).unsqueeze(-1)),dim=-1)
    return all_costs, all_structure_costs, attribute_cost,all_errors

from sklearn.metrics import average_precision_score   
def detect_anom_ap(errors,label):
    #import ipdb ;ipdb.set_trace()
    return average_precision_score(label,errors)

def detect_anom(sorted_errors, label, top_nodes_perc,scores,thresh):
    import ipdb ; ipdb.set_trace()
    anom_sc1 = label[0][0]
    anom_sc2 = label[1][0]
    anom_sc3 = label[2][0]
    def redo(anom):
        for ind,i in enumerate(anom):
            if ind == 0:
                ret_anom = i[0]
            else:
                ret_anom = np.concatenate((ret_anom,i[0]))
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
    adj, adj_label = sparse_matrix_to_tensor(adj,attrs_det[0]), sparse_matrix_to_tensor(adj_label,attrs_det[0])
    model = EGCN(in_size = attrs_det[0].size(1), out_size = args.hidden_dim, scales = args.scales, recons = args.recons, mlp = args.mlp, d = args.d)
    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        model = model.cuda()

    '''
    # weigh positive edges for loss calculation
    weight_mask = torch.where(adj_train.flatten() == 1)[0]
    weight_tensor = torch.ones(adj_train.flatten().shape).to(device)
    pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
    weight_tensor[weight_mask] = pos_weight
    '''
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    data_loader = DataLoader(torch.arange(adj.num_edges()), batch_size=args.batch_size,shuffle=True) 
    score = None
    
    best_loss = torch.tensor(float('inf')).cuda()
    model.train()
    for epoch in range(args.epoch):
        for iter, bg in enumerate(data_loader):
            pos_edges = adj.find_edges(bg.cuda())
            pos_edge_weights = adj.edata['w'][bg.cuda()]

            sel_nodes = torch.unique(torch.cat((pos_edges[0],pos_edges[1])))
            neg_edges = None
            for idx, a in enumerate(sel_nodes):
                for b in sel_nodes[idx+1:]:
                    idx_a1, idx_a2 = torch.where(a==pos_edges[0])[0], torch.where(a==pos_edges[1])[0]
                    idx_b1, idx_b2 = torch.where(b==pos_edges[0])[0], torch.where(b==pos_edges[1])[0]
                    if find_intersection(idx_a1,idx_b2).shape[0] > 0 or find_intersection(idx_a2,idx_b1).shape[0] > 0:
                        continue
                    if neg_edges is None:
                        neg_edges = torch.cat((a.expand(1,1),b.expand(1,1)))
                    else:
                        neg_edges = torch.cat((neg_edges,torch.cat((a.expand(1,1),b.expand(1,1)))),dim=1)
            neg_edges_backprop = neg_edges[:,:args.batch_size].T
            pos_edges_= torch.cat((pos_edges[0].unsqueeze(-1),pos_edges[1].unsqueeze(-1)),dim=1)
            batch_feats = adj.ndata['feature'][sel_nodes]
            batch_dict_for,batch_dict_rev = {k.item():v.item() for k,v in zip(sel_nodes,torch.arange(sel_nodes.shape[0]))},{k:v for k,v in zip(torch.arange(sel_nodes.shape[0]),sel_nodes)}
            for ind,i in enumerate(pos_edges_):
                pos_edges_[ind][0],pos_edges_[ind][1] = batch_dict_for[i[0].item()],batch_dict_for[i[1].item()]
            for ind,i in enumerate(neg_edges_backprop):
                neg_edges_backprop[ind][0],neg_edges_backprop[ind][1] = batch_dict_for[i[0].item()],batch_dict_for[i[1].item()]
            batch_adj = dgl.graph((pos_edges_[:,0],pos_edges_[:,1]))
            batch_adj.ndata['feature'] = batch_feats
            batch_adj.edata['w'] = pos_edge_weights
            A_hat_scales, X_hat = model(batch_adj,sc_label)
            loss, struct_loss, feat_loss,all_errors = loss_func(batch_adj, A_hat_scales, pos_edges_,neg_edges_backprop, args.alpha, args.recons)
            '''
            #self.linear.train()
            batched_x, batched_edges = attrs[0][bg].cuda(),list(nx_graph.edges(bg.numpy()))
            #import ipdb ; ipdb.set_trace()
            batched_edges = torch.Tensor([[torch.where(bg==a)[0][0].item() for a,b in batched_edges if a in bg and b in bg],[torch.where(bg==b)[0][0].item() for a,b in batched_edges if b in bg and a in bg]]).long()
            for bg_ind,b in enumerate(bg):
                #import ipdb ; ipdb.set_trace()
                if bg_ind not in batched_edges:
                    #import ipdb ; ipdb.set_trace()
                    batched_edges=torch.cat((batched_edges.t(),torch.Tensor([[torch.where(bg==b)[0][0].item(),torch.where(bg==b)[0][0].item()]]))).t()
                    #print('found')
                
                if bg_ind in batched_edges[1][torch.where(batched_edges[0]==bg_ind)[0]] or bg_ind in batched_edges[0][torch.where(batched_edges[1]==bg_ind)[0]]:
                    #print('found')
                    continue
                #batched_edges=torch.cat((batched_edges.t(),torch.Tensor([[torch.where(bg==b)[0][0].item(),torch.where(bg==b)[0][0].item()]]))).t()
                
            #batched_edges=batched_adj.nonzero().t().contiguous()
            def normalize_adj(adj_i):
                adj_in = sp.coo_matrix(adj_i)
                rowsum = np.array(adj_in.sum(1))
                d_inv_sqrt = np.power(rowsum, -0.5).flatten()
                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
                #import ipdb ; ipdb.set_trace()
                return adj_in.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
            
            batched_adj=torch_geometric.utils.to_dense_adj(batched_edges.type(torch.int64))[0].cuda()
            batched_adj = normalize_adj(batched_adj.detach().cpu().numpy()).todense()
            batched_adj = torch.Tensor(batched_adj+np.eye(batched_adj.shape[0])).to(device)
            if 2 in batched_adj:
                #print('2')
                #import ipdb ; ipdb.set_trace()
                for i in torch.where(batched_adj==2)[0]:
                    batched_adj[i][i]=1
                #import ipdb ; ipdb.set_trace()
            '''
            optimizer.zero_grad()
            
            #import ipdb ; ipdb.set_trace()
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
    
    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        adj=adj_label
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
    #import ipdb ; ipdb.set_trace()
    #lipschitz = calc_lipschitz(A_hat_scales, A_hat_scales_pert, factor_pert)
    #scores = lipschitz.detach().cpu().numpy()
    #scores = loss.detach().cpu().numpy()
    thresh=[1.3,2.1,2.1,2.1,2.1,2.1,2.1,2.1]
    #import ipdb ; ipdb.set_trace()
    # anomaly evaluation
    data_loader = DataLoader(torch.arange(adj.shape[0]), batch_size=1681,shuffle=True) 
    score = None
    #adj=torch_geometric.utils.to_dense_adj(edge_index)[0]
    nx_graph = nx.from_numpy_matrix(adj_label.detach().cpu().numpy())
     
    #X_pert, A_pert, factor_pert = perturb_adj(attrs[0],adj,3,0.5)
    #A_hat_scales, X_hat = model(attrs,adj,sc_label)

    scores, l_scores = torch.zeros(adj.shape[0],args.d+1),torch.zeros((adj.shape[0],args.d+1))
    for iter, bg in enumerate(data_loader):
        print(iter)
        #self.linear.train()
        batched_x, batched_edges = attrs[0][bg].cuda(),list(nx_graph.edges(bg.numpy()))
        batched_edges = torch.Tensor([[torch.where(bg==a)[0][0].item() for a,b in batched_edges if a in bg and b in bg],[torch.where(bg==b)[0][0].item() for a,b in batched_edges if b in bg and a in bg]]).long()
        
        for bg_ind,b in enumerate(bg):
            #import ipdb ; ipdb.set_trace()
            if bg_ind not in batched_edges:
                #import ipdb ; ipdb.set_trace()
                batched_edges=torch.cat((batched_edges.t(),torch.Tensor([[torch.where(bg==b)[0][0].item(),torch.where(bg==b)[0][0].item()]]))).t()
                #print('found')
            
            if bg_ind in batched_edges[1][torch.where(batched_edges[0]==bg_ind)[0]] or bg_ind in batched_edges[0][torch.where(batched_edges[1]==bg_ind)[0]]:
                #print('found')
                continue
            #batched_edges=torch.cat((batched_edges.t(),torch.Tensor([[torch.where(bg==b)[0][0].item(),torch.where(bg==b)[0][0].item()]]))).t()
            
        #batched_edges=batched_adj.nonzero().t().contiguous()
        def normalize_adj(adj_i):
            adj_in = sp.coo_matrix(adj_i)
            rowsum = np.array(adj_in.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            #import ipdb ; ipdb.set_trace()
            return adj_in.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        
        batched_adj=torch_geometric.utils.to_dense_adj(batched_edges.type(torch.int64))[0].cuda()
        batched_adj = normalize_adj(batched_adj.detach().cpu().numpy()).todense()
        batched_adj = torch.Tensor(batched_adj+np.eye(batched_adj.shape[0])).to(device)
        if 2 in batched_adj:
            #print('2')
            #import ipdb ; ipdb.set_trace()
            for i in torch.where(batched_adj==2)[0]:
                batched_adj[i][i]=1
            #import ipdb ; ipdb.set_trace() 
        
        # !!! EMBEDDINGS RETRIEVED !!!
        X_hat,A_hat_scales = model([batched_x],batched_adj,sc_label)
        #import ipdb ; ipdb.set_trace()
        loss, struct_loss, feat_loss,all_errors = loss_func(batched_adj, A_hat_scales, batched_x, X_hat, args.alpha, args.recons)
    
        '''
        for i in np.arange(batched_adj.shape[0]):
            adj_pert = batched_adj.clone()
            attr_pert = batched_x.clone().detach().cpu().numpy()
            #lipschitz_og = calc_lipschitz(adj, A_pert, factor_pert.cuda(), label)
            attr_pert[i] += (np.random.binomial(1,.5,1)*np.random.normal(0,0.1,1))
            
            edges = adj[i]
            for j in edges:
                if j == 0: continue
                if np.random.binomial(1,.5,1) == 1:
                    try:
                        adj_pert[i][int(j.item())] = 0
                        adj_pert[int(j.item())][i]= 0
                    except:
                        import ipdb ; ipdb.set_trace()
                        print('hi')
            #import ipdb ; ipdb.set_trace()
            A_hat_scales_pert, X_hat_pert = model([torch.Tensor(attr_pert).cuda()], adj_pert, sc_label)
            for sc_ind,sc in enumerate(A_hat_scales):
                lipschitz = calc_lipschitz(X_hat[sc_ind],X_hat_pert[sc_ind],1,label).unsqueeze(0).detach().cpu()
                #lipschitz = calc_lipschitz(A_hat_scales[sc_ind], A_hat_scales_pert[sc_ind], 1,label).unsqueeze(0).detach().cpu()
                if sc_ind == 0:
                    scores_l = lipschitz
                else:
                    scores_l = torch.cat((scores_l,lipschitz),dim=0)
            #import ipdb ; ipdb.set_trace()
            #scores = scores.unsqueeze(0)
            for j in bg:
                l_scores[j] = scores_l
            #l_scores = l_scores.detach().cpu().numpy()
            del adj_pert
            del attr_pert
            torch.cuda.empty_cache()

        '''
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
                recons_errors = F.binary_cross_entropy(A_hat.detach().cpu(), batched_adj.detach().cpu(), reduction="none")
                scores_ = torch.mean(recons_errors,axis=0).detach().cpu().numpy() 
                #scores_ = scipy.stats.skew(recons_errors.numpy(),axis=1)
                for ind,score in enumerate(scores_):
                    scores[bg[ind].item()][sc] = float(score)
                #recons_errors = F.binary_cross_entropy(A_hat.detach().cpu(), adj.detach().cpu(),reduction="none")
            elif args.recons == 'feat':
                import ipdb ; ipdb.set_trace()
                recons_errors = F.mse_loss(A_hat.detach().cpu(), attrs[0].detach().cpu(), reduction='none')
            #recons_errors = F.mse_loss(A_hat.detach().cpu(), adj.detach().cpu(),reduction="none")
            #scores=recons_errors
            #lipschitz = calc_lipschitz(A_hat_scales[sc], A_hat_scales_pert[sc], factor_pert)
            #scores = lipschitz.detach().cpu().numpy()
    import ipdb ; ipdb.set_trace() 
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
        print(f'SCALE {sc+1}',torch.mean(scores_recons))
        try:
            print('AP',detect_anom_ap(scores_recons,label))
        except:
            import ipdb ; ipdb.set_trace()
        #prec,anom1,anom2,anom3,all_anom=detect_anom(sorted_errors, label, 1)
        #print(detect_anom(sorted_errors, sc_label, 1))
        #print(detect_anom(sorted_errors, sc_label, .75))
        print('RECONSTRUCTION')
        
        print(detect_anom(sorted_errors, sc_label, 1,scores_recons[sorted_errors],thresh[sc]))
        print('reverse')
        print(detect_anom(rev_sorted_errors, sc_label, 1,scores_recons[rev_sorted_errors],thresh[sc]))
        #print(detect_anom(sorted_errors, label, .25))
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
        import ipdb ; ipdb.set_trace()
        
    
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
    parser.add_argument('--dataset', default='tfinance', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--scales', default='4', type=int, help='number of scales for multi-scale analysis')
    parser.add_argument('--batch_size', type=int, default=32, help='number of edges to use for batching (default: 32)')
    parser.add_argument('--recons', default='struct', type=str, help='reconstruct features or structure')
    parser.add_argument('--mlp', default=False, type=bool, help='include features for mlp or not')
    parser.add_argument('--parity', default=None, type=str, help='even, odd, or regular scales')
    parser.add_argument('--cutoff',default=50, type=float, help='validation cutoff')
    parser.add_argument('--d', default=4, type=int, help='d parameter for BWGNN filters')
    args = parser.parse_args()

    train_dominant(args)
