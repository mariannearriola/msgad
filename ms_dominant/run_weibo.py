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
from model import EGCN
from utils import * 
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KernelDensity
import random 


def calc_lipschitz(a, a_pert, x, x_pert, embed, embed_pert, weights, node):
    norm_weights = []
    for weight in weights:
        norm_weights.append(torch.linalg.norm(weight,2))
    A_hat_diff = torch.linalg.norm((a@x)[node]-(a_pert@x_pert)[node],2)
    embed_diff = torch.linalg.norm(embed[node]-embed_pert[node],2)
    for ind,norm_weight in enumerate(norm_weights):
        if ind == 0:
            upper_bound = norm_weight
        else:
            upper_bound *= norm_weight
        upper_bound *= norm_weight*A_hat_diff
    print(embed_diff,upper_bound)
    return embed_diff <= upper_bound

def perturb_adj(attribute_dense,adj,x_prob,adj_prob,node):
    # feature perturbation
    adj_dense = adj.clone()
    r_x = torch.distributions.binomial.Binomial(1,torch.tensor([x_prob])).sample(attribute_dense[node].shape)[:,0]
    noise = torch.mul(r_x,torch.normal(torch.full(r_x.shape,0.),torch.full(r_x.shape,1.)))
    x_pert = attribute_dense[node]+noise.cuda()
     
    # structure perturbation
    half_a = torch.triu(adj)
    node_edges = torch.nonzero(half_a[node]).flatten()
    r = torch.distributions.binomial.Binomial(1,torch.tensor([1-adj_prob])).sample(node_edges.shape)[:,0]
    broken_edges = torch.where(r==0)[0]
    
    for idx in broken_edges:
        try:
            adj_dense[node][node_edges[idx]] = 0
            adj_dense[node_edges[idx]][node] = 0
        except:
            import ipdb ; ipdb.set_trace()
    adj_norm = normalize_adj(adj_dense.detach().cpu().numpy() + sp.eye(adj_dense.shape[0]))
    return x_pert,torch.tensor(adj_norm.todense()).cuda()


def loss_func(adj, A_hat_scales, attrs, X_hat, alpha, recons, weight=None):
    attribute_cost = torch.tensor(0., dtype=torch.float32).cuda()

    all_errors= []
    all_costs, all_structure_costs = None, None
    for ind, A_hat in enumerate(A_hat_scales): 
        weight = None
        if weight is not None:
            if recons == 'struct':
                structure_reconstruction_errors = F.binary_cross_entropy(A_hat.flatten(), adj.flatten(), weight=weight)
            elif recons == 'feat':
                structure_reconstruction_errors = F.mse_loss(A_hat, attrs[0], reduction='none', weight= weight)
        else:
            if recons == 'struct':
                structure_reconstruction_errors = F.binary_cross_entropy(A_hat.flatten(),adj.flatten())
            elif recons == 'feat':
                structure_reconstruction_errors = F.mse_loss(A_hat, attrs[0], reduction='none')
                
        structure_cost = torch.mean(structure_reconstruction_errors)
        if all_costs is None:
            all_costs = structure_reconstruction_errors
            all_structure_costs = structure_cost
        else:
            all_costs = torch.add(all_costs,structure_reconstruction_errors)
            all_structure_costs = torch.add(all_structure_costs,structure_cost)
        all_errors.append(structure_cost)
        
    return all_costs, all_structure_costs, attribute_cost, all_errors

from sklearn.metrics import average_precision_score   
def detect_anom_ap(errors,label):
    return average_precision_score(label,errors)

def detect_anom(sorted_errors, label, top_nodes_perc,scores,thresh):
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
            
    print(cor_1/anom_sc1.shape[0],cor_2/anom_sc2.shape[0],cor_3/anom_sc3.shape[0])
    return true_anoms/int(all_anom.shape[0]*top_nodes_perc), cor_1, cor_2, cor_3, true_anoms
     
    anom_sc1 = label[0][0]#[0]
    anom_sc2 = label[1][0]#[0]
    anom_sc3 = label[2][0]#[0]
    #non_anom = label[3]
    #all_anom = np.concatenate((anom_sc1,np.concatenate((anom_sc2,np.concatenate((anom_sc3,non_anom),axis=None)),axis=None)),axis=None)
    all_anom = np.concatenate((anom_sc1,np.concatenate((anom_sc1,anom_sc3),axis=None)),axis=None)
    anom_scores = []
    norm_scores = []
    for ind,error in enumerate(sorted_errors):
        if error in all_anom:
            anom_scores.append(scores[ind])
        else:
            norm_scores.append(scores[ind])
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
        return top_anom
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

    # weigh positive edges for loss calculation
    weight_mask = torch.where(adj_train.flatten() == 1)[0]
    weight_tensor = torch.ones(adj_train.flatten().shape).to(device)
    pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
    weight_tensor[weight_mask] = pos_weight
    
    #weight_tensor = 0
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    #data_loader = DataLoader(torch.arange(attrs[0].shape[0]), batch_size=1681,shuffle=True) 
    score = None
    #adj=torch_geometric.utils.to_dense_adj(edge_index)[0]
    nx_graph = nx.from_numpy_matrix(adj_label.detach().cpu().numpy())
    
    best_loss = torch.tensor(float('inf')).cuda()
    model.train()
    for epoch in range(args.epoch):
        data_loader = DataLoader(torch.arange(attrs[0].shape[0]), batch_size=1681,shuffle=True) 
       
        for iter, bg in enumerate(data_loader):
            print(iter)
            
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
            optimizer.zero_grad()
            #A_hat, X_hat = model(attrs, adj)
            #A_hat_scales, X_hat = model(attrs_train,adj_train,sc_label)
            #import ipdb ; ipdb.set_trace()
            X_hat,A_hat_scales = model([batched_x],batched_adj,sc_label)
            #A_hat_scales_val, X_hat_val = model(attrs_val,adj_val,sc_label)
            
            #import ipdb ; ipdb.set_trace()
            loss, struct_loss, feat_loss,all_errors = loss_func(batched_adj, A_hat_scales, batched_x, X_hat, args.alpha, args.recons, weight_tensor)
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
        #adj=adj_label
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
        ''' 
        for node in range(batched_adj.shape[0]):
            #import ipdb ; ipdb.set_trace()
            x_pert,a_pert = perturb_adj(batched_x,batched_adj,0.3,0.3,node)
            
            #import ipdb ; ipdb.set_trace()
            a_pert = a_pert.to(device).cuda().type(torch.float32)
            x_pert_full = batched_x.clone()
            x_pert_full[node]=x_pert
            x_pert_full = x_pert_full.type(torch.float32)
            weights = [model.conv.linear.weight.data]#,model.conv.linear2.weight.data]
            #import ipdb ; ipdb.set_trace()
            #x_pert_full = attrs[0]
            #a_pert = adj
            emb_pert,_ = model([x_pert_full],a_pert,sc_label)
            #import ipdb ; ipdb.set_trace()
            bound_met=calc_lipschitz(batched_adj, a_pert, batched_x, x_pert_full, X_hat, emb_pert, weights, node)
            #import ipdb ; ipdb.set_trace()
            if not bound_met:
                print('node',node,'does not meet bound. anomaly label is ',label[bg[node]])
            else:
                print('node',node,'meets bound! label is',label[bg[node]])
        print('done')
        '''
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
                recons_errors = torch.tensor(recons_errors)
                '''
                
                print('lipschitz ranking')
                for node in batched_adj.shape[0]:
                    x_pert,a_pert = perturb_adj(batched_x,batched_adj,0.3,0.3,node)
                
                    a_pert = a_pert.to(device).cuda().type(torch.float32)
                    x_pert_full = batched_x.clone()
                    x_pert_full[node]=x_pert
                    x_pert_full = x_pert_full.type(torch.float32)
                    weights = []
                    for lin in model.conv:
                        weights.append(lin.weight.data)
                    emb_pert,_ = model([x_pert_full],a_pert,sc_label)
                    bound_met,s=calc_lipschitz(batched_adj, a_pert, batched_x, x_pert_full, X_hat, emb_pert, weights, node)
                    lip_scores[bg[ind].item()]
                recons_errors = F.binary_cross_entropy(A_hat.detach().cpu(), batched_adj.detach().cpu(), reduction="none")
                scores_ = torch.mean(recons_errors,axis=0).detach().cpu().numpy() 
                #scores_ = scipy.stats.skew(recons_errors.numpy(),axis=1)
                for ind,score in enumerate(scores_):
                    scores[bg[ind].item()][sc] = float(score)
            elif args.recons == 'feat':
                import ipdb ; ipdb.set_trace()
                recons_errors = F.mse_loss(A_hat.detach().cpu(), attrs[0].detach().cpu(), reduction='none')
            #scores=recons_errors
            #lipschitz = calc_lipschitz(A_hat_scales[sc], A_hat_scales_pert[sc], factor_pert)
            #scores = lipschitz.detach().cpu().numpy()
    import ipdb ; ipdb.set_trace() 
    for sc, A_hat in enumerate(A_hat_scales):  
        scores_recons = scores[:,sc]
        sorted_errors = np.argsort(-scores_recons)
        rev_sorted_errors = np.argsort(scores_recons)
        rankings = []
        for error in sorted_errors:
            rankings.append(label[error])
        rankings = np.array(rankings)

        print(f'SCALE {sc+1}',torch.mean(scores_recons))
        try:
            print('AP',detect_anom_ap(scores_recons,label))
        except:
            import ipdb ; ipdb.set_trace()

        print('RECONSTRUCTION: by mean')
        print(detect_anom(sorted_errors, sc_label, 1,scores_recons[sorted_errors],thresh[sc]))
        print('reverse')
        print(detect_anom(rev_sorted_errors, sc_label, 1,scores_recons[rev_sorted_errors],thresh[sc]))
        
        print('RECONSTRUCTION: by kernel density')
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(scores[:,sc].reshape(-1,1))
        scores_dens = kde.score_samples(scores[:,sc].reshape(-1,1))
        sorted_dens = np.argsort(scores_dens)
        rev_sorted_dens = np.argsort(-scores_dens)
        print(detect_anom(sorted_dens, sc_label, 1,scores_dens[sorted_dens],thresh[sc]))
        print('reverse')
        print(detect_anom(rev_sorted_dens, sc_label, 1,scores_dens[sorted_dens],thresh[sc]))
        
        print('')
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
    parser.add_argument('--dataset', default='weibo', help='dataset name: Flickr/ACM/BlogCatalog')
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
    parser.add_argument('--d', default=4, type=int, help='d parameter for BWGNN filters')
    args = parser.parse_args()

    train_dominant(args)
