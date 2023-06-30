import torch
import numpy as np
import dgl
from utils import *
import random


def generate_unique_numbers(n, start_range, end_range):
    if n > (end_range - start_range + 1):
        raise ValueError("The range is not large enough to generate n unique numbers.")
    
    numbers = np.arange(start_range, end_range + 1)  # Create an array with numbers within the specified range
    np.random.shuffle(numbers)  # Shuffle the array randomly
    
    return numbers[:n]  # Return the first n elements of the shuffled array

def loss_func(graph, feat, A_hat, X_hat, res_a, edge_ids, attract_sel,repel_sel, sample=False, recons='struct', alpha=None, clusts=None, regularize=True):
    """
    Calculate reconstruction error given a graph reconstruction and its corresponding reconstruction
    label.

    Parameters
    ----------
        adj: array-like, shape=[n, n]
            normalized adjacency matrix
        feat: array-like, shape=[n, k]
            feature matrix
        pos_edges: array=like, shape=[n, 2?]
            positive edge list
        neg_edges: array=like, shape=[n, 2?]
            negative edge list
        sample: bool
            if true, use sampled edges/non-edges for loss calculation. if false, use all edges/non-edges
        recons: str
            structure reconstruction, feature reconstruction, or structure & feature reconstruction
        alpha: float
            if structure & feature reconstruction, scalar to weigh importance of structure vs feature reconstruction

    Returns
    ----------
        all_costs : array-like, shape=[]
            total loss for backpropagation
        all_struct_error : array-like, shape=[]
            node-wise errors at each scale
    """
    if not alpha: alpha = 1 if recons=='struct' else 0

    # node sampling: full reconstruction error
    all_costs, all_struct_error, all_feat_error = None, None, None
    scale_weights=[1,1,1]
    total_struct_error,total_feat_error=torch.tensor(-1.),torch.tensor(-1.)
    
    # DEBUG
    #loss=torch.nn.functional.binary_cross_entropy_with_logits(A_hat[0], graph[0].adjacency_matrix().to_dense().to(A_hat[0].device), reduction='none')

    
    #loss=torch.nn.functional.mse_loss(A_hat[0], graph[0].adjacency_matrix().to_dense().to(A_hat[0].device), reduction='none')
    #loss=torch.nn.functional.mse_loss(A_hat[0], graph[0].ndata['feature'], reduction='none')
    #loss = torch.abs(A_hat[0] - graph[0].ndata['feature'])
    #loss = A_hat[0] - graph[0].adjacency_matrix().to_dense().to(A_hat[0].device)
    print(A_hat.min(),A_hat.max())
    #if A_hat.max() > 0.9:
    #    print('checking')
    #import ipdb ; ipdb.set_trace()
    #return loss.sum(),loss.unsqueeze(0),loss.unsqueeze(0)

    clusts = torch.tensor(clusts)
    all_clust = clusts.unique().sort()
    margin = 1.
    #import ipdb ; ipdb.set_trace()

    edge_ids_tot = []
    for recons_ind,preds in enumerate([A_hat, X_hat]):
        if preds == None: continue
        for ind, sc_pred in enumerate(preds):
            # structure loss
            if recons_ind == 0 and recons in ['struct','both']:
                if sample:
                    sampled_pred = sc_pred
                    lbl_edges = torch.zeros(sampled_pred.shape).to(sampled_pred.device).to(torch.float64)
                    check_gpu_usage('before edge idx')
                    edge_idx=graph[ind].has_edges_between(edge_ids[ind,0,:],edge_ids[ind,1,:])
                    edge_idx = edge_idx.nonzero()
                    edge_idx = edge_idx.T[0]
                    
                    # bug?
                    lbl_edges[edge_idx] = graph[ind].edata['w'][graph[ind].edge_ids(edge_ids[ind,0,:][edge_idx],edge_ids[ind,1,:][edge_idx])].to(torch.float64)
                    total_struct_error, edge_struct_errors,regloss,clustloss,nonclustloss,sc_idx_inside,sc_idx_outside,entropies = get_sampled_losses(sampled_pred,edge_ids[ind],lbl_edges,ind,attract_sel[ind],repel_sel[ind],clusts[ind],regularize)

                    del sampled_pred, lbl_edges, edge_idx
                    torch.cuda.empty_cache()
                    check_gpu_usage('after edge idx')
                else:
                    # collect loss for all edges/non-edges in reconstruction
                    if type(graph) != list:
                        if not pos_edges == None:
                            adj_label=graph.adjacency_matrix().to_dense()
                        else:
                            if type(graph) == torch.Tensor:
                                adj_label = graph
                                num_nodes = adj_label.shape[0]
                            else:
                                adj_label = graph.adjacency_matrix().to_dense().to(graph.device)
                                num_nodes = graph.num_dst_nodes()
                    else:
                        adj_label = graph[ind].to(pos_edges.device)
                    edge_struct_errors = torch.sqrt(torch.sum(torch.pow(sc_pred - adj_label, 2),1))
                    total_struct_error = torch.mean(edge_struct_errors)
            

            # feature loss
            if recons_ind == 1 and recons in ['feat','both']:
                #import ipdb ; ipdb.set_trace()
                feat_error = torch.nn.functional.mse_loss(sc_pred,feat.to(feat.device),reduction='none')
                total_feat_error = torch.mean(feat_error,dim=1)
            
            if total_struct_error <= -1:
                print('negative loss')
                import ipdb ; ipdb.set_trace()

            # accumulate errors
            if all_costs is None:
                if recons_ind == 0 and total_struct_error > -1:
                    all_struct_error = (edge_struct_errors).unsqueeze(0)
                    all_reg_error = (regloss).unsqueeze(0)
                    all_clust_error = (clustloss).unsqueeze(0)
                    all_nonclust_error = (nonclustloss).unsqueeze(0)
                    all_sc_idx_inside = [sc_idx_inside]
                    all_sc_idx_outside = [sc_idx_outside]
                    all_costs = total_struct_error.unsqueeze(0)*alpha
                    all_entropies = [entropies]
                if recons_ind == 1 and total_feat_error > -1:
                    all_feat_error = (feat_error).unsqueeze(0)
                    all_costs = (torch.mean(all_feat_error))*(1-alpha)
            else:
                if recons_ind == 0 and total_struct_error > -1:
                    all_struct_error = torch.cat((all_struct_error,(edge_struct_errors).unsqueeze(0)))
                    all_reg_error = torch.cat((all_reg_error,(regloss).unsqueeze(0)))
                    all_clust_error = torch.cat((all_clust_error,(clustloss).unsqueeze(0)))
                    all_nonclust_error = torch.cat((all_nonclust_error,(nonclustloss).unsqueeze(0)))
                    all_sc_idx_inside.append(sc_idx_inside)
                    all_sc_idx_outside.append(sc_idx_outside)
                    all_entropies.append(entropies)
                elif recons_ind == 1 and total_feat_error > -1:
                    if all_feat_error is None:
                        all_feat_error = (feat_error).unsqueeze(0)
                    else:
                        all_feat_error = torch.cat((all_feat_error,(feat_error).unsqueeze(0)))
                else:
                    continue

                # assuming both is NOT multi-scale
                if recons != 'both':
                    all_costs = torch.cat((all_costs,torch.add(total_struct_error*alpha*scale_weights[recons_ind],torch.mean(total_feat_error)*(1-alpha)).unsqueeze(0)))
                else:
                    all_costs = torch.add(total_struct_error*alpha,torch.mean(total_feat_error)*(1-alpha)).unsqueeze(0)
 
    #import ipdb ; ipdb.set_trace()
    del feat, total_feat_error, total_struct_error ; torch.cuda.empty_cache()
    return all_costs, all_struct_error, all_feat_error, all_reg_error, all_clust_error, all_nonclust_error, all_sc_idx_inside, all_sc_idx_outside, all_entropies#, edge_ids_tot

def get_sampled_losses(pred,edges,label,ind,attract_sel,repel_sel,clusts=None,regularize=True):
    """description

    Parameters
    ----------
        pred: array-like, shape=[]
            normalized adjacency matrix
        edges: array-like, shape=[]
            feature matrix
        label: array=like, shape=[]
            positive edge list
    
    Returns
    ----------
        total_error : array-like, shape=[]
            total loss for backpropagation
        edge_errors : array-like, shape=[]
            edge-wise errors
    """
    edge_errors=torch.nn.functional.mse_loss(pred.to(torch.float64), label, reduction='none')
    
    regloss=torch.zeros(edge_errors.shape).to(edge_errors.device)
    reg_clust = torch.zeros(edge_errors.shape).to(edge_errors.device)
    reg_nonclust = torch.zeros(edge_errors.shape).to(edge_errors.device)
    reg_nonclust_sample = torch.zeros(edge_errors.shape).to(edge_errors.device)

    #in_reg_weights = [0,1,]
    #in_reg_weights = np.ones(4)
    #out_reg_weights = [1,1,0]
    #out_reg_weights = np.ones(4)
    #import ipdb ; ipdb.set_trace()
    if regularize is True:      
        #import ipdb ; ipdb.set_trace()
        epsilon = 0.
        attract_edges = torch.where(clusts[edges][0]==clusts[edges[1]])[0]
        repel_edges = torch.where(clusts[edges][0]!=clusts[edges[1]])[0]
        repel_edges = repel_edges[torch.tensor(np.random.randint(0,repel_edges.shape[0],attract_edges.shape))]
        alpha = 0.8
        sc_idx = attract_edges
        clust_loss =  (1-pred[sc_idx]+epsilon)

        edge_errors[sc_idx]= edge_errors[sc_idx] + alpha * clust_loss
        reg_clust[sc_idx] = alpha * clust_loss
        nonclust=repel_edges
        nonclust_loss = pred[nonclust]
        reg_nonclust[nonclust] = alpha * nonclust_loss
        #reg_nonclust_sample[nonclust[sample_non]] = alpha * pred[sample_non]

        #edge_errors[sample_non] = edge_errors[sample_non] + alpha * pred[sample_non]
        edge_errors[nonclust] = edge_errors[nonclust] + alpha * nonclust_loss

        regloss = reg_clust + reg_nonclust
        print((alpha*clust_loss)[(alpha*clust_loss).nonzero()[:,0]].mean(),(alpha*pred[nonclust])[(alpha*pred[nonclust]).nonzero()[:,0]].mean())
        print(f'correct edges for clust {ind}',torch.where(pred[sc_idx]>=0.5)[0].shape[0]/pred[sc_idx].shape[0])
        print(f'correct non-edges {ind}',torch.where(pred[nonclust]<0.5)[0].shape[0]/pred[nonclust].shape[0])
    print('correct edges',torch.where(pred[label.nonzero()][:,0]>=0.5)[0].shape[0]/pred[label.nonzero()][:,0].shape[0])
    print('correct non-edges',torch.where(pred[~label.nonzero()][:,0]<0.5)[0].shape[0]/pred[~label.nonzero()][:,0].shape[0])
            
    total_error = torch.sum(edge_errors)
    #import ipdb ; ipdb.set_trace()
    recons_e = pred

    # get node-wise + cluster-wise entropy of loss?
    entropies = {}
    for clust in clusts.unique():
        idx = attract_edges[torch.where(clusts[edges][0][attract_edges]==clust)]
        clust_recons = recons_e[idx]
        entropies[clust.item()]=scipy.stats.entropy(clust_recons.detach().cpu())
        if np.isnan(entropies[clust.item()]):
            entropies[clust.item()] = 0.
        del clust_recons,idx ; torch.cuda.empty_cache()
    
    #total_error = torch.sum(edge_errors)
    if torch.isnan(total_error):
        print('nan')
        import ipdb ; ipdb.set_trace()
    #import ipdb ; ipdb.set_trace()
        
    return total_error, edge_errors, regloss, reg_clust, reg_nonclust,sc_idx,nonclust,entropies

def average_with_indices(input_tensor, indices_tensor):
    input_tensor,indices_tensor=input_tensor,indices_tensor
    average_tensor = torch.scatter_reduce(input_tensor, 0, indices_tensor, reduce="sum")
    expanded_average = average_tensor[indices_tensor]
    
    return expanded_average