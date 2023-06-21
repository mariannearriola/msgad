import torch
import numpy as np
import dgl
from utils import *

def generate_unique_numbers(n, start_range, end_range):
    if n > (end_range - start_range + 1):
        raise ValueError("The range is not large enough to generate n unique numbers.")
    
    numbers = np.arange(start_range, end_range + 1)  # Create an array with numbers within the specified range
    np.random.shuffle(numbers)  # Shuffle the array randomly
    
    return numbers[:n]  # Return the first n elements of the shuffled array

def loss_func(graph, feat, A_hat, X_hat, res_a, edge_ids, sample=False, recons='struct', alpha=None, clusts=None):
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
                    '''
                    if type(graph) == list:
                        import ipdb ; ipdb.set_trace()
                        edge_labels = graph[ind].adjacency_matrix().to_dense().to(graph[ind].device)

                    print('loss',torch.cuda.memory_allocated()/torch.cuda.max_memory_reserved())
                    '''
                    #sampled_pred=sc_pred[edge_ids[:,0],edge_ids[:,1]]
                    
                    sampled_pred = sc_pred#[:,0]
                    lbl_edges = torch.zeros(sampled_pred.shape).to(sampled_pred.device).to(torch.float64)
                    check_gpu_usage('before edge idx')
                    edge_idx=graph[ind].has_edges_between(edge_ids[ind,0,:],edge_ids[ind,1,:])
                    edge_idx = edge_idx.nonzero()
                    edge_idx = edge_idx.T[0]
                    
                    # bug?
                    lbl_edges[edge_idx] = graph[ind].edata['w'][graph[ind].edge_ids(edge_ids[ind,0,:][edge_idx],edge_ids[ind,1,:][edge_idx])].to(torch.float64)
                    #import ipdb ; ipdb.set_trace()
                    total_struct_error, edge_struct_errors,regloss,clustloss,nonclustloss = get_sampled_losses(sampled_pred,edge_ids[ind],lbl_edges,clusts[-1])
                    
                    #check_gpu_usage('b4')
                    '''
                    ctr_loss=torch.zeros(clusts[ind].unique().shape[0])

                    #import ipdb ; ipdb.set_trace()
                    for node_ind,emb in enumerate(res_a[ind]):
                        #print(node_ind)
                        clust_idx = clusts[ind][node_ind]
                        clust_idxs = torch.where(clusts[ind]==clust_idx)[0]
                        #clust_idxs = clust_idxs[clust_idxs!=node_ind]

                        # difference between embeddings, sum over features
                        alldiffs =torch.abs((emb.tile((res_a[ind].shape[0],1))-res_a[ind])).sum(1)
                        #diffs=torch.abs((emb.tile((clust_idxs.shape[0],1))-res_a[ind][clust_idxs])).sum(1)
                        alldiffs = 1/alldiffs#torch.log(alldiffs)
                        alldiffs[torch.where(torch.isinf(alldiffs))] = 0.
                        try:
                            nonclust = torch.where(clusts[ind]!=clust_idx)[0]
                            nonclust = nonclust[generate_unique_numbers(clust_idxs.shape[0]-1,0,nonclust.shape[0]-1)]
                        except Exception as e:
                            import ipdb ; ipdb.set_trace()
                            print(e)
                        ctr_loss[clust_idx] += torch.maximum(torch.tensor(0),0.8-alldiffs[nonclust]).mean()
                        ctr_loss[clust_idx] += alldiffs[clust_idxs][torch.where(clust_idxs!=node_ind)].mean()
                        del alldiffs #; torch.cuda.empty_cache()
                        #ctr_loss+=
                    import ipdb ; ipdb.set_trace()
                    total_struct_error += 0.5*ctr_loss[~torch.where(torch.isnan(ctr_loss))[0]].mean()
                    '''
                    del sampled_pred, lbl_edges, edge_idx
                    torch.cuda.empty_cache()
                    check_gpu_usage('after edge idx')
                    # sample some random edges, will be the same from dgl seed? check if sampling will be the same for each 
                    '''
                    num_samp=neg_edges.shape[0]
                    pos_edge_samp = torch.stack(list(graph[ind].edges())).T
                    pos_edge_samp = pos_edge_samp[np.random.randint(0,pos_edge_samp.shape[0],num_samp)]
                    neg_edge_samp = dgl.sampling.global_uniform_negative_sampling(graph[ind],num_samp)
                    edge_ids_samp = torch.vstack((pos_edge_samp,torch.stack(list(neg_edge_samp)).T))
                    edge_ids_tot.append(edge_ids_samp.detach().cpu().numpy().T)
                    total_struct_error, edge_struct_errors = get_sampled_losses(sc_pred,edge_ids_samp,graph[ind].adjacency_matrix().to_dense().to(sc_pred.device))
                    del pos_edge_samp,neg_edge_samp,edge_ids_samp ; torch.cuda.empty_cache()
                    '''
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
                    all_costs = total_struct_error.unsqueeze(0)*alpha
                if recons_ind == 1 and total_feat_error > -1:
                    all_feat_error = (feat_error).unsqueeze(0)
                    all_costs = (torch.mean(all_feat_error))*(1-alpha)
            else:
                if recons_ind == 0 and total_struct_error > -1:
                    all_struct_error = torch.cat((all_struct_error,(edge_struct_errors).unsqueeze(0)))
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
    return all_costs, all_struct_error, all_feat_error, regloss,clustloss,nonclustloss#, edge_ids_tot

def get_sampled_losses(pred,edges,label,clusts=None):
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
    reg_clust = torch.zeros(edge_errors.shape).to(edge_errors.device)
    reg_nonclust = torch.zeros(edge_errors.shape).to(edge_errors.device)
    
    epsilon = 1.e-8
    clusts = torch.tensor(clusts)
    #clusts = torch.arange(2708)
    edge_clusts=clusts[edges.detach().cpu()]
    sc_idx=torch.where(edge_clusts[0]==edge_clusts[1])[0]
    sc_idx = sc_idx[label[sc_idx].nonzero()][:,0]
    #sc_idx = torch.arange(edge_clusts.shape[1])
    cl = edge_clusts[:,sc_idx][0]
    average_tensor = torch.scatter_reduce(pred[sc_idx].detach().cpu(), 0, cl, reduce="mean")
    expanded_average = average_tensor[cl].to(edge_errors.device)
    alpha = 0.2

    # edges outside of cluster
    nonclust = torch.where(edge_clusts[0]!=edge_clusts[1])[0]
    nonclust = nonclust[generate_unique_numbers(nonclust.shape[0]-1,0,nonclust.shape[0]-1)]
    #import ipdb ; ipdb.set_trace()
    clust_loss =  torch.log(pred[sc_idx]+epsilon)
    #clust_loss =  torch.log((expanded_average-pred[sc_idx]).abs()+epsilon)
    #clust_loss = pred[sc_idx]
    clust_loss[torch.where(torch.isinf(clust_loss))[0]] = 0.

    #import ipdb  ; ipdb.set_trace()
    
    edge_errors[sc_idx]= edge_errors[sc_idx] - alpha * clust_loss
    reg_clust[sc_idx] = -alpha * clust_loss
    
    nonclust_loss = torch.log(1-pred[nonclust]+epsilon)
    #nonclust_loss = 1-pred[nonclust]
    nonclust_loss[torch.where(torch.isinf(nonclust_loss))[0]] = 0.

    edge_errors[nonclust] = edge_errors[nonclust] - alpha * nonclust_loss
    reg_nonclust[nonclust] = -alpha * nonclust_loss
    
    #edge_errors[sc_idx] = edge_errors[sc_idx] - (alpha*torch.abs((edge_errors[sc_idx]-expanded_average)))
    #edge_errors[~sc_idx] = edge_errors[~sc_idx] - (alpha*torch.abs((edge_errors[~sc_idx]-expanded_average)))
    #import ipdb ; ipdb.set_trace()
    #edge_errors[sc_idx] = edge_errors[sc_idx] + (alpha*torch.abs((edge_errors[sc_idx]-expanded_average)))
    #edge_errors[~sc_idx] = edge_errors[~sc_idx] + (alpha*torch.abs((edge_errors[~sc_idx]-expanded_average)))
    
    #print(pred[sc_idx][torch.where(edge_clusts[:,sc_idx][0]==10)].mean(),pred[~sc_idx].mean())

    regloss = reg_clust + reg_nonclust
    total_error = torch.sum(edge_errors)
    
    #total_error = torch.sum(edge_errors)
    if torch.isnan(total_error):
        print('nan')
        import ipdb ; ipdb.set_trace()
        

    return total_error, edge_errors, regloss, reg_clust, reg_nonclust

def average_with_indices(input_tensor, indices_tensor):


    input_tensor,indices_tensor=input_tensor,indices_tensor
    average_tensor = torch.scatter_reduce(input_tensor, 0, indices_tensor, reduce="sum")
    expanded_average = average_tensor[indices_tensor]
    
    return expanded_average