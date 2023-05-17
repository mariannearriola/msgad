import torch
import numpy as np
import dgl

def loss_func(graph, feat, A_hat, X_hat, pos_edges, neg_edges, sample=False, recons='struct', alpha=None):
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
    if not pos_edges == None:
        edge_ids = torch.vstack((pos_edges,neg_edges))
 
        if type(graph) != list:
            feat = graph.ndata['feature']
            edge_labels = torch.cat((torch.full((pos_edges.shape[0],),1.),(torch.full((neg_edges.shape[0],),0.))))
            edge_labels = edge_labels.to(graph.device)
    edge_ids_tot = []
    for recons_ind,preds in enumerate([A_hat, X_hat]):
        if preds == None: continue
        for ind, sc_pred in enumerate(preds):
            # structure loss
            if recons_ind == 0 and recons in ['struct','both']:
                if sample:
                    # collect loss for selected positive/negative edges. adjacency not used

                    total_struct_error, edge_struct_errors = get_sampled_losses(sc_pred,edge_ids,graph[ind].adjacency_matrix().to_dense().to(sc_pred.device))
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
                    import ipdb ; ipdb.set_trace()
                    total_struct_error = torch.mean(edge_struct_errors)
     

            # feature loss
            if recons_ind == 1 and recons in ['feat','both']:
                #import ipdb ; ipdb.set_trace()
                feat_error = torch.nn.functional.mse_loss(sc_pred,feat.to(feat.device),reduction='none')
                total_feat_error = torch.mean(feat_error,dim=1)
            
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
    del feat, total_feat_error, total_struct_error ; torch.cuda.empty_cache()
    return all_costs, all_struct_error, all_feat_error#, edge_ids_tot

def get_sampled_losses(pred,edges,label):
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
    sampled_pred=pred[edges[:,0],edges[:,1]]
    #edge_errors = pred[edges[:,0],edges[:,1]]
    # BUG: edges not right
    label = label[edges[:,0],edges[:,1]]
    #return torch.nn.functional.binary_cross_entropy_with_logits(edge_errors,label), torch.nn.functional.binary_cross_entropy_with_logits(edge_errors,label,reduction='none')

    edge_errors = torch.pow(torch.abs(sampled_pred-label),2)

    #edge_errors = torch.abs(edge_errors-label)
    #total_error = torch.mean(torch.sqrt(edge_errors))
    total_error = torch.mean(edge_errors)
    return total_error, edge_errors
