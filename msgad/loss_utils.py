import torch

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
    all_costs, all_struct_error, all_feat_error = None, torch.tensor(0.), torch.tensor(0.)
    total_struct_error,total_feat_error=None,None
    if not pos_edges == None:
        edge_ids = torch.vstack((pos_edges,neg_edges))
        if type(graph) != list:
            feat = graph.ndata['feature']
            edge_labels = torch.cat((torch.full((pos_edges.shape[0],),1.),(torch.full((neg_edges.shape[0],),0.))))
            edge_labels = edge_labels.to(graph.device)
    #import ipdb ; ipdb.set_trace()
    for recons_ind,preds in enumerate([A_hat, X_hat]):
        if preds == None: continue
        for ind, sc_pred in enumerate(preds):
            # structure loss
            if recons_ind == 0 and recons in ['struct','both']:
                if sample:
                    # collect loss for selected positive/negative edges. adjacency not used
                    if type(graph) == list:
                        #edge_labels = torch.round(graph[inds_label[ind]][edge_ids[:,0],edge_ids[:,1]])
                        #import ipdb; ipdb.set_trace()
                        edge_labels = graph[ind]

                    total_struct_error, edge_struct_errors = get_sampled_losses(sc_pred,edge_ids,edge_labels)
  
                else:
                    # collect loss for all edges/non-edges in reconstruction
                    if type(graph) != list:
                        if not pos_edges == None:
                            adj_label=torch.sparse_coo_tensor(edge_ids.T,edge_labels)
                            num_nodes = graph.num_dst_nodes()
                            adj_label=adj_label.sparse_resize_((num_nodes,num_nodes),adj_label.sparse_dim(),adj_label.dense_dim())
                            adj_label=adj_label.to_dense()
                        else:
                            if type(graph) == torch.Tensor:
                                adj_label = graph
                                num_nodes = adj_label.shape[0]
                            else:
                                adj_label = graph.adjacency_matrix().to_dense().to(graph.device)
                                num_nodes = graph.num_dst_nodes()
                    else:
                        adj_label = graph[ind].to(pos_edges.device)
                        
                    edge_struct_errors = torch.pow(sc_pred - adj_label, 2)
                    total_struct_error = torch.mean(torch.sqrt(torch.sum(edge_struct_errors,1)))

            # feature loss
            if recons_ind == 1 and recons in ['feat','both']:
                #import ipdb ; ipdb.set_trace()
                feat_error = torch.nn.functional.mse_loss(sc_pred,feat.to(feat.device),reduction='none')
                total_feat_error = torch.mean(feat_error,dim=0)
           
            # accumulate errors
            if all_costs is None:
                if recons_ind == 0 and total_struct_error is not None:
                    all_struct_error = (edge_struct_errors).unsqueeze(0)
                    all_costs = total_struct_error.unsqueeze(0)*alpha
                if recons_ind == 1 and total_feat_error is not None:
                    all_feat_error = (feat_error).unsqueeze(0)
                    all_costs = (torch.mean(all_feat_error))*(1-alpha)
            else:
                if recons_ind == 0 and total_struct_error is not None:
                    all_struct_error = torch.cat((all_struct_error,(edge_struct_errors).unsqueeze(0)))
                if recons_ind == 1 and total_feat_error is not None:
                    all_feat_error = torch.cat((all_feat_error,(feat_error).unsqueeze(0)))
                all_costs = torch.cat((all_costs,torch.add(total_struct_error*alpha,torch.mean(all_feat_error)*(1-alpha)).unsqueeze(0)))
    return all_costs, all_struct_error, all_feat_error

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
    edge_errors = pred[edges[:,0],edges[:,1]]
    label = label[edges[:,0],edges[:,1]]
    label[torch.where(label<0)]=0.
    label[torch.where(label>0)]=1.
    edge_errors = torch.pow(torch.abs(edge_errors-label),2)
    epsilon = 1e-8
    total_error = torch.sqrt(edge_errors+epsilon)
    #edge_errors = edge_errors-label
    #total_error = edge_errors
    return total_error, edge_errors
