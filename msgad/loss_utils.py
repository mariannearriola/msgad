import torch


def loss_func(graph, A_hat, X_hat, pos_edges, neg_edges, sample=False, recons='struct', alpha=None):
    '''
    Input:
        adj: normalized adjacency matrix
        feat: feature matrix
        pos_edges: positive edge list
        neg_edges: negative edge list
        sample: if true, use sampled edges/non-edges for loss calculation. if false, use all edges/non-edges
        recons: structure reconstruction, feature reconstruction, or structure & feature reconstruction (dominant)
        alpha: if structure & feature reconstruction, scalar to weigh importance of structure vs feature reconstruction
    Output:
        all_costs: total loss for backpropagation
        all_struct_error: node-wise errors at each scale
    '''
    if not alpha: alpha = 1 if recons=='struct' else 0

    # node sampling: full reconstruction error
    all_costs, all_struct_error, all_feat_error = None, torch.tensor(0.), torch.tensor(0.)
    struct_error,feat_error=None,None
    if pos_edges:
        pos_edges = torch.vstack((pos_edges[0],pos_edges[1])).T
        neg_edges = torch.vstack((neg_edges[0],neg_edges[1])).T
        edge_ids = torch.cat((pos_edges,neg_edges))
        if type(graph) != list:
            feat = graph.ndata['feature']
            edge_labels = torch.cat((torch.full((pos_edges.shape[0],),1.),(torch.full((neg_edges.shape[0],),0.))))
            edge_labels = edge_labels.to(graph.device)

    # DEBUG
    #edge_ids = 

    for recons_ind,preds in enumerate([A_hat, X_hat]):
        if not preds: continue
        for ind, sc_pred in enumerate(preds):
            # structure loss
            if recons_ind == 0:
                if sample:
                    # collect loss for selected positive/negative edges. adjacency not used
                    if type(graph) == list:
                        edge_labels = torch.round(graph[ind].cuda()[edge_ids[:,0],edge_ids[:,1]])
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
                            adj_label = graph.adjacency_matrix().to_dense().cuda()
                            num_nodes = graph.num_dst_nodes()
                    else:
                        adj_label = graph[ind].cuda()
                    #import ipdb ; ipdb.set_trace()
                    #if sample == True:
                    #edge_struct_errors = torch.sparse.sum(torch.pow(sc_pred - adj_label, 2),1).to_dense()
                    # TODO: SUM VS MEAN !!!
                    #import ipdb ; ipdb.set_trace()
                    # issue on SPARSE SUM !!
                    edge_struct_errors = torch.pow(sc_pred - adj_label, 2)
                    total_struct_error = torch.mean(torch.sqrt(torch.sum(edge_struct_errors,1)))
                    #total_struct_error = torch.mean(torch.sqrt(edge_struct_errors))
                    #total_struct_error = torch.mean(torch.sqrt(torch.sum(edge_struct_errors,1)))
                    #import ipdb ; ipdb.set_trace()

            # feature loss
            if recons_ind == 1:
                feat_error = torch.nn.functional.mse_loss(sc_pred,feat.to(graph.device),reduction='none')
                feat_error = torch.mean(feat_error,dim=0)

            # accumulate errors
            if all_costs is None:
                if recons_ind == 0:
                    all_struct_error = (edge_struct_errors).unsqueeze(0)
                    all_costs = total_struct_error.unsqueeze(0)*alpha
                if recons_ind == 1:
                    all_feat_error = (feat_error).unsqueeze(0)
                    all_costs = (torch.mean(all_feat_error))*(1-alpha)
            else:
                if recons_ind == 0: all_struct_error = torch.cat((all_struct_error,(edge_struct_errors).unsqueeze(0)))
                if recons_ind == 1: all_feat_error = torch.cat((all_feat_error,(feat_error).unsqueeze(0)))
                all_costs = torch.cat((all_costs,torch.add(total_struct_error*alpha,torch.mean(all_feat_error)*(1-alpha)).unsqueeze(0)))
    #import ipdb ; ipdb.set_trace()
    return all_costs, all_struct_error, all_feat_error

def get_sampled_losses(pred,edges,label):
    edge_errors = pred[edges[:,0],edges[:,1]]
    edge_errors = torch.pow(edge_errors-label,2)
    total_error = torch.sqrt(edge_errors)
    return total_error, edge_errors
