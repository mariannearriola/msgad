import torch
import numpy as np
import dgl
from utils import *
import torch_scatter
import random


class loss_func:
    def __init__(self,graph,feat,exp_params,sample=False, recons='struct', alpha=None, clusts=None, regularize=True):
        self.graph = graph
        self.feat = feat
        self.sample = sample
        self.recons = recons
        self.alpha = alpha
        self.clusts = clusts
        self.regularize=regularize
        self.model_ind= exp_params['MODEL']['IND']
        
    def calc_loss(self,lbl,A_hat,X_hat,res_a,edge_ids,attract_sel,repel_sel,clusts):
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
        if not self.alpha: self.alpha = 1 if self.recons=='struct' else 0

        # node sampling: full reconstruction error
        all_costs, all_struct_error, all_feat_error = None, None, None
        scale_weights=[1,1,1]
        total_struct_error,total_feat_error=torch.tensor(-1.),torch.tensor(-1.)
   
        # simple node contrastive loss; nodes in the same cluster should have similar reps
        # DEBUG
        #loss=torch.nn.functional.binary_cross_entropy_with_logits(A_hat[0], graph[0].adjacency_matrix().to_dense().to(A_hat[0].device), reduction='none')
        #loss=torch.nn.functional.mse_loss(A_hat[0], graph[0].adjacency_matrix().to_dense().to(A_hat[0].device), reduction='none')
        #loss=torch.nn.functional.mse_loss(A_hat[0], graph[0].ndata['feature'], reduction='none')
        #loss = torch.abs(A_hat[0] - graph[0].ndata['feature'])
        #loss = A_hat[0] - graph[0].adjacency_matrix().to_dense().to(A_hat[0].device)
        #import ipdb ; ipdb.set_trace()


        print(A_hat.min(),A_hat.max())
        clusts = torch.tensor(clusts)
        edge_ids_tot = []
        for recons_ind,preds in enumerate([A_hat, X_hat]):
            if preds == None: continue
            for ind, sc_pred in enumerate(preds):
                # structure loss
                if recons_ind == 0 and self.recons in ['struct','both']:
                    if sample:
                        if self.model_ind is not None:
                            samp_ind = self.model_ind
                        else:
                            samp_ind = ind
                        sampled_pred = sc_pred
                        lbl_edges = torch.zeros(sampled_pred.shape).to(sampled_pred.device).to(torch.float64)
                        check_gpu_usage('before edge idx')
                        edge_idx=lbl[samp_ind].has_edges_between(edge_ids[samp_ind][:,0],edge_ids[samp_ind][:,1]).nonzero().T[0]
                        # bug?
                        lbl_edges[edge_idx] = lbl[samp_ind].edata['w'][lbl[samp_ind].edge_ids(edge_ids[samp_ind][:,0][edge_idx],edge_ids[samp_ind][:,1][edge_idx])].to(torch.float64)
                        
                        # TODO: lbl edges should be balanced
                        
                        
                        total_struct_error, edge_struct_errors,regloss,clustloss,nonclustloss,sc_idx_inside,sc_idx_outside = self.get_sampled_losses(lbl[samp_ind],sampled_pred,edge_ids[samp_ind],lbl_edges,ind,attract_sel,repel_sel,clusts[samp_ind])
                        del sampled_pred, lbl_edges, edge_idx
                        torch.cuda.empty_cache()
                        check_gpu_usage('after edge idx')
                    else:
                        # collect loss for all edges/non-edges in reconstruction
                        if type(lbl) != list:
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
                            adj_label = lbl[ind].to(pos_edges.device)
                        edge_struct_errors = torch.sqrt(torch.sum(torch.pow(sc_pred - adj_label, 2),1))
                        total_struct_error = torch.mean(edge_struct_errors)
                
                # feature loss
                if recons_ind == 1 and self.recons in ['feat','both']:
                    #import ipdb ; ipdb.set_trace()
                    feat_error = torch.nn.functional.mse_loss(sc_pred,feat.to(feat.device),reduction='none')
                    total_feat_error = torch.mean(feat_error,dim=1)

                # accumulate errors
                if all_costs is None:
                    if recons_ind == 0 and total_struct_error > -1:
                        all_struct_error = (edge_struct_errors).unsqueeze(0)
                        all_reg_error = (regloss).unsqueeze(0)
                        all_clust_error = (clustloss).unsqueeze(0)
                        all_nonclust_error = (nonclustloss).unsqueeze(0)
                        all_sc_idx_inside = [sc_idx_inside]
                        all_sc_idx_outside = [sc_idx_outside]
                        all_costs = total_struct_error.unsqueeze(0)*self.alpha
                    if recons_ind == 1 and total_feat_error > -1:
                        all_feat_error = (feat_error).unsqueeze(0)
                        all_costs = (torch.mean(all_feat_error))*(1-self.alpha)
                else:
                    if recons_ind == 0 and total_struct_error > -1:
                        all_struct_error = torch.cat((all_struct_error,(edge_struct_errors).unsqueeze(0)))
                        all_reg_error = torch.cat((all_reg_error,(regloss).unsqueeze(0)))
                        all_clust_error = torch.cat((all_clust_error,(clustloss).unsqueeze(0)))
                        all_nonclust_error = torch.cat((all_nonclust_error,(nonclustloss).unsqueeze(0)))
                        all_sc_idx_inside.append(sc_idx_inside)
                        all_sc_idx_outside.append(sc_idx_outside)
                    elif recons_ind == 1 and total_feat_error > -1:
                        if all_feat_error is None:
                            all_feat_error = (feat_error).unsqueeze(0)
                        else:
                            all_feat_error = torch.cat((all_feat_error,(feat_error).unsqueeze(0)))
                    else:
                        continue

                    # assuming both is NOT multi-scale
                    if self.recons != 'both':
                        all_costs = torch.cat((all_costs,torch.add(total_struct_error*self.alpha*scale_weights[recons_ind],torch.mean(total_feat_error)*(1-self.alpha)).unsqueeze(0)))
                    else:
                        all_costs = torch.add(total_struct_error*self.alpha,torch.mean(total_feat_error)*(1-self.alpha)).unsqueeze(0)
    
        del total_feat_error, total_struct_error ; torch.cuda.empty_cache()
        return all_costs, all_struct_error, all_feat_error, all_reg_error, all_clust_error, all_nonclust_error, all_sc_idx_inside, all_sc_idx_outside#, edge_ids_tot

    def get_group_idx(self,edge_ids,clust,i,anom_wise=True):
        """Get all edges associated with an anomaly group OR of the cluster(s) of the anomaly group"""
        dgl_g = dgl.graph((edge_ids[:,0],edge_ids[:,1]))
        if anom_wise:
            return dgl_g.out_edges(i,form='eid')
        else:
            anom_clusts = clust[i].unique()
            return dgl_g.out_edges(np.intersect1d(clust,anom_clusts,return_indices=True)[-2],form='eid')

    def get_sampled_losses(self,lbl,pred,edges,label,ind,attract_sel,repel_sel,clusts=None):
        """description

        Parameters
        ----------
            lbl: DGL graph
                DGL graph (label)
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

        # gets even # of edge/nonedge for each node in backprop
        #import ipdb ; ipdb.set_trace() # perform gather based on node id
        edge_errors=torch.nn.functional.mse_loss(pred.to(torch.float64), label, reduction='none')
        # if neg nodes doesn't cover, this means that there are disconnected nodes in label
        pos=(clusts[edges[:,0]]==clusts[edges[:,1]]).nonzero()
        pos_nodes = edges[pos.T[0]][:,1]
        pos_losses = edge_errors[pos.T[0]]
        pos_losses = torch_scatter.scatter_add(pos_losses,pos_nodes)
        neg=(clusts[edges[:,0]]!=clusts[edges[:,1]]).nonzero()
        neg_nodes = edges[neg.T[0]][:,0]
        neg_losses = edge_errors[neg.T[0]]
        neg_losses = torch_scatter.scatter_add(neg_losses,neg_nodes)
        pos_losses_tot = torch.zeros(edges.unique().shape[0]).to(float).to(pred.device) ; neg_losses_tot = torch.zeros(edges.unique().shape[0]).to(float).to(pred.device)
        pos_losses_tot[torch.arange(pos_losses.shape[0])] = pos_losses ; neg_losses_tot[torch.arange(neg_losses.shape[0])] = neg_losses
       
        tot_error = pos_losses_tot + neg_losses_tot
        attract_edges = pos ; repel_edges = neg
        return tot_error.mean(),tot_error, tot_error,pos_losses_tot,neg_losses_tot,attract_edges,repel_edges

        repel_edges = torch.where(clusts[edges][:,0]!=clusts[edges[:,1]])[0]
        for ind,i in enumerate(lbl.nodes()):
            #neighbors = lbl.successors(i)
            out_e = self.get_group_idx(edges,None,i,anom_wise=True)
            lbl_out = label[out_e]
            try:
                
                pos_e = out_e[label[out_e].nonzero().T[0]]
                if pos_e.shape[0] == 0:
                    pos_loss = torch.tensor(0).to(pos_e.device)
                else:
                    pos_e = pos_e[np.random.randint(0,pos_e.shape[0],5)]
                    pos_loss=edge_errors[pos_e].sum()
                neg_e = out_e[label[out_e]==0]
                if neg_e.shape[0] == 0:
                    neg_loss = torch.tensor(0).to(neg_e.device)
                else:
                    neg_e = neg_e[np.random.randint(0,neg_e.shape[0],5)]
      
                    neg_loss = edge_errors[neg_e].sum()
            except Exception as e:
                print(e)
                import ipdb ; ipdb.set_trace()
            
            #pos_loss=edge_errors[lbl.edge_ids(i,neighbors[np.random.randint(0,neighbors.shape[0],5)])]
            #neg_loss=
            if ind == 0:
                tot_error = (pos_loss+neg_loss).unsqueeze(0)
                reg_clust = torch.tensor(pos_loss).unsqueeze(0)
                reg_nonclust = neg_loss.unsqueeze(0)
                attract_edges = pos_e
                repel_edges = neg_e
            else:
                tot_error = torch.cat((tot_error,(pos_loss+neg_loss).unsqueeze(0)),dim=0)
                attract_edges = torch.cat((attract_edges,pos_e))
                reg_clust = torch.cat((reg_clust,pos_loss.unsqueeze(0)))
                reg_nonclust = torch.cat((reg_nonclust,neg_loss.unsqueeze(0)))
                repel_edges = torch.cat((repel_edges,neg_e))
        return tot_error.mean(),tot_error, tot_error,reg_clust,reg_nonclust,attract_edges,repel_edges
        
        regloss=torch.zeros(edge_errors.shape).to(edge_errors.device)
        reg_clust = torch.zeros(edge_errors.shape).to(edge_errors.device)
        reg_nonclust = torch.zeros(edge_errors.shape).to(edge_errors.device)
        if self.regularize is True:
            attract_edges = torch.where(clusts[edges][:,0]==clusts[edges[:,1]])[0]
            repel_edges = torch.where(clusts[edges][:,0]!=clusts[edges[:,1]])[0]
            assert(attract_edges.shape[0]==repel_edges.shape[0])
            assert(label[label==0].shape[0]==label[label!=0].shape[0])
            #repel_edges = repel_edges[torch.tensor(np.random.randint(0,repel_edges.shape[0],attract_edges.shape))]
            
            alpha = 1.0
            clust_loss =  (1-pred[attract_edges])
            reg_clust[attract_edges] = alpha * clust_loss #* 1/(attract_edges.shape[0]/edge_errors.shape[0])
            #edge_errors[attract_edges]= edge_errors[attract_edges] + reg_clust[attract_edges]
            #edge_errors[attract_edges]=  clust_loss


            nonclust=repel_edges
            nonclust_loss = pred[nonclust]
            reg_nonclust[nonclust] = alpha * nonclust_loss #* 1/(repel_edges.shape[0]/edge_errors.shape[0])
            #edge_errors[nonclust] = edge_errors[nonclust] + reg_nonclust[nonclust]
            #edge_errors[nonclust]=  nonclust_loss
            regloss = reg_clust + reg_nonclust

            print((alpha*clust_loss)[(alpha*clust_loss).nonzero()[:,0]].mean(),(alpha*pred[nonclust])[(alpha*pred[nonclust]).nonzero()[:,0]].mean())
            print(f'correct edges for clust {ind}',torch.where(pred[attract_edges]>=0.5)[0].shape[0]/pred[attract_edges].shape[0])
            print(f'correct non-edges {ind}',torch.where(pred[nonclust]<0.5)[0].shape[0]/pred[nonclust].shape[0])
        
        print('correct edges',torch.where(pred[label.nonzero()][:,0]>=0.5)[0].shape[0]/pred[label.nonzero()][:,0].shape[0])
        print('correct non-edges',torch.where(pred[~label.nonzero()][:,0]<0.5)[0].shape[0]/pred[~label.nonzero()][:,0].shape[0])
                

        total_error = torch.mean(edge_errors)
        if torch.isnan(total_error):
            raise('reconstruction contains nan')
            
        return total_error, edge_errors, regloss, reg_clust, reg_nonclust, attract_edges, nonclust

def average_with_indices(input_tensor, indices_tensor):
    input_tensor,indices_tensor=input_tensor,indices_tensor
    average_tensor = torch.scatter_reduce(input_tensor, 0, indices_tensor, reduce="sum")
    expanded_average = average_tensor[indices_tensor]
    
    return expanded_average