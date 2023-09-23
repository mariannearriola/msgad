
import numpy as np
import scipy.sparse as sp
import torch
import torch_scatter
import scipy
import random
import networkx as nx
from igraph import Graph
import dgl
import copy
import matplotlib.ticker as mtick
import pickle as pkl
import model
import matplotlib.pyplot as plt
import os
import yaml
import torch
from pygspcopy import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from anom_detector import *
from sklearn.ensemble import IsolationForest
from itertools import combinations

def check_batch(pos_edges,neg_edges,clusts,batch_nodes,neg_batch_nodes):
    """Ensure that intra-cluster & inter-cluster edges selected align with cluster assignments"""
    try:
        for i in range(len(pos_edges)):
            attract_e=torch.where(clusts[i][batch_nodes[i][pos_edges[i]]][:,0]==clusts[i][batch_nodes[i][pos_edges[i]]][:,1])[0]
            
            assert(attract_e.shape[0]==pos_edges[i].shape[0])
            repel_e=torch.where(clusts[i][batch_nodes[i][pos_edges[i]]][:,0]!=clusts[i][batch_nodes[i][pos_edges[i]][:,1]])[0]
            attract_e_neg=torch.where(clusts[i][neg_batch_nodes[i][neg_edges[i]]][:,0]==clusts[i][neg_batch_nodes[i][neg_edges[i]][:,1]])[0]
            assert(attract_e_neg.shape[0]==0)
            assert(repel_e.shape[0]==0)
            del attract_e,repel_e,attract_e_neg
    except Exception as e:
        import ipdb ; ipdb.set_trace()
        raise "batching incorrect"

def get_labels(adj,feats,clusts,exp_params):
    """
    Prepare intra-cluster edge labels according to cluster assignments to pass into dataloader
    
    Input:
        adj: {DGL graph}
            Input graph.
        feats: {array-like}, shape=[n, f]
            Feature matrix.
        clusts: {array-like}, shape=[k, n]
            Scale-wise cluster assignments.
        exp_params: {dictionary}
            Experiment parameters.
    Output:
        lbls: {array-like}, shape=[k, ]
            Array of DGL graphs corresponding to unique scale-specific clusterings
        pos_edges_full: {array-like}, shape=[k, n, 2]
            Array of edge lists ccorresponding to unique scale-specific clusterings
    """
    exp = exp_params['EXP']
    dataset = exp_params['DATASET']['NAME']
    dataset_scales, model_scales = exp_params['DATASET']['SCALES'], exp_params['MODEL']['SCALES']
    transform = dgl.AddReverse(copy_edata=True)
    # sample an even number of intra-cluster & inter-cluster edges for each node
    if not os.path.exists(f'{dataset}_full_adj_{model_scales}_dataset{dataset_scales}_exp{exp}.mat'):
        lbls,neg_lbls,pos_edges_full,neg_edges_full = [],[],[],[]
        for clust_ind,clust in enumerate(clusts):
            print(clust_ind)
            clusters = {}
            for node, cluster_id in enumerate(clust,0):
                clusters.setdefault(cluster_id.item(), []).append(node)
            pos_edges= torch.tensor([(x, y) for nodes in clusters.values() for x, y in combinations(nodes, 2)])
            # for each positive edge, replace connection with a negative edge. during dataloading, index both simultaneously
            assert(torch.where(clust[pos_edges[:,0]]==clust[pos_edges[:,1]])[0].shape[0]==pos_edges.shape[0])
            
            pos_clusts = clust[pos_edges[:,1]]
            clust_offset=np.random.randint(1,(clust.max()),pos_clusts.shape[0])
            if torch.where(pos_clusts==pos_clusts+clust_offset)[0].shape[0] != 0:
                raise "wrong offset"
            pos_clusts += clust_offset ; opp_clusts = pos_clusts % (clust.max()+1)
            
            neg_edges = copy.deepcopy(pos_edges)
            largest_clust_size = np.unique(clust,return_counts=True)[-1][0]
            el_offset = torch.randint(0,largest_clust_size,neg_edges[:,0].shape)
            
            neg_edges[:,1] = torch.tensor([clusters[i.item()][el_offset[ind]%len(clusters[i.item()])] for ind,i in enumerate(opp_clusts)])
            assert(torch.where(clust[neg_edges[:,0]]==clust[neg_edges[:,1]])[0].shape[0]==0)

            lbl_adj = dgl.graph((pos_edges[:,0],pos_edges[:,1]),num_nodes=adj.number_of_nodes()).to(exp_params['DEVICE'])
            lbl_adj.ndata['feature'] = feats.to(lbl_adj.device)
            lbl_adj.edata['w'] = torch.ones(pos_edges.shape[0]).to(lbl_adj.device)
            
            neg_adj = dgl.graph((neg_edges[:,0],neg_edges[:,1]),num_nodes=adj.number_of_nodes()).to(exp_params['DEVICE'])
            neg_lbls.append(transform(neg_adj))

            lbls.append(transform(lbl_adj))
            pos_edges_full.append(torch.vstack((pos_edges,pos_edges.flip(1))))
            neg_edges_full.append(torch.vstack((neg_edges,neg_edges.flip(1))))
            
        save_mat = {'lbls':[i for i in lbls],'neg_lbls':[i for i in neg_lbls],'pos_edges':[i.to_sparse() for i in pos_edges_full]}#,'neg_edges':[i.to_sparse() for i in neg_edges_full]}
        with open(f'{dataset}_full_adj_{model_scales}_dataset{dataset_scales}_exp{exp}.mat','wb') as fout:
            pkl.dump(save_mat,fout)
    else:
        with open(f'{dataset}_full_adj_{model_scales}_dataset{dataset_scales}_exp{exp}.mat','rb') as fin:
            mat =pkl.load(fin)
        lbls,neg_lbls,pos_edges_full = mat['lbls'],mat['neg_lbls'],[i.to_dense() for i in mat['pos_edges']]#,[i.to_dense() for i in mat['neg_edges']]
    return lbls, neg_lbls, pos_edges_full

def get_counts(clust,edge_ids):
    """
    ...

    Input:
        clust: array-like, shape=[k, n]
            Scale-wise cluster assignments
        edge_ids: array-like, shape=[n, 2]
            Edge IDs used for sampling
    Output:
        intra_edges: array-like, shape=[e]
            Indices of edge IDs corresponding to intra-cluster edges
        pos_counts: dict
            Dictionary mapping cluster IDs to the number of intra-cluster edges sampled
        neg_counts: dict
            Dictionary mapping cluster IDs to the number of inter-cluster edges sampled
        pos_clust_counts: dict
            ...
        neg_clust_counts: dict
            ...
    """
    intra_edges = torch.where(clust[edge_ids[:,0]]==clust[edge_ids[:,1]])[0]

    unique_elements, counts = torch.unique(edge_ids[:,1][intra_edges], return_counts=True)
    pos_clust_counts = torch.zeros(clust.shape[0]) ; pos_clust_counts[unique_elements.detach().cpu()] = counts.to(torch.float32).detach().cpu()
    pos_counts = dict(zip(unique_elements.tolist(), counts.tolist()))
    
    unique_elements, counts = torch.unique(edge_ids[:,1][torch.where(clust[edge_ids[:,0]]!=clust[edge_ids[:,1]])[0]], return_counts=True)
    neg_counts = dict(zip(unique_elements.tolist(), counts.tolist()))
    neg_clust_counts = torch.zeros(clust.shape[0]) ; neg_clust_counts[unique_elements.detach().cpu()] = counts.to(torch.float32).detach().cpu()

    return intra_edges, pos_counts, neg_counts, pos_clust_counts, neg_clust_counts

def score_multiscale_anoms(clustloss,nonclustloss, clusts, res, batch_edges, agg='std'):
    """
    Assign node-wise scores based on intra-cluster residuals

    Input:
        clustloss: {array-like}, shape=[k, n]
            Scale-wise intra-cluster losses.
        nonclustloss: {array-like}, shape=[k, n]
            Scale-wise inter-cluster losses.
        clusts: {array-like}, shape=[k, n]
            Scale-wise cluster assignments. 
        res: {array-like}, shape=[k, n]
            Scale-wise model embeddings.
    Output:
        scores_all: {array-like}, shape=[k, n]
            Computed scale-wise anomaly scores.
    """
    struct_loss = clustloss+nonclustloss
    for sc,sc_loss in enumerate(struct_loss):
        #score = gather_clust_info(sc_clustloss.detach().cpu(),clusts[sc],agg)
        score = gather_clust_info(sc_loss.detach().cpu(),clusts[sc],'std')*gather_clust_info(sc_loss.detach().cpu(),clusts[sc],'mean')
        scores_all = score if sc == 0 else torch.vstack((scores_all,score))
        
    return scores_all

class TBWriter:
    def __init__(self, tb, sc_label,truth,clust,exp_params):
        self.tb = tb
        self.clust = clust
        self.sc_labels = sc_label
        self.truth = truth
        self.anoms = (truth==1).nonzero()[0]
        self.norms = (truth==0).nonzero()[0]
        self.a_clf = anom_classifier(exp_params,exp_params['DATASET']['SCALES'],'output',dataset=exp_params['DATASET']['NAME'],exp_name=exp_params['EXP'],model='msgad')

    def collect_clust_loss(self,edge_ids,sc_idx,loss):
        """Get average loss for each cluster, and assign back to edge-wise losses"""
        #sc_idx = np.intersect1d(sc_idx,sc_idx_)
        cl = edge_ids[:,sc_idx][0].detach().cpu()
        average_tensor = torch.scatter_reduce(loss[sc_idx].detach().cpu(), 0, cl, reduce="mean")
        expanded_average = average_tensor[cl].to(edge_ids.device).mean()#/cl.shape[0]
        return expanded_average

    def get_group_idx(self,edge_ids,clust,i,anom_wise=True):
        """Get all edges associated with an anomaly group OR of the cluster(s) of the anomaly group"""
        dgl_g = dgl.graph((edge_ids[:,0],edge_ids[:,1]))
        if anom_wise:
            return dgl_g.in_edges(i,form='eid')
        else:
            anom_clusts = clust[i].unique()
            return dgl_g.in_edges(np.intersect1d(clust,anom_clusts,return_indices=True)[-2],form='eid')

    def update_dict(self,gr_dict,k,v):
        """Updates logging dictionary for a node group."""
        if k not in gr_dict.keys():
            gr_dict[k] = [v]
        else:
            gr_dict[k].append(v)
        return gr_dict

    def tb_write_anom(self,sc_label,edge_ids,pred,scores_all,loss,sc,epoch,clustloss,nonclustloss,clusts,anom_wise=True,log=False):
        """Log loss evolution for anomaly group"""
        self.tb.add_scalar(f'Loss_{sc}', loss[sc].mean(), epoch)
        self.tb.add_scalar(f'ClustLoss_{sc}', clustloss[sc].mean(), epoch)
        self.tb.add_scalar(f'NonClustLoss_{sc}', nonclustloss[sc].mean(), epoch)
        clust = clusts[sc]
        intra_edges, pos_counts_dict, neg_counts_dict, _, _ = get_counts(clust,edge_ids[sc])
    
        sc_labels = np.unique(sc_label) ; sc_labels = np.append(sc_labels,-1)

        self.tb.add_histogram(f'Loss_inclust_{sc}',clustloss[sc].detach().cpu().mean(), epoch)
        self.tb.add_histogram(f'Loss_outclust_{sc}',nonclustloss[sc].detach().cpu().mean(), epoch)
        pos_counts_all,neg_counts_all = [],[]
        for ind,(nc,cl) in enumerate(zip(nonclustloss,clustloss)):
            _, _, _, pos_counts,neg_counts = get_counts(clusts[ind],edge_ids[ind])
            pos_counts_all.append(pos_counts) ; neg_counts_all.append(neg_counts)
            sil = torch.nan_to_num((1-torch.nan_to_num(nc.detach().cpu()/neg_counts,posinf=0,neginf=0)-(torch.nan_to_num(cl.detach().cpu()/pos_counts,posinf=0,neginf=0)))/torch.max(1-torch.nan_to_num(nc.detach().cpu()/neg_counts,posinf=0,neginf=0),(torch.nan_to_num(cl.detach().cpu()/pos_counts,posinf=0,neginf=0))),posinf=0,neginf=0)
            self.tb.add_histogram(f'Sil{ind+1}',sil.mean(), epoch)
            sils = sil if ind == 0 else torch.vstack((sils,sil))

        for ind,i in enumerate(sc_labels):
            
            group = torch.tensor(self.anoms[np.where(np.array(sc_label)==i)]) if i != -1 else torch.tensor(self.norms)
            sc_truth = np.zeros(clust.shape).astype(float) ; sc_truth[group] = 1.
            # gets all edges related to a group
            gr_dict = {}
            gr_dict = self.update_dict(gr_dict,f'Pred_fraction',(pred>=.5).nonzero().shape[0]/pred.shape[0])
            gr_dict = self.update_dict(gr_dict,f'True_fraction',intra_edges.shape[0]/pred.shape[0])     
            for l_ind,score in enumerate(scores_all):       
                gr_dict = self.update_dict(gr_dict,f'L{l_ind+1}_{anom_wise}',(score[group].detach().cpu()))
            anom_sc = ind if ind != len(self.sc_labels)-1 else 'norm'
            
            gr_clusts = clusts[sc][group] ; gr_intra = clustloss[sc][group].detach().cpu() ; gr_inter = nonclustloss[sc][group].detach().cpu()
            gr_dict = self.update_dict(gr_dict,f'Loss',gather_clust_info(loss[sc].detach().cpu(),clusts[sc])[group])
            gr_dict = self.update_dict(gr_dict,f'Loss_inclust',gr_intra)
            gr_dict = self.update_dict(gr_dict,f'Loss_inclust_mean',gather_clust_info(gr_intra,gr_clusts,'mean'))

            gr_dict = self.update_dict(gr_dict,f'Loss_inclust_std',gather_clust_info(gr_intra,gr_clusts,'std'))
            for sil_ind,sil in enumerate(sils):
                gr_dict = self.update_dict(gr_dict,f'Sil{sil_ind+1}',gather_clust_info(sil,clusts[sc])[group])
            
            group_pos_counts = np.array([pos_counts_dict[uid] if uid in pos_counts_dict.keys() else 1 for uid in group])
            group_neg_counts = np.array([neg_counts_dict[uid] if uid in neg_counts_dict.keys() else 1 for uid in group])
            if (group_neg_counts > 10).nonzero()[0].shape[0] > 0 or (group_pos_counts > 10).nonzero()[0].shape[0] > 0:
                raise('counted more than sampled')
            gr_dict = self.update_dict(gr_dict,f'Loss_outclust',gr_inter)
            gr_dict = self.update_dict(gr_dict,f'Loss_outclust_mean',gather_clust_info(gr_inter,gr_clusts,'mean'))
            gr_dict = self.update_dict(gr_dict,f'Loss_outclust_std',gather_clust_info(gr_inter,gr_clusts,'std'))
            
            for k,v in gr_dict.items():
                kname = k + f'_{sc}/Anom{anom_sc}_mean'
                self.tb.add_scalar(kname,np.array(v[0]).mean(), epoch)
                
                kname = k + f'_{sc}/Anom{anom_sc}_hist'
                self.tb.add_histogram(kname,np.array(v[0])[~np.isnan(np.array(v[0]))], epoch)

        _,prec1,ra1=self.a_clf.calc_anom_stats(scores_all.detach().cpu(),self.truth,sc_label,verbose=True,log=log)
        #_,prec1,ra1=self.a_clf.calc_anom_stats(scores_all.detach().cpu(),self.truth,sc_label,verbose=log,log=log)
        for sc in range(len(prec1)):
            for anom,prec in enumerate(prec1[sc]):
                self.tb.add_scalar(f'Precsc{sc+1}/anom{anom+1}', prec, epoch)
                self.tb.add_scalar(f'ROC{sc+1}/anom{anom+1}', ra1[sc][anom], epoch)
        return np.stack(pos_counts_all),np.stack(neg_counts_all)

def gather_clust_info(mat,clust,reduce="mean"):
    """Aggregate scores by cluster assignments according to reduction scheme"""
    if reduce=='std':
        mat = torch_scatter.scatter_std(mat,clust.to(mat.device))
    else:
        mat = torch.scatter_reduce(mat, 0, clust, reduce=reduce)
    return mat[clust]

def dgl_to_mat(g,device='cpu'):
    """Get sparse adjacency matrix from DGL graph"""
    src, dst = g.edges()
    block_adj = torch.sparse_coo_tensor(torch.stack((src,dst)),g.edata['w'].squeeze(-1),size=(g.number_of_nodes(),g.number_of_nodes()))
    return block_adj

def process_graph(graph):
    """Obtain graph information from input"""
    feats = graph[0].ndata['feature']
    return [torch.vstack((i.edges()[0],i.edges()[1])) for i in graph], feats

def check_gpu_usage(tag):
    """Logging GPU usage for debugging"""
    allocated_bytes = torch.cuda.memory_allocated(torch.device('cuda'))
    cached_bytes = torch.cuda.memory_cached(torch.device('cuda'))

    allocated_gb = allocated_bytes / 1e9
    cached_gb = cached_bytes / 1e9
    print(f"{tag} -> GPU Memory - Allocated: {allocated_gb:.2f} GB, Cached: {cached_gb:.2f} GB")

def prep_args(args):
    """Retrieve arguments from config file"""
    with open(f'configs/{args.config}.yaml') as file:
        yaml_list = yaml.load(file,Loader=yaml.FullLoader)
    # args.epoch will be populated if datasaving
    if args.epoch is not None: yaml_list['MODEL']['EPOCH'] = args.epoch
    yaml_list['DATASET']['DATASAVE'] = args.datasave ; yaml_list['DATASET']['DATALOAD'] = args.dataload
    return yaml_list

def dgl_to_nx(g):
    """Convert DGL graph to NetworkX graph"""
    nx_graph = nx.to_undirected(dgl.to_networkx(g.cpu(),edge_attrs='w'))
    node_ids = np.arange(g.num_nodes())
    return nx_graph,node_ids

def seed_everything(seed=1234):
    """Set random seeds for run"""
    random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_model(feat_size,exp_params,args):
    """Intialize model with configuration parameters"""
    struct_model,feat_model,params=None,None,None
    loaded=False
    exp_name = exp_params['EXP']
    try:
        if 'weibo' not in exp_name or 'elliptic' not in exp_name:
            struct_model = torch.load(f'{exp_name}.pt')
            loaded=True
    except:
        struct_model = model.GraphReconstruction(feat_size, exp_params)
  
    device = torch.device(exp_params['DEVICE'])
    if struct_model:
        struct_model = struct_model.to(exp_params['DEVICE']) ; struct_model.requires_grad_(True) ; struct_model.train() ; params = struct_model.parameters()
    if feat_model:
        feat_model = feat_model.to(exp_params['DEVICE']) ; feat_model.train() ; params = feat_model.parameters()
    
    if exp_params['MODEL']['NAME'] == 'gcad':
        gcad = GCAD(2,100,4)
    elif exp_params['MODEL']['NAME'] == 'madan':
        pass

    return struct_model,params,loaded

def sparse_matrix_to_dgl(coo,feat):
    """Converts sparse adjacency/feature information to DGL graph"""
    coo = scipy.sparse.coo_matrix(coo)
    v = torch.DoubleTensor(coo.data)
    i = torch.LongTensor(np.vstack((coo.row, coo.col)))
    dgl_graph = dgl.graph((i[0],i[1]),num_nodes=feat.shape[0])

    dgl_graph.edata['w'] = v
    dgl_graph.ndata['feature'] = feat
    return dgl_graph

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    adj += sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
