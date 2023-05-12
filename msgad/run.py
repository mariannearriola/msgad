import torch
import numpy as np
import argparse
import networkx as nx
from utils import *
from anom_detector import *
from dataloading import *
from loss_utils import *
import random 
import time
import gc
from models.gcad import *
from model import *
import torch.nn.functional as F
import MADAN.Madan as md
from visualization import *

import warnings
warnings.filterwarnings("ignore")

def graph_anomaly_detection(args):
    """Pipeline for autoencoder-based graph anomaly detection"""
    # load data
    seed_everything()
    dataloading = DataLoading(args)

    sp_adj, edge_idx, feats, truth, sc_label = dataloading.load_anomaly_detection_dataset()
    anoms,norms=np.where(truth==1)[0],np.where(truth==0)[0]

    if sp_adj is not None:
        adj = sparse_matrix_to_tensor(sp_adj,feats)

    lbl,last_batch_node,pos_edges,neg_edges=None,None,None,None
    
    # initialize data loading
    if edge_idx is not None:
        adj = dgl.graph((edge_idx[0],edge_idx[1]),num_nodes=feats.shape[0])
        adj.ndata['feature'] = feats
    edges=adj.edges('eid')

    dataloader = dataloading.fetch_dataloader(adj, edges)
    print('sample train',args.sample_train,'sample test',args.sample_test, 'epochs',args.epoch, 'saving?', args.datasave, 'loading?',args.dataload)
    
    # intialize model (on given device)
    adj = adj.to(args.device) ; feats = feats#.to(args.device)
    struct_model,feat_model=None,None
    struct_model,params = init_model(feats.size(1),args) ; struct_model.requires_grad_(True)

    #print(torch.cuda.memory_allocated()/torch.cuda.memory_reserved())

    if not args.model in ['gcad','madan']:
        optimizer = torch.optim.Adam(struct_model.parameters(), lr = args.lr)

    # begin model training
    #best_loss = torch.tensor(float('inf')).to(args.device)    
    A_hat, X_hat = None,None
    struct_loss,feat_loss=None,None
    res_a = None
    seconds = time.time()
    # epoch x 3 x num filters x nodes
    if 'multi-scale-amnet' in args.model:
        train_attn_w = torch.zeros((args.epoch,3,struct_model.module_list[0].filter_num,adj.number_of_nodes())).to(torch.float64)#.to(args.device)
    if args.vis_filters == True:
        visualizer = Visualizer(adj,feats,args,sc_label,norms,anoms)

    print(dataloader.__len__(),'batches')
    
    for epoch in range(args.epoch):
        epoch_l = 0
        if args.model == 'gcad': break
        iter=0
        for data_ind in dataloader:
            if (args.datadir is not None and args.dataload):
                loaded_input,lbl=dataloading.load_batch(data_ind,'train')
            else:
                loaded_input = data_ind
            if args.batch_type == 'node':
                all_nodes,in_nodes,g_batch=loaded_input
            else:
                in_nodes, pos_edges, neg_edges, g_batch, last_batch_node = dataloading.get_edge_batch(loaded_input)
                if 'cora' not in args.dataset: print('size of batch',g_batch.num_dst_nodes(),'nodes')

            if args.model == 'madan':
                if args.debug:
                    if iter % 100 == 0:
                        print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%')
                if iter == 0:
                    adj = adj.adjacency_matrix()
                    adj = adj.sparse_resize_((adj.size(0), adj.size(0)), adj.sparse_dim(), adj.dense_dim())
                idx=adj.coalesce().indices()
                nx_graph=nx.from_edgelist([(i[0].item(),i[1].item()) for i in idx.T])
                node_dict = None
                
                nodes = list(max(nx.connected_components(nx_graph), key=len))

                nx_graph = nx.subgraph(nx_graph, nodes)
                nx_graph = nx.from_numpy_matrix(nx.adjacency_matrix(nx_graph))
                feats_ = feats[nodes]

                #node_dict = {k.item():v for k,v in zip(list(in_nodes.detach().cpu()),np.arange(len(list(nx_graph.nodes))))}
                node_dict = {k:v for k,v in zip(nodes,np.arange(len(list(nx_graph.nodes))))}
                rev_node_dict = {v: k for k, v in node_dict.items()}
                try:
                    madan = md.Madan(nx_graph, attributes=feats_, sigma=0.08)
                except Exception as e:
                    print('eigenvalues dont converge',e)
                    continue
                
                time_scales   =   np.concatenate([np.array([0]), 10**np.linspace(0,5,5)])
                batch_anoms=[node_dict[j] for j in np.intersect1d(anoms,np.array(list(node_dict.keys()))).tolist()]
                #madan.anomalous_nodes=[node_dict[j] for j in np.intersect1d(anoms,np.array(list(node_dict.keys()))).tolist()]
                #madan.anomalous_nodes=[node_dict[j] for j in anoms]
                #madan.scanning_relevant_context_time(time_scales)
                #import ipdb ; ipdb.set_trace()
                madan.compute_concentration(10000000)
                madan.compute_context_for_anomalies()
                madan.plot_concentration()
                print('anoms detected',np.intersect1d(np.where(madan.concentration==1)[0],batch_anoms).shape[0]/np.where(madan.concentration==1)[0].shape[0])
                #import ipdb ; ipdb.set_trace()
                madan.interp_com
                anoms_detected=madan.anomalous_nodes
                if node_dict and len(anoms_detected)>0:
                    anoms_detected = rev_node_dict[anoms_detected]
                if len(anoms_detected)>0:
                    print('anom found')
                    import ipdb ; ipdb.set_trace()
                iter += 1
                continue
            optimizer.zero_grad()
            #node_dict = {k.item():v.item() for k,v in zip(in_nodes[g_batch.dstnodes()],np.arange(len(list(g_batch.dstnodes()))))}
            batch_sc_label = dataloading.get_batch_sc_label(in_nodes.detach().cpu(),sc_label,g_batch)
            #rev_node_dict = {v: k for k, v in node_dict.items()}
 
            if struct_model:
                vis = True if (epoch == 0 and iter == 0 and args.vis_filters == True) else False
            
                A_hat,X_hat,model_lbl,res_a = struct_model(g_batch,last_batch_node,pos_edges,neg_edges,batch_sc_label,vis=vis,vis_name='epoch1')
                if args.datasave:
                    lbl = model_lbl

                diffs = []
                ranges =[]
                for ind_,A_ in enumerate(A_hat):
                    ranges.append(A_.max()-A_.min())
                    if ind_ == len(A_hat)-1:
                        break
                    diffs.append((A_-A_hat[ind_+1]).max())

                print("diffs",diffs)
                print("ranges",ranges)
                    
            recons_label = g_batch if lbl is None else collect_recons_label(lbl,args.device)
 
            loss, struct_loss, feat_cost, _ = loss_func(recons_label, g_batch.ndata['feature'], A_hat, X_hat, pos_edges, neg_edges, sample=args.sample_test, recons=args.recons,alpha=args.alpha)
            
            #(new_scores.max(0).values-new_scores.mean(0))/(new_scores.max(0).values-new_scores.mean(0)).sum()
            if args.model == 'gradate':
                loss = A_hat[0]
            l = torch.sum(loss) if 'multi-scale' in args.model else torch.mean(loss)
            '''
            if l < best_loss:
                best_loss = dl
                torch.save(model,'best_model.pt')
            '''
            epoch_l = loss.unsqueeze(0) if iter == 0 else torch.cat((epoch_l,l.unsqueeze(0)))

            # save batch info
            if args.datasave: dataloading.save_batch(loaded_input,lbl,iter,'train')
            if args.debug:
                if iter % 100 == 0:
                    print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=', round(l.item(),3))
                
            iter += 1
            if 'cora' not in args.dataset: print('iter',iter)
            
            del g_batch, batch_sc_label, struct_loss, pos_edges, neg_edges
            for k in range(len(loaded_input)): del loaded_input[0]
            for k in range(len(A_hat)): del A_hat[0]
            if res_a:
                for k in range(len(res_a)): del res_a[0]
            for k in range(len(model_lbl)): del model_lbl[0]
            if X_hat is not None:
                for k in range(len(X_hat)): del X_hat[0]
            if feat_loss is not None: del feat_loss
            
            torch.cuda.empty_cache()
            gc.collect()
            l.backward()
            optimizer.step()
        '''
        print("Seconds since epoch =", (time.time()-seconds)/60)
        seconds = time.time()
        if args.model != 'madan' and 'multi-scale' in args.model:
            print("Epoch:", '%04d' % (epoch), "train_loss=", round(torch.sum(loss),3), "losses=",torch.round(loss,decimals=4).detach().cpu())
        else:
             print("Epoch:", '%04d' % (epoch), "train_loss=", round(epoch_l.item(),3))
        #print('avg loss',torch.mean(epoch_l/dataloader.__len__()))
        '''
        print('epoch done',epoch,loss)
        del loss
        if epoch == 0:
            tot_loss = epoch_l.sum(0).unsqueeze(0)
        else:
            tot_loss = torch.cat((tot_loss,epoch_l.sum(0).unsqueeze(0)),dim=0)
        epoch_l = torch.sum(epoch_l)
        #epoch_l.backward()
        #optimizer.step()
        
        if struct_model:
            if struct_model.attn_weights != None:
                # epoch x 3 x num filters x nodes
                train_attn_w[epoch,:,:,in_nodes]=torch.unsqueeze(struct_model.attn_weights,0).detach().cpu()
    
    #model = torch.load('best_model.pt')

    # accumulate node-wise anomaly scores via model evaluation
    if args.model not in ['madan','gcad']:
        if struct_model: struct_model.eval()
    #if 'elliptic' in args.dataset:
    #    struct_model.dataload = False
    
    edges=adj.edges('eid')
    #dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,device=args.device)
    if args.model != 'gcad':
        struct_scores, feat_scores = torch.zeros(len(A_hat),adj.number_of_nodes()).to(args.device),torch.zeros(args.d,adj.number_of_nodes()).to(args.device)
    iter = 0


    edge_anom_mats,node_anom_mats,recons_a,res_a_all = init_recons_agg(adj.number_of_nodes(),adj.ndata['feature'].shape[1],args)

    all_samps = []
    if (args.datadir is not None and args.dataload):# or 'elliptic' not in args.model:
        pass
        #random.shuffle(dataloader)
    else:
        dataloader = dataloading.fetch_dataloader(adj, edges)

    #import ipdb ; ipdb.set_trace()
    for loaded_input in dataloader:
        vis = True if (args.vis_filters == True and iter == 0) else False
        if args.dataload:# or 'elliptic' not in args.model:
            loaded_input,lbl=dataloading.load_batch(loaded_input,'test')
        # collect input
        if args.batch_type == 'node':
            all_nodes,in_nodes,g_batch= loaded_input
            pos_edges,neg_edges=None,None
        else:
            in_nodes, pos_edges, neg_edges, g_batch, last_batch_node = dataloading.get_edge_batch(loaded_input)
            edge_ids = torch.vstack((pos_edges,neg_edges))
    
        # run evaluation
        if struct_model and args.model != 'gcad':
            #node_dict = {k.item():v.item() for k,v in zip(in_nodes[g_batch.dstnodes()],np.arange(len(list(g_batch.dstnodes()))))}
            batch_sc_label = dataloading.get_batch_sc_label(in_nodes.detach().cpu(),sc_label,g_batch)
            #rev_node_dict = {v: k for k, v in node_dict.items()}
            A_hat,X_hat,model_lbl,res_a = struct_model(g_batch,last_batch_node,pos_edges,neg_edges,batch_sc_label,vis=vis,vis_name='test')
            if args.datasave:# or 'elliptic' in args.model:
                lbl = model_lbl

        if args.model == 'gcad':
            adj_ = g_batch.adjacency_matrix() 
            adj_=adj_.sparse_resize_((g_batch.num_src_nodes(),g_batch.num_src_nodes()), adj_.sparse_dim(),adj_.dense_dim())
            feat = g_batch.ndata['feature']
            if type(feat) == dict:
                feat = feat['_N']
            struct_loss = struct_model.fit_transform(adj_,feat)
            struct_loss = np.array((struct_loss.T)[g_batch.dstnodes().detach().cpu().numpy()].T)

        # collect anomaly scores
        edge_ids_,node_ids_ = torch.cat((pos_edges,neg_edges)).T.detach().cpu().numpy(),in_nodes[:g_batch.num_dst_nodes()]
        # save batch info
        if args.datasave:# or 'elliptic' in args.dataset:
            dataloading.save_batch(loaded_input,lbl,iter,'test')

        recons_label = g_batch if lbl is None else collect_recons_label(lbl,args.device)

        loss, struct_loss, feat_cost, edge_ids_tot = loss_func(recons_label, g_batch.ndata['feature'], A_hat, X_hat, pos_edges, neg_edges, sample=args.sample_test, recons=args.recons,alpha=args.alpha)
        if args.model == 'gradate':
            loss = A_hat[0]

        if args.model in ['gradate']:
            loss = A_hat[0]
            struct_loss = [A_hat[1]]
            if args.sample_test:
                l = torch.sum(loss)
            else:
                l = torch.mean(loss)
        elif args.model in ['gcad']:
            l = np.mean(struct_loss)

        node_anom_mats,edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,struct_loss,feat_cost,node_ids_,edge_ids,edge_ids_tot,node_anom_mats,edge_anom_mats,recons_a,res_a_all,args)

        #if iter % 100 == 0 and args.debug:
        #    print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=',round(l.item(),3))
        iter += 1
    
    
    # anomaly detection with final scores
    a_clf = anom_classifier()
    print('structure scores')
    if args.recons == 'both':
        print('struct scores')
        a_clf = anom_classifier()
        a_clf.calc_prec(None, [np.mean(edge_anom_mats[0],axis=1)], truth, sc_label, train_attn_w[-1].sum(axis=1).numpy(), args, cluster=False, input_scores=True)
        print('feat scores')
        a_clf.calc_prec(None, [np.mean(node_anom_mats[0],axis=1)], truth, sc_label, train_attn_w[-1].sum(axis=1).numpy(), args, cluster=False, input_scores=True)
        print('combined scores')
        a_clf.calc_prec(None, loss.detach().cpu().numpy(), truth, sc_label, train_attn_w[-1].sum(axis=1).numpy(), args, cluster=False, input_scores=True)
    else:
        if args.batch_type == 'node':
            if -1 in node_anom_mats[0]:
                raise('node not sampled')
            a_clf.calc_prec(None,node_anom_mats, truth, sc_label, args, cluster=False, input_scores=True)
        else:
            #lams = np.array([F.softmax(i.lam).detach().cpu().numpy() for i in struct_model.module_list])
            lams = np.array([(i.filter_weights).detach().cpu().numpy() for i in struct_model.module_list])
            tiled=np.tile(lams,((train_attn_w.shape[-1],1,1))).T
            tiled=np.moveaxis(tiled, 0, 1)
            attns = (tiled*train_attn_w[-1].numpy()).sum(axis=1)
            attns = train_attn_w[-1].numpy().sum(axis=1)
            if not args.sample_test:
                a_clf.calc_prec(adj.adjacency_matrix().to_dense(), struct_loss.detach().cpu().numpy(), truth, sc_label, attns, args, cluster=False, input_scores=True)
            else:
                a_clf.calc_prec(adj.adjacency_matrix().to_dense(), edge_anom_mats, truth, sc_label, attns,args, cluster=False)

    if args.vis_filters == True:
        #visualizer.plot_final_filters([np.array([i.weight.detach().cpu().numpy() for i in j.filters]) for j in struct_model.module_list],[F.softmax(i.lam).detach().cpu().numpy() for i in struct_model.module_list])
        #visualizer.plot_final_filters([np.array([i.weight.detach().cpu().numpy() for i in j.filters]) for j in struct_model.module_list])#,[(i.lam).detach().cpu().numpy() for i in struct_model.module_list])
        if 'multi-scale-amnet' == args.model and struct_model.attn_weights != None:
            visualizer.plot_attn_scores(train_attn_w.numpy(),[(i.filter_weights).detach().cpu().numpy() for i in struct_model.module_list])
            #visualizer.plot_attn_scores(train_attn_w.numpy(),[F.softmax(i.lam).detach().cpu().numpy() for i in struct_model.module_list])
            #visualizer.plot_attn_scores(train_attn_w.numpy(),None)
        # plot losses
        visualizer.plot_filter_weights([(i.filter_weights).detach().cpu().numpy() for i in struct_model.module_list])
        visualizer.plot_loss_curve((tot_loss.T).detach().cpu().numpy())
        visualizer.plot_recons(recons_a,recons_label)
        visualizer.plot_filters(res_a_all)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora_triple_sc_all', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--datadir', default=None, type=str, help='directory to load data from')
    parser.add_argument('--datasave', default=False, type=bool, help='whether to save data')
    parser.add_argument('--dataload', default=False, type=bool, help='whether to load data')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=15, help='Training epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--scales', default='4', type=int, help='number of scales for multi-scale analysis')
    parser.add_argument('--batch_size', type=int, default=32, help='number of edges to use for batching (default: 32)')
    parser.add_argument('--recons', default='struct', type=str, help='reconstruct features or structure')
    parser.add_argument('--cutoff',default=50, type=float, help='validation cutoff')
    parser.add_argument('--d', default=5, type=int, help='d parameter for BWGNN filters')
    parser.add_argument('--model', default='multi-scale-amnet', type=str, help='encoder to use')
    parser.add_argument('--sample_train', default=False, type=bool, help="whether or not to sample edges in training")
    parser.add_argument('--sample_test', default=False, type=bool, help="whether or not to sample edges in testing")
    parser.add_argument('--batch_type', default='edge', type=str, help="node or edge sampling for batching")
    parser.add_argument('--debug', default=False, type=bool, help="if true, prints intermediate output")
    parser.add_argument('--label_type', default='single', type=str, help="if true, prints intermediate output")
    parser.add_argument('--vis_filters', default=False, type=bool, help="if true, visualize model filters (spectral only)")
    parser.add_argument('--exp_name', default=None, type=str, help='experiment name for visualization')
    args = parser.parse_args()

    graph_anomaly_detection(args)
