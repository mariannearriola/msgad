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

def graph_anomaly_detection(exp_params):
    """Pipeline for autoencoder-based graph anomaly detection"""
    # load data
    seed_everything()
    dataloading = DataLoading(exp_params)

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
    print('sample train',exp_params['MODEL']['SAMPLE_TRAIN'],'sample test',exp_params['MODEL']['SAMPLE_TEST'], 'epochs',exp_params['EPOCH'],'saving?', exp_params['DATASET']['DATASAVE'], 'loading?', exp_params['DATASET']['DATALOAD'])
    
    # intialize model (on given device)
    adj = adj.to(exp_params['DEVICE']) ; feats = feats#.to(exp_params['DEVICE'])
    struct_model,feat_model=None,None
    struct_model,params = init_model(feats.size(1),exp_params)

    #print(torch.cuda.memory_allocated()/torch.cuda.memory_reserved())

    if not exp_params['MODEL']['NAME'] in ['gcad','madan']:
        optimizer = torch.optim.Adam(struct_model.parameters(), lr = float(exp_params['MODEL']['LR']))

    # begin model training
    #best_loss = torch.tensor(float('inf')).to(exp_params['DEVICE'])    
    A_hat, X_hat = None,None
    struct_loss,feat_loss=None,None
    res_a = None
    seconds = time.time()
    # epoch x 3 x num filters x nodes
    if 'multi-scale-amnet' in exp_params['MODEL']['NAME']:
        train_attn_w = torch.zeros((exp_params['MODEL']['EPOCH'],3,struct_model.module_list[0].filter_num,adj.number_of_nodes())).to(torch.float64)#.to(exp_params['DEVICE'])

    if exp_params['VIS_FILTERS'] == True:
        visualizer = Visualizer(adj,feats,exp_params,sc_label,norms,anoms)
    else:
        visualizer = None

    print(dataloader.__len__(),'batches')
    
    for epoch in range(int(exp_params['MODEL']['EPOCH'])):
        epoch_l = 0
        if exp_params['MODEL']['NAME'] == 'gcad': break
        iter=0
        for data_ind in dataloader:
            if (exp_params['DATASET']['DATADIR'] is not None and exp_params['DATASET']['DATALOAD']):
                loaded_input,lbl=dataloading.load_batch(data_ind,'train')
            else:
                loaded_input = data_ind
            if exp_params['DATASET']['BATCH_TYPE'] == 'node':
                all_nodes,in_nodes,g_batch=loaded_input
            else:
                in_nodes, pos_edges, neg_edges, g_batch, last_batch_node = dataloading.get_edge_batch(loaded_input)
                if 'cora' not in exp_params['DATASET']['NAME']: print('size of batch',g_batch.num_dst_nodes(),'nodes')

            #node_dict = {k.item():v.item() for k,v in zip(in_nodes[g_batch.dstnodes()],np.arange(len(list(g_batch.dstnodes()))))}
            batch_sc_label = dataloading.get_batch_sc_label(in_nodes.detach().cpu(),sc_label,g_batch)
            #rev_node_dict = {v: k for k, v in node_dict.items()}

            # load labels
            if struct_model:
                vis = True if (epoch == 0 and iter == 0 and exp_params['VIS_FILTERS'] == True) else False

                if not exp_params['DATASET']['DATALOAD']:
                    lg = LabelGenerator(g_batch,g_batch.ndata['feature'],vis,'train',batch_sc_label,exp_params,visualizer)
                    model_lbl=lg.construct_labels()
                    pos_edges = torch.stack(list(model_lbl[0].edges())).T
                    pos_edges = pos_edges[np.random.randint(0,pos_edges.shape[0],pos_edges.shape[0])]
                    neg_edges = dgl.sampling.global_uniform_negative_sampling(model_lbl[0],pos_edges.shape[0])
                    neg_edges = torch.stack(list(neg_edges)).T
                    #edge_ids_samp = torch.vstack((pos_edge_samp,torch.stack(list(neg_edge_samp)).T))
            # save batch info/labels
            if exp_params['DATASET']['DATASAVE']:
                lbl = model_lbl
            if exp_params['DATASET']['DATASAVE']: dataloading.save_batch(loaded_input,lbl,iter,'train')
            
            if exp_params['MODEL']['NAME'] == 'madan':
                if exp_params['MODEL']['DEBUG']:
                    if iter % 100 == 0:
                        print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%')
                #if iter == 0:
                #    adj = adj.adjacency_matrix()
                #    adj = adj.sparse_resize_((adj.size(0), adj.size(0)), adj.sparse_dim(), adj.dense_dim())
                #import ipdb ; ipdb.set_trace()
                nx_graph,node_ids = dgl_to_nx(adj)
                nodes = list(max(nx.connected_components(nx_graph), key=len))
                #node_dict = {k.item():v for k,v in zip(list(nx_graph.nodes))}
                #nx_anoms = np.vectorize(node_dict.get)(anoms)

                anom_nodes = np.intersect1d(node_ids[nodes],anoms)
                nx_graph = nx.subgraph(nx_graph, nodes)
                nx_adj = nx.from_numpy_matrix(nx.adjacency_matrix(nx_graph))
                #madan_node_dict = {k:v for k,v in zip(nodes,np.arange(len(list(nx_graph.nodes))))}
                #feats_ = feats[nodes]
                feats_ = feats
                madan = md.Madan(nx_adj, attributes=feats_)
                #madan.anomalous_nodes = np.vectorize(madan_node_dict.get)(anoms)

                madan.anomalous_nodes = np.intersect1d(anom_nodes,nx_graph.nodes,return_indices=True)[2]
                time_scales   =   np.linspace(0.,200.,400)
                #import ipdb ; ipdb.set_trace()
                #time_scales = np.array([150])
                #import ipdb ; ipdb.set_trace()
                #import ipdb ; ipdb.set_trace()
                madan.scanning_relevant_context(time_scales, n_jobs=15)
                print('scanning context tiems')
                madan.scanning_relevant_context_time(time_scales)
                #import ipdb ; ipdb.set_trace()
                #madan.anomalous_nodes=[node_dict[j] for j in np.intersect1d(anoms,np.array(list(node_dict.keys()))).tolist()]
                #madan.anomalous_nodes=[node_dict[j] for j in anoms]
                #batch_anoms = batch_sc_label
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
            if struct_model:
                nx_graph,node_ids = dgl_to_nx(g_batch)
                nx_adj = nx.from_numpy_matrix(nx.adjacency_matrix(nx_graph))
                nodes = list(max(nx.connected_components(nx_graph), key=len))
                anom_nodes = np.intersect1d(in_nodes[nodes].detach().cpu().numpy(),anoms)
                '''
                nx_graph = nx.subgraph(nx_graph, nodes)
                nx_adj = nx.from_numpy_matrix(nx.adjacency_matrix(nx_graph))
                #madan_node_dict = {k:v for k,v in zip(nodes,np.arange(len(list(nx_graph.nodes))))}
                #feats_ = feats[nodes]
                feats_ = feats
                '''
                madan = md.Madan(nx_adj, exp_params['EXP'], attributes=g_batch.ndata['feature'].detach().cpu().numpy())
                madan.anomalous_nodes = anom_nodes
                #madan.anomalous_nodes = np.vectorize(madan_node_dict.get)(anoms)

                #madan.anomalous_nodes = np.intersect1d(anom_nodes,nx_graph.nodes,return_indices=True)[2]
                time_scales   =   np.linspace(0.,10.)
                mats = [np.array(nx.adjacency_matrix(dgl_to_nx(i)[0]).todense()).astype(np.float64) for i in lbl]
                madan.scanning_relevant_context(time_scales, mats=mats, n_jobs=1)
                import ipdb ; ipdb.set_trace()
                print('scanning context tiems')
                madan.scanning_relevant_context_time(time_scales, mats= mats)

                A_hat,X_hat,res_a = struct_model(g_batch,last_batch_node,pos_edges,neg_edges,batch_sc_label,vis=vis,vis_name='epoch1')
                diffs = []
                ranges =[]
                for ind_,A_ in enumerate(A_hat):
                    ranges.append(A_.max()-A_.min())
                    if ind_ == len(A_hat)-1:
                        break
                    diffs.append((A_-A_hat[ind_+1]).max())

                print("diffs",diffs)
                print("ranges",ranges)
                    
            recons_label = g_batch if lbl is None else collect_recons_label(lbl,exp_params['DEVICE'])

            loss, struct_loss, feat_cost = loss_func(recons_label, g_batch.ndata['feature'], A_hat, X_hat, pos_edges, neg_edges, sample=exp_params['DATASET']['SAMPLE_TEST'], recons=exp_params['MODEL']['RECONS'],alpha=exp_params['MODEL']['ALPHA'])
            
            #(new_scores.max(0).values-new_scores.mean(0))/(new_scores.max(0).values-new_scores.mean(0)).sum()
            if exp_params['MODEL']['NAME'] == 'gradate':
                loss = A_hat[0]
                
            num_reg = 0.1
            '''
            if 'multi-scale' in exp_params['MODEL']['NAME']:
                for sc in range(loss.shape[0]):
                    final_att = struct_model.module_list[sc].att[:,0].tile(128,1).T * struct_model.module_list[sc].filter_weights[0]#*lams[0]#*self.filter_weights[0]).tile(128,1).T#* new_scores[:,0].tile(128,1).T * self.filter_weights[0]# score[:,0] * lams[0].tile(128,1).T#score.tile(128,1).T#[:, 0]
                    for i in range(1, len(struct_model.module_list[sc].filter_weights)):
                        final_att += struct_model.module_list[sc].att[:,i].tile(128,1).T * struct_model.module_list[sc].filter_weights[i]
                    reg_term = final_att
                    if reg_term.max() == reg_term.min():
                        loss[sc] *= (num_reg/(0.0001))*(1/2)
                    else:
                        loss[sc] *= (num_reg/(reg_term.max()-reg_term.min()))**(1/2)
            '''
            l = torch.sum(loss) if 'multi-scale' in exp_params['MODEL']['NAME'] else torch.mean(loss)
            '''
            if l < best_loss:
                best_loss = dl
                torch.save(model,'best_model.pt')
            '''
            epoch_l = loss.unsqueeze(0) if iter == 0 else torch.cat((epoch_l,l.unsqueeze(0)))


            if exp_params['MODEL']['DEBUG']:
                if iter % 100 == 0:
                    print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=', round(l.item(),3))
                
            iter += 1
            if 'cora' not in exp_params['DATASET']['NAME']: print('iter',iter)
            
            del g_batch, batch_sc_label, struct_loss, pos_edges, neg_edges
            for k in range(len(loaded_input)): del loaded_input[0]
            for k in range(len(A_hat)): del A_hat[0]
            if res_a:
                for k in range(len(res_a)): del res_a[0]
            #for k in range(len(model_lbl)): del model_lbl[0]
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
        if exp_params['MODEL']['NAME'] != 'madan' and 'multi-scale' in exp_params['MODEL']['NAME']:
            print("Epoch:", '%04d' % (epoch), "train_loss=", round(torch.sum(loss),3), "losses=",torch.round(loss,decimals=4).detach().cpu())
        else:
             print("Epoch:", '%04d' % (epoch), "train_loss=", round(epoch_l.item(),3))
        #print('avg loss',torch.mean(epoch_l/dataloader.__len__()))
        '''
        
        print('epoch done',epoch,loss)
        if 'weibo' in exp_params['DATASET']['NAME']:
            print(torch.cuda.memory_allocated()/torch.cuda.memory_reserved())

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
    if exp_params['MODEL']['NAME'] not in ['madan','gcad']:
        if struct_model: struct_model.eval()
    #if 'elliptic' in exp_params['DATASET']['NAME']:
    #    struct_model.dataload = False
    
    edges=adj.edges('eid')
    #dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,device=exp_params['DEVICE'])
    if exp_params['MODEL']['NAME'] != 'gcad':
        struct_scores, feat_scores = torch.zeros(len(A_hat),adj.number_of_nodes()).to(exp_params['DEVICE']),torch.zeros(exp_params['MODEL']['D'],adj.number_of_nodes()).to(exp_params['DEVICE'])
    iter = 0


    edge_anom_mats,node_anom_mats,recons_a,res_a_all = init_recons_agg(adj.number_of_nodes(),adj.ndata['feature'].shape[1],exp_params)

    all_samps = []
    if (exp_params['DATASET']['DATADIR'] is not None and exp_params['DATASET']['DATALOAD']):# or 'elliptic' not in exp_params['MODEL']['NAME']:
        pass
        #random.shuffle(dataloader)
    else:
        dataloader = dataloading.fetch_dataloader(adj, edges)

    #import ipdb ; ipdb.set_trace()
    for loaded_input in dataloader:
        vis = True if (exp_params['VIS_FILTERS'] == True and iter == 0) else False
        if exp_params['DATASET']['DATALOAD']:# or 'elliptic' not in exp_params['MODEL']['NAME']:
            loaded_input,lbl=dataloading.load_batch(loaded_input,'test')
        # collect input
        if exp_params['DATASET']['BATCH_TYPE'] == 'node':
            all_nodes,in_nodes,g_batch= loaded_input
            pos_edges,neg_edges=None,None
        else:
            in_nodes, pos_edges, neg_edges, g_batch, last_batch_node = dataloading.get_edge_batch(loaded_input)
            edge_ids = torch.vstack((pos_edges,neg_edges))
        
        # load data/labels
        if struct_model and exp_params['MODEL']['NAME'] != 'gcad':
            batch_sc_label = dataloading.get_batch_sc_label(in_nodes.detach().cpu(),sc_label,g_batch)
            if not exp_params['DATASET']['DATALOAD']:
                lg = LabelGenerator(g_batch,g_batch.ndata['feature'],vis,'test',batch_sc_label,exp_params,visualizer)
                model_lbl=lg.construct_labels()
                pos_edges = torch.stack(list(model_lbl[0].edges())).T
                pos_edges = pos_edges[np.random.randint(0,pos_edges.shape[0],pos_edges.shape[0])]
                neg_edges = dgl.sampling.global_uniform_negative_sampling(model_lbl[0],pos_edges.shape[0])
                neg_edges = torch.stack(list(neg_edges)).T
                edge_ids = torch.vstack((pos_edges,neg_edges))
                #edge_ids_samp = torch.vstack((pos_edge_samp,torch.stack(list(neg_edge_samp)).T))
        if exp_params['DATASET']['DATASAVE']:# or 'elliptic' in exp_params['MODEL']['NAME']:
                lbl = model_lbl
         # save batch info
        if exp_params['DATASET']['DATASAVE']:# or 'elliptic' in exp_params['DATASET']['NAME']:
            dataloading.save_batch(loaded_input,lbl,iter,'test')
    
        # run evaluation
        if struct_model and exp_params['MODEL']['NAME'] != 'gcad':
            #node_dict = {k.item():v.item() for k,v in zip(in_nodes[g_batch.dstnodes()],np.arange(len(list(g_batch.dstnodes()))))}
            #rev_node_dict = {v: k for k, v in node_dict.items()}
            A_hat,X_hat,res_a = struct_model(g_batch,last_batch_node,pos_edges,neg_edges,batch_sc_label,vis=vis,vis_name='test')

        if exp_params['MODEL']['NAME'] == 'gcad':
            adj_ = g_batch.adjacency_matrix() 
            adj_=adj_.sparse_resize_((g_batch.num_src_nodes(),g_batch.num_src_nodes()), adj_.sparse_dim(),adj_.dense_dim())
            feat = g_batch.ndata['feature']
            if type(feat) == dict:
                feat = feat['_N']
            struct_loss = struct_model.fit_transform(adj_,feat)
            struct_loss = np.array((struct_loss.T)[g_batch.dstnodes().detach().cpu().numpy()].T)

        # collect anomaly scores
        edge_ids_,node_ids_ = torch.cat((pos_edges,neg_edges)).T.detach().cpu().numpy(),in_nodes[:g_batch.num_dst_nodes()]

        recons_label = g_batch if lbl is None else collect_recons_label(lbl,exp_params['DEVICE'])

        loss, struct_loss, feat_cost = loss_func(recons_label, g_batch.ndata['feature'], A_hat, X_hat, pos_edges, neg_edges, sample=exp_params['DATASET']['SAMPLE_TEST'], recons=exp_params['MODEL']['RECONS'],alpha=exp_params['MODEL']['ALPHA'])
        if exp_params['MODEL']['NAME'] == 'gradate':
            loss = A_hat[0]

        if exp_params['MODEL']['NAME'] in ['gradate']:
            loss = A_hat[0]
            struct_loss = [A_hat[1]]
            if exp_params['DATASET']['SAMPLE_TEST']:
                l = torch.sum(loss)
            else:
                l = torch.mean(loss)
        elif exp_params['MODEL']['NAME'] in ['gcad']:
            l = np.mean(struct_loss)
        node_anom_mats,edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,struct_loss,feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats,recons_a,res_a_all,exp_params)

        #if iter % 100 == 0 and exp_params['MODEL']['DEBUG']:
        #    print(f'Batch: {round(iter/dataloader.__len__()*100, 3)}%', 'train_loss=',round(l.item(),3))
        iter += 1
    
    
    # anomaly detection with final scores
    a_clf = anom_classifier()
    print('structure scores')
    if exp_params['MODEL']['RECONS'] == 'both':
        print('struct scores')
        a_clf = anom_classifier()
        a_clf.calc_prec(None, [np.mean(edge_anom_mats[0],axis=1)], truth, sc_label, train_attn_w[-1].sum(axis=1).numpy(), exp_params, cluster=False, input_scores=True)
        print('feat scores')
        a_clf.calc_prec(None, [np.mean(node_anom_mats[0],axis=1)], truth, sc_label, train_attn_w[-1].sum(axis=1).numpy(), exp_params, cluster=False, input_scores=True)
        print('combined scores')
        a_clf.calc_prec(None, loss.detach().cpu().numpy(), truth, sc_label, train_attn_w[-1].sum(axis=1).numpy(), exp_params, cluster=False, input_scores=True)
    else:
        if exp_params['DATASET']['BATCH_TYPE'] == 'node':
            if -1 in node_anom_mats[0]:
                raise('node not sampled')
            a_clf.calc_prec(None,node_anom_mats, truth, sc_label, exp_params, cluster=False, input_scores=True)
        else:
            if 'multi-scale' in exp_params['MODEL']['NAME']:
                #lams = np.array([(i.filter_weights).detach().cpu().numpy() for i in struct_model.module_list])
                #tiled=np.tile(lams,((train_attn_w.shape[-1],1,1))).T
                #tiled=np.moveaxis(tiled, 0, 1)
                #attns = (tiled*train_attn_w[-1].numpy()).sum(axis=1)
                attns = train_attn_w[-1].numpy().sum(axis=1)
            else:
                attns = None
            if not exp_params['DATASET']['SAMPLE_TEST']:
                a_clf.calc_prec(adj.adjacency_matrix().to_dense(), struct_loss.detach().cpu().numpy(), truth, sc_label, attns, exp_params, cluster=False, input_scores=True)
            else:
                a_clf.calc_prec(adj.adjacency_matrix().to_dense(), edge_anom_mats, truth, sc_label, attns,exp_params, cluster=False)

    if exp_params['VIS_FILTERS'] == True:
        #visualizer.plot_final_filters([np.array([i.weight.detach().cpu().numpy() for i in j.filters]) for j in struct_model.module_list],[F.softmax(i.lam).detach().cpu().numpy() for i in struct_model.module_list])
        #visualizer.plot_final_filters([np.array([i.weight.detach().cpu().numpy() for i in j.filters]) for j in struct_model.module_list])#,[(i.lam).detach().cpu().numpy() for i in struct_model.module_list])
        if 'multi-scale-amnet' == exp_params['MODEL']['NAME'] and struct_model.attn_weights != None:
            #visualizer.plot_attn_scores(train_attn_w.numpy(),[(i.filter_weights).detach().cpu().numpy() for i in struct_model.module_list])

            visualizer.plot_attn_scores(train_attn_w.numpy(),None,edge_anom_mats)
        # plot losses
        #visualizer.plot_filter_weights([(i.filter_weights).detach().cpu().numpy() for i in struct_model.module_list])
        visualizer.plot_loss_curve((tot_loss.T).detach().cpu().numpy())
        visualizer.plot_recons(recons_a,recons_label)
        visualizer.plot_filters(res_a_all)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasave', default=False, type=bool, help='whether to save data')
    parser.add_argument('--dataload', default=False, type=bool, help='whether to load data')
    parser.add_argument('--epoch', default=None, help='Training epoch')
    parser.add_argument('--config', default='cora', type=str, help='path to config file')
    args = parser.parse_args()
    exp_params = prep_args(args)

    graph_anomaly_detection(exp_params)
