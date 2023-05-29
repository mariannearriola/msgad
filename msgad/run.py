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
from torch.utils.tensorboard import SummaryWriter

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
    print('sample train',exp_params['MODEL']['SAMPLE_TRAIN'],'sample test',exp_params['MODEL']['SAMPLE_TEST'], 'epochs',exp_params['MODEL']['EPOCH'],'saving?', exp_params['DATASET']['DATASAVE'], 'loading?', exp_params['DATASET']['DATALOAD'])
    
    # intialize model (on given device)
    adj = adj.to(exp_params['DEVICE']) ; feats = feats#.to(exp_params['DEVICE'])
    struct_model,feat_model=None,None
    struct_model,params = init_model(feats.size(1),exp_params)
    torch.cuda.synchronize()
    tb = SummaryWriter()

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
        train_attn_w = torch.zeros((int(exp_params['MODEL']['EPOCH']),3,adj.number_of_nodes(),5)).to(torch.float64)#.to(exp_params['DEVICE'])

    if exp_params['VIS_FILTERS'] or exp_params['VIS_LOSS']:
        visualizer = Visualizer(adj,feats,exp_params,sc_label,norms,anoms)
    else:
        visualizer = None

    print(dataloader.__len__(),'batches')
    
    if 'tfinance' in exp_params['EXP']:
        if exp_params['VIS_CONCENTRATION'] == True:
            nx_graph,node_ids = dgl_to_nx(adj)
            #import ipdb ; ipdb.set_trace()
            nx_adj = nx.from_numpy_matrix(nx.adjacency_matrix(nx_graph))
            # as a result, there may be few to no single node anomalies
            nodes = list(max(nx.connected_components(nx_graph), key=len))
            anom_nodes = np.intersect1d(nodes,anoms,return_indices=True)[0]
            
            nx_graph = nx.subgraph(nx_graph, nodes)
            nx_adj = nx.from_numpy_matrix(nx.adjacency_matrix(nx_graph))
            nx_dict = {k:v for k,v in zip(nodes,np.arange(len(nodes)))}
            nx_graph=nx.relabel_nodes(nx_graph,nx_dict)

            #madan_node_dict = {k:v for k,v in zip(nodes,np.arange(len(list(nx_graph.nodes))))}
            #feats_ = feats[nodes]
            #feats_ = feats
            madan = md.Madan(nx_adj, exp_params, attributes=feats.numpy())
            madan.anomalous_nodes = np.vectorize(nx_dict.get)(anom_nodes)
            madan.sc_label = batch_sc_label ; madan.organize_ms_labels(nodes,nx_dict)
            madan.find_sc_label = {}
            for k,v in madan.sc_label.items():
                madan.find_sc_label[v] = k

            #import ipdb ; ipdb.set_trace()
            #madan.anomalous_nodes = np.vectorize(madan_node_dict.get)(anoms)
            #madan.anomalous_nodes = np.intersect1d(anom_nodes,nx_graph.nodes,return_indices=True)[2]
            print('scanning..')
            if 'madan' in exp_params['MODEL']['NAME']:
                #if 'cora' in exp_params['DATASET']['NAME']:
                #    time_scales = np.array([2,15,100])
                #else:
                time_scales   =   np.linspace(100.,200.,5)
                #madan.scanning_relevant_context(time_scales, n_jobs=1)
                print('scanning context times')
                madan.scanning_relevant_context_time(time_scales)
            else:
                time_scales   =   np.linspace(0.,len(lbl)+1,num=len(lbl))
                mats = [np.array(nx.adjacency_matrix(dgl_to_nx(i.subgraph(nodes))[0]).todense()).astype(np.float64) for i in lbl]
                madan.scanning_relevant_context(time_scales, mats=mats, n_jobs=1)
                print('scanning context times')
                madan.scanning_relevant_context_time(time_scales, mats= mats)
            import ipdb ; ipdb.set_trace()


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
            #if struct_model:
            if True:
                vis = True if (epoch == 0 and iter == 0 and exp_params['VIS_FILTERS'] == True) else False

                if not exp_params['DATASET']['DATALOAD']:
                    if exp_params['VIS_FILTERS']:
                        lg = LabelGenerator(g_batch,g_batch.ndata['feature'],vis,'train',batch_sc_label,exp_params,visualizer)
                    else:
                        lg = LabelGenerator(g_batch,g_batch.ndata['feature'],vis,'train',batch_sc_label,exp_params,None)
                    model_lbl=lg.construct_labels()
                    pos_edges = torch.stack(list(model_lbl[0].edges())).T
                    pos_edges = pos_edges[np.random.randint(0,pos_edges.shape[0],pos_edges.shape[0])]
                    neg_edges = dgl.sampling.global_uniform_negative_sampling(model_lbl[0],pos_edges.shape[0])
                    neg_edges = torch.stack(list(neg_edges)).T
                    #edge_ids_samp = torch.vstack((pos_edge_samp,torch.stack(list(neg_edge_samp)).T))
            # save batch info/labels
            if exp_params['DATASET']['DATASAVE']:
                lbl = model_lbl
            if exp_params['DATASET']['DATASAVE'] and 'weibo' not in exp_params['DATASET']['NAME']:
                dataloading.save_batch(loaded_input,lbl,iter,'train')
                continue
            for k in range(len(loaded_input)): del loaded_input[0]
            torch.cuda.empty_cache() ; gc.collect()
            if struct_model:
                optimizer.zero_grad()
            #if struct_model:
            if True:
                if exp_params['VIS_CONCENTRATION'] == True and 'madan' in exp_params['MODEL']['NAME']:
                    nx_graph,node_ids = dgl_to_nx(g_batch)
                    #import ipdb ; ipdb.set_trace()
                    nx_adj = nx.from_numpy_matrix(nx.adjacency_matrix(nx_graph))
                    nodes = list(max(nx.connected_components(nx_graph), key=len))
                    anom_nodes = np.intersect1d(nodes,in_nodes[anoms].detach().cpu().numpy(),return_indices=True)[0]
                    
                    nx_graph = nx.subgraph(nx_graph, nodes)
                    nx_adj = nx.from_numpy_matrix(nx.adjacency_matrix(nx_graph))
                    nx_dict = {k:v for k,v in zip(nodes,np.arange(len(nodes)))}
                    nx_graph=nx.relabel_nodes(nx_graph,nx_dict)

                    #madan_node_dict = {k:v for k,v in zip(nodes,np.arange(len(list(nx_graph.nodes))))}
                    #feats_ = feats[nodes]
                    #feats_ = feats
                    #import ipdb ; ipdb.set_trace()

                    madan = md.Madan(nx_adj, exp_params, attributes=g_batch.ndata['feature'].detach().cpu().numpy())
                    madan.anomalous_nodes = np.vectorize(nx_dict.get)(anom_nodes)
                    madan.sc_label = batch_sc_label ; madan.organize_ms_labels(nodes,nx_dict)
                    
                    # sc label and anoms nodes? interesect
                    #import ipdb ; ipdb.set_trace()
                    #madan.anomalous_nodes = np.vectorize(madan_node_dict.get)(anoms)
                    #madan.anomalous_nodes = np.intersect1d(anom_nodes,nx_graph.nodes,return_indices=True)[2]
                    print('scanning..')
                    if 'single' in exp_params['DATASET']['LABEL_TYPE']:
                        #if 'cora' in exp_params['DATASET']['NAME']:
                        #    time_scales = np.array([2,15,100])
                        #else:
                        time_scales   =   np.linspace(1.,200.,20)
                        #madan.scanning_relevant_context(time_scales, n_jobs=1)
                        print('scanning context times')
                        madan.scanning_relevant_context_time(time_scales)
                    else:
                        time_scales   =   np.linspace(0.,len(lbl)+1,num=len(lbl))
                        mats = [np.array(nx.adjacency_matrix(dgl_to_nx(i.subgraph(nodes))[0]).todense()).astype(np.float64) for i in lbl]
                        madan.scanning_relevant_context(time_scales, mats=mats, n_jobs=1)
                        print('scanning context times')
                        madan.scanning_relevant_context_time(time_scales, mats= mats)
                    #import ipdb ; ipdb.set_trace()
                recons_label = g_batch if lbl is None else collect_recons_label(lbl,exp_params['DEVICE'])
                edges, feats, graph_ = process_graph(g_batch)
                A_hat,X_hat,res_a = struct_model(edges,feats,vis=vis,vis_name='epoch1')
                edge_ids = torch.vstack((pos_edges,neg_edges)).to(pos_edges.device)
                A_hat = A_hat[:,edge_ids[:,0],edge_ids[:,1]]
                #A_hat,X_hat,res_a = struct_model(g_batch,last_batch_node,pos_edges,neg_edges,batch_sc_label,recons_label,vis=vis,vis_name='epoch1')
                tb.add_graph(struct_model,(edges,feats))
                del res_a
                torch.cuda.empty_cache()
                gc.collect()
                '''
                diffs = []
                ranges =[]
                for ind_,A_ in enumerate(A_hat):
                    ranges.append(A_.max()-A_.min())
                    if ind_ == len(A_hat)-1:
                        break
                    diffs.append((A_-A_hat[ind_+1]).max())

                print("diffs",diffs)
                print("ranges",ranges)
                '''
            check_gpu_usage('collecting recons label')
            check_gpu_usage('recons label collected, starting loss')
            loss, struct_loss, feat_cost = loss_func(recons_label, g_batch.ndata['feature'], A_hat, X_hat, pos_edges, neg_edges, sample=exp_params['MODEL']['SAMPLE_TEST'], recons=exp_params['MODEL']['RECONS'],alpha=exp_params['MODEL']['ALPHA'])
            check_gpu_usage('loss collected')
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
            del A_hat
            #for k in range(len(model_lbl)): del model_lbl[0]
            del X_hat
            if feat_loss is not None: del feat_loss
            
            torch.cuda.empty_cache()
            gc.collect()
            check_gpu_usage('about to backward')
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
        if exp_params['DATASET']['DATASAVE']: continue 
        
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
                # epoch x 3 x nodes x num filters
                train_attn_w[epoch,:,in_nodes,:]=struct_model.attn_weights.detach().cpu()#torch.unsqueeze(struct_model.attn_weights,0).detach().cpu()

    #model = torch.load('best_model.pt')
    # accumulate node-wise anomaly scores via model evaluation
    if exp_params['MODEL']['NAME'] not in ['madan','gcad']:
        if struct_model: struct_model.eval()
    #if 'elliptic' in exp_params['DATASET']['NAME']:
    #    struct_model.dataload = False
    
    edges=adj.edges('eid')
    #dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,device=exp_params['DEVICE'])
    if exp_params['MODEL']['NAME'] != 'gcad' and exp_params['DATASET']['DATASAVE'] == False:
        struct_scores, feat_scores = torch.zeros(3,adj.number_of_nodes()).to(exp_params['DEVICE']),torch.zeros(exp_params['MODEL']['D'],adj.number_of_nodes()).to(exp_params['DEVICE'])
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
                if exp_params['VIS_FILTERS']:
                    lg = LabelGenerator(g_batch,g_batch.ndata['feature'],vis,'train',batch_sc_label,exp_params,visualizer)
                else:
                    lg = LabelGenerator(g_batch,g_batch.ndata['feature'],vis,'train',batch_sc_label,exp_params,None)
                model_lbl=lg.construct_labels()
                pos_edges = (torch.stack(list(model_lbl[0].edges())).T)
                pos_edges = pos_edges[np.random.randint(0,pos_edges.shape[0],pos_edges.shape[0])].to(exp_params['DEVICE'])
                neg_edges = (dgl.sampling.global_uniform_negative_sampling(model_lbl[0],pos_edges.shape[0]))
                neg_edges = (torch.stack(list(neg_edges)).T).to(exp_params['DEVICE'])
                edge_ids = torch.vstack((pos_edges,neg_edges))
                #edge_ids_samp = torch.vstack((pos_edge_samp,torch.stack(list(neg_edge_samp)).T))
        if exp_params['DATASET']['DATASAVE']:# or 'elliptic' in exp_params['MODEL']['NAME']:
                lbl = model_lbl
         # save batch info
        if exp_params['DATASET']['DATASAVE']:# or 'elliptic' in exp_params['DATASET']['NAME']:
            dataloading.save_batch(loaded_input,lbl,iter,'test')
            continue
    
        recons_label = g_batch if lbl is None else collect_recons_label(lbl,exp_params['DEVICE'])

        # run evaluation
        if struct_model and exp_params['MODEL']['NAME'] != 'gcad':
            #node_dict = {k.item():v.item() for k,v in zip(in_nodes[g_batch.dstnodes()],np.arange(len(list(g_batch.dstnodes()))))}
            #rev_node_dict = {v: k for k, v in node_dict.items()}
            A_hat,X_hat,res_a = struct_model(g_batch,last_batch_node,pos_edges,neg_edges,batch_sc_label,recons_label,vis=vis,vis_name='test')

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


        loss, struct_loss, feat_cost = loss_func(recons_label, g_batch.ndata['feature'], A_hat, X_hat, pos_edges, neg_edges, sample=exp_params['MODEL']['SAMPLE_TEST'], recons=exp_params['MODEL']['RECONS'],alpha=exp_params['MODEL']['ALPHA'])
        if exp_params['MODEL']['NAME'] == 'gradate':
            loss = A_hat[0]

        if exp_params['MODEL']['NAME'] in ['gradate']:
            loss = A_hat[0]
            struct_loss = [A_hat[1]]
            if exp_params['MODEL']['SAMPLE_TEST']:
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
    a_clf = anom_classifier(exp_params)
    print('structure scores')
    if exp_params['MODEL']['RECONS'] == 'both':
        print('struct scores')
        a_clf = anom_classifier(exp_params)
        a_clf.calc_prec(None, [np.mean(edge_anom_mats[0],axis=1)], truth, sc_label, train_attn_w[-1].sum(axis=1).numpy(), cluster=False, input_scores=True)
        print('feat scores')
        a_clf.calc_prec(None, [np.mean(node_anom_mats[0],axis=1)], truth, sc_label, train_attn_w[-1].sum(axis=1).numpy(), cluster=False, input_scores=True)
        print('combined scores')
        a_clf.calc_prec(None, loss.detach().cpu().numpy(), truth, sc_label, train_attn_w[-1].sum(axis=1).numpy(), cluster=False, input_scores=True)
    else:
        if exp_params['DATASET']['BATCH_TYPE'] == 'node':
            if -1 in node_anom_mats[0]:
                raise('node not sampled')
            a_clf.calc_prec(None,node_anom_mats, truth, sc_label, cluster=False, input_scores=True)
        else:
            if 'multi-scale' in exp_params['MODEL']['NAME']:
                #lams = np.array([(i.filter_weights).detach().cpu().numpy() for i in struct_model.module_list])
                #tiled=np.tile(lams,((train_attn_w.shape[-1],1,1))).T
                #tiled=np.moveaxis(tiled, 0, 1)
                #attns = (tiled*train_attn_w[-1].numpy()).sum(axis=1)
                attns = train_attn_w[-1].numpy().sum(axis=1)
            else:
                attns = None
            if not exp_params['MODEL']['SAMPLE_TEST']:
                a_clf.calc_prec(adj.adjacency_matrix().to_dense(), struct_loss.detach().cpu().numpy(), truth, sc_label, attns, cluster=False, input_scores=True)
            else:
                a_clf.calc_prec(adj.adjacency_matrix().to_dense(), edge_anom_mats, truth, sc_label, attns, cluster=False)
    if visualizer is not None:
        #if exp_params['VIS_FILTERS'] == True:
        #visualizer.plot_final_filters([np.array([i.weight.detach().cpu().numpy() for i in j.filters]) for j in struct_model.module_list],[F.softmax(i.lam).detach().cpu().numpy() for i in struct_model.module_list])
        #visualizer.plot_final_filters([np.array([i.weight.detach().cpu().numpy() for i in j.filters]) for j in struct_model.module_list])#,[(i.lam).detach().cpu().numpy() for i in struct_model.module_list])
        if 'multi-scale-amnet' == exp_params['MODEL']['NAME'] and struct_model.attn_weights != None:
            #visualizer.plot_attn_scores(train_attn_w.numpy(),[(i.filter_weights).detach().cpu().numpy() for i in struct_model.module_list])

            visualizer.plot_attn_scores(train_attn_w.numpy(),None,edge_anom_mats)
        # plot losses
        #visualizer.plot_filter_weights([(i.filter_weights).detach().cpu().numpy() for i in struct_model.module_list])
        visualizer.plot_loss_curve((tot_loss.T).detach().cpu().numpy())
        #visualizer.plot_recons(recons_a,recons_label)
        #visualizer.plot_filters(res_a_all)
    tb.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasave', default=False, type=bool, help='whether to save data')
    parser.add_argument('--dataload', default=False, type=bool, help='whether to load data')
    parser.add_argument('--epoch', default=None, help='Training epoch')
    parser.add_argument('--config', default='cora', type=str, help='path to config file')
    args = parser.parse_args()
    exp_params = prep_args(args)

    graph_anomaly_detection(exp_params)
