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
from madan_analysis import madan_analysis
from label_analysis import LabelAnalysis

import warnings ; warnings.filterwarnings("ignore")

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
    if 'tfinance' in exp_params['DATASET']['NAME'] and exp_params['VIS']['VIS_FILTERS'] == True:
        dataloader = None
    else:
        dataloader = dataloading.fetch_dataloader(adj, edges)
    
    # intialize model (on given device)
    adj = adj.to(exp_params['DEVICE']) ; feats = feats#.to(exp_params['DEVICE'])
    struct_model,feat_model=None,None
    struct_model,params = init_model(feats.size(1),exp_params)
    exp_name = exp_params['EXP'] ; scales = exp_params['SCALES']
    regularize = True if exp_params['MODEL']['CONTRASTIVE'] == True else False

    if 'weibo' in exp_params['DATASET']['NAME']:
        tb = SummaryWriter(log_dir=f'runs/{exp_name}')
        
    if not exp_params['MODEL']['NAME'] in ['gcad','madan']:
        optimizer = torch.optim.Adam(struct_model.parameters(), lr = float(exp_params['MODEL']['LR']))

    # begin model training
    #best_loss = torch.tensor(float('inf')).to(exp_params['DEVICE'])    
    train_attn_w,train_losses=None,None
    
    # epoch x 3 x num filters x nodes
    if 'multi-scale-amnet' in exp_params['MODEL']['NAME']:
        train_attn_w = torch.FloatTensor(int(exp_params['MODEL']['EPOCH']),scales,adj.number_of_nodes()).to(torch.float64)#int(exp_params['MODEL']['D'])).to(torch.float64)
    train_losses = torch.FloatTensor(int(exp_params['MODEL']['EPOCH']),scales,adj.number_of_nodes()).to(torch.float64)

    if exp_params['VIS']['VIS_FILTERS'] or exp_params['VIS']['VIS_LOSS']:
        visualizer = Visualizer(adj,feats,exp_params,sc_label,norms,anoms)
    else:
        visualizer = None

    if dataloader is not None:
        print(dataloader.__len__(),'batches')
    else:
        dataloader = [None]
    
    if 'tfinance' in exp_params['EXP']:
        if exp_params['VIS']['VIS_CONCENTRATION'] == True:
            madan_analysis(adj,lbl,sc_label,anoms,exp_params)

    la = LabelAnalysis(get_sc_label(sc_label),exp_params['DATASET']['NAME'])
    g = adj.to('cpu')#self.graph.device)#.to('cpu')
    g_nx = dgl_to_nx(g)[0]

    clust1,clust2,clust3=la.run_dend(g_nx,return_clusts=True)
    clusts =[clust1,clust2,clust3]

    for epoch in range(int(exp_params['MODEL']['EPOCH'])):
        epoch_l = 0
        if exp_params['MODEL']['NAME'] == 'gcad': break
        iter=0
        edge_anom_mats,node_anom_mats,recons_a,res_a_all = init_recons_agg(adj.number_of_nodes(),adj.ndata['feature'].shape[1],exp_params)
        
        for data_ind in dataloader:
            if data_ind is not None:
                if (exp_params['DATASET']['DATADIR'] is not None and exp_params['DATASET']['DATALOAD']):
                    loaded_input,lbl=dataloading.load_batch(data_ind,'train')
                else:
                    loaded_input = data_ind
                if exp_params['DATASET']['BATCH_TYPE'] == 'node':
                    all_nodes,in_nodes,g_batch=loaded_input
                else:
                    in_nodes, pos_edges, neg_edges, g_batch, last_batch_node, batch_sc_label = dataloading.get_edge_batch(loaded_input,sc_label)
                    if 'cora' not in exp_params['DATASET']['NAME']: print('size of batch',g_batch.num_dst_nodes(),'nodes')
            in_nodes = np.arange(0,in_nodes.shape[0])
            vis = True if (epoch == 0 and iter == 0 and exp_params['VIS']['VIS_FILTERS'] == True) else False

            if not exp_params['DATASET']['DATALOAD']:
                if not exp_params['VIS']['VIS_FILTERS']:
                    visualizer.sc_label = batch_sc_label
                    lg = LabelGenerator(adj,adj.ndata['feature'],vis,'train',batch_sc_label,norms,exp_params,None)
                else:
                    lg = LabelGenerator(g_batch,g_batch.ndata['feature'],vis,'train',get_sc_label(sc_label),norms,exp_params,visualizer)
                
                lbl=lg.construct_labels()
            else:
                lbl = lbl[1:]
            
            lbl_eids = lbl[0].edge_ids(pos_edges[:,0],pos_edges[:,1])
            pos_edges = torch.stack([torch.stack(i.find_edges(lbl_eids)) for i in lbl])
            neg_edges = torch.stack([torch.stack(dgl.sampling.global_uniform_negative_sampling(i,pos_edges.shape[-1])) for i in lbl])


            if exp_params['DATASET']['DATASAVE']:
                dataloading.save_batch(loaded_input,lbl,iter,'train')
                continue
            
            for k in range(len(loaded_input)): del loaded_input[0]
            torch.cuda.empty_cache() ; gc.collect()
            if struct_model:
                optimizer.zero_grad()

            if exp_params['VIS']['VIS_CONCENTRATION'] == True and 'madan' in exp_params['MODEL']['NAME']:
                madan_analysis(adj,lbl,sc_label,anoms,exp_params)
            recons_label = g_batch if lbl is None else collect_recons_label(lbl,exp_params['DEVICE'])
            
            #edges, feats, graph_ = process_graph(g_batch)
            edges, feats, graph_ = process_graph(adj)
            
            #edge_ids = torch.cat((pos_edges,neg_edges),axis=2).to(pos_edges.device)[:scales]
            # NOTE: POS EDGES AND NEG EDGES THE SAME FOR ALL SCALES
            edge_ids = torch.cat((pos_edges[0],neg_edges[0]),axis=1).unsqueeze(0).tile((scales,1,1))
            
            A_hat,X_hat,res_a = struct_model(edges,feats,edge_ids,vis=vis,vis_name='epoch1',clusts=clusts)
            
            #del res_a
            torch.cuda.empty_cache()
            gc.collect()

            check_gpu_usage('collecting recons label')
            check_gpu_usage('recons label collected, starting loss')
            
            loss, struct_loss, feat_cost,regloss,clustloss,nonclustloss = loss_func(recons_label, adj.ndata['feature'], A_hat, None, None, edge_ids, sample=exp_params['MODEL']['SAMPLE_TEST'], recons=exp_params['MODEL']['RECONS'],alpha=exp_params['MODEL']['ALPHA'], clusts=clusts, regularize=regularize)
            check_gpu_usage('loss collected')
            #(new_scores.max(0).values-new_scores.mean(0))/(new_scores.max(0).values-new_scores.mean(0)).sum()
            if exp_params['MODEL']['NAME'] == 'gradate':
                loss = A_hat[0]
            #import ipdb ; ipdb.set_trace()
            
            try:
                l = torch.sum(loss) if 'multi-scale' in exp_params['MODEL']['NAME'] else torch.mean(loss)
            except Exception as e:
                print(e)
                import ipdb ; ipdb.set_trace()
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
            
            check_gpu_usage('about to backward')
            l.backward()
            optimizer.step()

            if exp_params['VIS']['VIS_LOSS']:
                visualizer.sc_label = batch_sc_label
                edge_ids = torch.cat((pos_edges,neg_edges),axis=2).to(pos_edges.device)[:scales]
                edge_ids_ = edge_ids
                node_ids_ = g_batch.nodes()

                node_anom_mats,edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,struct_loss,feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats,recons_a,res_a_all,exp_params)

            del g_batch, pos_edges, neg_edges, l, res_a, A_hat,X_hat
            torch.cuda.empty_cache() ; gc.collect()
            
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
        if 'weibo' in exp_params['DATASET']['NAME']:
            if epoch == 0 and iter == 1:
                tb_writer_norm = [TBWriter(tb,edge_ids[0],norms,clusts[sc]) for sc in range(scales)]
                tb_writer_single = [TBWriter(tb,edge_ids[0],batch_sc_label['single'],clusts[sc]) for sc in range(scales)]
                tb_writer_sc1 = [TBWriter(tb,edge_ids[0],visualizer.flatten_label(batch_sc_label['anom_sc1']),clusts[sc]) for sc in range(scales)]
                tb_writer_sc2 = [TBWriter(tb,edge_ids[0],visualizer.flatten_label(batch_sc_label['anom_sc2']),clusts[sc]) for sc in range(scales)]
                tb_writer_sc3 = [TBWriter(tb,edge_ids[0],visualizer.flatten_label(batch_sc_label['anom_sc3']),clusts[sc]) for sc in range(scales)]
            for sc,l in enumerate(loss):
                tb.add_scalar(f'Loss_{sc}', l, epoch)
                tb.add_scalar(f'RegLoss_{sc}', regloss[sc].sum(), epoch)
                tb.add_scalar(f'ClustLoss_{sc}', clustloss[sc].sum(), epoch)
                tb.add_scalar(f'NonClustLoss_{sc}', nonclustloss[sc].sum(), epoch)
                #tb.add_scalar(f'AvgAtt_{sc}',F.softmax(struct_model.attn,0)[sc].mean())
                clust = clusts[sc]
                
                tb_writer_norm[sc].tb_write_anom(tb,edge_ids[0],norms,struct_loss[sc],F.softmax(struct_model.attn,0)[sc],'normal',sc,epoch,regloss[sc],clustloss[sc],nonclustloss[sc],clust)
                tb_writer_single[sc].tb_write_anom(tb,edge_ids[0],batch_sc_label['single'],struct_loss[sc],F.softmax(struct_model.attn,0)[sc],'single',sc,epoch,regloss[sc],clustloss[sc],nonclustloss[sc],clust)
                tb_writer_sc1[sc].tb_write_anom(tb,edge_ids[0],visualizer.flatten_label(batch_sc_label['anom_sc1']),struct_loss[sc],F.softmax(struct_model.attn,0)[sc],1,sc,epoch,regloss[sc],clustloss[sc],nonclustloss[sc],clust)
                tb_writer_sc2[sc].tb_write_anom(tb,edge_ids[0],visualizer.flatten_label(batch_sc_label['anom_sc2']),struct_loss[sc],F.softmax(struct_model.attn,0)[sc],2,sc,epoch,regloss[sc],clustloss[sc],nonclustloss[sc],clust)
                tb_writer_sc3[sc].tb_write_anom(tb,edge_ids[0],visualizer.flatten_label(batch_sc_label['anom_sc3']),struct_loss[sc],F.softmax(struct_model.attn,0)[sc],3,sc,epoch,regloss[sc],clustloss[sc],nonclustloss[sc],clust)
            #import ipdb ; ipdb.set_trace()
            if struct_model.final_attn is not None:
                tb.add_histogram(f'Filt_att',F.softmax(struct_model.final_attn,0),epoch)

        del loss,batch_sc_label,struct_loss

        if epoch == 0:
            tot_loss = epoch_l.sum(0).unsqueeze(0)
        else:
            tot_loss = torch.cat((tot_loss,epoch_l.sum(0).unsqueeze(0)),dim=0)
        epoch_l = torch.sum(epoch_l)
        #epoch_l.backward()
        #optimizer.step()
        if struct_model:
            if struct_model.final_attn != None:
                # epoch x 3 x nodes x num filters
                train_attn_w[epoch,:,in_nodes]=F.softmax(struct_model.final_attn.detach().cpu(),1)#torch.unsqueeze(struct_model.attn,0).detach().cpu()
                #train_attn_w[epoch].scatter_(1,in_nodes.unsqueeze(0).unsqueeze(-1).to('cpu'),struct_model.final_attn.to(torch.float32).to('cpu'))
            train_losses[epoch]=torch.stack([torch.stack([torch.mean(j[j.nonzero()],axis=0) for j in i]) for i in torch.tensor(edge_anom_mats)]).squeeze(-1)
        for name, param in struct_model.named_parameters():
            tb.add_histogram(name, param.flatten(), epoch)

    # save/load trained model model
    if exp_params['DATASET']['DATASAVE'] == True and args.dataload:
        torch.save(struct_model,f'{exp_name}.pt')


    # accumulate node-wise anomaly scores via model evaluation
    if exp_params['MODEL']['NAME'] not in ['madan','gcad'] and struct_model: struct_model.eval()

    edges=adj.edges('eid')
    #dataloader = dgl.dataloading.DataLoader(adj, edges, sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,device=exp_params['DEVICE'])
    if exp_params['MODEL']['NAME'] != 'gcad' and exp_params['DATASET']['DATASAVE'] == False:
        struct_scores, feat_scores = torch.zeros(scales,adj.number_of_nodes()).to(exp_params['DEVICE']),torch.zeros(exp_params['MODEL']['D'],adj.number_of_nodes()).to(exp_params['DEVICE'])
    iter = 0

    edge_anom_mats,node_anom_mats,recons_a,res_a_all = init_recons_agg(adj.number_of_nodes(),adj.ndata['feature'].shape[1],exp_params)

    all_samps = []
    if (exp_params['DATASET']['DATADIR'] is not None and exp_params['DATASET']['DATALOAD']):
        pass
        #random.shuffle(dataloader)
    else:
        dataloader = dataloading.fetch_dataloader(adj, edges)

    for loaded_input in dataloader:
        vis = True if (exp_params['VIS']['VIS_FILTERS'] == True and iter == 0) else False
        if exp_params['DATASET']['DATALOAD']:
            loaded_input,lbl=dataloading.load_batch(loaded_input,'test')
            
        # collect input
        if exp_params['DATASET']['BATCH_TYPE'] == 'node':
            all_nodes,in_nodes,g_batch= loaded_input
            pos_edges,neg_edges=None,None
        else:
            in_nodes, pos_edges, neg_edges, g_batch, last_batch_node,batch_sc_label = dataloading.get_edge_batch(loaded_input,sc_label)
            #edge_ids = torch.vstack((pos_edges,neg_edges))
        in_nodes = np.arange(0,in_nodes.shape[0])
        # load data/labels
        if struct_model and exp_params['MODEL']['NAME'] != 'gcad':
            if not exp_params['DATASET']['DATALOAD']:
                if exp_params['VIS']['VIS_FILTERS']:
                    visualizer.sc_label = batch_sc_label
                    lg = LabelGenerator(g_batch,g_batch.ndata['feature'],vis,'test',batch_sc_label,in_nodes[norms],exp_params,visualizer)
                else:
                    lg = LabelGenerator(g_batch,g_batch.ndata['feature'],vis,'test',batch_sc_label,in_nodes[norms],exp_params,None)
                lbl=lg.construct_labels()
            else:
                lbl = lbl[1:]
            lbl_eids = lbl[0].edge_ids(pos_edges[:,0],pos_edges[:,1])
            pos_edges = torch.stack([torch.stack(i.find_edges(lbl_eids)) for i in lbl])
            neg_edges = torch.stack([torch.stack(dgl.sampling.global_uniform_negative_sampling(i,pos_edges.shape[-1])) for i in lbl])
 

         # save batch info
        if exp_params['DATASET']['DATASAVE']:
            dataloading.save_batch(loaded_input,lbl,iter,'test')
            continue
    
        recons_label = g_batch if lbl is None else collect_recons_label(lbl,exp_params['DEVICE'])

        # run evaluation
        if struct_model and exp_params['MODEL']['NAME'] != 'gcad':
            #edges, feats, graph_ = process_graph(g_batch)
            edges, feats, graph_ = process_graph(adj)
     
            edge_ids = torch.cat((pos_edges,neg_edges),axis=2).to(pos_edges.device)[:scales]
            A_hat,X_hat,res_a = struct_model(edges,feats,edge_ids,vis=vis,vis_name='test',clusts=clusts)

        if exp_params['MODEL']['NAME'] == 'gcad':
            adj_ = dgl_to_mat(g_batch)
            adj_=adj_.sparse_resize_((g_batch.num_src_nodes(),g_batch.num_src_nodes()), adj_.sparse_dim(),adj_.dense_dim())
            feat = g_batch.ndata['feature']
            if type(feat) == dict:
                feat = feat['_N']
            struct_loss = struct_model.fit_transform(adj_,feat)
            struct_loss = np.array((struct_loss.T)[g_batch.dstnodes().detach().cpu().numpy()].T)

        # collect anomaly scores
        #edge_ids = torch.cat((pos_edges,neg_edges),axis=2).to(pos_edges.device)[:scales]
        edge_ids = torch.cat((pos_edges[0],neg_edges[0]),axis=1).unsqueeze(0).tile((scales,1,1))
        edge_ids_ = edge_ids
        node_ids_ = g_batch.nodes()

        #A_hat = A_hat[:,edge_ids[:,0],edge_ids[:,1]]
        #A_hat = torch.stack([A_hat[i,edge_ids[i,0],edge_ids[i,1]] for i in range(scales)])

        #loss, struct_loss, feat_cost = loss_func(recons_label, adj.ndata['feature'], A_hat, None, res_a.detach().cpu(), edge_ids, sample=exp_params['MODEL']['SAMPLE_TEST'], recons=exp_params['MODEL']['RECONS'],alpha=exp_params['MODEL']['ALPHA'], clusts=clusts)
        loss, struct_loss, feat_cost,_,_,_ = loss_func(recons_label, adj.ndata['feature'], A_hat, None, None, edge_ids, sample=exp_params['MODEL']['SAMPLE_TEST'], recons=exp_params['MODEL']['RECONS'],alpha=exp_params['MODEL']['ALPHA'], clusts=clusts, regularize=regularize)
        #loss, struct_loss, feat_cost = loss_func(recons_label, g_batch.ndata['feature'], A_hat, X_hat, res_a.detach().cpu(), edge_ids, sample=exp_params['MODEL']['SAMPLE_TEST'], recons=exp_params['MODEL']['RECONS'],alpha=exp_params['MODEL']['ALPHA'],clusts=clusts)
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
    #import ipdb ; ipdb.set_trace()
    if exp_params['MODEL']['RECONS'] == 'both':
        print('struct scores')
        a_clf = anom_classifier(exp_params)
        a_clf.calc_prec(None, [np.mean(edge_anom_mats[0],axis=1)], truth, sc_label, train_attn_w, cluster=False, input_scores=True)
        print('feat scores')
        a_clf.calc_prec(None, [np.mean(node_anom_mats[0],axis=1)], truth, sc_label, train_attn_w, cluster=False, input_scores=True)
        print('combined scores')
        a_clf.calc_prec(None, loss.detach().cpu().numpy(), truth, sc_label, train_attn_w, cluster=False, input_scores=True)
    else:
        if exp_params['DATASET']['BATCH_TYPE'] == 'node':
            if -1 in node_anom_mats[0]:
                raise('node not sampled')
            a_clf.calc_prec(None,node_anom_mats, truth, sc_label, cluster=False, input_scores=True)
        else:
            if 'multi-scale' in exp_params['MODEL']['NAME']:
                #attns = F.softmax(train_attn_w[-1],1).numpy()
                attns = train_attn_w[-1].numpy()
            else:
                attns = None
            
            if not exp_params['MODEL']['SAMPLE_TEST']:
                a_clf.calc_prec(dgl_to_mat(adj), struct_loss.detach().cpu().numpy(), truth, sc_label, attns, cluster=False, input_scores=True)
            else:
                a_clf.title = 'loss'
                a_clf.calc_prec(dgl_to_nx(adj)[0], edge_anom_mats, truth, sc_label, attns, clusts, cluster=False)
                

                a_clf.title = 'regloss' ; a_clf.stds = 3
                anoms_found = None
                losses = []

                # multiply by edges
                losses.append([np.zeros(edge_anom_mats[0].shape),nonclustloss[0]*edge_anom_mats[0],(clusts[0],clusts[0])])
                losses.append([clustloss[1]*edge_anom_mats[1],np.zeros(edge_anom_mats[0].shape),(clusts[1],clusts[1])])
                losses.append([clustloss[2]*edge_anom_mats[2],nonclustloss[-2]*edge_anom_mats[-2],(clusts[2],clusts[-2])])
                losses.append([clustloss[3]*edge_anom_mats[3],nonclustloss[-3]*edge_anom_mats[-3],(clusts[3],clusts[-3])])
                
                import ipdb ; ipdb.set_trace()
                a_clf.title = 'clustprec'
                for i in range(scales):
                    #edge_anom_mats = [np.zeros(edge_anom_mats[0].shape)]
                    #node_anom_mats,edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,regloss[i].unsqueeze(0),feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats,recons_a,res_a_all,exp_params)
                    #a_clf.calc_prec(dgl_to_nx(adj)[0], edge_anom_mats, truth, sc_label, attns, clusts, cluster=False)

                    edge_anom_mats_ = [np.zeros(edge_anom_mats[0].shape)]
                    node_anom_mats,clust_edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,losses[i][0].unsqueeze(0),feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats_,recons_a,res_a_all,exp_params)
                    edge_anom_mats_ = [np.zeros(edge_anom_mats[0].shape)]
                    node_anom_mats,nonclust_edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,losses[i][1].unsqueeze(0),feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats_,recons_a,res_a_all,exp_params)
                    scores,anoms_found = a_clf.calc_clust_prec(dgl_to_nx(adj)[0], clust_edge_anom_mats, nonclust_edge_anom_mats, i, anoms_found, sc_label, attns, losses[i][-1], cluster=False)
                    
                    '''
                    a_clf.title = 'nonclustloss'
                    edge_anom_mats = [np.zeros(edge_anom_mats[0].shape)]
                    node_anom_mats,edge_anom_mats,recons_a,res_a_all = agg_recons(A_hat,res_a,nonclustloss[i].unsqueeze(0),feat_cost,node_ids_,edge_ids,edge_ids_,node_anom_mats,edge_anom_mats,recons_a,res_a_all,exp_params)
                    a_clf.calc_clust_prec(dgl_to_nx(adj)[0], edge_anom_mats, truth, sc_label, attns, clusts, cluster=False)
                    '''
            print('done..')
            import ipdb ; ipdb.set_trace()

    if visualizer is not None and args.datasave is False:
        if train_attn_w is not None:
            train_attn_w = train_attn_w.detach().to('cpu').numpy()
            #train_attn_w=train_attn_w[-1].sum(axis=1).numpy()
        #if exp_params['VIS']['VIS_FILTERS'] == True:
        #visualizer.plot_final_filters([np.array([i.weight.detach().cpu().numpy() for i in j.filters]) for j in struct_model.module_list],[F.softmax(i.lam).detach().cpu().numpy() for i in struct_model.module_list])
        #visualizer.plot_final_filters([np.array([i.weight.detach().cpu().numpy() for i in j.filters]) for j in struct_model.module_list])#,[(i.lam).detach().cpu().numpy() for i in struct_model.module_list])
        #visualizer.plot_attn_scores(train_attn_w.numpy(),[(i.filter_weights).detach().cpu().numpy() for i in struct_model.module_list])
        visualizer.plot_attn_scores(train_attn_w,train_losses,clusts=clusts)
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