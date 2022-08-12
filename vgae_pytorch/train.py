import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, auc
import scipy.sparse as sp
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import os
import time

from input_data import load_data
from preprocessing import *
from utils import *
import args
import model

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from os.path import dirname, join as pjoin
import scipy.io as sio


# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

ori_mat_contents = sio.loadmat('../anom_data/cora_ori.mat')
adj = ori_mat_contents['Network']
print('ADJ SHAPE',adj.shape)
class_label = ori_mat_contents['Class']
ind = np.argsort(class_label,axis=0)
ind2 = np.argsort(class_label,axis=1)
class_label = np.take_along_axis(class_label,ind,axis=0)
str_anom_label, attr_anom_label = ori_mat_contents['str_anomaly_label'],ori_mat_contents['attr_anomaly_label']

anom_idx = np.where(attr_anom_label == 1)[0]

features = []
features_load = ori_mat_contents['Attributes']
features_load = features_load.tocoo()
features_load = sparse_to_tuple(features_load)
features_load = torch.sparse.FloatTensor(torch.LongTensor(features_load[0].T), 
                    torch.FloatTensor(features_load[1]), 
                    torch.Size(features_load[2])).cuda(args.device)
features.append(features_load)

for s in range(args.scales):
    fname = '../smoothed_graphs/' + str(s+1) + '_cora_ori.mat'
    # fname = '../smoothed_graphs/' + str(s+1) + '_cora.mat'
    # fname = '../anom_data/cora.mat'
    mat_contents = sio.loadmat(fname)

    # adj, features = load_data(args.dataset)
    features_load = mat_contents['Attributes']
    features_load = features_load.tocoo()
    features_load = sparse_to_tuple(features_load)
    features_load = torch.sparse.FloatTensor(torch.LongTensor(features_load[0].T), 
                        torch.FloatTensor(features_load[1]), 
                        torch.Size(features_load[2])).cuda(args.device)
    features.append(features_load)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)
adj_norm_np = preprocess_graph_np(adj)

num_nodes = adj.shape[0]

# features = sparse_to_tuple(features.tocoo())
# num_features = features[2][1]
# features_nonzero = features[1].shape[0]

# Create Model
# WEIGHT FOR EDGES
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


adj_label = adj_train + sp.eye(adj_train.shape[0])
# adj_label = sparse_to_tuple(adj_label)

print(sp.csr_matrix.todense(adj))
sorted_adj = np.take_along_axis(sp.csr_matrix.todense(adj),ind,axis=0)
# sorted_adj = adj
# sorted_adj = sorted_adj[:,ind]
# sorted_adj = np.squeeze(sorted_adj)
# adj_distr(sorted_adj,0)

print('ADJ SHAPE',adj.shape)

new_adj = sorted_adj
# print('NORM ADJ RANGE',np.min(new_adj),np.max(new_adj))
# print('adj distr')
# adj_distr(new_adj,1)
# print('edge distr')
# edge_distr(new_adj,0)
# for i in range(args.scales-1):
#     print('scale',i+1)
#     new_adj = new_adj@sorted_adj
#     new_adj = torch.where(new_adj > 1.0, 1.0), torch.where(new_adj < 0, 0)
#     adj_distr(new_adj,i+1)
#     edge_distr(new_adj,i+1)


adj_label = sparse_to_tuple(adj_label)
adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                            torch.FloatTensor(adj_norm[1]), 
                            torch.Size(adj_norm[2])).cuda(args.device)
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                            torch.FloatTensor(adj_label[1]), 
                            torch.Size(adj_label[2])).cuda(args.device)
# features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
#                             torch.FloatTensor(features[1]), 
#                             torch.Size(features[2]))

total_pos_edges=num_pos_edges(adj_label.cpu().to_dense())
print('NUMBER OF POSITIVE EDGES',total_pos_edges)

# weighs positive
weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0)).cuda(args.device)
weight_tensor[weight_mask] = pos_weight

writer = SummaryWriter()


#init model and optimizer
# model = getattr(model,args.model)(adj=adj_norm,in_channels=args.hidden1_dim,hidden_size=args.hidden2_dim,num_timesteps=args.scales)
# optimizer = Adam(model.parameters(), lr=args.learning_rate)

# # train model
# for epoch in range(args.num_epoch):
#     # TODO: REMOVE!!!!!!!!!!
#     # break
#     # return
#     t = time.time()
#     # print('features size',len(features))
#     # model_in = torch.stack([features,features],axis=0)
#     # print('model in size',model_in.shape)
#     A_preds = model(features)
#     optimizer.zero_grad()
#     loss = 0
#     for scale,A_pred in enumerate(A_preds):
#         scale_loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
#         # scale_norm = torch.norm(torch.round(A_pred.view(-1))-adj_label.to_dense().view(-1))
#         scale_prec_1 = calc_precision(A_pred,adj_label.to_dense(),int(total_pos_edges))
#         scale_prec_5 = calc_precision(A_pred,adj_label.to_dense(),int(round(total_pos_edges*.5)))
#         scale_prec_25 = calc_precision(A_pred,adj_label.to_dense(),int(round(total_pos_edges*.25)))
#         scale_norm = torch.norm(A_pred.view(-1)-adj_label.to_dense().view(-1))
#         scale_auc = roc_auc_score(adj_label.to_dense().view(-1).cpu(), A_pred.view(-1).detach().cpu())

#         # svd_recons = torch.sigmoid(svd(adj_label.to_dense(),args.hidden2_dim))
#         # svd_scale_norm = torch.norm(svd_recons.view(-1)-adj_label.to_dense().view(-1))
#         # print('---SVD---')
#         # svd_scale_prec_1 = calc_precision(svd_recons,adj_label.to_dense(),int(total_pos_edges))
#         # svd_scale_prec_5 = calc_precision(svd_recons,adj_label.to_dense(),int(round(total_pos_edges*.5)))
#         # svd_scale_prec_25 = calc_precision(svd_recons,adj_label.to_dense(),int(round(total_pos_edges*.25)))

#         # print('Density at scale',scale,get_density(adj_label.to_dense(),scale))
#         if epoch == args.num_epoch-1:
#         # if epoch == 0:
#             # edge_distr(A_pred.detach().cpu(),scale)
#             ces = F.binary_cross_entropy(A_pred, adj_label.to_dense(),reduction="none").detach().cpu().numpy()
#             print('making histogram')
#             adj_distr(ces,"norm",scale)
#             # rankings,node_idx = anom_classify(ces)
#             # fname = "figs/rankings"+str(scale)+'norm.csv'
#             # np.savetxt(fname, np.array([rankings,node_idx]), delimiter=",")
#             # print('rankings',rankings[0:10],node_idx[0:10])
#             # ces_distr(ces,scale)

#         # svd_auc = roc_auc_score(adj_label.to_dense().view(-1).cpu(),svd_recons.view(-1).cpu())

#         writer_out = 'Loss_Scale' + str(scale) + '/'
#         writer.add_scalar(writer_out, scale_loss.item(), epoch)

#         writer_out = 'Precision_100%_Scale' + str(scale) + '/'
#         writer.add_scalar(writer_out, scale_prec_1.item(), epoch)
#         writer_out = 'Precision_50%_Scale' + str(scale) + '/'
#         writer.add_scalar(writer_out, scale_prec_5.item(), epoch)
#         writer_out = 'Precision_25%_Scale' + str(scale) + '/'
#         writer.add_scalar(writer_out, scale_prec_25.item(), epoch)

#         # writer_out = 'SVD_Precision_100%_Scale' + str(scale+1) + '/'
#         # writer.add_scalar(writer_out, svd_scale_prec_1.item(), epoch)
#         # writer_out = 'SVD_Precision_50%_Scale' + str(scale+1) + '/'
#         # writer.add_scalar(writer_out, svd_scale_prec_5.item(), epoch)
#         # writer_out = 'SVD_Precision_25%_Scale' + str(scale+1) + '/'
#         # writer.add_scalar(writer_out, svd_scale_prec_25.item(), epoch)

#         writer_out = 'Norm_Scale' + str(scale) + '/'
#         writer.add_scalar(writer_out, scale_norm.item(), epoch)

#         # writer_out = 'SVD_Norm_Scale' + str(scale+1) + '/'
#         # writer.add_scalar(writer_out, svd_scale_norm.item(), epoch)

#         writer_out = 'AUC_Scale' + str(scale) + '/'
#         writer.add_scalar(writer_out, scale_auc.item(), epoch)

#         # writer_out = 'SVD_AUC_Scale' + str(scale+1) + '/'
#         # writer.add_scalar(writer_out, svd_auc.item(), epoch)


#         loss += scale_loss
#         if args.gae_model == 'VGAE':
#             kl_divergence = 0.5/ A_pred.size(0) * (1 + 2*model.logstd - model.mean**2 - torch.exp(model.logstd)**2).sum(1).mean()
#             loss -= kl_divergence
#     # for name, param in model.named_parameters():
#     #     if param.requires_grad:
#     #         print('requires grad')
#     #         print(name)
#     #     else:
#     #         print('no grad')
#     #         print(name)
#     # print('model params',model.parameters())
#     loss.backward()
#     optimizer.step()

#     train_acc = get_acc(A_pred,adj_label)

#     # val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
#     print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
#           "train_acc=", "{:.5f}".format(train_acc), #"val_roc=", "{:.5f}".format(val_roc),
#           #"val_ap=", "{:.5f}".format(val_ap),
#           "time=", "{:.5f}".format(time.time() - t))

#     writer.add_scalar('Loss/', loss.item(), epoch)
#     # writer.add_scalar('AUC/', val_roc, epoch)
#     # writer.add_scalar('Accuracy/', train_acc, epoch)
# torch.save(model, "./saved_models/0sc_model.pt")
device_name = 'cuda:' + str(args.device)
model = torch.load('./saved_models/model.pt',map_location=torch.device(device_name))

features = []

ori_mat_contents = sio.loadmat('../anom_data/cora_ori.mat')
adj = ori_mat_contents['Network']
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                            torch.FloatTensor(adj_label[1]), 
                            torch.Size(adj_label[2])).cuda(args.device)

fname = '../anom_data/cora.mat'
mat_contents = sio.loadmat(fname)
features_load = sparse_to_tuple(mat_contents['Attributes'].tocoo())
features_load = torch.sparse.FloatTensor(torch.LongTensor(features_load[0].T), 
                        torch.FloatTensor(features_load[1]), 
                        torch.Size(features_load[2])).cuda(args.device)
# features.append(features_load)

# features_load = ori_mat_contents['Attributes']
# features_load = features_load.tocoo()
# features_load = sparse_to_tuple(features_load)
# features_load = torch.sparse.FloatTensor(torch.LongTensor(features_load[0].T), 
#                     torch.FloatTensor(features_load[1]), 
#                     torch.Size(features_load[2])).cuda(args.device)
# features.append(features_load)

for s in range(args.scales):
    fname = '../smoothed_graphs/' + str(s+1) + '_cora_ori.mat'
    # fname = '../anom_data/cora.mat'
    mat_contents = sio.loadmat(fname)

    # adj, features = load_data(args.dataset)
    features_load = mat_contents['Attributes']
    features_load = features_load.tocoo()
    features_load = sparse_to_tuple(features_load)
    features_load = torch.sparse.FloatTensor(torch.LongTensor(features_load[0].T), 
                        torch.FloatTensor(features_load[1]), 
                        torch.Size(features_load[2])).cuda(args.device)
    features.append(features_load)
print('ALL FEATURES',len(features))
recons = model.forward(features)
total_pos_edges=num_pos_edges(adj_label.cpu().to_dense())


anom_sc1 = np.array([1653,879])
anom_sc2 = np.array([1276, 376, 804, 867, 906, 1143, 574, 1671, 2, 962, 2183, 643, 196, 636, 1446])
anom_sc3 = np.array([2355, 1422, 2557, 1222, 788, 1526, 2143, 1895, 1405, 731, 968, 657, 2300, 783, 2424, 1547, 2399, 361, 967, 703, 402, 1423, 583, 1924, 2585, 1624, 395, 862, 294, 0, 1195, 1270, 479, 1213, 2353, 1172, 2277, 1286, 935, 2136, 2623, 665, 2468, 2132, 414])
all_anom = np.concatenate((anom_sc1,np.concatenate((anom_sc2,anom_sc3),axis=None)),axis=None)
select1 = np.array([])
select2 = np.array([])
select3 = np.array([])
# for scale,A_pred in enumerate(recons):
#     print('scale',scale)
#     scale_loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
#     # scale_norm = torch.norm(torch.round(A_pred.view(-1))-adj_label.to_dense().view(-1))
#     scale_prec_1 = calc_precision(A_pred,adj_label.to_dense(),int(total_pos_edges))
#     scale_prec_5 = calc_precision(A_pred,adj_label.to_dense(),int(round(total_pos_edges*.5)))
#     scale_prec_25 = calc_precision(A_pred,adj_label.to_dense(),int(round(total_pos_edges*.25)))
#     print('25\%:',scale_prec_25)
#     print('50\%:',scale_prec_5)
#     print('100\%:',scale_prec_1)
#     scale_norm = torch.norm(A_pred.view(-1)-adj_label.to_dense().view(-1))
#     scale_auc = roc_auc_score(adj_label.to_dense().view(-1).cpu(), A_pred.view(-1).detach().cpu())

#     svd_recons = torch.sigmoid(svd(adj_label.to_dense(),args.hidden2_dim))
#     svd_scale_norm = torch.norm(svd_recons.view(-1)-adj_label.to_dense().view(-1))
#     print('---SVD---')
#     svd_scale_prec_1 = calc_precision(svd_recons,adj_label.to_dense(),int(total_pos_edges))
#     svd_scale_prec_5 = calc_precision(svd_recons,adj_label.to_dense(),int(round(total_pos_edges*.5)))
#     svd_scale_prec_25 = calc_precision(svd_recons,adj_label.to_dense(),int(round(total_pos_edges*.25)))

#     writer_out = 'SVD_Precision_100%_Scale' + str(scale+1) + '/'
#     writer.add_scalar(writer_out, svd_scale_prec_1.item(), 1)
#     writer_out = 'SVD_Precision_50%_Scale' + str(scale+1) + '/'
#     writer.add_scalar(writer_out, svd_scale_prec_5.item(), 1)
#     writer_out = 'SVD_Precision_25%_Scale' + str(scale+1) + '/'
#     writer.add_scalar(writer_out, svd_scale_prec_25.item(), 1)

#     # edge_distr(A_pred.detach().cpu(),"anom",scale)
#     ces = F.binary_cross_entropy(A_pred, adj_label.to_dense(),reduction="none").detach().cpu().numpy()
#     ces_svd = F.binary_cross_entropy(svd_recons, adj_label.to_dense(),reduction="none").detach().cpu().numpy()
#     print('making histogram')
#     # adj_distr(ces,"norm",scale)
#     rankings,node_idx = anom_classify(ces,A_pred.cpu())
#     rankings_svd,node_idx_svd = anom_classify(ces_svd,svd_recons)
#     fname = "figs/rankings"+str(scale)+'.csv'
#     np.savetxt(fname, np.array([rankings,node_idx]), delimiter=",")
#     print('rankings',rankings[0:10],node_idx[0:10])
#     # ces_distr(ces,scale)
#     rankings_distr(rankings,node_idx,"norm",scale+1)

#     intersect, ind_a, ind_b = np.intersect1d(node_idx,anom_sc1, return_indices=True)
#     all_intersect, all_ind_a, all_ind_b = np.intersect1d(node_idx,all_anom, return_indices=True)
#     if len(select1) == 0:
#         select1 = np.array([rankings[ind_a]]).T
#         select1_ex = np.array([np.delete(rankings,all_ind_a)]).T
#     else:
#         select1 = np.concatenate((select1,np.array([rankings[ind_a]]).T),axis=1)
#         select1_ex = np.concatenate((select1_ex,np.array([np.delete(rankings,all_ind_a)]).T),axis=1)

#     intersect, ind_a, ind_b = np.intersect1d(node_idx,anom_sc2, return_indices=True)
#     if len(select2) == 0:
#         select2 = np.array([rankings[ind_a]]).T
#         select2_ex = np.array([np.delete(rankings,all_ind_a)]).T
#     else:
#         select2 = np.concatenate((select2,np.array([rankings[ind_a]]).T),axis=1)
#         select2_ex = np.concatenate((select2_ex,np.array([np.delete(rankings,all_ind_a)]).T),axis=1)

#     intersect, ind_a, ind_b = np.intersect1d(node_idx,anom_sc3, return_indices=True)
#     if len(select3) == 0:
#         select3 = np.array([rankings[ind_a]]).T
#         select3_ex = np.array([np.delete(rankings,all_ind_a)]).T
#     else:
#         select3 = np.concatenate((select3,np.array([rankings[ind_a]]).T),axis=1)
#         select3_ex = np.concatenate((select3_ex,np.array([np.delete(rankings,all_ind_a)]).T),axis=1)

#     if scale == len(recons)-1:
#         plot_scale_info(select1,1,"norm")
#         plot_scale_info(select1_ex,1,"norm_ex")
#         plot_scale_info(select2,2,"norm")
#         plot_scale_info(select2_ex,2,"norm_ex")
#         plot_scale_info(select3,3,"norm")
#         plot_scale_info(select3_ex,3,"norm_ex")

#     writer_out = 'Loss_Scale' + str(scale+1) + '/'
#     writer.add_scalar(writer_out, scale_loss.item(), 1)
#     writer_out = 'Precision_100%_Scale' + str(scale+1) + '/'
#     writer.add_scalar(writer_out, scale_prec_1.item(), 1)
#     writer_out = 'Precision_50%_Scale' + str(scale+1) + '/'
#     writer.add_scalar(writer_out, scale_prec_5.item(), 1)
#     writer_out = 'Precision_25%_Scale' + str(scale+1) + '/'
#     writer.add_scalar(writer_out, scale_prec_25.item(), 1)
#     writer_out = 'Norm_Scale' + str(scale+1) + '/'
#     writer.add_scalar(writer_out, scale_norm.item(), 1)
#     writer_out = 'AUC_Scale' + str(scale+1) + '/'
#     writer.add_scalar(writer_out, scale_auc.item(), 1)


print('ANOM')

anom_features = []
ori_mat_contents = sio.loadmat('../anom_data/cora_sparse_anoms.mat')
adj = ori_mat_contents['Network']
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                            torch.FloatTensor(adj_label[1]), 
                            torch.Size(adj_label[2])).cuda(args.device)

features_load = ori_mat_contents['Attributes']
features_load = features_load.tocoo()
features_load = sparse_to_tuple(features_load)
features_load = torch.sparse.FloatTensor(torch.LongTensor(features_load[0].T), 
                    torch.FloatTensor(features_load[1]), 
                    torch.Size(features_load[2])).cuda(args.device)
# anom_features.append(features_load)
for s in range(args.scales):
    fname = '../smoothed_graphs/' + str(s+1) + '_cora_sparse_anoms.mat'
    # fname = '../anom_data/cora.mat'
    mat_contents = sio.loadmat(fname)

    # adj, features = load_data(args.dataset)
    features_load = mat_contents['Attributes']
    features_load = features_load.tocoo()
    features_load = sparse_to_tuple(features_load)
    features_load = torch.sparse.FloatTensor(torch.LongTensor(features_load[0].T), 
                        torch.FloatTensor(features_load[1]), 
                        torch.Size(features_load[2])).cuda(args.device)
    anom_features.append(features_load)
anom_recons = model.forward(anom_features)
total_pos_edges=num_pos_edges(adj_label.cpu().to_dense())

select1 = np.array([])
select2 = np.array([])
select3 = np.array([])


# baselines
dominant_rankings = np.loadtxt('../../msgad/GCN_AnomalyDetection/gae/output/cora-ranking.txt',dtype=int).T
dominant_scores = np.genfromtxt('../../msgad/GCN_AnomalyDetection/gae/output/cora-scores.csv',delimiter=',').T
idx = np.argsort(-dominant_scores)
print('d scores',dominant_scores)
d_node_idx = np.arange(len(dominant_scores))
d_node_idx = d_node_idx[idx]

print(' --- MADAN --- ')
import Madan as md
madan = md.Madan(adj, attributes=features, sigma=0.08)
time_scales = np.concatenate([np.array([0]), 10**np.linspace(0,5,500)])
madan.scanning_relevant_context(time_scales, n_jobs=4)
madan.scanning_relevant_context_time(time_scales)
madan.compute_concentration(1000)
print(madan.concentration,madan.anomalous_nodes)
madan.compute_context_for_anomalies()
print(madan.interp_com)
print(' ------------')

exp='skew'

for scale,A_pred in enumerate(anom_recons):
    print('scale',scale+1)
    scale_loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
    # scale_norm = torch.norm(torch.round(A_pred.view(-1))-adj_label.to_dense().view(-1))
    # scale_prec_1 = calc_precision(A_pred,adj_label.to_dense(),int(total_pos_edges))
    # scale_prec_5 = calc_precision(A_pred,adj_label.to_dense(),int(round(total_pos_edges*.5)))
    # scale_prec_25 = calc_precision(A_pred,adj_label.to_dense(),int(round(total_pos_edges*.25)))
    # print('25\%:',scale_prec_25)
    # print('50\%:',scale_prec_5)
    # print('100\%:',scale_prec_1)
    scale_norm = torch.norm(A_pred.view(-1)-adj_label.to_dense().view(-1))
    scale_auc = roc_auc_score(adj_label.to_dense().view(-1).cpu(), A_pred.view(-1).detach().cpu())

    # svd_recons = torch.sigmoid(svd(adj_label.to_dense(),args.hidden2_dim))
    # svd_scale_norm = torch.norm(svd_recons.view(-1)-adj_label.to_dense().view(-1))
    # print('---SVD---')
    # svd_scale_prec_1 = calc_precision(svd_recons,adj_label.to_dense(),int(total_pos_edges))
    # svd_scale_prec_5 = calc_precision(svd_recons,adj_label.to_dense(),int(round(total_pos_edges*.5)))
    # svd_scale_prec_25 = calc_precision(svd_recons,adj_label.to_dense(),int(round(total_pos_edges*.25)))

    # writer_out = 'SVD_Precision_100%_Anom_Scale' + str(scale+1) + '/'
    # writer.add_scalar(writer_out, svd_scale_prec_1.item(), 1)
    # writer_out = 'SVD_Precision_50%_Anom_Scale' + str(scale+1) + '/'
    # writer.add_scalar(writer_out, svd_scale_prec_5.item(), 1)
    # writer_out = 'SVD_Precision_25%_Anom_Scale' + str(scale+1) + '/'
    # writer.add_scalar(writer_out, svd_scale_prec_25.item(), 1)

    # edge_distr(A_pred.detach().cpu(),"anom",scale)
    ces = F.binary_cross_entropy(A_pred, adj_label.to_dense(),reduction="none").detach().cpu().numpy()
    mses = F.mse_loss(A_pred, adj_label.to_dense(),reduction="none").detach().cpu().numpy()

    metric = mses
    # ces_svd = F.binary_cross_entropy(svd_recons, adj_label.to_dense(),reduction="none").detach().cpu().numpy()
    print('making histogram')
    # adj_distr(ces,"anom",scale)

    rankings,node_idx = anom_classify(metric,A_pred.cpu())
    # rankings_svd,node_idx_svd = anom_classify(ces_svd,svd_recons.cpu().numpy())

    fname = "figs/anom_rankings"+str(scale)+'anom.csv'
    np.savetxt(fname, np.array([rankings,node_idx]), delimiter=",")
    # ces_distr(ces,scale)
    rankings_distr(rankings,node_idx,"anom",scale+1)

    ks = [0.001,0.025,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
    #calc_anom_precs(rankings,node_idx,all_anom,num_nodes,num_nodes,ks,scale,exp)
    #disp_test_rankings(metric,anom_sc2,all_anom,str(scale))
    #disp_score_distr(rankings,all_anom,str(scale),node_idx,exp)
    #disp_score_distr(dominant_rankings,all_anom,str(scale),node_idx,'dominant')

    anom_prec_1,sum1 = calc_anom_precision(rankings,node_idx,all_anom,num_nodes,len(all_anom))
    anom_prec_5,sum5 = calc_anom_precision(rankings,node_idx,all_anom,num_nodes,int(round(len(all_anom)*.75)))
    anom_prec_25,sum25 = calc_anom_precision(rankings,node_idx,all_anom,num_nodes,int(round(len(all_anom)*.5)))

    print('--- DOMINANT ---')
    anom_sc = all_anom
    d_sum1,d_sum5,d_sum25 = get_sum(d_node_idx,anom_sc,len(all_anom)),get_sum(d_node_idx,anom_sc,int(round(len(all_anom)*.75))),get_sum(d_node_idx,anom_sc,int(round(len(all_anom)*.5)))
    d_anom_prec_1,d_sum1 = calc_anom_precision(dominant_rankings,d_node_idx,all_anom,num_nodes,len(all_anom))
    d_anom_prec_5,d_sum5 = calc_anom_precision(dominant_rankings,d_node_idx,all_anom,num_nodes,int(round(len(all_anom)*.75)))
    d_anom_prec_25,d_sum25 = calc_anom_precision(dominant_rankings,d_node_idx,all_anom,num_nodes,int(round(len(all_anom)*.5)))

    # svd_anom_prec_1,svd_sum1 = calc_anom_precision(rankings_svd,node_idx_svd,anom_sc3,num_nodes,len(all_anom))
    # svd_anom_prec_5,svd_sum5 = calc_anom_precision(rankings_svd,node_idx_svd,anom_sc3,num_nodes,int(round(len(all_anom)*.75)))
    # svd_anom_prec_25,svd_sum25 = calc_anom_precision(rankings_svd,node_idx_svd,anom_sc3,num_nodes,int(round(len(all_anom)*.5)))


    # anom_prec_1 = calc_anom_precision(rankings,node_idx,all_anom,num_nodes,int(total_pos_edges))
    # anom_prec_5 = calc_anom_precision(rankings,node_idx,all_anom,num_nodes,int(round(total_pos_edges*.5)))
    # anom_prec_25 = calc_anom_precision(rankings,node_idx,all_anom,num_nodes,int(round(total_pos_edges*.25)))

    all_intersect, all_ind_a, all_ind_b = np.intersect1d(node_idx,all_anom, return_indices=True)
    intersect, ind_a, ind_b = np.intersect1d(node_idx,anom_sc1, return_indices=True)
    if len(select1) == 0:
        select1 = np.array([rankings[ind_a]]).T
        select1_ex = np.array([np.delete(rankings,all_ind_a)]).T
    else:
        select1 = np.concatenate((select1,np.array([rankings[ind_a]]).T),axis=1)
        select1_ex = np.concatenate((select1_ex,np.array([np.delete(rankings,all_ind_a)]).T),axis=1)

    intersect, ind_a, ind_b = np.intersect1d(node_idx,anom_sc2, return_indices=True)
    if len(select2) == 0:
        select2 = np.array([rankings[ind_a]]).T
        select2_ex = np.array([np.delete(rankings,all_ind_a)]).T
    else:
        select2 = np.concatenate((select2,np.array([rankings[ind_a]]).T),axis=1)
        select2_ex = np.concatenate((select2_ex,np.array([np.delete(rankings,all_ind_a)]).T),axis=1)

    intersect, ind_a, ind_b = np.intersect1d(node_idx,anom_sc3, return_indices=True)
    if len(select3) == 0:
        select3 = np.array([rankings[ind_a]]).T
        select3_ex = np.array([np.delete(rankings,all_ind_a)]).T
    else:
        select3 = np.concatenate((select3,np.array([rankings[ind_a]]).T),axis=1)
        select3_ex = np.concatenate((select3_ex,np.array([np.delete(rankings,all_ind_a)]).T),axis=1)

    # if scale == len(anom_recons)-1:
    #     plot_scale_info(select1,1,"anom")
    #     plot_scale_info(select1_ex,1,"anom_ex")
    #     plot_scale_info(select2,2,"anom")
    #     plot_scale_info(select2_ex,2,"anom_ex")
    #     plot_scale_info(select3,3,"anom")
    #     plot_scale_info(select3_ex,3,"anom_ex")
        
    # rankings_distr(rankings[ind_a],node_idx[ind_a],"anom_select_1",scale+1)
    # rankings_distr(rankings[ind_a],node_idx[ind_a],"anom_select_2",scale+1)
    # rankings_distr(rankings[ind_a],node_idx[ind_a],"anom_select_3",scale+1)

    print('WRITING FOR',str(scale+1))        
    # writer_out = 'Anom_Loss_Scale' + str(scale+1) + '/'
    # writer.add_scalar(writer_out, scale_loss.item(), 1)
    writer_out = 'Anom_Precision_100%_Scale' + str(scale+1) + '/'
    writer.add_scalar(writer_out, anom_prec_1.item(), 1)
    writer_out = 'Anom_Precision_50%_Scale' + str(scale+1) + '/'
    writer.add_scalar(writer_out, anom_prec_5.item(), 1)
    writer_out = 'Anom_Precision_25%_Scale' + str(scale+1) + '/'
    writer.add_scalar(writer_out, anom_prec_25.item(), 1)
    writer_out = 'Dominant_Anom_Precision_100%_Scale' + str(scale+1) + '/'
    writer.add_scalar(writer_out, d_anom_prec_1.item(), 1)
    writer_out = 'Dominant_Anom_Precision_50%_Scale' + str(scale+1) + '/'
    writer.add_scalar(writer_out, d_anom_prec_5.item(), 1)
    writer_out = 'Dominant_Anom_Precision_25%_Scale' + str(scale+1) + '/'
    writer.add_scalar(writer_out, d_anom_prec_25.item(), 1)
    # writer_out = 'Anom_Precision_SVD_100%_Scale' + str(scale+1) + '/'
    # writer.add_scalar(writer_out, svd_anom_prec_1.item(), 1)
    # writer_out = 'Anom_Precision_SVD_50%_Scale' + str(scale+1) + '/'
    # writer.add_scalar(writer_out, svd_anom_prec_5.item(), 1)
    # writer_out = 'Anom_Precision_SVD_25%_Scale' + str(scale+1) + '/'
    # writer.add_scalar(writer_out, svd_anom_prec_25.item(), 1)
    print('done writing')
    writer_out = 'Sum_Anom_Precision_100%_Scale' + str(scale+1) + '/'
    writer.add_scalar(writer_out, sum1, 1)
    writer_out = 'Sum_Anom_Precision_50%_Scale' + str(scale+1) + '/'
    writer.add_scalar(writer_out, sum5, 1)
    writer_out = 'Sum_Anom_Precision_25%_Scale' + str(scale+1) + '/'
    writer.add_scalar(writer_out, sum25, 1)
    writer_out = 'Dominant_Sum_Anom_Precision_100%_Scale' + str(scale+1) + '/'
    writer.add_scalar(writer_out, d_sum1, 1)
    writer_out = 'Dominant_Sum_Anom_Precision_50%_Scale' + str(scale+1) + '/'
    writer.add_scalar(writer_out, d_sum5, 1)
    writer_out = 'Dominant_Sum_Anom_Precision_25%_Scale' + str(scale+1) + '/'
    writer.add_scalar(writer_out, d_sum25, 1)
    # writer_out = 'Sum_Anom_Precision_SVD_100%_Scale' + str(scale+1) + '/'
    # writer.add_scalar(writer_out, svd_sum1, 1)
    # writer_out = 'Sum_Anom_Precision_SVD_50%_Scale' + str(scale+1) + '/'
    # writer.add_scalar(writer_out, svd_sum5, 1)
    # writer_out = 'Sum_Anom_Precision_SVD_25%_Scale' + str(scale+1) + '/'
    # writer.add_scalar(writer_out, svd_sum25, 1)
    # writer_out = 'Anom_Norm_Scale' + str(scale+1) + '/'
    # writer.add_scalar(writer_out, scale_norm.item(), 1)
    # writer_out = 'Anom_AUC_Scale' + str(scale+1) + '/'
    # writer.add_scalar(writer_out, scale_auc.item(), 1)



# test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
# print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
#       "test_ap=", "{:.5f}".format(test_ap))
writer.close()
