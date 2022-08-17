from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import scipy.io
from sklearn.metrics import roc_auc_score
from datetime import datetime
import argparse

#from model import Dominant
from model import EGCN
from utils import load_anomaly_detection_dataset



def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)


    cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost

def train_dominant(args):

    adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset, 2)

    adj = torch.FloatTensor(adj)
    adj_label = torch.FloatTensor(adj_label)
    attrs = torch.FloatTensor(attrs)

    model = EGCN(in_size = attrs.size(1), out_size = args.hidden_dim)
    #model = Dominant(feat_size = attrs.size(1), hidden_size = args.hidden_dim, dropout = args.dropout)


    if args.device == 'cuda':
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        attrs = attrs.to(device)
        model = model.cuda()
        
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    w = None
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        #A_hat, X_hat = model(attrs, adj)
        A_hat, X_hat, w = model(attrs, w)
        loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, args.alpha)
        l = torch.mean(loss)
        l.backward()
        w = w.detach()
        optimizer.step() 
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))

        #if epoch%10 == 0 or epoch == args.epoch - 1:
        if epoch == args.epoch-1:
            model.eval()
            #A_hat, X_hat = model(attrs, adj)
            A_hat, X_hat, w = model(attrs, w)
            loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, args.alpha)
            score = loss.detach().cpu().numpy()
            print("Epoch:", '%04d' % (epoch))#, 'Auc', roc_auc_score(label, score))
    '''
    #torch.save(model,'model.pt')
    model = torch.load('model.pt')
    model.eval()
    A_hat, X_hat = model(attrs, adj)
    '''
    loss, struct_loss, feat_loss = loss_func(adj_label, A_hat, attrs, X_hat, args.alpha)
    #score = loss.detach().cpu().numpy()
    scores = loss.detach().cpu().numpy()
    import scipy
    import torch.nn.functional as F
    
    recons_errors = F.mse_loss(A_hat.detach().cpu(), adj_label.detach().cpu(),reduction="none")
    scores = scipy.stats.skew(recons_errors.numpy(),axis=0)
    
    sorted_errors = np.argsort(scores)
    rankings = []
    for error in sorted_errors:
        rankings.append(label[error])
    rankings = np.array(rankings)
    import  ipdb ; ipdb.set_trace()

    with open('output/{}-ranking.txt'.format(args.dataset), 'w') as f:
        for index in sorted_errors:
            f.write("%s\n" % label[index])
    
    import pandas as pd
    #df = pd.DataFrame({'AD-GCA':reconstruction_errors})
    df = pd.DataFrame({'AD-GCA':scores})
    df.to_csv('output/{}-scores.csv'.format(args.dataset), index=False, sep=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='BlogCatalog', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.8, help='balance parameter')
    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')

    args = parser.parse_args()

    train_dominant(args)
