import numpy as np
from scipy.sparse import csc_matrix
from utility import load_graph
import argparse
from subgraph_centralization import subgraph_embeddings
from sklearn.metrics import roc_auc_score
from iNN_IK import iNN_IK
import sys
import scipy
import scipy.io as sio
import dgl
import networkx as nx
sys.path.append('../../msgad')
from msgad.anom_detector import *


def generate_scores(scores, M, args):
    score_weight = [np.math.pow(args.lamda, i) for i in range(6)]
    en_scores = np.zeros_like(scores)
    tot = np.zeros_like(scores)
    for i in range(len(M)):
        for key, values in M[i].items():
            en_scores[key] += score_weight[values] * scores[i]
            tot[key] += score_weight[values]
    return np.divide(en_scores, tot)


def main(args):
    attr, adj, nodes, label = load_graph(args.dataset)
    embedding, M = subgraph_embeddings(attr, adj, args.h)
    kmembeddings = iNN_IK(args.psi, 5).fit_transform(embedding)
    mean_embedding = np.mean(kmembeddings, axis=0)
    scores = kmembeddings.dot(mean_embedding.transpose())


    final_scores = np.zeros(label.shape)
    final_scores[nodes] = np.array(generate_scores(scores, M, args))
    #final_scores = np.array(generate_scores(scores, M, args))

    a_clf = anom_classifier(None,args.scales,'../msgad/output',args.dataset,0,'gcad','gcad')
    with open(f'../msgad/batch_data/labels/{args.dataset}_labels_{args.scales}.mat','rb') as fin:
        mat = pkl.load(fin)
    sc_all,clusts = mat['labels'],mat['clusts']
    a_clf.calc_prec(-final_scores.T,label,sc_all,clusts)
    
    print(
        f'dataset : {args.dataset}, auc = {roc_auc_score(label, -final_scores)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--psi', default=2, type=int)
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--h', default=1, type=int)
    parser.add_argument('--lamda', default=0.0625, type=float)
    parser.add_argument('--scales', type=int, default=3, help='for msgad comparison: which multi-scale labels to use')
    args = parser.parse_args()
    main(args)
