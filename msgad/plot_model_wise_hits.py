import pickle as pkl
import argparse
import matplotlib.pyplot as plt
import os
import glob
import numpy as np


def plot_hits(scale_dicts,type,dataset):
    model_names = scale_dicts[0][hits].keys()
    for scale,scale_dict in enumerate(scale_dicts):
        plt.figure()
        for model,hits in scale_dict['hits']:
            hits_plot = np.cumsum(hits)

            if type == 'precision':
                hits_plot /= hits.nonzero()[0].shape[0]
            else:
                raise "hit type not found"
            plt.plot(np.arange(hits_plot.shape),hits_plot)
        plt.legend(model_names)
        plt.savefig(f'output/{dataset}/figs/hits_{type}_totscales_{len(scale_dicts)}_scale{scale}.png')

def plot_bar_charts(scale_dicts,metric,dataset):
    plt.figure()
    plt.title(f'{metric}')
    model_names = scale_dicts[0][metric].keys()
    width = 0.25
    for idx,model_name in enumerate(model_names):
        # NOTE: last index is the result across all scales
        bar_data = [scale_dicts[i][metric][model_name][-1] for i in range(len(scale_dicts))]
        plt.bar(np.arange(len(bar_data))+(idx*width),bar_data,label=model_name)
    plt.legend(model_names)
    plt.savefig(f'output/{dataset}/figs/bar_{metric}_scales{len(scale_dicts)}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora_anom', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--scales', default='3', help='number of multi-scale anomalies used in labeling')
    args = parser.parse_args()
    
    if not os.path.exists('output/figs'): os.mkdirs('output/figs')

    fpaths = glob.glob(f'output/{args.dataset}-{args.scales}*')
    num_scales = glob.glob()
    
    # record all model-wise results across anomaly scales
    scale_dicts = []
    for scale in range(args.scales):
        model_rocs,model_precisions,model_hits={},{},{}
        scale_models = glob.glob(f'output/{args.dataset}/{args.scales}-sc{scale+1}*')
        for scale_model in scale_models:
            with open(scale_model,'rb') as fin:
                result_dict = pkl.load(fin)
                rocauc,precision,hits=result_dict['rocs'],result_dict['precs'],result_dict['hits']
                model_name = scale_model.split("_")[-1].split(".pkl")[0]
                model_rocs[model_name] = rocauc ; model_precisions[model_name] = model_precisions ; model_hits[model_name] = hits
        scale_dicts.append({'rocs':model_rocs,'precisions':model_precisions,'hits':model_hits})

    # bar charts of all rocs/aucs/precisions
    plot_bar_charts(scale_dicts,'rocs',args.dataset)
    plot_bar_charts(scale_dicts,'precisions',args.dataset)

    # scale-wise line plots of precision@k/hit@k
    plot_hits(scale_dicts,'hits')
    plot_hits(scale_dicts,'precision')