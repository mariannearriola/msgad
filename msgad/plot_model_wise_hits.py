import pickle as pkl
import argparse
import matplotlib.pyplot as plt
import os
import glob
import numpy as np


def plot_hits(scale_dicts,metric,dataset):
    model_names = scale_dicts[0]['hits'].keys()
    num_groups = scale_dicts[0]['hits'][list(scale_dicts[0]['hits'].keys())[0]].shape[0]
    for group in range(num_groups):
        for scale,scale_dict in enumerate(scale_dicts):
            plt.figure()
            if group == num_groups-1:
                plt.title(f'{(metric.capitalize())} @ K for scale all anomalies in {dataset.capitalize()}')
            else:
                plt.title(f'{(metric.capitalize())} @ K for scale {group} anomalies in {dataset.capitalize()}')
            plt.xlabel('# predictions')
            if metric == 'hits':
                plt.ylabel(f'# anomalies found')
            elif metric == 'precision':
                plt.ylabel(f'% anomalies found')
            for model,hits in scale_dict['hits'].items():
                hits_plot = np.cumsum(hits[group])

                if metric == 'precision':
                    hits_plot /= hits[group].nonzero()[0].shape[0]
                elif metric != 'hits':
                    raise "hit type not found"
                plt.plot(np.arange(hits_plot.shape[0]),hits_plot)
            model_names = [i.capitalize() for i in model_names]
            plt.legend(model_names,loc='lower right')
            plt.savefig(f'output/{dataset}/figs/hits_{metric}_totscales_{len(scale_dicts)}_scale{group}.png')

def plot_bar_charts(scale_dicts,metric,dataset):
    plt.figure()
    plt.title(f'{(metric.capitalize())} for {dataset.capitalize()}')
    plt.xlabel('Anomaly group')
    plt.ylabel(f'{metric.capitalize()}')
    model_names = scale_dicts[0][metric].keys() ; model_names = scale_dicts[0]['hits'].keys()
    width = 0.25
    for idx,model_name in enumerate(model_names):
        # NOTE: last index is the result across all scales
        bar_data = [scale_dicts[i][metric][model_name][-1] for i in range(len(scale_dicts))]

        plt.bar(np.arange(len(bar_data))+(idx*width),bar_data,label=model_name)
    model_names = [i.capitalize() for i in model_names]
    plt.legend(model_names,loc='upper right')
    plt.savefig(f'output/{dataset}/figs/bar_{metric}_scales{len(scale_dicts)}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora_anom', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--scales', default=3, type=int, help='number of multi-scale anomalies used in labeling')
    args = parser.parse_args()
    
    if not os.path.exists(f'output/{args.dataset}/figs'): os.makedirs(f'output/{args.dataset}/figs')

    fpaths = glob.glob(f'output/{args.dataset}-{args.scales}*')
    
    # record all model-wise results across anomaly scales
    scale_dicts = []
    for scale in range(args.scales):
        model_rocs,model_precisions,model_hits={},{},{}
        scale_models = glob.glob(f'output/{args.dataset}/{args.scales}-sc{scale+1}*')
        if len(scale_models) == 0: break
        for scale_model in scale_models:
            with open(scale_model,'rb') as fin:
                result_dict = pkl.load(fin)
                rocauc,precision,hits=result_dict['rocs'],result_dict['precs'],result_dict['hits']
                model_name = scale_model.split("_")[-1].split(".pkl")[0]
                model_rocs[model_name] = rocauc ; model_precisions[model_name] = precision ; model_hits[model_name] = hits
        scale_dicts.append({'rocs':model_rocs,'precisions':model_precisions,'hits':model_hits})

    # bar charts of all rocs/aucs/precisions
    plot_bar_charts(scale_dicts,'rocs',args.dataset)
    plot_bar_charts(scale_dicts,'precisions',args.dataset)

    # scale-wise line plots of precision@k/hit@k
    plot_hits(scale_dicts,'hits',args.dataset)
    plot_hits(scale_dicts,'precision',args.dataset)