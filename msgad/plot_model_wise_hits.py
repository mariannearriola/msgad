import pickle as pkl
import argparse
import matplotlib.pyplot as plt
import os
import glob
import numpy as np


def plot_hits(scale_dicts,metric,dataset,scale):
    model_names = list(scale_dicts.keys())
    num_groups = len(scale_dicts[model_names[0]][0]['hits'])
    for group in range(num_groups):
        #for scale,scale_dict in enumerate(scale_dicts):
        plt.figure()
        if group == 0:
            plt.title(f'{(metric)} @ K for single-node anomalies in {dataset.capitalize()}')
        elif group == num_groups-1:
            plt.title(f'{(metric)} @ K for all anomalies in {dataset.capitalize()}')
        else:
            plt.title(f'{(metric)} @ K for scale {group} anomalies in {dataset.capitalize()}')
        plt.xlabel('# predictions')
        if metric == 'hits':
            plt.ylabel(f'# anomalies found')
        elif metric == 'precision':
            plt.ylabel(f'% anomalies found')
        for model,info in scale_dicts.items():
            try:
                hits_plot = np.cumsum(info[scale]['hits'][group])
            except:
                raise Exception(f'score formatting for {model} failed')
            if metric == 'precision':
                hits_plot /= info[scale]['hits'][group].nonzero()[0].shape[0]
            elif metric != 'hits':
                raise "hit type not found"
            plt.plot(np.arange(hits_plot.shape[0]),hits_plot)
        model_names = [i for i in model_names]
        plt.legend(model_names,loc='lower right')
        plt.savefig(f'output/{dataset}/figs/hits_{metric}_totscales_{len(scale_dicts[model_names[0]])}_scale{group}.png')

def plot_bar_charts(scale_dicts,metric,dataset,scale):
    ax =plt.figure()
    plt.title(f'Scale {scale+1} {(metric)} for {dataset.capitalize()}')
    plt.xlabel('Anomaly group')
    plt.ylabel(f'{metric}')
    
    model_names = scale_dicts.keys() 
    width = 0.1
    
    for idx,model_name in enumerate(model_names):
        # NOTE: last index is the result across all scales
        #bar_data = [scale_dicts[i][metric][model_name] for i in range(len(scale_dicts))]
        bar_data = scale_dicts[model_name][scale][metric]
        plt.bar(np.arange(len(bar_data))+(idx*width),bar_data,label=model_name,width=width)
    model_names = [i for i in model_names]
    plt.legend(model_names,loc='upper right',fontsize='x-small')
    xticks = ['Single scale'] + [f'Scale {ind+1}' for ind in range(len(bar_data)-2)] + ['All anoms']
    plt.xticks(np.arange(len(xticks)),xticks)
    plt.yticks(np.arange(0.,1.,.1))
    #import ipdb ; ipdb.set_trace()
    plt.savefig(f'output/{dataset}/figs/bar_{metric}_scales{len(scale_dicts[model_name])}_scale{scale}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora_anom', help='dataset name: Flickr/ACM/BlogCatalog')
    parser.add_argument('--scales', default=3, type=int, help='number of multi-scale anomalies used in labeling')
    args = parser.parse_args()
    
    if not os.path.exists(f'output/{args.dataset}/figs'): os.makedirs(f'output/{args.dataset}/figs')


    all_scale_models = glob.glob(f'output/{args.dataset}/{args.scales}-sc1*')    
    # record all model-wise results across anomaly scales
    scale_dicts = {}
    for scale in range(args.scales):
        #scale_models = glob.glob(f'output/{args.dataset}/{args.scales}-sc{scale+1}*')
        if len(all_scale_models) == 0: break
        for model in all_scale_models:
            scale_model = f'output/{args.dataset}/{args.scales}-sc' + str(scale+1) +  model.split(f'{args.scales}-sc1')[1]
            if not os.path.exists(scale_model):
                scale_model = model
            with open(scale_model,'rb') as fin:
                result_dict = pkl.load(fin)
                rocauc,precision,hits=result_dict['rocs'],result_dict['precs'],result_dict['hits']
                full_anom = hits[:-1].nonzero()[0].shape[0]
                precision = hits[:,:full_anom].sum(1)/hits.sum(1)
                #import ipdb ; ipdb.set_trace()
                model_name = scale_model.split("_")[-1].split(".pkl")[0]
                if model_name.capitalize() not in scale_dicts.keys(): scale_dicts[model_name.capitalize()] = []
                scale_dicts[model_name.capitalize()].append({'rocs':rocauc,'precisions':precision,'hits':hits})
    # bar charts of all rocs/aucs/precisions3-sc2_weibo-3scales-nobatching-multiscaledominant-10neighbors-1000e-128hid-clustperst.pkl
    #for scale,scale_dict in enumerate(scale_dicts):
    for scale in range(args.scales):
        plot_bar_charts(scale_dicts,'rocs',args.dataset,scale)
        plot_bar_charts(scale_dicts,'precisions',args.dataset,scale)

        # scale-wise line plots of precision@k/hit@k
        plot_hits(scale_dicts,'hits',args.dataset,scale)
        plot_hits(scale_dicts,'precision',args.dataset,scale)