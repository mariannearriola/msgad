import random
from pygsp import *
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import scipy.io as sio
from scipy.interpolate import make_interp_spline
from scipy import sparse
import argparse
import os

np.random.seed(123)
random.seed(123)
os.environ['PYTHONHASHSEED'] = str(123)

def gen_anomaly(s, anoms, anom_type, prob=0.01, d=1):
    #random.seed(123)
    anomaly_id = []
    if anoms is None:
        selected=[]
    else:
        selected=anoms
    s2 = s
    weight_ind = 0
    idx_track = 0
    for idx,i in enumerate(selected):
        if type(i) == np.int64 and G.anom_w is None:
            s2[i] *= 400
        elif type(i) == np.int64 and G.anom_w is not None:
            s2[i] *= 400*((d/G.anom_w[weight_ind]))
        elif G.anom_w is None:
            s2[i] *= 400
            anomaly_id.append(i)
            continue
        else:
            s2[i] *= 400*((d/G.anom_w[weight_ind]))
            weight_ind += 1
            idx_track = 1
            anomaly_id.append(i)
    return anomaly_id, s2

'''
def gen_anomaly(s, prob=0.01, d=1):
    s2 = np.zeros_like(s)
    random.seed(123)
    anomaly_id = []
    for i in range(len(s)):
        if random.random()<prob:
            s2[i] = np.random.randn()*d+1
            anomaly_id.append(i)
        else:
            s2[i] = s[i]
    return anomaly_id, s2
'''

def convert_color(aid, x):
    co = np.zeros([len(x), 4])
    for i in aid:
        co[i][0]=65/255
        co[i][1]=105/255
        co[i][2]=225/255
    return co


def convert_shape(aid, x, scale=15, mins=30, maxs=180):
    x2 = x.copy()
    for i in aid:
        x2[i] = x[i]*scale
        x2[i] = max(x2[i],mins)
        x2[i] = min(x2[i],maxs)
    return x2

def plot_diag(G, sa, anoms, e, U, bar_scale=4):
    f = []
    x = np.linspace(0,2-2/bar_scale, bar_scale)
    import ipdb ; ipdb.set_trace()
    print(np.unique((e/(2/bar_scale)).astype(np.int),return_counts=True)[1])
    for i in range(1):
        width = 2/(bar_scale*5)
        c = np.array(np.dot(U.transpose(), sa[i]+1))[0]
        M = np.zeros(bar_scale)
        for j in range(G.N):
            idx = min(int(e[j] / 0.05), bar_scale-1)
            M[idx] += c[j]**2
        M = M/sum(M)
        #y = np.mean(M)#,axis=1)
        y = M
        print(y)
        x = np.arange(y.shape[0])[15:25]
        y = y[15:25]
        spline = make_interp_spline(x, y)
        X_ = np.linspace(x.min(), x.max(), 400)
        Y_ = spline(X_)
        plt.xlabel('lambda')
        #plt.xticks(np.arange(y.shape[0])/bar_scale)
        plt.plot(X_,Y_)
    return f


def plotg(G, anoms, gs, e, U, xs=0, ys=0, ft=14):
    #ax1 = plt.subplot(gs[xs+0,ys+0])
    #plt.xticks([])
    #plt.yticks([])
    #inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2,
    #            subplot_spec=ax1, wspace=0.0, hspace=0.0)
    s = np.random.randn(G.N)
    sa = np.zeros([4, G.N])

    sigma = [1, 2, 5, 20]
    txt = ['σ=1', 'σ=2', 'σ=5', 'σ=20']
    for i in range(4):
        #aid, sa[i] = gen_anomaly(s, 0.05, sigma[i])
        aid, sa[i] = gen_anomaly(s, G.anoms, anoms, sigma[i])
        #ax = plt.Subplot(fig, inner_grid[i])
        if i == 0:
            aid=[]
        #G.plot(vertex_color=convert_color(aid, sa[i]), vertex_size=convert_shape(aid, sa[i]), colorbar=False, ax=ax, highlight=aid)  # highlight
        #fig.add_subplot(ax)
        #plt.xticks([])
        #plt.yticks([])
        plt.title('')
        #plt.text(0.8, 0.93, txt[i], horizontalalignment='center',
        #     verticalalignment='center',
        #     transform = ax.transAxes,fontsize=ft)

    #ax1 = plt.subplot(gs[xs+1,ys+0])
    f = plot_diag(G, sa, anoms, e, U, bar_scale = 40)
    '''
    if ys == 0:
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.yticks(fontsize=ft)
        plt.xlabel('λ\n(a)', fontsize=ft+1)
    else:
        plt.yticks([])
        plt.xlabel('λ\n(c)', fontsize=ft+1)
    plt.xticks([0,0.5,1,1.5,2], ['0','0.5','1','1.5','2'])
    plt.xticks(fontsize=ft)
    plt.legend(handles=f, labels=txt, loc='upper left', fontsize=ft)

    p = [0., 0.01, 0.05, 0.2]
    ax1 = plt.subplot(gs[xs+0,ys+1])
    plt.xticks([])
    plt.yticks([])
    inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2,
                subplot_spec=ax1, wspace=0.0, hspace=0.0)

    txt = ['α=0%','α=1%','α=5%','α=20%']
    for i in range(4):
        aid, sa[i] = gen_anomaly(s, p[i], 5)
        ax = plt.Subplot(fig, inner_grid[i])
        if i == 0:
            aid=[]
        #G.plot(vertex_color=convert_color(aid, sa[i]), vertex_size=convert_shape(aid, sa[i]), colorbar=False, ax=ax, highlight=aid)  # highlight
        fig.add_subplot(ax)
        plt.xticks([])
        plt.yticks([])
        plt.title('')
        plt.text(0.8, 0.93, txt[i], horizontalalignment='center',
             verticalalignment='center', transform = ax.transAxes,fontsize=ft)


    ax1 = plt.subplot(gs[xs+1,ys+1])
    if ys==0:
        plt.xlabel('λ\n(b)', fontsize=ft+1)
    else:
        plt.xlabel('λ\n(d)', fontsize=ft+1)
    f = plot_diag(G, sa)
    plt.legend(handles=f, labels=txt, loc='upper left',fontsize=ft)
    plt.yticks([])
    plt.xticks([0,0.5,1,1.5,2], ['0','0.5','1','1.5','2'])
    plt.xticks(fontsize=ft)
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='weibo')
    parser.add_argument('--load_lapl', default=False)
    parser.add_argument('--scale',type=int)
    parser.add_argument('--drop_anom',type=bool,default=False)
    parser.add_argument('--exp',type=str,default='')
    args = parser.parse_args()

    #fig = plt.figure(figsize=(15.5, 8.5), dpi=300)
    anoms = ['all','none','sc1','sc2','sc3','single','random']
    #anoms = ['single','single','single','single']
    #anoms = ['none']
    plt.figure()
    for anom_ind,anom in enumerate(anoms):
        print(f'getting spectral energy distribution of {anom} for {args.dataset}')
        gs = gridspec.GridSpec(2, 4)
        gs.update(left=0.05, right=0.98, top=0.97, bottom=0.1, wspace=0, hspace=0)
        if anom_ind == 0 or args.drop_anom == True:
            G = graphs.MultiScale(args.dataset,anom,args.drop_anom)
            if args.load_lapl:
                #U,e,L_inv = sio.loadmat(f'{args.dataset}_lapl.mat')['U'],sio.loadmat(f'{args.dataset}_lapl.mat')['e'][0],sio.loadmat(f'{args.dataset}_lapl.mat')['L_inv']
                U,e = sio.loadmat(f'{args.dataset}_lapl_drop{args.drop_anom}_ind{anom_ind}.mat')['U'].todense(),np.array(sio.loadmat(f'{args.dataset}_lapl_drop{args.drop_anom}_ind{anom_ind}.mat')['e'].todense())[0]
                #U,e = sio.loadmat(f'{args.dataset}_lapl_drop{args.drop_anom}_ind{anom_ind}_exp{args.exp}.mat')['U'].todense(),np.array(sio.loadmat(f'{args.dataset}_lapl_drop{args.drop_anom}_ind{anom_ind}_exp{args.exp}.mat')['e'].todense())[0]
            else:
                G.compute_laplacian('normalized')
                G.compute_fourier_basis()
                U,e = G.U,G.e
                #sio.savemat(f'{args.dataset}_lapl_drop{args.drop_anom}.mat',{'U':G.U,'e':G.e,'L_inv':G.L_inv})
                sio.savemat(f'{args.dataset}_lapl_drop{args.drop_anom}_ind{anom_ind}_exp{args.exp}.mat',{'U':sparse.csr_matrix(G.U),'e':sparse.csr_matrix(G.e)})
        else:
            G.anoms,G.anom_tot,G.anom_w = G.update_anom(anom)
        plotg(G, anom, gs, e, U, 0, 2)
        if not os.path.exists('vis/'):
            os.makedirs('vis/')
        plt.savefig(f'vis/anoms_vis_{args.dataset}_drop{args.drop_anom}_{args.exp}.png')
    plt.legend(anoms)
    if not os.path.exists('vis/'):
        os.makedirs('vis/')
    plt.savefig(f'vis/anoms_vis_{args.dataset}_drop{args.drop_anom}_{args.exp}.png')
  