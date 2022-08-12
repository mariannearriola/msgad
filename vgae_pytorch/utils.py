import numpy as np
import scipy

def num_pos_edges(adj):
    return np.count_nonzero(np.triu(adj,k=1))

def anom_classify(loss,A_pred,metric):
    #import ipdb ; ipdb.set_trace()
    if metric == 'skew':
        loss = scipy.stats.skew(loss,axis=0)
    node_idx = np.arange(len(A_pred))
    idx = np.argsort(loss)
    rankings = loss[idx]
    node_idx = node_idx[idx]
    return rankings, node_idx

def calc_anom_precision(rankings,node_idx,all_anom,num_nodes,num_anoms):
    top_rankings = rankings[:num_anoms]
    top_nodes = node_idx[:num_anoms]

    correct_anoms = np.intersect1d(top_nodes,all_anom)

    prec = correct_anoms.shape[0]/num_anoms
    anom_sum = correct_anoms.shape[0]
    #import ipdb ; ipdb.set_trace()
    print(prec,anom_sum)
    return prec,anom_sum

def tic():
	#Homemade version of matlab tic and toc functions
	import time
	global startTime_for_tictoc
	startTime_for_tictoc = time.time()

def toc():
	import time
	if 'startTime_for_tictoc' in globals():
		print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
	else:
		print("Toc: start time not set"	)
