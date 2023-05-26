from pygsp import graphs, filters, plotting
import numpy as np
from sklearn import preprocessing
import networkx as nx
from MADAN.Plotters import *
import pandas as pd
from collections import Counter
from MADAN.LouvainClustering_fast import Clustering, norm_var_information
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
import numpy.linalg as npla
import copy
import os
from scipy.interpolate import make_interp_spline

import pdb	

class Madan(object):


	def __init__(self, net, exp_params, attributes=None, sigma=None):

		self.N          = net.order()
		self.network    = net
		self.attributes = attributes
		self.pl         = Plotters()
		self.f_matrix = attributes
		self.sc_label = []
		self.dataset = exp_params['DATASET']['NAME']
		self.epoch = exp_params['MODEL']['EPOCH']
		self.device = exp_params['DEVICE']
		self.exp_name = exp_params['EXP']
		self.label_type = exp_params['DATASET']['LABEL_TYPE']
		self.model = exp_params['MODEL']['NAME']
		'''
		self.f_matrix = np.zeros((self.N,len(attributes)))
		
		for i, att in enumerate(attributes):
			attribs       =  nx.get_node_attributes(net,att)
			self.f_matrix[:,i] =  list(attribs.values())

		'''
		f_scaled = preprocessing.MinMaxScaler().fit_transform(self.f_matrix)

		if sigma is None:
		   self.sigma = f_scaled.std()
		else:
		   self.sigma = sigma   

		#--------------------------------------------------------------------
		self.A = nx.adjacency_matrix(net).toarray()
		self.W = self._get_weigth_matrix_vectors(self.A,f_scaled,self.sigma)
		#self.G = graphs.Graph(self.W)
		self.G = graphs.Graph(self.A)
		
		#self.computing_fourier_basis()
		#------------------------------------------------------------------------
		# Random walk components
		#------------------------------------------------------------------------
		v_ones      =  np.matrix(np.ones((self.N,1)))
		degree_vect =  self.W.sum(axis=1)                        # strengths vector
		D           =  np.matrix(np.diag(degree_vect))    
		self.avg_d  =  (v_ones.T*D*v_ones)[0,0]/self.N           # average strength

		#------------------------------------------------------------------------
		# stationary distribution
		#------------------------------------------------------------------------
		self.pi     =  v_ones.A.T.reshape(self.N)/self.N
		#------------------------------------------------------------------------

	def flatten_label(self,anoms,nodes):
		anom_flat = np.intersect1d(nodes,anoms[0],return_indices=True)[0]#[0]
		for i in anoms[1:]:
			anom_flat=np.concatenate((anom_flat,np.intersect1d(nodes,i,return_indices=True)[0]))#[0]))
		return anom_flat

	def organize_ms_labels(self,nodes,nx_dict):
		self.color_scheme = {'normal':'green','anom_sc1':'red','anom_sc2':'blue','anom_sc3':'purple','anom_single':'cyan'}
		self.color_arr = np.full(self.N,self.color_scheme['normal'], dtype=object)
		self.color_arr[np.vectorize(nx_dict.get)(self.flatten_label(self.sc_label['anom_sc1'],nodes))] = self.color_scheme['anom_sc1']
		self.color_arr[np.vectorize(nx_dict.get)(self.flatten_label(self.sc_label['anom_sc2'],nodes))] = self.color_scheme['anom_sc2']
		self.color_arr[np.vectorize(nx_dict.get)(self.flatten_label(self.sc_label['anom_sc3'],nodes))] = self.color_scheme['anom_sc3']
		self.color_arr[np.vectorize(nx_dict.get)(np.intersect1d(self.sc_label['single'],nodes))] = self.color_scheme['anom_single']
		self.sc1_anom=np.vectorize(nx_dict.get)(self.flatten_label(self.sc_label['anom_sc1'],nodes))
		self.sc2_anom=np.vectorize(nx_dict.get)(self.flatten_label(self.sc_label['anom_sc2'],nodes))
		self.sc3_anom=np.vectorize(nx_dict.get)(self.flatten_label(self.sc_label['anom_sc3'],nodes))
		self.single_anom=np.vectorize(nx_dict.get)(np.intersect1d(self.sc_label['single'],nodes))
		self.find_anoms = {}
		
		for i in self.sc1_anom:
			self.find_anoms[i] = 'sc1'
		for i in self.sc2_anom:
			self.find_anoms[i] = 'sc2'
		for i in self.sc3_anom:
			self.find_anoms[i] = 'sc3'
		for i in self.single_anom:
			self.find_anoms[i] = 'single'


	def computing_fourier_basis(self, chebychev=False):

		self.G.compute_fourier_basis() 

		
	def evaluating_heat_kernel(self, tau=0):	
		#------------------------------------------------------------------------
		# Computing Fourier basis evaluated at tau
		#------------------------------------------------------------------------
								
		kernel     =  np.exp(-tau/(self.G.e))#np.exp(-tau/self.avg_d*(self.G.e))#/self.G.lmax)) 
		exp_sigma  =  np.diag(kernel)
		self.Ht    =  np.dot(np.dot(self.G.U,exp_sigma),self.G.U.T)
		self.tau   =  tau
		
		

	def compute_concentration(self, tau=0, mat=None):		
		#import ipdb ; ipdb.set_trace()
		if mat is None:
			self.evaluating_heat_kernel(tau)
			self.concentration =  np.linalg.norm(self.Ht, axis=0, ord=2)
		else:
			self.concentration =  np.linalg.norm(mat, ord=2)
			'''
			self.concentration = np.zeros(self.N)
			for node in self.network.nodes:
				# Get the cluster assignment for the current node
				node_cluster = self.partitions[node]
				
				# Get the neighbors of the current node
				neighbors = self.network.neighbors(node)
				
				# Count the number of neighbors that belong to different clusters
				cut_value = sum(1 for neighbor in neighbors if self.partitions[neighbor] != node_cluster)
				
				# Store the cut value for the current node
				self.concentration[node] = cut_value
			'''

		self.concentration + 2*self.concentration.std()
	   
		thre = self.concentration.mean() + 2*self.concentration.std()
		#self.anomalies_labels = (self.concentration>=thre)*1
		#self.anomalous_nodes  = [i for i in range(0,len(self.anomalies_labels)) if self.anomalies_labels[i]==1]

	
	def compute_context_for_anomalies(self, mat=None, random_seed=2):

		#------------------------------------------------------------------------
		#  Find clusters with Louvain algorithm 
		#------------------------------------------------------------------------
		if mat is None:
			clustering   =  Clustering(p1=self.pi, p2=self.pi, T=self.Ht)    
		else:
			clustering   =  Clustering(p1=self.pi, p2=self.pi, T=mat)
		
		clustering.find_louvain_clustering(rnd_seed=random_seed)
		self.partition = clustering.partition.node_to_cluster_dict

		clust_labels   = np.array([self.partition[i] for i in range(len(self.partition))]) # Re-order

		val_clusters       = self._interpolate_comm(self.network, clust_labels, self.anomalous_nodes)
		self.interp_com   = dict(zip(range(0, len(val_clusters)), val_clusters))
		self.num_clusters = len(set(self.interp_com.values()))
		
	def plot_graph_concentration(self, coord=None):

		if coord is None:
			self.G.set_coordinates(iterations=1000, seed=100)
		else:	
			self.G.set_coordinates(coord)

		self.pl.visualize_graph_signal(self.G,self.concentration,title='t='+str(self.tau))
		#self.pl.plot_matrix(W, inter=0, title="W")
		
	def plot_graph_context(self, coord=None):
		
		if coord is None:
			self.G.set_coordinates(iterations=1000, seed=100)
		else:	
			self.G.set_coordinates(coord)

		#node_labels = dict(zip(self.network.nodes(), np.array(self.interp_com, dtype='int')))
		#self.pl.visualize_graph_signal(self.G, self.interp_com, node_labels=node_labels)    
		self.pl.visualize_graph_signal(self.G, np.array(list(self.interp_com.values())))    

	def plot_curve(self,ys,sort_idx,color):
		y = ys[sort_idx]#[np.where(self.color_arr[sort_idx]==color)]
		y[np.where(self.color_arr[sort_idx]!=color)] = 0.
		x = np.arange(y.shape[0])
		try:
			spline = make_interp_spline(x, y)
			X_ = np.linspace(x.min(), x.max(), 500)
			Y_ = spline(X_)
			plt.plot(X_,Y_)
		except Exception as e:
			plt.scatter(x,y)

	def plot_concentration(self,tau,plot_grad=False):
		#------------------------------------------------------------------------
		# Plotting concentration
		#------------------------------------------------------------------------
		print('PLOTTING',tau)
		comm_concent = np.zeros((self.N,2))
		comm_concent[:,0] = list(self.interp_com.values())
		if plot_grad:
			comm_concent[:,1] = self.concentration_grad
		else:
			comm_concent[:,1] = self.concentration
		
		df_comm_concent   = pd.DataFrame(comm_concent, columns=['groups','concentration'])
		#ax = plt.subplot(1, 1, 1)
		std_val = df_comm_concent['concentration'].std()
		concentrations = df_comm_concent['concentration'].to_numpy()
		sorted_idx = np.argsort(concentrations)
		#sorted_idx = np.argsort(self.color_arr)
		
		plt.figure()
		plt.bar(np.arange(self.N),concentrations[sorted_idx],color=self.color_arr[sorted_idx],width=1.)#,edgecolor=self.color_arr[sorted_idx])
		
		'''
		#self.plot_curve(concentrations,sorted_idx,self.color_scheme['normal'])
		if not np.isnan(concentrations).any():
			self.plot_curve(concentrations,sorted_idx,self.color_scheme['anom_sc1'])
			self.plot_curve(concentrations,sorted_idx,self.color_scheme['anom_sc2'])
			self.plot_curve(concentrations,sorted_idx,self.color_scheme['anom_sc3'])
			self.plot_curve(concentrations,sorted_idx,self.color_scheme['anom_single'])
		'''
		#df_comm_concent.plot(kind='bar', title='Node concentration', grid=False, y='concentration',rot=0, ax=ax, color=self.color_arr, cmap='viridis', fontsize=8, legend=False)
		plt.hlines(df_comm_concent['concentration'].mean() + 2.0*std_val, xmin=-1, xmax=self.N, linestyles='dashed', alpha=1.0, color='blue')
		#ax.set_facecolor((1.0, 1.0, 1.0))

		# TODO: check sc label
		anoms_detected = self.anomalous_nodes[np.where(concentrations[self.anomalous_nodes] > concentrations.mean()+std_val)[0]]

		#anom_types = [self.find_anoms[i] for i in anoms_detected]
		#if len(anom_types) > 0:
		#	print(np.unique(anom_types,return_counts=True)[-1])
		#	print(np.unique(anom_types,return_counts=True)[-1]/np.array([self.sc1_anom.shape[0],self.sc2_anom.shape[0],self.sc3_anom.shape[0],self.single_anom.shape[0]]))
		#sc_anoms_detected = 
		plt.ylim(0,concentrations[sorted_idx].max())
		norm_patch = mpatches.Patch(color='green', label='normal')
		sc1_patch = mpatches.Patch(color='red', label='anom_sc1')
		sc2_patch = mpatches.Patch(color='blue', label='anom_sc2')
		sc3_patch = mpatches.Patch(color='purple', label='anom_sc3')
		single_patch = mpatches.Patch(color='cyan', label='anom_single')

		plt.legend(handles=[norm_patch,sc1_patch,sc2_patch,sc3_patch,single_patch])
		fpath = f'vis/concentration/{self.dataset}/{self.model}/{self.label_type}/{self.epoch}/{self.exp_name}'
		if not os.path.exists(fpath):
			os.makedirs(fpath)
		if plot_grad: fname = f'{fpath}/concentration_grad_{tau}.png'
		else: fname = f'{fpath}/concentration_{tau}.png'
		plt.title(f'{tau}')
		plt.savefig(fname)
	
	# Only interpolate anomalies
	def _interpolate_comm(self, G,part, true_nodes):

		new_part = part
		for node in true_nodes:
			neigh_n =  list(G.neighbors(node))
			
			flag = len(set(part[neigh_n])) == len(part[neigh_n])
			if not(flag): 
				res            =  dict(Counter(part[neigh_n]))
				max_key        =  max(res, key=lambda k: res[k])
				new_part[node] =  max_key

		keys_comm = list(set(new_part))
		vals_comm = range(0,len(keys_comm))
		mapping = dict(zip(keys_comm, vals_comm))
		interp_comm_ok = np.array([mapping[val] for val in new_part])
		
		return interp_comm_ok	

	def _get_weigth_matrix_vectors(self, A,attrib_mat,sigma=1):
	
		N = A.shape[0]
		W = np.zeros((N,N))

		for i in range(0,N):
			for j in range(i,N):
				if A[i,j] == 1:	
								
					W[i,j] = np.linalg.norm(attrib_mat[i] - attrib_mat[j])
					W[j,i] = W[i,j]
			
		W_res  = np.exp(-0.5*np.square(W)/(sigma**2))
		
		return np.multiply(W_res,A)		

	def compute_voi(self, list_partitions): #M x N
		res = []
		for i in range(len(list_partitions)):
			for j in range(i, len(list_partitions)):
				if i!=j:
					voi = norm_var_information(list_partitions[i],list_partitions[j])
					res.append(voi)
				
		return np.mean(res)	


	def __processInput(self, i):
		clustering =  Clustering(p1=self.pi, p2=self.pi, T=self.Ht)      
		clustering.find_louvain_clustering(rnd_seed=i)
		return clustering.partition.node_to_cluster_dict

	def scanning_relevant_context(self, time, mats=None, n_jobs=1):
		'''
		#self.G.compute_laplacian('normalized')
		#self.computing_fourier_basis()
		eigenvalues = np.sort(self.G.e)[::-1]
		# Compute the decay rate of the eigenvalues
		decay_rate = np.abs(eigenvalues) / np.abs(eigenvalues[0])

		# Plot the decay rate
		plt.plot(decay_rate)
		plt.xlabel('Eigenvalue Index')
		plt.ylabel('Decay Rate')
		plt.title('Decay Rate of Graph Eigenvalues')
		plt.grid(True)
		plt.savefig('eval_decay.png')
		'''
		def processInput(i,mat):
			clustering =  Clustering(p1=self.pi, p2=self.pi, T=mat) 
			clustering.find_louvain_clustering(rnd_seed=i)
			return clustering.partition.node_to_cluster_dict
		self.num_com   = []
		self.voi_list  = []
		self.time      = time
		if mats is None:
			self.computing_fourier_basis()
		
		for inx, t in enumerate(time):
			print(inx)
			if inx%50==0:
				print("Processed %d/%d"%(inx,len(time)))
			#------------------------------------------------------------------     
			if mats is None:    
				self.evaluating_heat_kernel(t)
				#------------------------------------------------------------------ 
				# NOTE: does not find partitions?? too much heat?
				#import ipdb ; ipdb.set_trace()
				list_partitoins = Parallel(n_jobs=n_jobs)(delayed(processInput)(i,self.Ht) for i in range(1))
				#import ipdb ; ipdb.set_trace()
			else:
				list_partitoins = Parallel(n_jobs=n_jobs)(delayed(processInput)(i,mats[inx]) for i in range(1))
			#import ipdb ; ipdb.set_trace()
			
				self.concentration = np.zeros(self.N)
				net = nx.from_numpy_matrix(mats[inx])
				for node in net.nodes:
					# Get the cluster assignment for the current node
					node_cluster = list_partitoins[0][node]
					
					# Get the neighbors of the current node
					neighbors = [n for n in net.neighbors(node)]
					
					# Count the number of neighbors that belong to different clusters
					cut_value = sum(1 for neighbor in neighbors if list_partitoins[0][neighbor] != node_cluster)
					#cut_value = nx.cut_size(net,[node],neighbors)
					# Store the cut value for the current node
					self.concentration[node] = cut_value
				clust_labels   = np.array([list_partitoins[0][i] for i in range(len(list_partitoins[0]))]) # Re-order
				val_clusters       = self._interpolate_comm(self.network, clust_labels, self.anomalous_nodes)
				self.interp_com   = dict(zip(range(0, len(val_clusters)), val_clusters))
				self.num_clusters = len(set(self.interp_com.values()))
				self.plot_concentration(inx)
		
			#import ipdb ; ipdb.set_trace()
			'''
			self.X = np.zeros((self.N,  len(set(list_partitoins[0].values()))))
			for node, cluster in list_partitoins[0].items():
				self.X[node][cluster] = 1
			'''
			self.voi_list.append(self.compute_voi(list_partitoins))
			if mats is None:
				self.compute_context_for_anomalies()
			else:
				self.compute_context_for_anomalies(mat=mats[inx])
			self.num_com.append(self.num_clusters)

		#-----------------------------------------------------------------------                
		self.plot_relevant_context()
		#-----------------------------------------------------------------------            
		
		
	def plot_relevant_context(self):

		fig, ax1 = plt.subplots(figsize=(9,5))

		color = 'tab:blue'
		ax1.set_xlabel('time (s)', size=16)
		ax1.set_ylabel('Num communities', color=color, size=16)
		#ax1.set_yscale('log')
		#ax1.set_ylim([1,np.max(self.num_com)+10])
		#ax1.set_xscale('log')
		ax1.plot(self.time, self.num_com, color=color, linewidth=2)
		ax1.tick_params(axis='y', labelcolor=color)
		ax1.grid(True)
		#ax1.set_facecolor((1.0, 1.0, 1.0))

		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

		color = 'tab:red'
		ax2.set_ylabel('Variation of Information', color=color, size=16)  # we already handled the x-label with ax1
		ax2.plot(self.time, self.voi_list, color=color, linewidth=2, alpha=0.5)
		ax2.tick_params(axis='y', labelcolor=color)
		ax2.grid(True)

		fig.tight_layout()  # otherwise the right y-label is slightly clipped
		fpath = f'vis/contexts/{self.dataset}/{self.model}/{self.label_type}/{self.epoch}/{self.exp_name}'
		if not os.path.exists(fpath):
			os.makedirs(fpath)
		plt.savefig(f'{fpath}/contexts.png')
		#plt.show()


	def scanning_relevant_context_time(self,time,mats=None):
		#--------------------------------------------
		# Compute temporal partitions
		#--------------------------------------------
		mat_parts  = []
		for inx, t in enumerate(time):
			if inx%50==0:
				print("Processed %d/%d"%(inx,len(time)))
			if mats is None:    
				self.evaluating_heat_kernel(t)
				self.compute_context_for_anomalies()
				clustering =  Clustering(p1=self.pi, p2=self.pi, T=self.Ht)
				if inx > 0: self.prev_concentration = copy.deepcopy(self.concentration)
				self.compute_concentration(tau=t)
				if inx > 0:
					self.concentration_grad = self.concentration - self.prev_concentration
					self.plot_concentration(t, plot_grad=True)
				self.plot_concentration(t)
			else:
				clustering =  Clustering(p1=self.pi, p2=self.pi, T=mats[inx])
				self.compute_concentration(tau=t, mat=mats[inx])
				self.plot_concentration(t)
			clustering.find_louvain_clustering(rnd_seed=42)
			partition  = clustering.partition.node_to_cluster_dict 
			mat_parts.append(partition)

		#-----------------------------------------------------------------------
		self.voi_mat = np.zeros((len(time),len(time)))
		
		print("Computing variation of information between V(t,t')...")           
		for i in range(len(mat_parts)):                       
			
			for j in range(i, len(mat_parts)):
				if i!=j:
					voi = norm_var_information(mat_parts[i],mat_parts[j])
					self.voi_mat[i,j] = voi
					self.voi_mat[j,i] = voi       

		#-----------------------------------------------------------------------            
		self.plot_voi_matrix()            
		#-----------------------------------------------------------------------            


	def plot_voi_matrix(self):

		fig, ax = plt.subplots(figsize=(8,8))
		im = ax.imshow(self.voi_mat, origin='lower')
		ax.grid(False)
		#ax.set_yscale('log')
		#ax.set_xscale('log')
		fig.colorbar(im, orientation='vertical')
		fpath = f'vis/concentration/{self.dataset}/{self.model}/{self.label_type}/{self.epoch}/{self.exp_name}'
		if not os.path.exists(fpath):
			os.makedirs(fpath)
		plt.savefig(f'{fpath}/voi.png')
		#plt.show()
