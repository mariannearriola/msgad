import MADAN.Madan as md

class MADAN():
    def forward():
        adj = adj.adjacency_matrix()
        adj = adj.sparse_resize_((adj.size(0), adj.size(0)), adj.sparse_dim(), adj.dense_dim())
        idx=adj.coalesce().indices()
        nx_graph=nx.from_edgelist([(i[0].item(),i[1].item()) for i in idx.T])
        #feats = g_batch.ndata['feature'].cpu()
        node_dict = None

        nodes = list(max(nx.connected_components(nx_graph), key=len))
        #node_dict = {k.item():v for k,v in zip(list(in_nodes.detach().cpu()),np.arange(len(list(nx_graph.nodes))))}
        node_dict = {k:v for k,v in zip(nodes,np.arange(len(list(nx_graph.nodes))))}
        rev_node_dict = {v: k for k, v in node_dict.items()}
        nx_graph = nx.subgraph(nx_graph, nodes)
        nx_graph = nx.from_numpy_matrix(nx.adjacency_matrix(nx_graph))
        feats = feats[nodes]
        try:
            madan = md.Madan(nx_graph, attributes=feats, sigma=0.08)
        except Exception as e:
            print('eigenvalues dont converge',e)
            continue
        import ipdb ; ipdb.set_trace()
        
        time_scales   =   np.concatenate([np.array([0]), 10**np.linspace(0,5,50)])
        import ipdb ; ipdb.set_trace()
        #madan.anomalous_nodes=[node_dict[j] for j in np.intersect1d(anoms,np.array(list(node_dict.keys()))).tolist()]
        madan.anomalous_nodes=[node_dict[j] for j in anoms]
        madan.scanning_relevant_context_time(time_scales)
        import ipdb ; ipdb.set_trace()
        madan.compute_concentration(1000)
        #anoms_detected=madan.anomalous_nodes
        if node_dict and len(anoms_detected)>0:
            anoms_detected = rev_node_dict[anoms_detected]
        if len(anoms_detected)>0:
            print('anom found')
            import ipdb ; ipdb.set_trace()
            print('hi')
        iter += 1
        