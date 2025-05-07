### Functions to construct a projected graph ###

def build_adj_directed_graph(adj_matrix):
    ''' Input: pandas dataframe of adjacency matrix'''
    G = nx.from_numpy_array(adj_matrix.values, 
                             create_using=nx.DiGraph, 
                             parallel_edges=False, 
                             nodelist=adj_matrix.columns)
    #G = nx.DiGraph()
    # Add neurons as nodes
    #all_nodes = np.array(adj_matrix.columns)
    #G.add_nodes_from(all_nodes)

    #for (g1, g2), count in group_pair_counts.items():
     #   G.add_edge(g1, g2, weight=count)
    return G

def graph_normalize_weights(G, factor='mean'):
    ''' 
    Normalize weights of edges in graph G based on: 
    - 'mean': Mean weight of all edges
    - 'log': Logarithm of the mean weight
    - 'jaccard': Jaccard similarity of the group/ group overlap
    '''
    if factor == 'mean':
        mean_weight = np.mean([G[u][v]['weight'] for u, v in G.edges()])
        for u, v in G.edges():
            G[u][v]['weight'] /= mean_weight
    elif factor == 'log':
        mean_weight = np.mean([G[u][v]['weight'] for u, v in G.edges()])
        for u, v in G.edges():
            G[u][v]['weight'] = np.log10(G[u][v]['weight'] / mean_weight)
    elif factor == 'jaccard':
        for u, v in G.edges():
            w = G[u][v]['weight']
            deg_u = sum(d['weight'] for _, _, d in G.edges(u, data=True))
            deg_v = sum(d['weight'] for _, _, d in G.edges(v, data=True))
            G[u][v]['weight'] = w / (deg_u + deg_v - w)
    else: 
        print("Unknown normalization factor. No normalization applied.")
    return G

### Functions to plot projected group graphs ###

def plot_nx_digraph(G, node_size=10, node_colors=None, plot_scale=1, save_fig=False, path=''):

    #pos = nx.circular_layout(G.subgraph(G.nodes))
    pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    #get colors for nodes 
    if node_colors is None:
        node_colors = ['lightblue' for _ in G.nodes()]
    else:
        node_colors = [node_colors.get(node, 'lightblue') for node in G.nodes()]

    #scale all weights by a factor for visualization
    edge_weights= [i*plot_scale for i in edge_weights]
    nx.draw(
        G, pos,
        with_labels=False,
        width=edge_weights,  # Line thickness ~ frequency
        node_color=node_colors,
        node_size=node_size,
        arrowsize=0.5,
        font_size=10,
        edge_color='black' 
    )
    plt.title("Projected Interaction Graph")
    if save_fig:
        plt.savefig(path)

