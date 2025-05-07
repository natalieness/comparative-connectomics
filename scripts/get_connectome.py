#%% 
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Union, Any

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from contools import Celltype, Celltype_Analyzer, Promat
import pymaid
from pymaid_creds import url, name, password, token

from scripts.nx_graph_functions import build_adj_directed_graph, plot_nx_digraph
from scripts.change_class import ChangeLog

rm = pymaid.CatmaidInstance(url, token, name, password)

#get parent directory path
current_file = __file__  
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
path_for_data = parent_dir+'/data/'

#%% Get neurons to include
# select neurons to include 
wanted_neurons = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
all_neurons = list(np.setdiff1d(wanted_neurons, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)

print(f"Working with {len(all_neurons)} neurons")
# get all synaptic sites associated with neurons 
# links = pymaid.get_connector_links(all_neurons, chunk_size=50)
# %% get adjacency matrix between neurons
adj = pymaid.adjacency_matrix(all_neurons) #consider syn threshold?
print(f"Adjacency matrix shape: {adj.shape}")
print(f'Number of synapses: {sum(adj)}')

#save adjacency matrix 
adj.to_csv(path_for_data+'adj.csv')

# %%

G = build_adj_directed_graph(adj)

# %%
plot_nx_digraph(G, node_size=1, plot_scale=0.01)

# %%
seed = 42
#generate numpy random instance
rng = np.random.default_rng(seed=seed)

def generate_mirror_network(adj_):
    adj_mirror = adj_.copy()
    return adj_mirror

adj_mirror = generate_mirror_network(adj)
#%% class instance to save matrix changes 


# %%
log = ChangeLog()

def neuron_duplication(adj_, log, n_ops=1):
    new_adj = adj_.copy()
    #calling n_neurons here prevents duplication of new neurons
    n_neurons = new_adj.shape[0]
    for n in range(n_ops):
        # select a random neuron idx
        neuron_idx = rng.integers(low=0, high=n_neurons, size=1)[0]
        # get the neuron name
        neuron_name = new_adj.columns[neuron_idx]

        #get neuron outputs 
        n_source = new_adj.iloc[neuron_idx, :].values
        syn_out = np.where(n_source > 0)[0]
        w_out = n_source[syn_out]
        new_w_out = rng.shuffle(w_out)
        #set new weights 
        new_n_source = n_source.copy()
        new_n_source[syn_out] = new_w_out

        #set new neuron name
        new_n_name = f'new_{n}'
        new_adj.loc[new_n_name] = new_n_source

        #get neuron inputs
        n_target = new_adj.iloc[:, neuron_idx].values
        n_in = np.where(n_target > 0)[0]
        w_in = n_target[n_in]
        new_w_in = rng.shuffle(w_in)
        #set new weights
        new_n_target = n_target.copy()
        new_n_target[n_in] = new_w_in
    
        new_adj[new_n_name] = new_n_target

        log.add_neuron_change(
            operation='neuron_duplication',
            source_index=neuron_name,
            new_index=new_n_name,
            weight_handling='shuffling existing'
        )

    return new_adj, log

def neuron_deletion(adj_, log, n_ops=1):
    new_adj = adj_.copy()
    n_neurons = new_adj.shape[0]
    for n in range(n_ops):
        # select a random neuron idx
        neuron_idx = rng.integers(low=0, high=n_neurons, size=1)[0]
        # get the neuron name
        neuron_name = new_adj.columns[neuron_idx]

        # delete the neuron from the adjacency matrix
        new_adj.drop(columns=[neuron_name], inplace=True)
        new_adj.drop(index=[neuron_name], inplace=True)

        log.add_neuron_change(
            operation='neuron_deletion',
            source_index=neuron_name,
            new_index=None,
            weight_handling='NA'
        )

    return new_adj, log


n_ops_neuron = 30
adj_mirror, log = neuron_duplication(adj_mirror, log, n_ops=n_ops_neuron)
adj_mirror, log = neuron_deletion(adj_mirror, log, n_ops=n_ops_neuron)







# %% save altered adjacency matrix and change log 

#save new adjacency matrix
adj_mirror.to_csv(path_for_data+'adj_mirror.csv')

print(log)

# %%
