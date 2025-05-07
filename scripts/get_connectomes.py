#%% 
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Union, Any
import pickle

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
from scripts.network_manipulation_functions import generate_mirror_network, neuron_duplication, neuron_deletion

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

# %%  view as graph (currently not really used)

#G = build_adj_directed_graph(adj)

# warning: plotting takes a few minutes
#plot_nx_digraph(G, node_size=1, plot_scale=0.01)

# %% set random seed, initiate log and generate mirror adjacency matrix to manipulate
seed = 42
#generate numpy random instance
rng = np.random.default_rng(seed=seed)

#initiate instance of ChangeLog class
log = ChangeLog()

# get mirror adjacency matrix
adj_mirror = generate_mirror_network(adj)

# %% apply network manipulations 

# neuron-level manipulations
n_ops_neuron = 30
adj_mirror, log = neuron_duplication(adj_mirror, log, rng, n_ops=n_ops_neuron)
adj_mirror, log = neuron_deletion(adj_mirror, log, rng, n_ops=n_ops_neuron)


# %% save altered adjacency matrix and change log 

#sanity check to see that changes have been made 
print(log)
#save new adjacency matrix
adj_mirror.to_csv(path_for_data+'adj_mirror.csv')

#save change log class instance
with open(path_for_data+'change_log.pkl', 'wb') as f:
    pickle.dump(log, f)

print('adjancey matrix and log saved')

# %%
