#%% 
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Union, Any
import pickle
from itertools import chain

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from contools import Celltype, Celltype_Analyzer, Promat
import pymaid
from pymaid_creds import url, name, password, token

from generate_connectomes.get_pairs_dict import create_pairs_dict
from generate_connectomes.nx_graph_functions import build_adj_directed_graph, plot_nx_digraph
from generate_connectomes.change_class import ChangeLog
from generate_connectomes.network_manipulation_functions import generate_mirror_network, neuron_duplication, neuron_deletion, new_rand_neurons

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
print(f"Adjacency from pymaid, matrix shape: {adj.shape}")
print(f'Number of synaptic connections: {(adj > 0).sum().sum()}')

# also get information about the neurons ? position of soma?



# %%  get and filter by pairs data to only include neurons matched across hemispheres

pairs_data = pd.read_csv(path_for_data+'pairs-2022-02-14.csv')
#get all neurons that are matched across hemispheres
all_matched = np.concat((pairs_data['leftid'].values, pairs_data['rightid'].values), axis=0)
adj_ids = list(adj.columns)
n_adj_ids = len(adj_ids)

#get overlap between matched neurons and neurons in adjacency matrix
matched_in_adj = np.intersect1d(all_matched, adj_ids)
unique_to_adj = np.setdiff1d(adj_ids, matched_in_adj)
unique_to_pairs = np.setdiff1d(all_matched, matched_in_adj)

bilateral_skids_in_adj = []
for row in range(pairs_data.shape[0]):
    left = pairs_data['leftid'].loc[row]
    right = pairs_data['rightid'].loc[row]
    if left in adj_ids and right in adj_ids:
        bilateral_skids_in_adj.append((left, right))

n_bilateral_skids_in_adj = len(bilateral_skids_in_adj)
print(f"Number of pairs in adjacency matrix: {n_bilateral_skids_in_adj}, total number of neurons: {n_bilateral_skids_in_adj*2} out of {n_adj_ids}")
bilateral_skids_in_adj_flat = np.unique(np.array(list(chain.from_iterable(bilateral_skids_in_adj))))

bilateral_adj = adj.loc[bilateral_skids_in_adj_flat, bilateral_skids_in_adj_flat]

print(f"Filtered adjacency matrix shape: {bilateral_adj.shape}")

#save adjacency matrix 
bilateral_adj.to_csv(path_for_data+'b_adj.csv')
print('Adjacency matrix (with bilateral pairs) saved')

#get dict to match neuron ids to hemisphere, region and partners 

pairs_dict = create_pairs_dict(pairs_data)

# %% set random seed, initiate log and generate mirror adjacency matrix to manipulate
seed = 42
#generate numpy random instance
rng = np.random.default_rng(seed=seed)

#initiate instance of ChangeLog class
log = ChangeLog()

# get mirror adjacency matrix
adj_mirror = generate_mirror_network(bilateral_adj)

# %% apply network manipulations 
#get original number of neurons, so only those will be deleted 
n_og_neurons = adj_mirror.shape[0]

# neuron-level manipulations
n_ops_neuron = 30
adj_mirror, log = neuron_duplication(adj_mirror, log, rng, n_ops=n_ops_neuron)
adj_mirror, log = neuron_deletion(adj_mirror, log, rng, n_ops=n_ops_neuron, n_og_neurons=n_og_neurons)
adj_mirror, log = new_rand_neurons(adj_mirror, log, rng, n_ops=1)

'''
To do: 
- shuffle rows and columns of adjacency matrix
- replace index/column names with realistic neuron names
'''

# %% save altered adjacency matrix and change log 

#sanity check to see that changes have been made 
print(log)
#save new adjacency matrix
adj_mirror.to_csv(path_for_data+'b_adj_mirror.csv')

#save change log class instance
with open(path_for_data+'change_log.pkl', 'wb') as f:
    pickle.dump(log, f)

print('Adjancency matrix and log saved')
### TODO: also save log as csv file to be safe it will be accessible

# %%
