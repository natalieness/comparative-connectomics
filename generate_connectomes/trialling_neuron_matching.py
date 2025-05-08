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
print(f"Adjacency matrix shape: {adj.shape}")
print(f'Number of synapses: {sum(adj > 0)}')
# %% get pairs 

pairs_data = pd.read_csv(path_for_data+'pairs-2022-02-14.csv')
n_matched = pairs_data.shape[0]
print(f"Number of matched pairs: {n_matched}, total number of neurons: {n_matched*2}")
u_reg = pairs_data['region'].unique()
print(f"Unique regions: {u_reg}")

# %% 
adj_ids = list(adj.columns)
n_adj_ids = len(adj_ids)
print(f"Number of neurons in adj matrix: {n_adj_ids}")

# %% 
#get all neurons that are matched across hemispheres
all_matched = np.concat((pairs_data['leftid'].values, pairs_data['rightid'].values), axis=0)

#get overlap between matched neurons and neurons in adjacency matrix
matched_in_adj = np.intersect1d(all_matched, adj_ids)
unique_to_adj = np.setdiff1d(adj_ids, matched_in_adj)
unique_to_pairs = np.setdiff1d(all_matched, matched_in_adj)
print(f"Number of matched neurons in adjacency matrix: {len(matched_in_adj)}")
print(f"Number of neurons unique to adjacency matrix (not in pairs): {len(unique_to_adj)}")
print(f"Number of neurons unique to pairs (not in adj): {len(unique_to_pairs)}")

# %% get overlap with all neurons where both left and right are matched 

bilateral_skids_in_adj = []
for row in range(pairs_data.shape[0]):
    left = pairs_data['leftid'].loc[row]
    right = pairs_data['rightid'].loc[row]
    if left in adj_ids and right in adj_ids:
        bilateral_skids_in_adj.append((left, right))

n_bilateral_skids_in_adj = len(bilateral_skids_in_adj)
print(f"Number of pairs in adjacency matrix: {n_bilateral_skids_in_adj}, total number of neurons: {n_bilateral_skids_in_adj*2} out of {n_adj_ids}")

# %%
