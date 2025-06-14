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

from graspologic.inference.density_test import density_test
from graspologic.simulations import er_np
from graspologic.plot import heatmap
from graspologic.utils import binarize
from contools import Celltype, Celltype_Analyzer, Promat
import pymaid
from pymaid_creds import url, name, password, token

rm = pymaid.CatmaidInstance(url, token, name, password)

#get parent directory path
current_file = __file__  
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
path_for_data = parent_dir+'/data/'

#%% Load adjacency matrix
adj1 = pd.read_csv(path_for_data+'b_adj.csv', index_col=0)
print(f"Adjacency matrix 1 shape: {adj1.shape}")

adj2 = pd.read_csv(path_for_data+'b_adj_mirror.csv', index_col=0)
print(f"Adjacency matrix 2 shape: {adj2.shape}")

# check for NaN values in adjacency matrices
if adj1.isnull().values.any():
    print("Warning: Adjacency matrix contains NaN values. This may cause issues in further analysis.")
if adj2.isnull().values.any():
    print("Warning: Adjacency matrix contains NaN values. This may cause issues in further analysis.")
#%% 
adj1 = adj1.values
adj2 = adj2.values


# %% density 
np.random.seed(42)
stat, pval, misc = density_test(adj1, adj2, method='fisher')
print(f'Comparison with generated mirror network, pvalue = {pval}')

#compare to random network (sanity check)
p=0.0124
random_adj = er_np(n=adj1.shape[0], p=p, directed=True)
stat_random, pval_random, misc_random = density_test(adj1, random_adj, method='fisher')
print(f'Comparison with random network of same size with probability {p}, pvalue = {pval_random}')

# %% plot binarized adjacency matrices
adj1_bi = binarize(adj1)
adj2_bi = binarize(adj2)
heatmap(adj1_bi, title='Binarized Adjacency Matrix 1')
heatmap(adj2_bi, title='Binarized Adjacency Matrix 2')
# %%
