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

rm = pymaid.CatmaidInstance(url, token, name, password)

#get parent directory path
current_file = __file__  
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
path_for_data = parent_dir+'/data/'

#%% Load adjacency matrix
adj1 = pd.read_csv(path_for_data+'adj.csv', index_col=0)
print(f"Adjacency matrix 1 shape: {adj1.shape}")

adj2 = pd.read_csv(path_for_data+'adj_mirror.csv', index_col=0)
print(f"Adjacency matrix 2 shape: {adj2.shape}")

# %%


