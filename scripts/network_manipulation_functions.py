''' Functions to generate an artificially manipulated connectome from 
an existing connectome.'''

import numpy as np
import pandas as pd

### Basics ###

def generate_mirror_network(adj_):
    adj_mirror = adj_.copy()
    return adj_mirror

### Neuron-level manipulations ###


def neuron_duplication(adj_, log, rng, n_ops=1):
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

def neuron_deletion(adj_, log, rng, n_ops=1, n_og_neurons=None):
    new_adj = adj_.copy()
    if n_og_neurons != None:
        # if n_og_neurons is provided, restrict deletions to the original neurons
        n_neurons = n_og_neurons
    else:
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