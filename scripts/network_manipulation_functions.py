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

def get_connectivity_properties(adj_):
    # get mean number of out connections
    mean_n_out = (adj_ > 0).sum(axis=1).mean()
    # get mean weight of out connections 
    mean_w_out = adj_.sum(axis=1).mean()

    # get mean number of in connections
    mean_n_in = (adj_ > 0).sum(axis=0).mean()
    # get mean weight of in connections
    mean_w_in = adj_.sum(axis=0).mean()
    return mean_n_out, mean_w_out, mean_n_in, mean_w_in

def random_partition_weights(rng, n_targets, total_weight):
    partition = rng.dirichlet([1] * n_targets)
    weights = [total_weight * p for p in partition]
    return weights

def new_rand_neurons(adj_, log, rng, n_ops=1):
    new_adj = adj_.copy()
    #get general network properties to copy 
    mean_n_out, mean_w_out, mean_n_in, mean_w_in = get_connectivity_properties(adj_)

    for n in range(n_ops):
        #get current neurons in adj 
        n_neurons = new_adj.shape[0]
        new_row = np.zeros((n_neurons))
        new_col = np.zeros((n_neurons))
        # select random neurons to form connections with 
        neurons_out = rng.integers(low=0, high=n_neurons, size=int(mean_n_out))
        neurons_in = rng.integers(low=0, high=n_neurons, size=int(mean_n_in))

        #get new weights 
        new_weights_out = random_partition_weights(rng, len(neurons_out), mean_w_out)
        new_weights_in = random_partition_weights(rng, len(neurons_in), mean_w_in)

        #set weights 
        new_row[neurons_out] = new_weights_out
        new_col[neurons_in] = new_weights_in
        #add weights to new row
        new_col = np.concat((new_col, np.array([0])))

        # get new neuron name 
        new_n_name = f'newnew_{n}'
        #add new row and column to adjacency matrix

        new_adj.loc[new_n_name] = new_row
        new_adj[new_n_name] = new_col

        log.add_neuron_change(
            operation='new_rand_neurons',
            source_index=None,
            new_index=new_n_name,
            weight_handling='random partition mean weights'
        )
    return new_adj, log