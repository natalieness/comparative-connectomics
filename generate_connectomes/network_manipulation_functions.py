''' Functions to generate an artificially manipulated connectome from 
an existing connectome.'''

import numpy as np
import pandas as pd

### Basics ###

def generate_mirror_network(adj_):
    adj_mirror = adj_.copy()
    return adj_mirror

### Neuron-level manipulations ###


#smaller functions used in neuron duplication
def get_dup_neuron_row(neuron_idx, new_adj, rng):
    n_source = new_adj.iloc[neuron_idx, :].values
    syn_out = np.where(n_source > 0)[0]
    w_out = n_source[syn_out]
    new_w_out = w_out.copy()
    rng.shuffle(new_w_out)
    #set new weights 
    new_n_source = n_source.copy()
    new_n_source[syn_out] = new_w_out
    return new_n_source
def get_dup_neuron_col(neuron_idx, new_adj, rng):
    n_target = new_adj.iloc[:, neuron_idx].values
    n_in = np.where(n_target > 0)[0]
    w_in = n_target[n_in]
    new_w_in = w_in.copy()
    rng.shuffle(new_w_in)
    #set new weights
    new_n_target = n_target.copy()
    new_n_target[n_in] = new_w_in
    return new_n_target

def neuron_duplication(adj_, log, rng, n_ops=1, bilateral=False, pairs_dict=None):
    # if bilateral is True, duplicate partner neuron on contralteral side
    # if so, pairs_dict must be provided 
    if bilateral:
        #check pairs_dict is provided 
        if pairs_dict == None:
            raise ValueError("pairs_dict must be provided if bilateral is True")
    #if bilateral is False, give warning if pairs_dict is not provided as new neuron will not be added to it 
    if pairs_dict == None:
        print("Warning: pairs_dict is not provided, new neurons will not be added to it")

    new_adj = adj_.copy()
    #calling n_neurons here prevents duplication of new neurons
    n_neurons = new_adj.shape[0]
    for n in range(n_ops):
        # select a random neuron idx
        neuron_idx = rng.integers(low=0, high=n_neurons, size=1)[0]
        # get the neuron name
        neuron_name = new_adj.columns[neuron_idx]
        #set new neuron name
        new_n_name = f'new_{n}'

        #set neuron outputs 
        new_n_source = get_dup_neuron_row(neuron_idx, new_adj, rng)
        new_adj.loc[new_n_name] = new_n_source

        #set neuron inputs
        new_n_target = get_dup_neuron_col(neuron_idx, new_adj, rng)
 
        new_adj[new_n_name] = new_n_target
        

        #if bilateral is True, also duplicated partner neuron 
        if bilateral:
            #get partner neuron name
            partner_name = pairs_dict[neuron_name]['partner']
            #if multiple partners, randomly select one
            if len(partner_name) > 1:
                partner_name = rng.choice(partner_name)
            
            #get partner neuron idx
            partner_idx = new_adj.columns.get_loc(partner_name[0])
            #give new partner neuron a new name 
            new_p_name = f'new_{n}_partner'
 
            #get partner neuron row and column
            new_p_source = get_dup_neuron_row(partner_idx, new_adj, rng)
            new_adj.loc[new_p_name] = new_p_source

            new_p_target = get_dup_neuron_col(partner_idx, new_adj, rng)

            new_adj[new_p_name] = new_p_target
            #add index back
            new_adj.index = new_index


        #Need to add to pairs_dict, if unilateral and bilateral
        if pairs_dict != None: 
            if bilateral == False:
                #add new neuron to pairs_dict
                pairs_dict[new_n_name] = {
                    'side': pairs_dict[neuron_name]['side'],
                    'region': pairs_dict[neuron_name]['region'],
                    'partner': pairs_dict[neuron_name]['partner']
                }
            elif bilateral == True:
                #add new neuron ton pairs_dict with new partner
                pairs_dict[new_n_name] = {
                    'side': pairs_dict[neuron_name]['side'],
                    'region': pairs_dict[neuron_name]['region'],
                    'partner': [new_p_name]
                }
                #add new partner to pairs_dict
                pairs_dict[new_p_name] = {
                    'side': pairs_dict[partner_name[0]]['side'],
                    'region': pairs_dict[partner_name[0]]['region'],
                    'partner': [new_n_name]
                }

        #Need to add pair to log if 2 neurons duplicated 
        log.add_neuron_change(
            operation='neuron_duplication',
            source_index=neuron_name,
            new_index=new_n_name,
            weight_handling='shuffling existing'
        )
        if bilateral:
            log.add_neuron_change(
                operation='neuron_duplication',
                source_index=partner_name,
                new_index=new_p_name,
                weight_handling='shuffling existing'
            )

    return new_adj, log, pairs_dict

def neuron_deletion(adj_, log, rng, n_ops=1, n_og_neurons=None, bilateral=False, pairs_dict=None):
    if bilateral:
        if pairs_dict == None:
            raise ValueError("pairs_dict must be provided if bilateral is True")
    
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

        if bilateral:
            #get partner neuron and delete too 
            partner_name = pairs_dict[neuron_name]['partner']
            #if multiple partners, randomly select one
            if len(partner_name) > 1:
                partner_name = rng.choice(partner_name)
            # delete the partner neuron from the adjacency matrix
            new_adj.drop(columns=[partner_name[0]], inplace=True)
            new_adj.drop(index=[partner_name[0]], inplace=True)

        log.add_neuron_change(
            operation='neuron_deletion',
            source_index=neuron_name,
            new_index=None,
            weight_handling='NA'
        )
        if bilateral:
            log.add_neuron_change(
                operation='neuron_deletion',
                source_index=partner_name[0],
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

def new_rand_neurons(adj_, log, rng, n_ops=1, pairs_dict=None):
    new_adj = adj_.copy()
    #get general network properties to copy 
    mean_n_out, mean_w_out, mean_n_in, mean_w_in = get_connectivity_properties(adj_)

    for n in range(n_ops):
        #get current neurons in adj 
        n_neurons = new_adj.shape[0]
        new_row = np.zeros((n_neurons))
        new_col = np.zeros((n_neurons + 1))
        # select random neurons to form connections with 
        neurons_out = rng.integers(low=0, high=n_neurons, size=int(mean_n_out))
        neurons_in = rng.integers(low=0, high=n_neurons, size=int(mean_n_in))

        #get new weights 
        new_weights_out = random_partition_weights(rng, len(neurons_out), mean_w_out)
        new_weights_in = random_partition_weights(rng, len(neurons_in), mean_w_in)

        #set weights 
        new_row[neurons_out] = new_weights_out 
        new_col[neurons_in] = new_weights_in
 
        
        # get new neuron name 
        new_n_name = f'newnew_{n}'
        #get old index 
        old_index = new_adj.index.tolist()
        new_index = old_index + [new_n_name]
        #add new row and column to adjacency matrix

        new_adj.loc[new_n_name] = new_row
        new_adj.reset_index(drop=False, inplace=True)  # reset index to ensure new row is added correctly
        new_adj[new_n_name] = new_col

        #set new index 
        new_adj.index = new_index
        

        #add neuron to pairs_dict to avoid ValueError
        if pairs_dict != None:
            #randomly assign side and region 
            nside = rng.choice(['left', 'right'])
            nregion = rng.choice(['A1','A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'SEZ', 'T2', np.nan, 'sensory'])
            pairs_dict[new_n_name] = {
                'side': nside,
                'region': nregion,
                'partner': [] #is this empty thing an issue? need to recheck
            }

        log.add_neuron_change(
            operation='new_rand_neurons',
            source_index=None,
            new_index=new_n_name,
            weight_handling='random partition mean weights'
        )
    return new_adj, log, pairs_dict