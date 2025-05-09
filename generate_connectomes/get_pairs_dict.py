import numpy as np
import pandas as pd
from itertools import chain

def create_pairs_dict(pairs_data, verbose=True):
    pairs_df = pd.DataFrame()
    pairs_df['neuron_id'] = pairs_data['leftid'].values
    pairs_df['side'] = 'left'
    pairs_df['region'] = pairs_data['region'].values
    pairs_df['partner'] = pairs_data['rightid'].values

    pairs_df2 = pd.DataFrame()
    pairs_df2['neuron_id'] = pairs_data['rightid'].values
    pairs_df2['side'] = 'right'
    pairs_df2['region'] = pairs_data['region'].values
    pairs_df2['partner'] = pairs_data['leftid'].values

    # merge the two dataframes 
    pairs_df = pd.concat((pairs_df, pairs_df2), axis=0)
    pairs_df = pairs_df.reset_index(drop=True)
    #set partners to list 
    pairs_df['partner'] = pairs_df['partner'].apply(lambda x: [x])

    # handle duplicates 
    non_unique = np.unique(pairs_df['neuron_id'].values[pairs_df['neuron_id'].duplicated(keep=False)])

    if verbose:
        print(f"Warning: Some neurons have multiple contralateral partners.")
        print(f"Number of non-unique neurons pairs: {len(non_unique)}")
    for n in non_unique: 
        occs = np.where(pairs_df['neuron_id'].values == n)[0]
        partners = pairs_df['partner'].values[occs]
        #add to the first occurance of the neuron 

       #print(list(chain.from_iterable(partners))
        partners_ = list(chain.from_iterable(partners))
 
        pairs_df.at[occs[0], 'partner'] = partners_
        #remove the rest
        pairs_df = pairs_df.drop(occs[1:], axis=0)
        pairs_df = pairs_df.reset_index(drop=True)

    pairs_df = pairs_df.set_index('neuron_id')
    pairs_dict = pairs_df.to_dict(orient='index')
    return pairs_dict