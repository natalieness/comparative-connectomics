''' This is a sanity check to make sure change log can be viewed 
and edited.'''
#%%
from scripts.change_class import ChangeLog
import pickle 

log = ChangeLog()


with open('data/change_log.pkl', 'rb') as f:
    log = pickle.load(f)

print(log)
# %%
