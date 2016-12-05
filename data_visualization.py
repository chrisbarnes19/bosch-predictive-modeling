

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')



def load_data(fileID,preproc_func=None):
    
    if fileID == 'categorical':
        dtype = str
    else:
        dtype = np.float32
        
    chunks = pd.read_csv('~/data/train_'+fileID+'.csv',chunksize=50000,dtype=dtype)
        
    if preproc_func is not None:
        df = pd.concat([preproc_func(chunk) for chunk in chunks])
    
    else:
        df = pd.concat([chunk for chunk in chunks])
        
    return df

def to_bool(df):
    return pd.isnull(df)
    

def plot(df):
    indices = np.linspace(0,df.shape[0]-1,df.shape[1]).astype(int)
    plt.matshow(df.iloc[indices],cmap = plt.cm.gray)



nums = load_data('numeric',to_bool)
plot(nums)




cats = load_data('categorical',to_bool)
plot(cats)



