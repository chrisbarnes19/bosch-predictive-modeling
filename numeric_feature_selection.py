import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import pickle



def load_features():
    
    
    # Examine significant columns for date and numeric datasets  
    date_chunks = pd.read_csv("~/data/train_date.csv", index_col=0, chunksize=100000, dtype=np.float32)
    num_chunks = pd.read_csv("~/data/train_numeric.csv", index_col=0,
                             usecols=list(range(969)), chunksize=100000, dtype=np.float32)

    # Sample the dataset
    X = pd.concat([pd.concat([dchunk, nchunk], axis=1).sample(frac=0.05)
                   for dchunk, nchunk in zip(date_chunks, num_chunks)])

    y = pd.read_csv("~/data/train_numeric.csv", index_col=0, dtype=np.float32, usecols=[0,969]).loc[X.index].values.ravel()
    X = X.values
        
        
    return X, y


X,y = load_features()


prior = sum(y) / len(y)

clf = XGBClassifier(base_score=prior)
clf.fit(X, y)


plt.hist(clf.feature_importances_[clf.feature_importances_>0])
important_indices = np.where(clf.feature_importances_>0.005)[0]
print(important_indices)



pickle.dump(important_indices,open('important_indices.pkl','wb'))


