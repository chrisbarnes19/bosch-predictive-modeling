import pandas as pd
import numpy as np
from functools import reduce
import xgboost as xgb
import gc

from sklearn.metrics import matthews_corrcoef




def read_data(file,cols,preproc_func=None,params=None):
    """
    Load data from csv files in chunks and process the data
    using preproc_func

    Returns: dataframe of processed data
    """


    directory = "~/data/"
    train_filename = directory+"train_"+file+".csv"
    test_filename = directory+"test_"+file+".csv"
    
    chunksize = 100000
    dtype = np.float32
    if file == 'categorical':
        dtype = str
    
    
    print('Fetching chunks..')
    if preproc_func is not None:
        print('Preprocessing data..')
        train_chunks = [preproc_func(chunk,params) for chunk in
                  pd.read_csv(train_filename,
                              chunksize=chunksize,
                              usecols=cols, dtype=dtype)]
        
        test_chunks = [preproc_func(chunk,params) for chunk in
                  pd.read_csv(test_filename,
                              chunksize=chunksize,
                              usecols=cols, dtype=dtype)]
        
    else:
        train_chunks = [chunk for chunk in
                  pd.read_csv(train_filename,
                              chunksize=chunksize,
                              usecols=cols, dtype=dtype)]
        
        test_chunks = [chunk for chunk in
                  pd.read_csv(test_filename,
                              chunksize=chunksize,
                              usecols=cols, dtype=dtype)]
    
    print('Building training set..')
    train_df = pd.concat(train_chunks)
    
    print('Building test set..')
    test_df = pd.concat(test_chunks)
    
    train_df['partition'] = 'train'
    test_df['partition'] = 'test'
    
    print('Finishing up..')
    complete_df = pd.concat([train_df,test_df])
    
    del train_df
    del test_df
    gc.collect()
        
    print('Done.')
    return complete_df




def get_first_non_null(df):
    """Returns first non-null value in each row of a dataframe"""
    a = df.values
    n_rows, n_cols = a.shape
    col_index = np.isnan(a).argmin(axis=1)
    flat_index = n_cols * np.arange(n_rows) + col_index
    return a.ravel()[flat_index]

def get_last_non_null(df):
    """Returns last non-null value in each row of a dataframe"""
    a = df.values
    n_rows, n_cols = a.shape
    col_index = np.isnan(a).argmax(axis=1)
    flat_index = n_cols * np.arange(n_rows) + col_index
    return a.ravel()[flat_index]


def date_preproc(df, params=None):
    """Processes the date data and generates new features"""


    feats = pd.DataFrame(df['Id'])
    
   
   # Extract the first measurement from each station
    for station in range(52):
        station_cols = [col for col in df.columns if 'S'+str(station) in col]
        feats['Station'+str(station)+'StartTime'] = df[station_cols].min(axis=1).values
        
    date_cols = [col for col in df.columns if col not in ['Id']]
    
    # Generate various date based features
    feats['FirstTime'] = get_first_non_null(df[date_cols])
    feats['LastTime'] = get_last_non_null(df[date_cols])
    feats['StartTime'] = df[date_cols].min(axis=1).values
    feats['EndTime'] = df[date_cols].max(axis=1).values
    feats['Duration'] = feats['EndTime'] - feats['StartTime']
    feats['NATimes'] = np.isnan(df[date_cols]).sum(axis=1)

    # A time increment of 0.1 likely corresponds to 1 hr
    feats['StartTimeOfDay'] = (feats['StartTime'].multiply(10)) % 24
    feats['StartTimeOfWeek'] = (feats['StartTime'].multiply(10)) % 168
    feats['EndTimeOfDay'] = (feats['EndTime'].multiply(10)) % 24
    feats['EndTimeOfWeek'] = (feats['EndTime'].multiply(10)) % 168
    
    date_cols = [col for col in df.columns if 'L3' in col]
    
    # Extracts the same statistics for Line 3 specifically
    feats['L3FirstTime'] = get_first_non_null(df[date_cols])
    feats['L3LastTime'] = get_last_non_null(df[date_cols])
    feats['L3StartTime'] = df[date_cols].min(axis=1).values
    feats['L3EndTime'] = df[date_cols].max(axis=1).values
    feats['L3Duration'] = feats['EndTime'] - feats['StartTime']
    feats['L3NATimes'] = np.isnan(df[date_cols]).sum(axis=1)
    feats['L3StartTimeOfDay'] = (feats['StartTime'].multiply(10)) % 24
    feats['L3StartTimeOfWeek'] = (feats['StartTime'].multiply(10)) % 168
    feats['L3EndTimeOfDay'] = (feats['EndTime'].multiply(10)) % 24
    feats['L3EndTimeOfWeek'] = (feats['EndTime'].multiply(10)) % 168
    
    del df
    gc.collect()
    
    return feats
    





print("Getting date data")
cols = pd.read_csv('~/data/train_date.csv',nrows=1).columns
date_df = read_data('date',cols,date_preproc)




print("Getting numeric data")
# Columns were selected based on feature importance from the entire feature set
cols = ['Id', 'L0_S0_F0', 'L0_S0_F16', 'L0_S0_F20', 'L0_S0_F22',
       'L0_S10_F224', 'L0_S11_F290', 'L0_S14_F358', 'L0_S14_F374',
       'L0_S15_F400', 'L0_S15_F409', 'L0_S15_F418', 'L0_S22_F556',
       'L0_S23_F663', 'L0_S2_F44', 'L0_S2_F64', 'L0_S4_F104', 'L0_S7_F136',
       'L0_S9_F190', 'L0_S9_F205', 'L1_S24_F1130', 'L1_S24_F1520',
       'L1_S24_F1567', 'L1_S24_F1604', 'L1_S24_F1632', 'L1_S24_F1637',
       'L1_S24_F1695', 'L1_S24_F1723', 'L1_S24_F1728', 'L1_S24_F1763',
       'L1_S24_F1846', 'L1_S24_F1848', 'L2_S26_F3069', 'L2_S27_F3155',
       'L3_S29_F3330', 'L3_S29_F3333', 'L3_S29_F3342', 'L3_S29_F3348',
       'L3_S29_F3351', 'L3_S29_F3354', 'L3_S29_F3357', 'L3_S29_F3376',
       'L3_S29_F3407', 'L3_S29_F3439', 'L3_S29_F3482', 'L3_S30_F3499',
       'L3_S30_F3529', 'L3_S30_F3559', 'L3_S30_F3579', 'L3_S30_F3609',
       'L3_S30_F3649', 'L3_S30_F3694', 'L3_S30_F3749', 'L3_S30_F3759',
       'L3_S30_F3764', 'L3_S30_F3774', 'L3_S30_F3799', 'L3_S30_F3809',
       'L3_S30_F3814', 'L3_S31_F3834', 'L3_S32_F3850', 'L3_S33_F3855',
       'L3_S33_F3859', 'L3_S33_F3861', 'L3_S33_F3865', 'L3_S33_F3867',
       'L3_S35_F3898', 'L3_S36_F3922', 'L3_S38_F3952', 'L3_S45_F4126']
num_df = read_data('numeric',cols)




print('Getting categorical data')
# Columns again selected based on feature importance from S24 and S32 features
cols = ['Id',
         'L1_S24_F1559', 'L3_S32_F3851',
         'L1_S24_F1827', 'L1_S24_F1582',
         'L3_S32_F3854', 'L1_S24_F1510',
         'L1_S24_F1525', 'L3_S32_F3853']
cat_df = read_data('categorical',cols)




print('Getting reponse')
response = pd.read_csv('~/data/train_numeric.csv',usecols=['Id','Response'],dtype=np.float32)




print('Merging Datasets')
cat_df['Id'] = cat_df['Id'].astype(np.float32)
df = pd.merge(date_df,num_df,on='Id')
df = pd.merge(df, cat_df, on='Id')
df = pd.merge(df,response,how='left', left_on='Id',right_on='Id')

del cat_df, date_df, num_df, response
gc.collect()


# More feature engineering
print('Engineering features..')
cols = ['Id','Response','FirstTime']
tmp_df = df[cols].copy()
tmp_df = tmp_df.sort_values(by='Id')
row_indices = np.isnan(tmp_df['FirstTime'])
tmp_df.loc[row_indices,'FirstTime'] = tmp_df.loc[row_indices,'Id']
tmp_df['Next'] = (tmp_df['FirstTime'] == tmp_df['FirstTime'].shift(1)).astype(np.int)
tmp_df['Previous'] = (tmp_df['FirstTime'] == tmp_df['FirstTime'].shift(-1)).astype(np.int)
tmp_df['NextOrPrev'] = tmp_df['Next'] + tmp_df['Previous']
tmp_df['P'] = (tmp_df['NextOrPrev'] > 0)
tmp_df['ord'] = reduce(lambda x,y: sum(np.array((x,y)))*y,
                       tmp_df['Previous']) \
                + tmp_df['P']



cols = ['Id','ord','P','FirstTime']
new_df = pd.merge(df,tmp_df[cols],on='Id')
new_df = new_df.sort_values(by='Id')
new_df['L3TimeDiff'] = new_df['L3Duration'].diff()
new_df['L3InvTimeDiff'] = new_df['L3Duration'].iloc[::-1].diff()
new_df['L3NATimeDiff'] = new_df['L3NATimes'].diff()
new_df['L3NAInvTimeDiff'] = new_df['L3NATimes'].iloc[::-1].diff()



# One hot encode the categorical features
cols = ['L1_S24_F1559', 'L3_S32_F3851',
         'L1_S24_F1827', 'L1_S24_F1582',
         'L3_S32_F3854', 'L1_S24_F1510',
         'L1_S24_F1525', 'L3_S32_F3853']

for col in cols:
    new_df[col] = pd.get_dummies(new_df[col])



# Generate features based on the next response in the dataframe sorted by ID
new_df['NextResponse'] = new_df['Response'].shift(1)
new_df['PreviousResponse'] = new_df['Response'].shift(-1)


# Faron's magic features. 
new_df['magic1'] = new_df['Id'].diff().fillna(9999999).astype(int)
new_df['magic2'] = new_df['Id'].iloc[::-1].diff().fillna(9999999).astype(int)

new_df = new_df.sort_values(by=['StartTime', 'Id'], ascending=True)

new_df['magic3'] = new_df['Id'].diff().fillna(9999999).astype(int)
new_df['magic4'] = new_df['Id'].iloc[::-1].diff().fillna(9999999).astype(int)

new_df = new_df.sort_values(by='Id')




new_df.drop(['FirstTime_x', 'FirstTime_y', 'partition_x', 'partition_y'],inplace=True,axis=1)




del df
del tmp_df
gc.collect()



# Pickle the dataframe for faster loading 
new_df.to_pickle('processed_data.pkl')




print('Train & Predict')
train_indices = new_df['partition'] == 'train'
test_indices = new_df['partition'] == 'test'

cols = [col for col in new_df.columns if col not in ['Id','Response','partition']]


print('Creating dtrain')
dtrain = xgb.DMatrix(
    data = new_df.loc[train_indices, cols],
    label = new_df.loc[train_indices, 'Response'].astype(int))

print('Creating dtest')
dtest = xgb.DMatrix(
    data = new_df.loc[test_indices, cols])



 
params = {'alpha':0,
                 'booster':'gbtree',
                 'colsample_bytree':0.6,
                 'min_child_weight':5,
                 'subsample':0.9,
                 'eta':0.03,
                 'objective':'binary:logistic',
                 'max_depth':14,
                 'lambda':4,
                 'eval_metric':'auc'}

watchlist = [(dtrain,'train')]
model = xgb.train(params, dtrain, 100,watchlist)

pred = model.predict(dtest)



def find_prob(y_prob, index):
    return np.sort(y_prob)[::-1][index]



prob = find_prob(pred,2700)


print("Train MCC: {}".format(matthews_corrcoef(dtrain.get_label(),(model.predict(dtrain) > prob).astype(int))))



# Return the predictions for the test set
pd.DataFrame({"Id": new_df.loc[test_indices, 'Id'].astype(int).values, 
    "Response": (pred > prob).astype(int)}).to_csv(
        'final_entry.csv', index=False) 




