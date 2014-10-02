'''
Standalone functions for Africa Soil Kaggle
Dan Morris
9/24/14 -
'''
import numpy as np
import pandas as pd
import simplejson as json

def r2(h, y):
    '''
    Input: identical-sized arrays - predicted value and true value
    Output: the r^2 prediction score
    '''
    return 1 - sum((h - y) ** 2) / sum((y - np.mean(y)) ** 2)

def split_df_cols(df, cols):
    '''
    Input: dataframe, list of columns
    Output: two dataframes.
          0) original with those columns removed
          1) new df with just those columns
    '''
    df1 = df.drop(cols, axis = 1)
    df2 = df[cols]
    return df1, df2

def MCRMSE(pred_df, true_df):
    '''
    Calculates the scoring metric for the competition on a given CV set
    pred_df and true_df are dataframes with identical columns and row IDs
    '''
    cols = pred_df.columns
    error = [RMSE(pred_df[c].values, true_df[c].values) for c in cols]
    for i, c in enumerate(cols):
        print 'RMSE of column ' + c + ': ' + str(error[i])
    print 'MCRMSE total: ' + str(np.mean(error))
    return np.mean(error)

def RMSE(pred_arr, true_arr):
    return np.sqrt(np.mean((pred_arr - true_arr) ** 2))

def pearson_bests(n, pfile = 'pearson.json'):
    '''
    returns the best n columns for each target in the pearson test json file
    '''
    with open(pfile) as rf:
        d = json.loads(rf.read())
    bests = {}
    for k in d:
        b = sorted(d[k].iteritems(), key = lambda x: abs(x[1]), reverse = True)
        bests[k] = b[:n]
    return bests

def no_neighbors_in_set(i, s, d):
    '''
    i = prospective index (int), s = current indices (set), d = distance (int)
    returns True if i is >= d from every index in s.
    '''
    for index in s:
        if abs(index - i) < d:
            return False
    return True
