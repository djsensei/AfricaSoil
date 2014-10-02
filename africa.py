'''
Africa Soil Kaggle code
Dan Morris
9/24/14 -

Links:
http://arkansasagnews.uark.edu/558-20.pdf
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.cross_validation import KFold
from scipy.stats import pearsonr
import simplejson as json

from functions import r2, split_df_cols, RMSE, MCRMSE, no_neighbors_in_set

class AfricaSoil(object):
    def __init__(self):
        # Load training and test dataframes
        self.train = pd.read_csv('training.csv')
        self.test = pd.read_csv('sorted_test.csv')
        self.ntrain = len(self.train)
        self.ntest = len(self.test)

        # Separate target columns
        self.targetcols = ['Ca', 'P', 'pH', 'SOC', 'Sand']
        self.train, self.train_target = split_df_cols(self.train, self.targetcols)

        # Basic feature selection
        self.train = self.basic_feature_select(self.train)
        self.train = self.train.drop('PIDN', axis = 1) # who cares about labels
        self.test = self.basic_feature_select(self.test)

        # initialize column name variables
        self.colnames = list(self.train.columns)
        self.numcols = [c for c in self.colnames if c[0] == 'm']
        self.namedcols = [c for c in self.colnames if c[0] != 'm']

    def predict_test(self, model_dict, output_file):
        '''
        Predicts the test set with the given model and outputs the results in
          proper format.
        '''
        X = self.test.drop('PIDN', axis = 1)
        pred_array = self.test['PIDN'].values.reshape((self.ntest, 1))
        for k in self.targetcols:
            print 'predicting test set: ' + k
            h = model_dict[k].predict(X).reshape((self.ntest, 1))
            pred_array = np.concatenate((pred_array, h), axis = 1)

        preddf = pd.DataFrame(pred_array, columns = ['PIDN'] + self.targetcols)
        preddf.to_csv(output_file, index = False)

    def basic_LRs(self):
        X = np.array(self.train)
        lrs = {}
        for k in self.targetcols:
            print 'training basic LR on target: ' + k
            y = self.train_target[k].values
            lrs[k] = LinearRegression()
            lrs[k].fit(X, y)
        return lrs

    def basic_feature_select(self, df):
        '''
        Input: training or test dataframe (with target features removed)
        Output: dataframe with simple transforms applied
        '''
        # drop CO2 spectra columns as per Kaggle recommendation
        CO2colnames = ['m2379.76', 'm2377.83', 'm2375.9', 'm2373.97',
                       'm2372.04', 'm2370.11', 'm2368.18', 'm2366.26',
                       'm2364.33', 'm2362.4', 'm2360.47', 'm2358.54',
                       'm2356.61', 'm2354.68', 'm2352.76']
        df = df.drop(CO2colnames, axis = 1)

        # Binarize Depth column
        df['Depth'] = df['Depth'].apply(lambda x: 1 if x == 'Topsoil' else 0)
        return df

    def pearson_test_features(self):
        '''
        Finds the pearson r coefficient for each combination of feature and
        target. Outputs the whole thing to a json file for later reference.
        '''
        d = {}
        m = len(self.colnames)
        for t in self.targetcols:
            print 'determining pearson correlation for: ' + t
            td = {} # dict of (colname, correlation) tuples keyed by col index
            for i, c in enumerate(self.colnames):
                r, _ = pearsonr(self.train[c].values,
                             self.train_target[t].values)
                td[i] = (c, r)
                if i % 500 == 0:
                    print 'calculating for column ' + str(i)
            d[t] = td
        with open('pearson.json', 'w') as wf:
            json.dump(d, wf)

    def find_best_features(self, target):
        '''
        Subset selection! Currently just searches namecols, but should search
          numcols for sure.
        '''
        r2s = {}
        y = self.train_target[target].values
        for c in self.namecols:
            scores = []
            X = self.train[c].values
            for train_i, test_i in KFold(self.n, n_folds=5, shuffle=True):
                lr = LinearRegression()
                lr.fit(X[train_i].reshape((len(train_i), 1)), y[train_i])
                scores.append(lr.score(X[test_i].reshape((len(test_i), 1)), y[test_i]))
            r2s[c] = np.mean(scores)
        return r2s
        #return sorted(r2s.keys(), key=lambda x: r2s[x])

    def best_pearsons(self, n_cols = 100, c_dist = 20, pfile = 'pearson.json'):
        '''
        Returns the top n_cols columns for each target feature.
        Metric: pearson correlation rankins (from pfile). To reduce colinearity,
        each column appended must be at least c_dist away from any columns already
        in the top rankings. Named columns are excluded from such proximity testing.
        '''
        with open(pfile) as rf:
            corr_dict = json.loads(rf.read())
        train_cols = {} # key = target column, value = best n_cols columns
        for target in corr_dict:
            best_cols = []
            best_col_indices = set()
            # get list of all column-correlation tuples, sorted by correlation
            l = sorted([(i, v) for i, v in corr_dict[target].iteritems()],
                       key = lambda x: abs(x[1][1]),
                       reverse = True)
            for row in l:
                if no_neighbors_in_set(int(row[0]), best_col_indices, c_dist):
                    best_col_indices.add(int(row[0]))
                    best_cols.append(row[1])
                if len(best_col_indices) == n_cols:
                    break
            train_cols[target] = best_cols
        return train_cols

    def pearsons_to_Xy(self, **kwargs):
        '''
        Builds X and y arrays for training, selecting only columns from the
          best_pearsons function. Feed any arguments for that function in
          through **kwargs.
        '''
        bests = self.best_pearsons(**kwargs)
        X = {}
        y = {}
        for t in self.targetcols:
            indexcolnames = [col[0] for col in bests[t]]
            X[t] = np.array(self.train[indexcolnames])
            y[t] = self.train_target[t].values
        return X, y

if __name__=='__main__':
  # a = AfricaSoil()
  bests = a.best_pearsons()
  for k in bests:
    print k
    for r in bests[k]:
      print r
