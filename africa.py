'''
Africa Soil Kaggle code
Dan Morris
9/24/14 -

Relevant Links:
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
          proper format. The model for each target column should be in the
          model_dict keyed by its name.
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

    def single_best_pearson(self, corr_dict, target, n_cols = 100,
                            c_dist = 20):
        best_cols = []
        best_col_indices = set()
        # get list of all column-correlation tuples, sorted by correlation
        l = sorted([(i, v) for i, v in corr_dict[target].iteritems()],
                   key = lambda x: abs(x[1][1]),
                   reverse = True)
        # each row in l: [index, (colname, correlation)]
        for row in l:
            if row[1][0][0] != 'm': # it's not a spectra column
                best_cols.append(row[1])
            else: # it IS a spectra column
                if no_neighbors_in_set(int(row[0]), best_col_indices, c_dist):
                    best_col_indices.add(int(row[0]))
                    best_cols.append(row[1])
            if len(best_cols) == n_cols:
                break
        return best_cols

    def best_pearsons(self, n_cols = 100, c_dist = 20, target = 'all',
                      pfile = 'pearson.json'):
        '''
        Returns the top n_cols columns for either one or all target features.
        Metric: pearson correlation rankins (from pfile). To reduce colinearity,
        each column appended must be at least c_dist away from any columns already
        in the top rankings. Named columns are excluded from such proximity testing.
        '''
        with open(pfile) as rf:
            corr_dict = json.loads(rf.read())
        if target == 'all':
            train_cols = {} # key = target column, value = best n_cols columns
            for target in corr_dict:
                best_cols = self.single_best_pearson(corr_dict, target,
                                                     n_cols, c_dist)
                train_cols[target] = best_cols
        else:
            train_cols = self.single_best_pearson(corr_dict, target,
                                                  n_cols, c_dist)
        return train_cols

    def single_pearson_to_Xy(self, target, solo = True,
                             bestcols=None, **kwargs):
        '''
        solo True if pulling a single column, False if coming from
          pearsons_to_Xy. Make sure to specify n_cols and c_dist if solo call.
        '''
        if solo:
            bestcols = self.best_pearsons(target = target, **kwargs)
        icn = [col[0] for col in bestcols]
        X = np.array(self.train[icn])
        y = self.train_target[target].values
        return X, y, icn

    def pearsons_to_Xy(self, **kwargs):
        '''
        Builds X and y arrays for training, selecting only columns from the
          best_pearsons function. Feed any arguments for that function in
          through **kwargs.
        '''
        bests = self.best_pearsons(**kwargs)
        X = {}
        y = {}
        icn = {} # lists of column names. keep them around for testing!
        for t in self.targetcols:
            X[t], y[t], icn[t] = self.single_pearsons_to_Xy(bests[t], t)
        return X, y, icn

    def grid_search_pearsons(self, model, n_cols_list, c_dist_list,
                             n_folds = 5, target = None):
        '''
        Tests all permutations of n_cols and c_dict with the given model.
        Prints the best hyperparams for each target feature and returns a sorted
          list of n, c, score tuples.
        '''
        # scores = dict of lists of (n_cols, c_dict, score) tuples
        if target == None:
            scores = {t:[] for t in self.targetcols}
            target = self.targetcols
        else:
            scores = {target:[]}
            target = [target]
        for t in target:
            for n in n_cols_list:
                for c in c_dist_list:
                    print 'predicting for t:' + t + ' n:' + str(n) + ' c:' + str(c)
                    temp_scores = []
                    X, y, icn = self.pearsons_to_Xy(n_cols = n, c_dist = c)
                    for train, test in KFold(self.ntrain, n_folds, shuffle = True):
                        model.fit(X[t][train], y[t][train])
                        temp_scores.append(model.score(X[t][test], y[t][test]))
                    scores[t].append((n, c, np.mean(temp_scores)))
        for t in scores:
            scores[t] = sorted(scores[t], key = lambda x: x[2], reverse = True)
            print 'Best score for target [' + t + ']: ' + str(scores[t][0][2])
            print 'Best params for target [' + t + ']:'
            print '  n_cols = ' + str(scores[t][0][0])
            print '  c_dist = ' + str(scores[t][0][1])
        return scores

if __name__=='__main__':
  # a = AfricaSoil()
  bests = a.best_pearsons()
  for k in bests:
    print k
    for r in bests[k]:
      print r
