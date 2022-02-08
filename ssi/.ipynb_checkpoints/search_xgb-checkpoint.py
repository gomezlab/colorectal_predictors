import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# %%
#Open csv file
data = pd.read_feather('/nas/longleaf/home/kchen315/Documents/ssi/procol_train.feather')

# %%
#Split into training and test data
y = data['SUPINFEC']
X = data.drop(['SUPINFEC','WNDINFD','ORGSPCSSI','ssi'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)


clf_xgb = XGBClassifier(tree_method='gpu_hist', use_label_encoder=False)

param_dist = {'n_estimators': [20, 50, 100],
              'learning_rate': [0.05, 0.1],
              'subsample': [0.2, 0.4, 0.6, 1.0],
              'max_depth': [4, 6, 8, 12, 16],
              'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0],
              'min_child_weight': [1, 5, 10]
             }


clf = RandomizedSearchCV(clf_xgb, 
                         param_distributions = param_dist,
                         cv = 5,  
                         n_iter = 100, 
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 1, 
                         n_jobs = -1)
clf.fit(X, y, eval_metric="auc", verbose=False)
results = pd.DataFrame(clf.cv_results_)
results.sort_values(by='rank_test_score').to_csv('results_xgb.csv')