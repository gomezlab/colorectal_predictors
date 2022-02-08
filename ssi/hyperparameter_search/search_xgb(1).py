import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# %%
#Open csv file
data = pd.read_csv('procol_train.csv', index_col=0)


# %%
#Split into training and test data
y = data['wndinf']
X = data.drop(['incssi','wndinf'], axis=1)

clf_xgb = XGBClassifier(tree_method='gpu_hist',use_label_encoder=True)

param_dist = {'n_estimators': [20, 50, 100, 200],
              'learning_rate': [0.1, 0.3, 0.5],
              'subsample': [0.4, 0.6, 1.0],
              'max_depth': [6, 8, 12, 20],
              'colsample_bytree': [0.6, 0.8, 1.0],
              'min_child_weight': [2, 4, 6]
             }


clf = RandomizedSearchCV(clf_xgb, 
                         param_distributions = param_dist,
                         cv = 5,  
                         n_iter = 100, 
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 3, 
                         n_jobs = -1)
clf.fit(X, y)
results = pd.DataFrame(clf.cv_results_)
results.sort_values(by='rank_test_score').to_csv('results_xgb.csv')