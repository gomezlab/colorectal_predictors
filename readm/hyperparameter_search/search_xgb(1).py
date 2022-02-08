import numpy as np
import pandas as pd
import feather
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# %%
#Open csv file
data = feather.read_dataframe('procol_train.feather')

# %%
#Split into training and test data
y = data['READMISSION1']
X = data.drop(['READMISSION1','CASEID'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

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