import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# %%
#Open csv file
data = pd.read_feather('/nas/longleaf/home/kchen315/Documents/ssi/procol_train.feather')


# %%
#Split into training and test data
y = data['SUPINFEC']
X = data.drop(['SUPINFEC','WNDINFD','ORGSPCSSI','ssi'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

# Number of trees in random forest
n_estimators = [200, 500, 750, 1000, 1250, 1500]
# Number of features to consider at every split
max_features = ['auto','sqrt']
# Maximum number of levels in tree
max_depth = [10, 20, 40, 60, 80, 100]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 4]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4, 6, 8]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, scoring='roc_auc', random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X, y)
results = pd.DataFrame(rf_random.cv_results_)
results.sort_values(by='rank_test_score').to_csv('results_rf.csv')