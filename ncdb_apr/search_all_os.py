import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from skopt import BayesSearchCV


# In[ ]:
data = pd.read_csv('data/train_os.csv')

# drop_years = '2010'

y = data['3-yr Mortality']
X = data.drop(['3-yr Mortality'], axis=1)
# X = X[X['Year of Diagnosis'] > int(drop_years)]
X.drop(['Year of Diagnosis','Total Dose'], axis=1, inplace=True)
y = y[X.index]
X, X_valid, y, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)




# Number of trees in random forest
n_estimators = [500, 750, 1000, 1250, 1500]
# Number of features to consider at every split
max_features = ['auto','sqrt']
# Maximum number of levels in tree
max_depth = [20, 40, 60, 80, 100, 120]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 4, 6]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4, 6, 8]
# Method of selecting samples for training each tree
bootstrap = [True]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
rf_random = BayesSearchCV(rf, random_grid, n_iter = 100, cv = 5, verbose=2, scoring='roc_auc', random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X, y)
results = pd.DataFrame(rf_random.cv_results_)
results.sort_values(by='rank_test_score').to_csv('results_os/results_rf_nodose.csv')

best_rf = rf_random.best_estimator_


clf_xgb = XGBClassifier(tree_method='gpu_hist', use_label_encoder=False)

param_dist = {'n_estimators': [20, 50, 100, 200],
              'learning_rate': [0.03, 0.05, 0.075, 0.1, 0.3, 0.5],
              'subsample': [0.4, 0.6, 1.0],
              'max_depth': [6, 8, 12, 20],
              'colsample_bytree': [0.6, 0.8, 1.0],
              'min_child_weight': [2, 4, 6]
             }

clf = BayesSearchCV(clf_xgb, 
                         param_dist,
                         cv = 5,  
                         n_iter = 100, 
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 0, 
                         n_jobs = -1)
clf.fit(X, y)
results = pd.DataFrame(clf.cv_results_)
results.sort_values(by='rank_test_score').to_csv('results_os/results_xgb_nodose.csv')

best_xgb = clf.best_estimator_


# In[ ]:
input_shape = [X.shape[1]]

def build_model(n_hidden=1, n_neurons=100, dropout=0.4, activation = "relu", learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Activation(activation))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(loss="binary_crossentropy", metrics=['AUC'], optimizer=optimizer)
    return model

keras_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model)

param_distribs = {
    "n_hidden": [1, 2, 3, 4],
    "n_neurons": [25, 50, 200, 500, 1000, 1500],
    "dropout": [0.2, 0.4, 0.6, 0.8],
    "activation": ["relu", "elu"],
    "learning_rate": [3e-5, 3e-4, 3e-3, 3e-2],
}

# In[ ]:
early_stopping = keras.callbacks.EarlyStopping(
    patience=15,
    min_delta=1e-6,
    restore_best_weights=True,)

rnd_search_cv = BayesSearchCV(keras_clf, param_distribs, n_iter=50, scoring='roc_auc', cv=5, verbose=2)

rnd_search_cv.fit(X, y, epochs=100, batch_size=512,
                  validation_data=(X_valid, y_valid),
                  callbacks=[early_stopping])


results = pd.DataFrame(rnd_search_cv.cv_results_)

results.sort_values(by='rank_test_score').to_csv('results_os/results_keras_nodose.csv')

best_keras = rnd_search_cv.best_estimator_

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow import keras
from math import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import csv

# %%
pd.options.display.max_rows = 20
pd.options.display.max_columns = 200

def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, AUC, upper)

def roc_prc_ci(y_true, y_score, positive=1):
    AUC = average_precision_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, AUC, upper)

def auroc_ci(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    mean = roc_auc
    std = sqrt(roc_auc * (1.0 - roc_auc) / len(y_true))
    low  = mean - std
    high = mean + std
    return low, mean, high
#calculate auprc 95% ci for each model
def auprc_ci(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    mean = pr_auc
    std = sqrt(pr_auc * (1.0 - pr_auc) / len(y_true))
    low  = mean - std
    high = mean + std
    return low, mean, high

test = pd.read_csv('data/test_os.csv')

y_test = test['3-yr Mortality']
X_test = test.drop(['Year of Diagnosis','Total Dose'], axis=1, inplace=True)

lr = LogisticRegression(penalty='none', max_iter=9000)

lr.fit(X, y)

lr_y_pred = lr.predict_proba(X_test)[:, 1]
rf_y_pred = best_rf.predict_proba(X_test)[:, 1]
xgb_y_pred = best_xgb.predict_proba(X_test)[:, 1]
nn_y_pred = best_keras.predict_proba(X_test)[:, 1]

rf_confidence = auroc_ci(y_test, rf_y_pred)
xgb_confidence = auroc_ci(y_test, xgb_y_pred)
lr_confidence = auroc_ci(y_test, lr_y_pred)
nn_confidence = auroc_ci(y_test, nn_y_pred)
print('Random Forest AUROC:', rf_confidence, 'AUROC CI:', rf_confidence)
print('XGBoost AUROC:', xgb_confidence, 'AUROC CI:', xgb_confidence)
print('Logistic Regression AUROC:', lr_confidence, 'AUROC CI:', lr_confidence)
print('Neural Network AUROC:', nn_confidence, 'AUROC CI:', nn_confidence)

#create labels for roc curves
rf_label = 'RF AUROC: ' + str(round(rf_confidence[1], 3)) + ' (' + str(round(rf_confidence[0], 3)) + ' - ' + str(round(rf_confidence[2], 3)) + ')'
xgb_label = 'XGB AUROC: ' + str(round(xgb_confidence[1], 3)) + ' (' + str(round(xgb_confidence[0], 3)) + ' - ' + str(round(xgb_confidence[2], 3)) + ')'
nn_label = 'NN AUROC: ' + str(round(nn_confidence[1], 3)) + ' (' + str(round(nn_confidence[0], 3)) + ' - ' + str(round(nn_confidence[2], 3)) + ')'
lr_label = 'LR AUROC: ' + str(round(lr_confidence[1], 3)) + ' (' + str(round(lr_confidence[0], 3)) + ' - ' + str(round(lr_confidence[2], 3)) + ')'
#calculate tpr and fpr for each model
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_pred)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_y_pred)
nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_y_pred)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_y_pred)

import matplotlib
matplotlib.rcParams.update({'font.size': 16})
#plot the ROC curves for each model
plt.figure(figsize=(10,10))
plt.plot(lr_fpr, lr_tpr, color='red', label=lr_label)
plt.plot(rf_fpr, rf_tpr, color='deepskyblue', label=rf_label)
plt.plot(xgb_fpr, xgb_tpr, color='steelblue', label=xgb_label)
plt.plot(nn_fpr, nn_tpr, color='dodgerblue', label=nn_label)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([0.0, 1.0])
plt.savefig('results_os/roc_nodose.png', dpi=300, bbox_inches='tight')
plt.clf()

# %%
rf_auprc_ci = auprc_ci(y_test, rf_y_pred)
xgb_auprc_ci = auprc_ci(y_test, xgb_y_pred)
lr_auprc_ci = auprc_ci(y_test, lr_y_pred)
nn_auprc_ci = auprc_ci(y_test, nn_y_pred)
#calculate precision and recall for each model
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_y_pred)
xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_y_pred)
nn_precision, nn_recall, _ = precision_recall_curve(y_test, nn_y_pred)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_y_pred)
#create labels for precision recall curves
rf_prc_label = 'RF AUPRC: ' + str(round(rf_auprc_ci[1], 3)) + ' (' + str(round(rf_auprc_ci[0], 3)) + ' - ' + str(round(rf_auprc_ci[2], 3)) + ')'
xgb_prc_label = 'XGB AUPRC: ' + str(round(xgb_auprc_ci[1], 3)) + ' (' + str(round(xgb_auprc_ci[0], 3)) + ' - ' + str(round(xgb_auprc_ci[2], 3)) + ')'
nn_prc_label = 'NN AUPRC: ' + str(round(nn_auprc_ci[1], 3)) + ' (' + str(round(nn_auprc_ci[0], 3)) + ' - ' + str(round(nn_auprc_ci[2], 3)) + ')'
lr_prc_label = 'LR AUPRC: ' + str(round(lr_auprc_ci[1], 3)) + ' (' + str(round(lr_auprc_ci[0], 3)) + ' - ' + str(round(lr_auprc_ci[2], 3)) + ')'
#plot the precision recall curves for each model
matplotlib.rcParams.update({'font.size': 16})
#plot the ROC curves for each model
plt.figure(figsize=(10,10))
plt.plot(lr_recall, lr_precision, color='red', label=lr_prc_label)
plt.plot(rf_recall, rf_precision, color='deepskyblue', label=rf_prc_label)
plt.plot(xgb_recall, xgb_precision, color='steelblue', label=xgb_prc_label)
plt.plot(nn_recall, nn_precision, color='dodgerblue', label=nn_prc_label)
plt.legend(loc="upper right")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.savefig('results_os/prc_nodose.png', dpi=300, bbox_inches='tight')
plt.clf()

import shap
import seaborn as sns
from matplotlib import pyplot as plt

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test.sample(n=500))
shap_obj = explainer(X_test.sample(n=500))

shap.summary_plot(shap_values = shap_values,
                  features = X_test.sample(n=500),
                  feature_names = X_test.columns,
                  plot_size = (15,10),
                  #cmap=plt.get_cmap("flare"),
                  sort=True, show=False)
plt.savefig('results_os/shap_xgb_waterfall_nodose.png', dpi=300, bbox_inches='tight')
plt.clf()

shap.summary_plot(shap_values = shap_values,
                  features = X_test.sample(n=500),
                  feature_names = X_test.columns,
                  plot_size = (15,10),
                  #cmap=plt.get_cmap("flare"),
                  plot_type='bar',
                  sort=True, show=False)
plt.savefig('results_os/shap_xgb_bar_nodose.png', dpi=300, bbox_inches='tight')
plt.clf()