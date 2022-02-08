import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, auc, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from tensorflow import keras
from tensorflow.keras import layers
from math import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import csv

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

data = pd.read_feather('/nas/longleaf/home/kchen315/Documents/ssi/procol_train.feather')
y = data['SUPINFEC']
X = data.drop(['SUPINFEC','WNDINFD','ORGSPCSSI','ssi'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

test = pd.read_feather('/nas/longleaf/home/kchen315/Documents/ssi/procol_test.feather')
y_test = test['SUPINFEC']
X_test = test.drop(['SUPINFEC','WNDINFD','ORGSPCSSI','ssi'], axis=1)

input_shape = [X_train.shape[1]]
model4 = keras.models.Sequential()
model4.add(keras.layers.Flatten(input_shape=input_shape))
model4.add(keras.layers.BatchNormalization())
for _ in range(2):
    model4.add(keras.layers.Dense(200))
    model4.add(keras.layers.BatchNormalization())
    model4.add(keras.layers.Dropout(0.8))
    model4.add(keras.layers.Activation("relu"))
model4.add(keras.layers.Dense(1, activation="sigmoid"))

opt = keras.optimizers.Adam(learning_rate=3e-3)

metrics = [keras.metrics.Recall(name='Sensitivity'), keras.metrics.TrueNegatives(name='tn'), keras.metrics.AUC(name='auc'), keras.metrics.AUC(name='prc', curve='PR')]

model4.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    metrics=metrics,)

early_stopping = keras.callbacks.EarlyStopping(
    patience=25,
    min_delta=1e-8,
    restore_best_weights=True,)

history = model4.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=500,
    callbacks=[early_stopping],)
ann_preds_sssi = model4.predict(X_test)
ann_auc_sssi = roc_auc_ci(y_test, ann_preds_sssi)
ann_prc_sssi = roc_prc_ci(y_test, ann_preds_sssi)

rf = RandomForestClassifier(n_estimators=1000, min_samples_split=4, min_samples_leaf=8, max_features='sqrt', max_depth=20, bootstrap=False)
rf.fit(X, y)
rf_preds_sssi = rf.predict_proba(X_test)[:,1]
rf_auc_sssi = roc_auc_ci(y_test, rf_preds_sssi)
rf_prc_sssi = roc_prc_ci(y_test, rf_preds_sssi)
xgb = XGBClassifier(subsample=1.0, n_estimators=100, min_child_weight=10, max_depth=6, learning_rate=0.1, colsample_bytree=0.6)
xgb.fit(X, y)
xgb_preds_sssi = xgb.predict_proba(X_test)[:,1]
xgb_auc_sssi = roc_auc_ci(y_test, xgb_preds_sssi)
xgb_prc_sssi = roc_prc_ci(y_test, xgb_preds_sssi)
lr = LogisticRegression(penalty='none')
lr.fit(X, y)
lr_preds_sssi = lr.predict_proba(X_test)[:,1]
lr_auc_sssi = roc_auc_ci(y_test, lr_preds_sssi)
lr_prc_sssi = roc_prc_ci(y_test, lr_preds_sssi)

#write results to txt file
with open('s_ssi.txt', 'w') as f:
    f.write('AUROC\n')
    f.write('LR: '+str(round(lr_auc_sssi[1], 3))+' (95% CI'+str(round(lr_auc_sssi[0], 3))+'-'+str(round(lr_auc_sssi[2], 3))+')\n')
    f.write('RF: '+str(round(rf_auc_sssi[1], 3))+' (95% CI'+str(round(rf_auc_sssi[0], 3))+'-'+str(round(rf_auc_sssi[2], 3))+')\n')
    f.write('XGB: '+str(round(xgb_auc_sssi[1], 3))+' (95% CI'+str(round(xgb_auc_sssi[0], 3))+'-'+str(round(xgb_auc_sssi[2], 3))+')\n')
    f.write('NN: '+str(round(ann_auc_sssi[1], 3))+' (95% CI'+str(round(ann_auc_sssi[0], 3))+'-'+str(round(ann_auc_sssi[2], 3))+')\n')
#write results to csv file
with open('s_ssi.csv', 'w', newline='') as csvfile:
    fieldnames = ['model','AUROC mean', 'AUROC 95% CI', 'AUPRC mean', 'AUPRC 95% CI']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'model':'Logistic Regression', 'AUROC mean':round(lr_auc_sssi[1], 3), 'AUROC 95% CI':str(round(lr_auc_sssi[0], 3))+'-'+str(round(lr_auc_sssi[2], 3)), 'AUPRC mean':round(lr_prc_sssi[1], 3), 'AUPRC 95% CI':str(round(lr_prc_sssi[0], 3))+'-'+str(round(lr_prc_sssi[2], 3))})
    writer.writerow({'model':'Random Forest', 'AUROC mean':round(rf_auc_sssi[1], 3), 'AUROC 95% CI':str(round(rf_auc_sssi[0], 3))+'-'+str(round(rf_auc_sssi[2], 3)), 'AUPRC mean':round(rf_prc_sssi[1], 3), 'AUPRC 95% CI':str(round(rf_prc_sssi[0], 3))+'-'+str(round(rf_prc_sssi[2], 3))})
    writer.writerow({'model':'XGBoost', 'AUROC mean':round(xgb_auc_sssi[1], 3), 'AUROC 95% CI':str(round(xgb_auc_sssi[0], 3))+'-'+str(round(xgb_auc_sssi[2], 3)), 'AUPRC mean':round(xgb_prc_sssi[1], 3), 'AUPRC 95% CI':str(round(xgb_prc_sssi[0], 3))+'-'+str(round(xgb_prc_sssi[2], 3))})
    writer.writerow({'model':'Neural Network', 'AUROC mean':round(ann_auc_sssi[1], 3), 'AUROC 95% CI':str(round(ann_auc_sssi[0], 3))+'-'+str(round(ann_auc_sssi[2], 3)), 'AUPRC mean':round(ann_prc_sssi[1], 3), 'AUPRC 95% CI':str(round(ann_prc_sssi[0], 3))+'-'+str(round(ann_prc_sssi[2], 3))})
    
y = data['WNDINFD']
X = data.drop(['SUPINFEC','WNDINFD','ORGSPCSSI','ssi'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

y_test = test['WNDINFD']
X_test = test.drop(['SUPINFEC','WNDINFD','ORGSPCSSI','ssi'], axis=1)

input_shape = [X_train.shape[1]]
model4 = keras.models.Sequential()
model4.add(keras.layers.Flatten(input_shape=input_shape))
model4.add(keras.layers.BatchNormalization())
for _ in range(2):
    model4.add(keras.layers.Dense(100))
    model4.add(keras.layers.BatchNormalization())
    model4.add(keras.layers.Dropout(0.8))
    model4.add(keras.layers.Activation("relu"))
model4.add(keras.layers.Dense(1, activation="sigmoid"))

opt = keras.optimizers.Adam(learning_rate=3e-3)

metrics = [keras.metrics.Recall(name='Sensitivity'), keras.metrics.TrueNegatives(name='tn'), keras.metrics.AUC(name='auc'), keras.metrics.AUC(name='prc', curve='PR')]

model4.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    metrics=metrics,)

early_stopping = keras.callbacks.EarlyStopping(
    patience=25,
    min_delta=1e-8,
    restore_best_weights=True,)

history = model4.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=500,
    callbacks=[early_stopping],)
ann_preds_dssi = model4.predict(X_test)
ann_auc_dssi = roc_auc_ci(y_test, ann_preds_dssi)
ann_prc_dssi = roc_prc_ci(y_test, ann_preds_dssi)

rf = RandomForestClassifier(n_estimators=1250, min_samples_split=4, min_samples_leaf=8, max_features='auto', max_depth=10, bootstrap=True)
rf.fit(X, y)
rf_preds_dssi = rf.predict_proba(X_test)[:,1]
rf_auc_dssi = roc_auc_ci(y_test, rf_preds_dssi)
rf_prc_dssi = roc_prc_ci(y_test, rf_preds_dssi)
xgb = XGBClassifier(subsample=0.4, n_estimators=100, min_child_weight=5, max_depth=4, learning_rate=0.1, colsample_bytree=0.6)
xgb.fit(X, y)
xgb_preds_dssi = xgb.predict_proba(X_test)[:,1]
xgb_auc_dssi = roc_auc_ci(y_test, xgb_preds_dssi)
xgb_prc_dssi = roc_prc_ci(y_test, xgb_preds_dssi)
lr = LogisticRegression(penalty='none')
lr.fit(X, y)
lr_preds_dssi = lr.predict_proba(X_test)[:,1]
lr_auc_dssi = roc_auc_ci(y_test, lr_preds_dssi)
lr_prc_dssi = roc_prc_ci(y_test, lr_preds_dssi)

#write results to txt file
with open('d_ssi.txt', 'w') as f:
    f.write('AUROC\n')
    f.write('LR: '+str(round(lr_auc_dssi[1], 3))+' (95% CI'+str(round(lr_auc_dssi[0], 3))+'-'+str(round(lr_auc_dssi[2], 3))+')\n')
    f.write('RF: '+str(round(rf_auc_dssi[1], 3))+' (95% CI'+str(round(rf_auc_dssi[0], 3))+'-'+str(round(rf_auc_dssi[2], 3))+')\n')
    f.write('XGB: '+str(round(xgb_auc_dssi[1], 3))+' (95% CI'+str(round(xgb_auc_dssi[0], 3))+'-'+str(round(xgb_auc_dssi[2], 3))+')\n')
    f.write('NN: '+str(round(ann_auc_dssi[1], 3))+' (95% CI'+str(round(ann_auc_dssi[0], 3))+'-'+str(round(ann_auc_dssi[2], 3))+')\n')
#write results to csv file
with open('d_ssi.csv', 'w', newline='') as csvfile:
    fieldnames = ['model','AUROC mean', 'AUROC 95% CI', 'AUPRC mean', 'AUPRC 95% CI']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'model':'Logistic Regression', 'AUROC mean':round(lr_auc_dssi[1], 3), 'AUROC 95% CI':str(round(lr_auc_dssi[0], 3))+'-'+str(round(lr_auc_dssi[2], 3)), 'AUPRC mean':round(lr_prc_dssi[1], 3), 'AUPRC 95% CI':str(round(lr_prc_dssi[0], 3))+'-'+str(round(lr_prc_dssi[2], 3))})
    writer.writerow({'model':'Random Forest', 'AUROC mean':round(rf_auc_dssi[1], 3), 'AUROC 95% CI':str(round(rf_auc_dssi[0], 3))+'-'+str(round(rf_auc_dssi[2], 3)), 'AUPRC mean':round(rf_prc_dssi[1], 3), 'AUPRC 95% CI':str(round(rf_prc_dssi[0], 3))+'-'+str(round(rf_prc_dssi[2], 3))})
    writer.writerow({'model':'XGBoost', 'AUROC mean':round(xgb_auc_dssi[1], 3), 'AUROC 95% CI':str(round(xgb_auc_dssi[0], 3))+'-'+str(round(xgb_auc_dssi[2], 3)), 'AUPRC mean':round(xgb_prc_dssi[1], 3), 'AUPRC 95% CI':str(round(xgb_prc_dssi[0], 3))+'-'+str(round(xgb_prc_dssi[2], 3))})
    writer.writerow({'model':'Neural Network', 'AUROC mean':round(ann_auc_dssi[1], 3), 'AUROC 95% CI':str(round(ann_auc_dssi[0], 3))+'-'+str(round(ann_auc_dssi[2], 3)), 'AUPRC mean':round(ann_prc_dssi[1], 3), 'AUPRC 95% CI':str(round(ann_prc_dssi[0], 3))+'-'+str(round(ann_prc_dssi[2], 3))})
y = data['ORGSPCSSI']
X = data.drop(['SUPINFEC','WNDINFD','ORGSPCSSI','ssi'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

y_test = test['ORGSPCSSI']
X_test = test.drop(['SUPINFEC','WNDINFD','ORGSPCSSI','ssi'], axis=1)
input_shape = [X_train.shape[1]]
model4 = keras.models.Sequential()
model4.add(keras.layers.Flatten(input_shape=input_shape))
model4.add(keras.layers.BatchNormalization())
for _ in range(4):
    model4.add(keras.layers.Dense(2000))
    model4.add(keras.layers.BatchNormalization())
    model4.add(keras.layers.Dropout(0.8))
    model4.add(keras.layers.Activation("relu"))
model4.add(keras.layers.Dense(1, activation="sigmoid"))

opt = keras.optimizers.Adam(learning_rate=3e-3)

metrics = [keras.metrics.Recall(name='Sensitivity'), keras.metrics.TrueNegatives(name='tn'), keras.metrics.AUC(name='auc'), keras.metrics.AUC(name='prc', curve='PR')]

model4.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    metrics=metrics,)

early_stopping = keras.callbacks.EarlyStopping(
    patience=25,
    min_delta=1e-8,
    restore_best_weights=True,)

history = model4.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=500,
    callbacks=[early_stopping],)
ann_preds_ossi = model4.predict(X_test)
ann_auc_ossi = roc_auc_ci(y_test, ann_preds_ossi)
ann_prc_ossi = roc_prc_ci(y_test, ann_preds_ossi)

rf = RandomForestClassifier(n_estimators=1250, min_samples_split=3, min_samples_leaf=6, max_features='auto', max_depth=20, bootstrap=False)
rf.fit(X, y)
rf_preds_ossi = rf.predict_proba(X_test)[:,1]
rf_auc_ossi = roc_auc_ci(y_test, rf_preds_ossi)
rf_prc_ossi = roc_prc_ci(y_test, rf_preds_ossi)
xgb = XGBClassifier(subsample=1.0, n_estimators=100, min_child_weight=10, max_depth=6, learning_rate=0.1, colsample_bytree=0.6)
xgb.fit(X, y)
xgb_preds_ossi = xgb.predict_proba(X_test)[:,1]
xgb_auc_ossi = roc_auc_ci(y_test, xgb_preds_ossi)
xgb_prc_ossi = roc_prc_ci(y_test, xgb_preds_ossi)
lr = LogisticRegression(penalty='none')
lr.fit(X, y)
lr_preds_ossi = lr.predict_proba(X_test)[:,1]
lr_auc_ossi = roc_auc_ci(y_test, lr_preds_ossi)
lr_prc_ossi = roc_prc_ci(y_test, lr_preds_ossi)

#write results to txt file
with open('o_ssi.txt', 'w') as f:
    f.write('AUROC\n')
    f.write('LR: '+str(round(lr_auc_ossi[1], 3))+' (95% CI'+str(round(lr_auc_ossi[0], 3))+'-'+str(round(lr_auc_ossi[2], 3))+')\n')
    f.write('RF: '+str(round(rf_auc_ossi[1], 3))+' (95% CI'+str(round(rf_auc_ossi[0], 3))+'-'+str(round(rf_auc_ossi[2], 3))+')\n')
    f.write('XGB: '+str(round(xgb_auc_ossi[1], 3))+' (95% CI'+str(round(xgb_auc_ossi[0], 3))+'-'+str(round(xgb_auc_ossi[2], 3))+')\n')
    f.write('NN: '+str(round(ann_auc_ossi[1], 3))+' (95% CI'+str(round(ann_auc_ossi[0], 3))+'-'+str(round(ann_auc_ossi[2], 3))+')\n')
#write results to csv file
with open('o_ssi.csv', 'w', newline='') as csvfile:
    fieldnames = ['model','AUROC mean', 'AUROC 95% CI', 'AUPRC mean', 'AUPRC 95% CI']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'model':'Logistic Regression', 'AUROC mean':round(lr_auc_ossi[1], 3), 'AUROC 95% CI':str(round(lr_auc_ossi[0], 3))+'-'+str(round(lr_auc_ossi[2], 3)), 'AUPRC mean':round(lr_prc_ossi[1], 3), 'AUPRC 95% CI':str(round(lr_prc_ossi[0], 3))+'-'+str(round(lr_prc_ossi[2], 3))})
    writer.writerow({'model':'Random Forest', 'AUROC mean':round(rf_auc_ossi[1], 3), 'AUROC 95% CI':str(round(rf_auc_ossi[0], 3))+'-'+str(round(rf_auc_ossi[2], 3)), 'AUPRC mean':round(rf_prc_ossi[1], 3), 'AUPRC 95% CI':str(round(rf_prc_ossi[0], 3))+'-'+str(round(rf_prc_ossi[2], 3))})
    writer.writerow({'model':'XGBoost', 'AUROC mean':round(xgb_auc_ossi[1], 3), 'AUROC 95% CI':str(round(xgb_auc_ossi[0], 3))+'-'+str(round(xgb_auc_ossi[2], 3)), 'AUPRC mean':round(xgb_prc_ossi[1], 3), 'AUPRC 95% CI':str(round(xgb_prc_ossi[0], 3))+'-'+str(round(xgb_prc_ossi[2], 3))})
    writer.writerow({'model':'Neural Network', 'AUROC mean':round(ann_auc_ossi[1], 3), 'AUROC 95% CI':str(round(ann_auc_ossi[0], 3))+'-'+str(round(ann_auc_ossi[2], 3)), 'AUPRC mean':round(ann_prc_ossi[1], 3), 'AUPRC 95% CI':str(round(ann_prc_ossi[0], 3))+'-'+str(round(ann_prc_ossi[2], 3))})