import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras

# %%
#Open csv file
data = pd.read_feather('/nas/longleaf/home/kchen315/Documents/ssi/procol_train.feather')

y = data['SUPINFEC']
X = data.drop(['SUPINFEC','WNDINFD','ORGSPCSSI','ssi'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

# In[ ]:
input_shape = [X_train.shape[1]]

def build_model(n_hidden=1, n_neurons=30, dropout=0.3, learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=['AUC'], optimizer=optimizer)
    return model


# In[ ]:


keras_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model)


# In[ ]:


param_distribs = {
    "n_hidden": [0, 2, 4],
    "n_neurons": [100, 200, 500, 1000, 2000],
    "dropout": [0.2, 0.4, 0.6, 0.8],
    "learning_rate": [3e-5, 3e-4, 3e-3],
}



rnd_search_cv = RandomizedSearchCV(keras_clf, param_distribs, n_iter=75, scoring='roc_auc', cv=5, n_jobs=-1, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=500, batch_size=512,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=15, min_delta=1e-6, restore_best_weights=True)])

results = pd.DataFrame(rnd_search_cv.cv_results_)
results.sort_values(by='rank_test_score').to_csv('results_keras.csv')
