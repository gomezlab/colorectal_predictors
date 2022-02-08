This repository accompanies the manuscript "Development and Validation of Machine Learning Models to Predict Readmission after Colorectal Surgery" submitted to Surgical Endoscopy and contains the code which can be used to reproduce work.

The notebooks in the combine_puf folder can be used to combine the colectomy and proctectomy datasets. Once the combined csv is created, it can be pre-processed using 'preproc.ipynb'. 'table1.ipynb' can be used to generate summary statistics. Scripts in the hyperparameter_search folder can be used to find optimal hyperparameters for each model. Then these parameters can be inputted into 'all_models.ipynb' and metrics calculated. These notebooks also produces TPR/FPR's and precision/recall's to be used in 'curves.ipynb'. Finally, 'shap.ipynb' can be used to build a NN model and perform SHAP analysis.


Then these parameters can be inputted into 'all_models.ipynb' and metrics calculated. This notebook also produces TPR/FPR's and precision/recall's to be used in 'curves.ipynb'. Finally, 'shap.ipynb' can be used to build a NN model and perform SHAP analysis.

