Machine Learning NCDB APR Repository
========================================

This is a repository containing code to produce models from NCDB to predict locoregional failure and overall survival for patients with anal cancer 

Due to a data use agreement with the NCDB, the underlying data must be downloaded from the NCDB itself (https://www.facs.org/quality-programs/cancer-programs/national-cancer-database/puf/) after completing an application. For this project, data was downloaded in the SAS format and SAS was used to convert the rectal cancer PUF to csv ('ascc.csv' used here) for use in Python/Pandas.

There are two groups of pre-processing steps that we used to prepare the data for modeling. The first is for abdominoperineal resection (APR) and the second is for 3-yr overall survival. The overall survival notebooks follow the same steps as the APR ones, but have '_os' at the end of the file name.
    - First, 'preproc.ipynb' and 'preproc_os.ipynb' are used to clean the variables of interest
    - Second, 'rename.ipynb' and 'rename_os.ipynb' are used to rename the category names to recongizable names and generate descriptive statistics
    - Third, 'impscale.ipynb' and 'impscale_os.ipynb' are used to impute missing variables, label encode the categorical variables, and scale the continuous variables. SimpleImputer with 'constant' is used for categorical variables and 'median' is used for continuous variables. Sklearn's StandardScaler is applied to the continuous variables.

For hyperparameter search, we use the 'search_all.py' and 'search_all_os.py' scripts. We recommend running these scripts with a GPU.
    - Each script uses BayesSearchCV from scikit-optimize to identify the optimal hyperparameters for each outcome. It then generates ROC and PR curves for each model based on the test set and uses SHAP to analyze the XGB model. Finally, it generates accuracy metrics for the XGB and LR models.