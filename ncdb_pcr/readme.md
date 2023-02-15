Machine Learning NCDB pCR Repository
========================================
This is a repository containing code to produce models from NCDB to predict pathologic complete response following neoadjuvant chemoradiation for rectal cancer.

Due to a data use agreement with the NCDB, the underlying data must be downloaded from the NCDB itself (https://www.facs.org/quality-programs/cancer-programs/national-cancer-database/puf/) after completing an application. For this project, data was downloaded in the SAS format and SAS was used to convert the rectal cancer PUF to csv ('rectum.csv' used here) for use in Python/Pandas.

We use three pre-processing notebooks to prepare the data for modeling.
- First, we identify and prepare the variables of interest using 'preproc_clean.ipynb'
- Second, we rename all the variable categories to their recognizable names and generate descriptive statistics using 'rename.ipynb'
- Third, we apply impute missing variables (using sklearn's SimpleImputer - constant for categorical variables and SimpleImputer - median for continuous variables) and apply sklearn's LabelEncoder to the categorical variables in 'preproc_label_impute.ipynb'. This generates 'train.csv' and 'test.csv'

For hyperparameter search, we use the 'search_all.py' and 'search_small.py' scripts
- search_all.py uses BayesSearchCV from scikit-optimize to find the optimal hyperparameters for each model and then generates ROC and PR curves based on those hyperparameters as well as accuracy metrics. Logistic regression is implemented using sklearn with penalty='none' and max_iter=9000.
- search_small.py uses the same methods for the 5 variable model
