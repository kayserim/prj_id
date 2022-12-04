# prj_id
There are 4 colab notebooks:
The notebooks in general expects that MIMIC-III is available under 'cse6250_proj/data/all' directory in google drive (see the first few cells in each notebook).

Correct sequence to run notebooks:

1. [generate_labels](../main/generate_labels.ipynb): categorizes icu_stays as positive and negative as in the paper (knobs provided to randomly sample a fraction of the icu stays for testing purposes) and saves the labeled icu stays as ICUSTAYS_LITE.csv in 'cse6250_proj/data/all' directory. Also saves corresponding CHARTEVENTS_LITE.csv to the same directory. This new versio of chart events is much smaller in size.

2. [generate_features](../main/generate_features.ipynb): loads up ICUSTAYS_LITE.csv & CHARTEVENTS_LITE.csv and generate features (pre discharge/transfer 48hr: body temperature, blood pressure,... diagnostics: 17 categories, demographics: age, race,...). Then applies SMOTE resampling. Then, splits features and labels to 80-10-10 and saves as XY_train_LITE.csv, XY_test_LITE.csv and XY_validation_LITE.csv in 'cse6250_proj/data/all' directory.

3. [tune](../main/tune.ipynb): this script performs the hyperparameter tuning for the selected model and is optional. It has a range of parameter settings for hyperparameter tuning and requires weights & biases (which needs a sign-up & api-key). This colab notebook expects utils.py and mymodels.py to be in the same directory. 

4. [train_and_evaluate](../main/train_and_evaluate.ipynb): very similar to hw5's train seizure where a selected model is trained and evaluated based on accuracy. Additionally, validation curve for the training as well as confusion matrix for the evaluation is plotted. Also precision, recall and auc scores for the test data is printed out. Models available are logistic regression, LSTM and LSTM+CNN. This colab notebook expects utils.py and mymodels.py to be in the same directory. 
