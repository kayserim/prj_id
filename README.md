# prj_id
There are 3 colab notebooks:
The notebooks in general expects that MIMIC-III is available under 'cse6250_proj/data/all' directory in google drive (see the first few cells in each notebook).

Correct sequence to run notebooks:

1. [generate_labels](../generate_labels.ipynb): categorizes icu_stays as positive and negative as in the paper and randomly samples 10% of the icu stays and saves the labeled and downsampled icu stays as ICUSTAYS_LITE.csv in 'cse6250_proj/data/all' directory. Downsampling will be removed later. Also saves corresponding CHARTEVENTS_LITE.csv to the same directory. This new versio of chart events is much smaller in size.

2. [generate_labels](../generate_features.ipynb): loads up ICUSTAYS_LITE.csv & CHARTEVENTS_LITE.csv and generate features (pre discharge/transfer 48hr) (Body temperature and Blood Pressure only for now). Then, splits features and labels to 80-10-10 and saves as XY_train_LITE.csv, XY_test_LITE.csv and XY_validation_LITE.csv in 'cse6250_proj/data/all' directory.

3. [train_and_evaluate](../train_and_evaluate.ipynb): very similar to hw5's train seizure where a selected model is trained and evaluated based on accuracy. Additionally, validation curve for the training as well as confusion matrix for the evaluation is plotted. Models available are logistic regression and LSTM at this moment. 
