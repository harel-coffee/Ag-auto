# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd

from datetime import date
import time

import random
from random import seed
from random import random

import os, os.path
import shutil

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow
import pickle
import h5py
import sys

from sklearn.metrics import confusion_matrix

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core.py as rcp

# %%
from tslearn.metrics import dtw as dtw_metric

# https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

# %% [markdown]
# # Metadata

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
# print (len(meta.ID.unique()))
# meta_lessThan10Acr=meta[meta.ExctAcr<10]
# print (meta_lessThan10Acr.shape)

# %% [markdown]
# # Read Training Set Labels

# %%
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models_Oct17/"
os.makedirs(model_dir, exist_ok=True)

training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"

# %%
GT_labels = pd.read_csv(training_set_dir+"groundTruth_labels_Oct17_2022.csv")
print ("Unique Votes: ", GT_labels.Vote.unique())
print (len(GT_labels.ID.unique()))
GT_labels.head(2)

# %% [markdown]
# ### Detect how many fields are less than 10 acres and report in the paper

# %%
print (len(meta[meta.ID.isin(list(GT_labels.ID))].ID.unique()))
meta.head(2)

# %%
GT_labels.head(2)

# %% [markdown]
# # Read the data

# %%
VI_idx = "EVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/04_regularized_TS/"

# %%
file_names = ["regular_Walla2015_" + VI_idx + "_JFD.csv", "regular_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              "regular_Grant2017_" + VI_idx + "_JFD.csv", "regular_FranklinYakima2018_" + VI_idx + "_JFD.csv"]

data=pd.DataFrame()

for file in file_names:
    curr_file=pd.read_csv(data_dir + file)
    curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])
    
    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years)==3 or len(all_years)==2:
        proper_year = all_years[1]
    elif len(all_years)==1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
    data=pd.concat([data, curr_file])

data.reset_index(drop=True, inplace=True)
data.head(2)

# %%
GT_TS = data[data.ID.isin(list(GT_labels.ID.unique()))].copy()
len(GT_TS.ID.unique())

# %% [markdown]
# # Toss Smalls

# %%
print (len(meta_moreThan10Acr.ID.unique()))
GT_labels.head(2)

# %%
meta.head(2)

# %%
GT_labels_extended = pd.merge(GT_labels, meta, on=['ID'], how='left')
GT_labels = GT_labels_extended[GT_labels_extended.ExctAcr>=10].copy()
GT_labels.reset_index(drop=True, inplace=True)

A = len(GT_labels_extended)
B = GT_labels_extended.ExctAcr.sum()
print ("There are [{:.0f}] fields in total whose area adds up to [{:.2f}].".format(A , B))

A = len(GT_labels)
B = GT_labels.ExctAcr.sum()
print ("There are [{:.0f}] fields larger than 10 acres whose area adds up to [{:.2f}].".format(A, B))

# %%
GT_labels.head(2)

# %%
GT_TS = GT_TS[GT_TS.ID.isin((list(meta_moreThan10Acr.ID)))].copy()
print (GT_labels.shape)
GT_labels = GT_labels[GT_labels.ID.isin((list(meta_moreThan10Acr.ID)))].copy()

GT_TS.reset_index(drop=True, inplace=True)
GT_labels.reset_index(drop=True, inplace=True)
print (GT_labels.shape)

# %% [markdown]
# # Sort

# %%
GT_TS.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
GT_labels.sort_values(by=["ID"], inplace=True)

GT_TS.reset_index(drop=True, inplace=True)
GT_labels.reset_index(drop=True, inplace=True)

assert (len(GT_TS.ID.unique()) == len(GT_labels.ID.unique()))

print (list(GT_TS.ID)[0])
print (list(GT_labels.ID)[0])
print ("____________________________________")
print (list(GT_TS.ID)[-1])
print (list(GT_labels.ID)[-1])
print ("____________________________________")
print (list(GT_TS.ID.unique())==list(GT_labels.ID.unique()))

# %% [markdown]
# # Widen

# %%
EVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + EVI_colnames
GT_wide = pd.DataFrame(columns=columnNames, 
                                index=range(len(GT_TS.ID.unique())))
GT_wide["ID"] = GT_TS.ID.unique()

for an_ID in GT_TS.ID.unique():
    curr_df = GT_TS[GT_TS.ID==an_ID]
    
    GT_wide_indx = GT_wide[GT_wide.ID==an_ID].index
    GT_wide.loc[GT_wide_indx, "EVI_1":"EVI_36"] = curr_df.EVI.values[:36]
    
print (len(GT_wide.ID.unique()))
GT_wide.head(2)

# %%
print (GT_labels.ExctAcr.min())
GT_labels.head(2)

# %% [markdown]
# # Train and Test Set
#
# #### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order

# %%
# This cell and some edits after this are new compared to 00_SVM_SG_EVI.ipynb.
# I want to avoid splitting and just use the one I created earlier.

ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
train80 = pd.read_csv(ML_data_folder+"train80_split_2Bconsistent_Oct17.csv")
test20 = pd.read_csv(ML_data_folder+"test20_split_2Bconsistent_Oct17.csv")

GT_labels.head(2)

# %%

# %%
x_train_df = GT_wide[GT_wide.ID.isin(list(train80.ID))]
x_test_df = GT_wide[GT_wide.ID.isin(list(test20.ID))]

y_train_df = GT_labels[GT_labels.ID.isin(list(train80.ID))]
y_test_df = GT_labels[GT_labels.ID.isin(list(test20.ID))]

y_train_df=y_train_df[["ID", "Vote"]]
y_test_df=y_test_df[["ID", "Vote"]]

# %%
#### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order

print ((x_train_df.ID==y_train_df.ID).sum())
print ((x_test_df.ID==y_test_df.ID).sum())

# %%
# GT_labels = GT_labels.set_index('ID')
# GT_labels = GT_labels.reindex(index=GT_wide['ID'])
# GT_labels = GT_labels.reset_index()
# GT_labels = GT_labels[["ID", "Vote"]]

# x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(GT_wide, 
#                                                                 GT_labels, 
#                                                                 test_size=0.2, 
#                                                                 random_state=0,
#                                                                 shuffle=True,
#                                                                 stratify=GT_labels.Vote.values)
# x_test_df.shape

# %% [markdown]
# # Start Random Forest

# %% [markdown]
# # Definitions
#
#   - **Precision** Of all instances we predict $\hat y = 1$, what fraction is actually 1.
#      \begin{equation}\label{eq:precision}
#         \text{Precision} = \frac{TP}{TP + FP}
#      \end{equation}
#
#   - **Recall** Of all instances that are actually $y = 1$, what fraction we predict 1.
#      \begin{equation}\label{eq:recall}
#          \text{Recall} = \text{TPR} = \frac{TP}{TP + FN}
#      \end{equation}
#      
#   - **Specifity** Fraction of all negative instances that are correctly predicted positive.
#      \begin{equation}\label{eq:specifity}
#         \text{Specifity} = TNR = \frac{TN}{TN + FP}\\
#      \end{equation}
#      
#   - **F-Score** Adjust $\beta$ for trade off between  precision and recall. For precision oriented task $\beta = 0.5$.
#      \begin{equation}\label{eq:Fscore}
#         F_\beta = \frac{(1+\beta^2) TP}{ (1+\beta^2) TP + \beta^2 FN + FP}
#      \end{equation}

# %%
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

# %%
# %%time
regular_forest_1_default = RandomForestClassifier(n_estimators=100, 
                                                  criterion='gini', max_depth=None, 
                                                  min_samples_split=2, min_samples_leaf=1, 
                                                  min_weight_fraction_leaf=0.0,
                                                  max_features='sqrt', max_leaf_nodes=None, 
                                                  min_impurity_decrease=0.0, 
                                                  bootstrap=True, oob_score=False, n_jobs=None, 
                                                  random_state=1, verbose=0, 
                                                  warm_start=False, class_weight=None, 
                                                  ccp_alpha=0.0, max_samples=None)

regular_forest_1_default.fit(x_train_df.iloc[:, 1:], y_train_df.iloc[:, 1:].values.ravel())

# %% [markdown]
# # Test

# %%
regular_forest_1_default_predictions = regular_forest_1_default.predict(x_test_df.iloc[:, 1:])
regular_forest_1_default_y_test_df = y_test_df.copy()
regular_forest_1_default_y_test_df["prediction"]=list(regular_forest_1_default_predictions)
regular_forest_1_default_y_test_df.head(2)

# %%
true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in regular_forest_1_default_y_test_df.index:
    curr_vote=list(regular_forest_1_default_y_test_df[regular_forest_1_default_y_test_df.index==index_].Vote)[0]
    curr_predict=list(regular_forest_1_default_y_test_df[\
                                            regular_forest_1_default_y_test_df.index==index_].prediction)[0]
    if curr_vote==curr_predict:
        if curr_vote==1: 
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
        else:
            true_double_predicted_single+=1
            
regular_forest_default_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                                               index=range(2))
regular_forest_default_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
regular_forest_default_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
regular_forest_default_confus_tbl_test['Predict_Single']=0
regular_forest_default_confus_tbl_test['Predict_Double']=0

regular_forest_default_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
regular_forest_default_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
regular_forest_default_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
regular_forest_default_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
regular_forest_default_confus_tbl_test

# %%
# FD1_y_test_df_act_1_pred_2=regular_forest_1_default_y_test_df[regular_forest_1_default_y_test_df.Vote==1].copy()
# FD1_y_test_df_act_2_pred_1=regular_forest_1_default_y_test_df[regular_forest_1_default_y_test_df.Vote==2].copy()

# FD1_y_test_df_act_1_pred_2=FD1_y_test_df_act_1_pred_2[FD1_y_test_df_act_1_pred_2.prediction==2].copy()
# FD1_y_test_df_act_2_pred_1=FD1_y_test_df_act_2_pred_1[FD1_y_test_df_act_2_pred_1.prediction==1].copy()

# FD1_y_test_df_act_2_pred_1 = pd.merge(FD1_y_test_df_act_2_pred_1, \
#                                            GT_labels_extended, on=['ID'], how='left')
# FD1_y_test_df_act_1_pred_2 = pd.merge(FD1_y_test_df_act_1_pred_2, \
#                                       GT_labels_extended, on=['ID'], how='left')

# print (FD1_y_test_df_act_2_pred_1.ExctAcr.sum())
# print (FD1_y_test_df_act_1_pred_2.ExctAcr.sum())

# print (FD1_y_test_df_act_2_pred_1.ExctAcr.sum()-FD1_y_test_df_act_1_pred_2.ExctAcr.sum())

# %%
filename = model_dir + "regular_" + VI_idx + "_RF1_default" + "_Oct17_accuracyScoring.sav"
pickle.dump(regular_forest_1_default, open(filename, 'wb'))

# %%
# %%time
parameters = {'n_jobs':[6],
              'criterion': ["gini", "entropy"], # log_loss 
              # 'scoring':["roc_auc", "accuracy"],
              'max_depth':[1, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17],
              'min_samples_split':[4],
              'max_features': ["log2"],
              'class_weight':[None],
              'ccp_alpha':[0.0], 
             # 'min_impurity_decreasefloat':[0, 1, 2], # roblem with sqrt stuff?
              'max_samples':[None]
             } # , 
regular_forest_grid_1 = GridSearchCV(RandomForestClassifier(random_state=0), 
                                     parameters, cv=5, verbose=1,
                                     error_score='raise')

regular_forest_grid_1.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

print (regular_forest_grid_1.best_params_)
print (regular_forest_grid_1.best_score_)

# %%
filename = model_dir + "regular_" + VI_idx + "_RF_grid_1_Oct17_accuracyScoring.sav"
pickle.dump(regular_forest_grid_1, open(filename, 'wb'))

# %% [markdown]
# # Test

# %%
regular_forest_grid_1_predictions = regular_forest_grid_1.predict(x_test_df.iloc[:, 1:])
regular_forest_grid_1_y_test_df=y_test_df.copy()
regular_forest_grid_1_y_test_df["prediction"]=list(regular_forest_grid_1_predictions)

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in regular_forest_grid_1_y_test_df.index:
    curr_vote=list(regular_forest_grid_1_y_test_df[regular_forest_grid_1_y_test_df.index==index_].Vote)[0]
    curr_predict=list(regular_forest_grid_1_y_test_df[regular_forest_grid_1_y_test_df.index==index_].prediction)[0]
    if curr_vote==curr_predict:
        if curr_vote==1: 
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
        else:
            true_double_predicted_single+=1
            
regular_forest_grid_1_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
regular_forest_grid_1_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
regular_forest_grid_1_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
regular_forest_grid_1_confus_tbl_test['Predict_Single']=0
regular_forest_grid_1_confus_tbl_test['Predict_Double']=0

regular_forest_grid_1_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
regular_forest_grid_1_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
regular_forest_grid_1_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
regular_forest_grid_1_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
regular_forest_grid_1_confus_tbl_test

# %%
confusion_matrix(regular_forest_grid_1_y_test_df.Vote, regular_forest_grid_1_y_test_df.prediction)

# %% [markdown]
# ### Regular More parameters

# %%
# %%time
parameters = {'n_jobs':[6],
              'criterion': ["gini", "entropy"], # log_loss 
              'max_depth':[2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20],
              'min_samples_split':[2, 3, 4, 5],
              'max_features': ["sqrt", "log2", None],
              'class_weight':['balanced', 'balanced_subsample', None],
              'ccp_alpha':[0.0, 1, 2, 3], 
             # 'min_impurity_decreasefloat':[0, 1, 2], # roblem with sqrt stuff?
              'max_samples':[None, 1, 2, 3, 4, 5]
             }

# parameters = {'n_jobs':[5],
#               'criterion': ["gini", "entropy"], # log_loss 
#               'max_depth':[1, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17],
#               'min_samples_split':[4],
#               'max_features': ["log2"],
#               'class_weight':[None],
#               'ccp_alpha':[0.0], 
#              # 'min_impurity_decreasefloat':[0, 1, 2], # roblem with sqrt stuff?
#               'max_samples':[None]
#              }

# {'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'entropy', 
#   'max_depth': 13, 'max_features': 'log2', 'max_samples': None, 'min_samples_split': 4, 'n_jobs': 6}
# 0.9596446601181856


regular_forest_grid_2 = GridSearchCV(RandomForestClassifier(random_state=0), 
                                     parameters, cv=5, verbose=1,
                                     error_score='raise')

regular_forest_grid_2.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

print (regular_forest_grid_2.best_params_)
print (regular_forest_grid_2.best_score_)


regular_forest_grid_2_predictions = regular_forest_grid_2.predict(x_test_df.iloc[:, 1:])
regular_forest_grid_2_y_test_df=y_test_df.copy()
regular_forest_grid_2_y_test_df["prediction"]=list(regular_forest_grid_2_predictions)
regular_forest_grid_2_y_test_df.head(2)

# %%
true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in regular_forest_grid_2_y_test_df.index:
    curr_vote=list(regular_forest_grid_2_y_test_df[regular_forest_grid_2_y_test_df.index==index_].Vote)[0]
    curr_predict=list(regular_forest_grid_2_y_test_df[regular_forest_grid_2_y_test_df.index==index_].prediction)[0]
    if curr_vote==curr_predict:
        if curr_vote==1: 
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
        else:
            true_double_predicted_single+=1
            
regular_forest_grid_2_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
regular_forest_grid_2_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
regular_forest_grid_2_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
regular_forest_grid_2_confus_tbl_test['Predict_Single']=0
regular_forest_grid_2_confus_tbl_test['Predict_Double']=0

regular_forest_grid_2_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
regular_forest_grid_2_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
regular_forest_grid_2_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
regular_forest_grid_2_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
regular_forest_grid_2_confus_tbl_test

# %%
confusion_matrix(regular_forest_grid_2_y_test_df.Vote, regular_forest_grid_2_y_test_df.prediction)

# %%


print ("accuracy of more parameters on test set is [{:.4f}].".format(accuracy_score(y_test_df.Vote, 
                                                                     regular_forest_grid_1_y_test_df.prediction)))
print ("accuracy of less parameters on test set is [{:.4f}].".format(accuracy_score(y_test_df.Vote, 
                                                                     regular_forest_grid_2_y_test_df.prediction)))


# %%
# FG2_y_test_df_act_1_pred_2=regular_forest_grid_2_y_test_df[regular_forest_grid_2_y_test_df.Vote==1].copy()
# FG2_y_test_df_act_2_pred_1=regular_forest_grid_2_y_test_df[regular_forest_grid_2_y_test_df.Vote==2].copy()

# FG2_y_test_df_act_1_pred_2=FG2_y_test_df_act_1_pred_2[FG2_y_test_df_act_1_pred_2.prediction==2].copy()
# FG2_y_test_df_act_2_pred_1=FG2_y_test_df_act_2_pred_1[FG2_y_test_df_act_2_pred_1.prediction==1].copy()

# FG2_y_test_df_act_2_pred_1 = pd.merge(FG2_y_test_df_act_2_pred_1, 
#                                       GT_labels_extended, on=['ID'], how='left')
# FG2_y_test_df_act_1_pred_2 = pd.merge(FG2_y_test_df_act_1_pred_2, 
#                                       GT_labels_extended, on=['ID'], how='left')

# print (FG2_y_test_df_act_2_pred_1.ExctAcr.sum())
# print (FG2_y_test_df_act_1_pred_2.ExctAcr.sum())
# print (FG2_y_test_df_act_2_pred_1.ExctAcr.sum()-FG2_y_test_df_act_1_pred_2.ExctAcr.sum())

# %%
filename = model_dir + "regular_" + VI_idx + "_RF_grid_2_Oct17_accuracyScoring.sav"
pickle.dump(regular_forest_grid_2, open(filename, 'wb'))

# %% [markdown]
# # SG

# %%
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"

file_names = ["SG_Walla2015_" + VI_idx + "_JFD.csv", "SG_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              "SG_Grant2017_" + VI_idx + "_JFD.csv", "SG_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

data=pd.DataFrame()

for file in file_names:
    curr_file=pd.read_csv(data_dir + file)
    curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])
    
    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years)==3 or len(all_years)==2:
        proper_year = all_years[1]
    elif len(all_years)==1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
    data=pd.concat([data, curr_file])

data.reset_index(drop=True, inplace=True)
data.head(2)

# %%

# %%
# GT_labels_extended = pd.merge(GT_labels, meta, on=['ID'], how='left')
# GT_labels = GT_labels_extended[GT_labels_extended.ExctAcr>=10].copy()
# GT_labels.reset_index(drop=True, inplace=True)

# A = len(GT_labels_extended)
# B = GT_labels_extended.ExctAcr.sum()
# print ("There are [{:.0f}] fields in total whose area adds up to [{:.2f}].".format(A , B))

# A = len(GT_labels)
# B = GT_labels.ExctAcr.sum()
# print ("There are [{:.0f}] fields larger than 10 acres whose area adds up to [{:.2f}].".format(A, B))

# %%
GT_TS = data[data.ID.isin(list(GT_labels.ID.unique()))].copy()

# print (len(meta_moreThan10Acr.ID.unique()))
# GT_labels_extended = pd.merge(GT_labels, meta, on=['ID'], how='left')
# GT_labels = GT_labels_extended[GT_labels_extended.ExctAcr>=10].copy()
# GT_labels.reset_index(drop=True, inplace=True)

# print ("There are [{:.0f}] fields in total whose \
#        area adds up to [{:.2f}].".format(len(GT_labels_extended),\
#                                                             GT_labels_extended.ExctAcr.sum()))


# print ("There are [{:.0f}] fields larger than 10 \
#         acres whose area adds up to [{:.2f}].".format(len(GT_labels), \
#                                                                     GT_labels.ExctAcr.sum()))
GT_TS.head(2)

# %%
GT_TS = GT_TS[GT_TS.ID.isin((list(meta_moreThan10Acr.ID)))].copy()


GT_TS.reset_index(drop=True, inplace=True)
GT_labels.reset_index(drop=True, inplace=True)

GT_TS.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
GT_labels.sort_values(by=["ID"], inplace=True)

GT_TS.reset_index(drop=True, inplace=True)
GT_labels.reset_index(drop=True, inplace=True)

assert (len(GT_TS.ID.unique()) == len(GT_labels.ID.unique()))

print (list(GT_TS.ID)[0])
print (list(GT_labels.ID)[0])
print ("____________________________________")
print (list(GT_TS.ID)[-1])
print (list(GT_labels.ID)[-1])
print ("____________________________________")
print (list(GT_TS.ID.unique())==list(GT_labels.ID.unique()))

# %%
GT_labels.head(2)

# %% [markdown]
# # Widen

# %%
EVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + EVI_colnames
GT_wide = pd.DataFrame(columns=columnNames, 
                       index=range(len(GT_TS.ID.unique())))
GT_wide["ID"] = GT_TS.ID.unique()

for an_ID in GT_TS.ID.unique():
    curr_df = GT_TS[GT_TS.ID==an_ID]
    
    GT_wide_indx = GT_wide[GT_wide.ID==an_ID].index
    GT_wide.loc[GT_wide_indx, "EVI_1":"EVI_36"] = curr_df.EVI.values[:36]

# %%
GT_labels = GT_labels.set_index('ID')
GT_labels = GT_labels.reindex(index=GT_wide['ID'])
GT_labels = GT_labels.reset_index()
GT_labels = GT_labels[["ID", "Vote"]]
GT_labels.head(2)

# %%
x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(GT_wide, 
                                                                GT_labels, 
                                                                test_size=0.2, 
                                                                random_state=0,
                                                                shuffle=True,
                                                                stratify=GT_labels.Vote.values)

# %% [markdown]
# # Train

# %%
# %%time
forest_1_default_SG = RandomForestClassifier(n_estimators=100, 
                                             criterion='gini', max_depth=None, 
                                             min_samples_split=2, min_samples_leaf=1, 
                                             min_weight_fraction_leaf=0.0,
                                             max_features='sqrt', max_leaf_nodes=None, 
                                             min_impurity_decrease=0.0, 
                                             bootstrap=True, oob_score=False, n_jobs=None, 
                                             random_state=1, verbose=0, 
                                             warm_start=False, class_weight=None, 
                                             ccp_alpha=0.0, max_samples=None)

print (forest_grid_1_SG.best_params_)
print (forest_grid_1_SG.best_score_)

forest_1_default_SG.fit(x_train_df.iloc[:, 1:], y_train_df.iloc[:, 1:].values.ravel())

# %%
forest_1_default_SG_predictions = forest_1_default_SG.predict(x_test_df.iloc[:, 1:])
forest_1_default_SG_y_test_df=y_test_df.copy()
forest_1_default_SG_y_test_df["prediction"]=list(forest_1_default_SG_predictions)
forest_1_default_SG_y_test_df.head(2)

# %%
true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in forest_1_default_SG_y_test_df.index:
    curr_vote=list(forest_1_default_SG_y_test_df[forest_1_default_SG_y_test_df.index==index_].Vote)[0]
    curr_predict=list(forest_1_default_SG_y_test_df[forest_1_default_SG_y_test_df.index==index_].prediction)[0]
    if curr_vote==curr_predict:
        if curr_vote==1: 
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
        else:
            true_double_predicted_single+=1
            
forest_default_SG_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
forest_default_SG_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
forest_default_SG_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
forest_default_SG_confus_tbl_test['Predict_Single']=0
forest_default_SG_confus_tbl_test['Predict_Double']=0

forest_default_SG_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
forest_default_SG_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
forest_default_SG_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
forest_default_SG_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
forest_default_SG_confus_tbl_test

# %%
# FD1_yTest_df_act_1_pred_2=forest_1_default_SG_y_test_df[forest_1_default_SG_y_test_df.Vote==1].copy()
# FD1_yTest_df_act_2_pred_1=forest_1_default_SG_y_test_df[forest_1_default_SG_y_test_df.Vote==2].copy()

# FD1_yTest_df_act_1_pred_2=FD1_yTest_df_act_1_pred_2[FD1_yTest_df_act_1_pred_2.prediction==2].copy()
# FD1_yTest_df_act_2_pred_1=FD1_yTest_df_act_2_pred_1[FD1_yTest_df_act_2_pred_1.prediction==1].copy()

# FD1_yTest_df_act_2_pred_1 = pd.merge(FD1_yTest_df_act_2_pred_1, GT_labels_extended, on=['ID'], how='left')
# FD1_yTest_df_act_1_pred_2 = pd.merge(FD1_yTest_df_act_1_pred_2, GT_labels_extended, on=['ID'], how='left')

# aa=FD1_yTest_df_act_2_pred_1.ExctAcr.sum()
# bb=FD1_yTest_df_act_1_pred_2.ExctAcr.sum()
# print ("FD1_yTest_df_act_2_pred_1.ExctAcr.sum(): ", aa)
# print ("FD1_yTest_df_act_1_pred_2.ExctAcr.sum(): ", bb)
# print (aa-bb)

# %%

# %%
# %%time
parameters = {'n_jobs':[6],
              'criterion': ["gini", "entropy"], # log_loss 
              'max_depth':[2, 4, 6, 8, 9, 10, 11, 12, 14, 16, 18, 20],
              'min_samples_split':[2, 3, 4, 5],
              'max_features': ["sqrt", "log2", None],
              'class_weight':['balanced', 'balanced_subsample', None],
              'ccp_alpha':[0.0, 1, 2, 3], 
             # 'min_impurity_decreasefloat':[0, 1, 2], # roblem with sqrt stuff?
              'max_samples':[None, 1, 2, 3, 4, 5]
             } # , 
forest_grid_1_SG = GridSearchCV(RandomForestClassifier(random_state=0), 
                                parameters, cv=5, verbose=1,
                                error_score='raise')

forest_grid_1_SG.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

print (forest_grid_1_SG.best_params_)
print (forest_grid_1_SG.best_score_)

filename = model_dir + "SG_" + VI_idx + "_RF_grid_1_Oct17_accuracyScoring.sav"
pickle.dump(forest_grid_1_SG, open(filename, 'wb'))

# %%
forest_grid_1_SG_predictions = forest_grid_1_SG.predict(x_test_df.iloc[:, 1:])
forest_grid_1_SG_y_test_df=y_test_df.copy()
forest_grid_1_SG_y_test_df["prediction"]=list(forest_grid_1_SG_predictions)
forest_grid_1_SG_y_test_df.head(2)

# %%
true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in forest_grid_1_SG_y_test_df.index:
    curr_vote=list(forest_grid_1_SG_y_test_df[forest_grid_1_SG_y_test_df.index==index_].Vote)[0]
    curr_predict=list(forest_grid_1_SG_y_test_df[forest_grid_1_SG_y_test_df.index==index_].prediction)[0]
    if curr_vote==curr_predict:
        if curr_vote==1: 
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
        else:
            true_double_predicted_single+=1
            
forest_default_SG_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
forest_default_SG_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
forest_default_SG_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
forest_default_SG_confus_tbl_test['Predict_Single']=0
forest_default_SG_confus_tbl_test['Predict_Double']=0

forest_default_SG_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
forest_default_SG_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
forest_default_SG_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
forest_default_SG_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
forest_default_SG_confus_tbl_test

# %%
# FG1_yTest_df_act_1_pred_2=forest_grid_1_SG_y_test_df[forest_grid_1_SG_y_test_df.Vote==1].copy()
# FG1_yTest_df_act_2_pred_1=forest_grid_1_SG_y_test_df[forest_grid_1_SG_y_test_df.Vote==2].copy()

# FG1_yTest_df_act_1_pred_2=FG1_yTest_df_act_1_pred_2[FG1_yTest_df_act_1_pred_2.prediction==2].copy()
# FG1_yTest_df_act_2_pred_1=FG1_yTest_df_act_2_pred_1[FG1_yTest_df_act_2_pred_1.prediction==1].copy()

# FG1_yTest_df_act_2_pred_1 = pd.merge(FG1_yTest_df_act_2_pred_1, GT_labels_extended, on=['ID'], how='left')
# FG1_yTest_df_act_1_pred_2 = pd.merge(FG1_yTest_df_act_1_pred_2, GT_labels_extended, on=['ID'], how='left')

# print (FG1_yTest_df_act_2_pred_1.ExctAcr.sum())
# print (FG1_yTest_df_act_1_pred_2.ExctAcr.sum())
# print (np.abs(FG1_yTest_df_act_1_pred_2.ExctAcr.sum() - FG1_yTest_df_act_2_pred_1.ExctAcr.sum()))

# %%

# %%
