# ---
# jupyter:
#   jupytext:
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

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow
import pickle
import h5py
import sys

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core.py as rcp

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
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"

ground_truth_labels = pd.read_csv(training_set_dir+"train_labels.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))
ground_truth_labels.head(2)

# %% [markdown]
# ### Detect how many fields are less than 10 acres and report in the paper

# %%
print (len(meta[meta.ID.isin(list(ground_truth_labels.ID))].ID.unique()))
meta.head(2)

# %% [markdown]
# # Read the data

# %%
VI_idx = "EVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/04_regularized_TS/"

# %%
file_names = ["regular_Walla2015_EVI_JFD.csv", "regular_AdamBenton2016_EVI_JFD.csv", 
              "regular_Grant2017_EVI_JFD.csv", "regular_FranklinYakima2018_EVI_JFD.csv"]

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
ground_truth = data[data.ID.isin(list(ground_truth_labels.ID.unique()))].copy()
len(ground_truth.ID.unique())

# %% [markdown]
# # Toss Smalls

# %%
ground_truth_labels_extended = pd.merge(ground_truth_labels, meta, on=['ID'], how='left')
ground_truth_labels = ground_truth_labels_extended[ground_truth_labels_extended.ExctAcr>=10].copy()
ground_truth_labels.reset_index(drop=True, inplace=True)

print ("There are [{:.0f}] fields in total whose area adds up to [{:.2f}].".format(len(ground_truth_labels_extended), \
                                                                     ground_truth_labels_extended.ExctAcr.sum()))


print ("There are [{:.0f}] fields larger than 10 acres whose area adds up to [{:.2f}].".format(len(ground_truth_labels), \
                                                                    ground_truth_labels.ExctAcr.sum()))


# %%
ground_truth = ground_truth[ground_truth.ID.isin((list(meta_moreThan10Acr.ID)))].copy()
ground_truth_labels = ground_truth_labels[ground_truth_labels.ID.isin((list(meta_moreThan10Acr.ID)))].copy()

ground_truth.reset_index(drop=True, inplace=True)
ground_truth_labels.reset_index(drop=True, inplace=True)

# %% [markdown]
# # Sort

# %%
ground_truth.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
ground_truth_labels.sort_values(by=["ID"], inplace=True)

ground_truth.reset_index(drop=True, inplace=True)
ground_truth_labels.reset_index(drop=True, inplace=True)

assert (len(ground_truth.ID.unique()) == len(ground_truth_labels.ID.unique()))

print (list(ground_truth.ID)[0])
print (list(ground_truth_labels.ID)[0])
print ("____________________________________")
print (list(ground_truth.ID)[-1])
print (list(ground_truth_labels.ID)[-1])
print ("____________________________________")
print (list(ground_truth.ID.unique())==list(ground_truth_labels.ID.unique()))

# %% [markdown]
# # Widen

# %%
EVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + EVI_colnames
ground_truth_wide = pd.DataFrame(columns=columnNames, 
                                index=range(len(ground_truth.ID.unique())))
ground_truth_wide["ID"] = ground_truth.ID.unique()

for an_ID in ground_truth.ID.unique():
    curr_df = ground_truth[ground_truth.ID==an_ID]
    
    ground_truth_wide_indx = ground_truth_wide[ground_truth_wide.ID==an_ID].index
    ground_truth_wide.loc[ground_truth_wide_indx, "EVI_1":"EVI_36"] = curr_df.EVI.values[:36]

# %%
print (len(ground_truth_wide.ID.unique()))
ground_truth_wide.head(2)

# %% [markdown]
# # Split Train and Test Set
#
# #### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order

# %%
print (ground_truth_labels.CropTyp.unique())
ground_truth_labels.head(2)

# %%
ground_truth.head(2)

# %%
ground_truth_labels = ground_truth_labels.set_index('ID')
ground_truth_labels = ground_truth_labels.reindex(index=ground_truth_wide['ID'])
ground_truth_labels = ground_truth_labels.reset_index()

print (ground_truth_labels.ExctAcr.min())
ground_truth_labels.head(2)

# %%
ground_truth_labels=ground_truth_labels[["ID", "Vote"]]

# %%
x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(ground_truth_wide, 
                                                                ground_truth_labels, 
                                                                test_size=0.2, 
                                                                random_state=0,
                                                                shuffle=True,
                                                                stratify=ground_truth_labels.Vote.values)
x_test_df.shape

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

# %%
regular_forest_1_default_predictions = regular_forest_1_default.predict(x_test_df.iloc[:, 1:])
regular_forest_1_default_y_test_df = y_test_df.copy()
regular_forest_1_default_y_test_df["prediction"]=list(regular_forest_1_default_predictions)
regular_forest_1_default_y_test_df.head(2)

# %%

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
FD1_y_test_df_act_1_pred_2=regular_forest_1_default_y_test_df[regular_forest_1_default_y_test_df.Vote==1].copy()
FD1_y_test_df_act_2_pred_1=regular_forest_1_default_y_test_df[regular_forest_1_default_y_test_df.Vote==2].copy()

FD1_y_test_df_act_1_pred_2=FD1_y_test_df_act_1_pred_2[FD1_y_test_df_act_1_pred_2.prediction==2].copy()
FD1_y_test_df_act_2_pred_1=FD1_y_test_df_act_2_pred_1[FD1_y_test_df_act_2_pred_1.prediction==1].copy()

FD1_y_test_df_act_2_pred_1 = pd.merge(FD1_y_test_df_act_2_pred_1, \
                                           ground_truth_labels_extended, on=['ID'], how='left')
FD1_y_test_df_act_1_pred_2 = pd.merge(FD1_y_test_df_act_1_pred_2, \
                                      ground_truth_labels_extended, on=['ID'], how='left')

print (FD1_y_test_df_act_2_pred_1.ExctAcr.sum())
print (FD1_y_test_df_act_1_pred_2.ExctAcr.sum())

print (FD1_y_test_df_act_2_pred_1.ExctAcr.sum()-FD1_y_test_df_act_1_pred_2.ExctAcr.sum())

# %%
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models/"
filename = model_dir + 'regularEVI_forest_default.sav'
pickle.dump(regular_forest_1_default, open(filename, 'wb'))

# %%
# parameters = {'n_jobs':[4],
#               'criterion': ["gini", "entropy"], # log_loss
#               'max_depth':[1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
#               'min_samples_split':[2, 3, 4, 5],
#               'max_features': ["sqrt", "log2", None],
#               # 'min_impurity_decreasefloat':[0, 1, 2],
#               'class_weight':['balanced', 'balanced_subsample', None],
#               'ccp_alpha':[0.0, 1, 2, 3], 
#               'max_samples':[None, 1, 2, 3, 4, 5]} # , 
# forest_classifier_grid = GridSearchCV(RandomForestClassifier(random_state=0), 
#                                       parameters, cv=5, verbose=1,
#                                       error_score='raise')

# forest_classifier_grid.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

# %%
# RandomForestClassifier.get_params().keys()

# %%
# (n_estimators=100, 
# criterion='gini', 
# max_depth=None, 
# min_samples_split=2, 
# min_samples_leaf=1, 
# min_weight_fraction_leaf=0.0,
# max_features='sqrt', 
# max_leaf_nodes=None, 
# min_impurity_decrease=0.0, 
# bootstrap=True, 
# oob_score=False, 
# n_jobs=None, 
# random_state=1,
# warm_start=False, 
# class_weight=None, 
# ccp_alpha=0.0,
# max_samples=None)

# %%
# # %%time
# parameters = {'n_jobs':[4],
#               'criterion': ["gini", "entropy"], # log_loss 
#               'max_depth':[2, 4, 6, 8, 9, 10, 11, 12, 14, 16, 18, 20],
#               'min_samples_split':[2, 3, 4, 5],
#               'max_features': ["sqrt", "log2", None],
#               'class_weight':['balanced', 'balanced_subsample', None],
#               'ccp_alpha':[0.0, 1, 2, 3], 
#              # 'min_impurity_decreasefloat':[0, 1, 2], # roblem with sqrt stuff?
#               'max_samples':[None, 1, 2, 3, 4, 5]
#              } # , 
# forest_grid_1 = GridSearchCV(RandomForestClassifier(random_state=0), 
#                              parameters, cv=5, verbose=1,
#                              error_score='raise')

# forest_grid_1.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

# print (forest_grid_1.best_params_)
# print (forest_grid_1.best_score_)


# model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models/"
# filename = model_dir + 'forest_grid_1.sav'
# pickle.dump(forest_1_default, open(filename, 'wb')) <- why it is saving default as grid_1?

# %%

# %%
# %%time
parameters = {'n_jobs':[6],
              'criterion': ["gini", "entropy"], # log_loss 
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
regular_forest_grid_1_predictions = regular_forest_grid_1.predict(x_test_df.iloc[:, 1:])
regular_forest_grid_1_y_test_df=y_test_df.copy()
regular_forest_grid_1_y_test_df["prediction"]=list(regular_forest_grid_1_predictions)
regular_forest_grid_1_y_test_df.head(2)

# %%
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
FG1_y_test_df_act_1_pred_2=regular_forest_grid_1_y_test_df[regular_forest_grid_1_y_test_df.Vote==1].copy()
FG1_y_test_df_act_2_pred_1=regular_forest_grid_1_y_test_df[regular_forest_grid_1_y_test_df.Vote==2].copy()

FG1_y_test_df_act_1_pred_2=FG1_y_test_df_act_1_pred_2[FG1_y_test_df_act_1_pred_2.prediction==2].copy()
FG1_y_test_df_act_2_pred_1=FG1_y_test_df_act_2_pred_1[FG1_y_test_df_act_2_pred_1.prediction==1].copy()

FG1_y_test_df_act_2_pred_1 = pd.merge(FG1_y_test_df_act_2_pred_1, ground_truth_labels_extended, on=['ID'], how='left')
FG1_y_test_df_act_1_pred_2 = pd.merge(FG1_y_test_df_act_1_pred_2, ground_truth_labels_extended, on=['ID'], how='left')

print (FG1_y_test_df_act_2_pred_1.ExctAcr.sum())
print (FG1_y_test_df_act_1_pred_2.ExctAcr.sum())
print (FG1_y_test_df_act_2_pred_1.ExctAcr.sum()-FG1_y_test_df_act_1_pred_2.ExctAcr.sum())

# %%
filename = model_dir + 'regularEVI_forest_grid_1.sav'
pickle.dump(regular_forest_grid_1, open(filename, 'wb'))

# %% [markdown]
# ### Regular More parameters

# %%
# %%time
parameters = {'n_jobs':[5],
              'criterion': ["gini", "entropy"], # log_loss 
              'max_depth':[2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20],
              'min_samples_split':[2, 3, 4, 5],
              'max_features': ["sqrt", "log2", None],
              'class_weight':['balanced', 'balanced_subsample', None],
              'ccp_alpha':[0.0, 1, 2, 3], 
             # 'min_impurity_decreasefloat':[0, 1, 2], # roblem with sqrt stuff?
              'max_samples':[None, 1, 2, 3, 4, 5]
             }

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
FG2_y_test_df_act_1_pred_2=regular_forest_grid_2_y_test_df[regular_forest_grid_2_y_test_df.Vote==1].copy()
FG2_y_test_df_act_2_pred_1=regular_forest_grid_2_y_test_df[regular_forest_grid_2_y_test_df.Vote==2].copy()

FG2_y_test_df_act_1_pred_2=FG2_y_test_df_act_1_pred_2[FG2_y_test_df_act_1_pred_2.prediction==2].copy()
FG2_y_test_df_act_2_pred_1=FG2_y_test_df_act_2_pred_1[FG2_y_test_df_act_2_pred_1.prediction==1].copy()

FG2_y_test_df_act_2_pred_1 = pd.merge(FG2_y_test_df_act_2_pred_1, ground_truth_labels_extended, on=['ID'], how='left')
FG2_y_test_df_act_1_pred_2 = pd.merge(FG2_y_test_df_act_1_pred_2, ground_truth_labels_extended, on=['ID'], how='left')

print (FG2_y_test_df_act_2_pred_1.ExctAcr.sum())
print (FG2_y_test_df_act_1_pred_2.ExctAcr.sum())
print (FG2_y_test_df_act_2_pred_1.ExctAcr.sum()-FG2_y_test_df_act_1_pred_2.ExctAcr.sum())

# %%
filename = model_dir + 'regularEVI_forest_grid_2.sav'
pickle.dump(regular_forest_grid_2, open(filename, 'wb'))

# %% [markdown]
# # SG

# %%
VI_idx = "EVI"
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
ground_truth = data[data.ID.isin(list(ground_truth_labels.ID.unique()))].copy()

print (len(meta_moreThan10Acr.ID.unique()))
ground_truth_labels_extended = pd.merge(ground_truth_labels, meta, on=['ID'], how='left')
ground_truth_labels = ground_truth_labels_extended[ground_truth_labels_extended.ExctAcr>=10].copy()
ground_truth_labels.reset_index(drop=True, inplace=True)

print ("There are [{:.0f}] fields in total whose \
       area adds up to [{:.2f}].".format(len(ground_truth_labels_extended),\
                                                            ground_truth_labels_extended.ExctAcr.sum()))


print ("There are [{:.0f}] fields larger than 10 \
        acres whose area adds up to [{:.2f}].".format(len(ground_truth_labels), \
                                                                    ground_truth_labels.ExctAcr.sum()))


# %%
ground_truth = ground_truth[ground_truth.ID.isin((list(meta_moreThan10Acr.ID)))].copy()
ground_truth_labels = ground_truth_labels[ground_truth_labels.ID.isin((list(meta_moreThan10Acr.ID)))].copy()

ground_truth.reset_index(drop=True, inplace=True)
ground_truth_labels.reset_index(drop=True, inplace=True)

# %%
ground_truth.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
ground_truth_labels.sort_values(by=["ID"], inplace=True)

ground_truth.reset_index(drop=True, inplace=True)
ground_truth_labels.reset_index(drop=True, inplace=True)

assert (len(ground_truth.ID.unique()) == len(ground_truth_labels.ID.unique()))

print (list(ground_truth.ID)[0])
print (list(ground_truth_labels.ID)[0])
print ("____________________________________")
print (list(ground_truth.ID)[-1])
print (list(ground_truth_labels.ID)[-1])
print ("____________________________________")
print (list(ground_truth.ID.unique())==list(ground_truth_labels.ID.unique()))

# %% [markdown]
# # Widen

# %%
EVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + EVI_colnames
ground_truth_wide = pd.DataFrame(columns=columnNames, 
                                index=range(len(ground_truth.ID.unique())))
ground_truth_wide["ID"] = ground_truth.ID.unique()

for an_ID in ground_truth.ID.unique():
    curr_df = ground_truth[ground_truth.ID==an_ID]
    
    ground_truth_wide_indx = ground_truth_wide[ground_truth_wide.ID==an_ID].index
    ground_truth_wide.loc[ground_truth_wide_indx, "EVI_1":"EVI_36"] = curr_df.EVI.values[:36]

# %%
ground_truth_labels = ground_truth_labels.set_index('ID')
ground_truth_labels = ground_truth_labels.reindex(index=ground_truth_wide['ID'])
ground_truth_labels = ground_truth_labels.reset_index()

# %%
ground_truth_labels=ground_truth_labels[["ID", "Vote"]]
ground_truth_labels.head(2)

# %%
x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(ground_truth_wide, 
                                                                ground_truth_labels, 
                                                                test_size=0.2, 
                                                                random_state=0,
                                                                shuffle=True,
                                                                stratify=ground_truth_labels.Vote.values)

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
FD1_yTest_df_act_1_pred_2=forest_1_default_SG_y_test_df[forest_1_default_SG_y_test_df.Vote==1].copy()
FD1_yTest_df_act_2_pred_1=forest_1_default_SG_y_test_df[forest_1_default_SG_y_test_df.Vote==2].copy()

FD1_yTest_df_act_1_pred_2=FD1_yTest_df_act_1_pred_2[FD1_yTest_df_act_1_pred_2.prediction==2].copy()
FD1_yTest_df_act_2_pred_1=FD1_yTest_df_act_2_pred_1[FD1_yTest_df_act_2_pred_1.prediction==1].copy()

FD1_yTest_df_act_2_pred_1 = pd.merge(FD1_yTest_df_act_2_pred_1, ground_truth_labels_extended, on=['ID'], how='left')
FD1_yTest_df_act_1_pred_2 = pd.merge(FD1_yTest_df_act_1_pred_2, ground_truth_labels_extended, on=['ID'], how='left')

aa=FD1_yTest_df_act_2_pred_1.ExctAcr.sum()
bb=FD1_yTest_df_act_1_pred_2.ExctAcr.sum()
print ("FD1_yTest_df_act_2_pred_1.ExctAcr.sum(): ", aa)
print ("FD1_yTest_df_act_1_pred_2.ExctAcr.sum(): ", bb)
print (aa-bb)

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


model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models/"
filename = model_dir + 'SGEVI_forest_grid_1.sav'
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
FG1_yTest_df_act_1_pred_2=forest_grid_1_SG_y_test_df[forest_grid_1_SG_y_test_df.Vote==1].copy()
FG1_yTest_df_act_2_pred_1=forest_grid_1_SG_y_test_df[forest_grid_1_SG_y_test_df.Vote==2].copy()

FG1_yTest_df_act_1_pred_2=FG1_yTest_df_act_1_pred_2[FG1_yTest_df_act_1_pred_2.prediction==2].copy()
FG1_yTest_df_act_2_pred_1=FG1_yTest_df_act_2_pred_1[FG1_yTest_df_act_2_pred_1.prediction==1].copy()

FG1_yTest_df_act_2_pred_1 = pd.merge(FG1_yTest_df_act_2_pred_1, ground_truth_labels_extended, on=['ID'], how='left')
FG1_yTest_df_act_1_pred_2 = pd.merge(FG1_yTest_df_act_1_pred_2, ground_truth_labels_extended, on=['ID'], how='left')

print (FG1_yTest_df_act_2_pred_1.ExctAcr.sum())
print (FG1_yTest_df_act_1_pred_2.ExctAcr.sum())
print (np.abs(FG1_yTest_df_act_1_pred_2.ExctAcr.sum() - FG1_yTest_df_act_2_pred_1.ExctAcr.sum()))

# %%

# %%
