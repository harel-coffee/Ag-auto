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

# %% [markdown]
# This notebook is created on Oct 24. But the name includes Oct. 26
# to follow the same pattern as I cannot do everytihng in one day.
# It is a copy of the old notebook ```00_SVM_SG_NDVI.ipynb```.
#

# %%
import numpy as np
import pandas as pd
import scipy, scipy.signal

from datetime import date
import time

import random
from random import seed, random

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

import pickle #, h5py
import sys, os, os.path, shutil


# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core as rcp

# %%
from tslearn.metrics import dtw as dtw_metric

# https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

# %% [markdown]
# # Read Fields Metadata

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
len(meta.ID.unique())

# %% [markdown]
# # Read Training Set Labels

# %%
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
ground_truth_labels = pd.read_csv(training_set_dir+"groundTruth_labels_Oct17_2022.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))
ground_truth_labels.head(2)

# %% [markdown]
# ### Detect how many fields are less than 10 acres and report in the paper

# %%
print (len(meta[meta.ID.isin(list(ground_truth_labels.ID))].ID.unique()))
meta.head(2)

# %% [markdown]
# # Read the Data

# %%
VI_idx = "NDVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"

# %%
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
ground_truth_TS = data[data.ID.isin(list(ground_truth_labels.ID.unique()))].copy()
len(ground_truth_TS.ID.unique())

# %% [markdown]
# # Toss small fields

# %%
len(meta_moreThan10Acr.ID.unique())

# %%
ground_truth_labels_extended = pd.merge(ground_truth_labels, meta, on=['ID'], how='left')
ground_truth_labels = ground_truth_labels_extended[ground_truth_labels_extended.ExctAcr>=10].copy()
ground_truth_labels.reset_index(drop=True, inplace=True)

A = len(ground_truth_labels_extended)
B = ground_truth_labels_extended.ExctAcr.sum()
print ("There are [{:.0f}] fields in total whose area adds up to [{:.2f}].".format(A , B))

A = len(ground_truth_labels)
B = ground_truth_labels.ExctAcr.sum()
print ("There are [{:.0f}] fields larger than 10 acres whose area adds up to [{:.2f}].".format(A, B))

# ground_truth_labels = ground_truth_labels[["ID", "Vote"]]

# %%
ground_truth_TS = ground_truth_TS[ground_truth_TS.ID.isin((list(meta_moreThan10Acr.ID)))].copy()
ground_truth_labels = ground_truth_labels[ground_truth_labels.ID.isin((list(meta_moreThan10Acr.ID)))].copy()

ground_truth_TS.reset_index(drop=True, inplace=True)
ground_truth_labels.reset_index(drop=True, inplace=True)

# %% [markdown]
# # Sort the order of time-series and experts' labels identically

# %%
ground_truth_TS.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
ground_truth_labels.sort_values(by=["ID"], inplace=True)

ground_truth_TS.reset_index(drop=True, inplace=True)
ground_truth_labels.reset_index(drop=True, inplace=True)

assert (len(ground_truth_TS.ID.unique()) == len(ground_truth_labels.ID.unique()))

print (list(ground_truth_TS.ID)[0])
print (list(ground_truth_labels.ID)[0])
print ("____________________________________")
print (list(ground_truth_TS.ID)[-1])
print (list(ground_truth_labels.ID)[-1])
print ("____________________________________")
print (list(ground_truth_TS.ID.unique())==list(ground_truth_labels.ID.unique()))

# %% [markdown]
# # Widen Ground Truth Table

# %%
NDVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + NDVI_colnames
ground_truth_wide = pd.DataFrame(columns=columnNames, 
                                index=range(len(ground_truth_TS.ID.unique())))
ground_truth_wide["ID"] = ground_truth_TS.ID.unique()

for an_ID in ground_truth_TS.ID.unique():
    curr_df = ground_truth_TS[ground_truth_TS.ID==an_ID]
    
    ground_truth_wide_indx = ground_truth_wide[ground_truth_wide.ID==an_ID].index
    ground_truth_wide.loc[ground_truth_wide_indx, "NDVI_1":"NDVI_36"] = curr_df.NDVI.values[:36]
    
print (len(ground_truth_wide.ID.unique()))
ground_truth_wide.head(2)

# %% [markdown]
# # Split Train and Test Set
#
# #### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order

# %%
# This is cell and some edits after this are new compared to 00_SVM_regular_EVI.ipynb.
# I want to avoid splitting and just use the one I created earlier.

ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
train80 = pd.read_csv(ML_data_folder+"train80_split_2Bconsistent_Oct17.csv")
test20 = pd.read_csv(ML_data_folder+"test20_split_2Bconsistent_Oct17.csv")

# %%
ground_truth_labels.head(2)

# %%
ground_truth_wide.head(2)

# %%
x_train_df = ground_truth_wide[ground_truth_wide.ID.isin(list(train80.ID))]
x_test_df = ground_truth_wide[ground_truth_wide.ID.isin(list(test20.ID))]

y_train_df = ground_truth_labels[ground_truth_labels.ID.isin(list(train80.ID))]
y_test_df = ground_truth_labels[ground_truth_labels.ID.isin(list(test20.ID))]

y_train_df=y_train_df[["ID", "Vote"]]
y_test_df=y_test_df[["ID", "Vote"]]

# %%
print (ground_truth_labels.ExctAcr.min())
ground_truth_labels.head(2)

# %%
# ground_truth_labels = ground_truth_labels.set_index('ID')
# ground_truth_labels = ground_truth_labels.reindex(index=ground_truth_wide['ID'])
# ground_truth_labels = ground_truth_labels.reset_index()
# x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(ground_truth_wide, 
#                                                             ground_truth_labels, 
#                                                             test_size=0.2, 
#                                                             random_state=0,
#                                                             shuffle=True,
#                                                             stratify=ground_truth_labels.Vote.values)


# %%
x_test_df.tail(3)

# %%
y_test_df.tail(3)

# %% [markdown]
# # Start SVM

# %%
# %%time
##
##   Balanced
##
parameters = {'C':[5, 10, 13, 14, 15, 16, 17, 20, 40, 80],
              'kernel':['linear', 'poly', 'rbf', 'sigmoid'] # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
              } # , 
SVM_classifier_balanced_00 = GridSearchCV(SVC(random_state=0, class_weight='balanced'), 
                                          parameters, cv=5, verbose=1, scoring="roc_auc")

SVM_classifier_balanced_00.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values)

print (SVM_classifier_balanced_00.best_params_)
print (SVM_classifier_balanced_00.best_score_)

# %%
# %%time
##
##   None Weight
##
parameters = {'C':[5, 10, 13, 14, 15, 16, 17, 20, 40, 80],
              'kernel':['linear', 'poly', 'rbf', 'sigmoid'] # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
              } # , 
SVM_classifier_NoneWeight_00 = GridSearchCV(SVC(random_state=0), 
                                            parameters, cv=5, verbose=1, scoring="roc_auc")

SVM_classifier_NoneWeight_00.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values)

print (SVM_classifier_NoneWeight_00.best_params_)
print (SVM_classifier_NoneWeight_00.best_score_)

# %% [markdown]
# ## Test

# %%
SVM_classifier_NoneWeight_00_predictions = SVM_classifier_NoneWeight_00.predict(x_test_df.iloc[:, 1:])
SVM_classifier_balanced_00_predictions = SVM_classifier_balanced_00.predict(x_test_df.iloc[:, 1:])

# %%
balanced_y_test_df=y_test_df.copy()
None_y_test_df=y_test_df.copy()
balanced_y_test_df["prediction"] = list(SVM_classifier_balanced_00_predictions)
balanced_y_test_df.head(2)

# %%
None_y_test_df=y_test_df.copy()
None_y_test_df=y_test_df.copy()
None_y_test_df["prediction"] = list(SVM_classifier_NoneWeight_00_predictions)
None_y_test_df.head(2)

# %%
# Balanced Confusion Table

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index in balanced_y_test_df.index:
    curr_vote=list(balanced_y_test_df[balanced_y_test_df.index==index].Vote)[0]
    curr_predict=list(balanced_y_test_df[balanced_y_test_df.index==index].prediction)[0]
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
            
balanced_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
balanced_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
balanced_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
balanced_confus_tbl_test['Predict_Single']=0
balanced_confus_tbl_test['Predict_Double']=0

balanced_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
balanced_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
balanced_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
balanced_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
balanced_confus_tbl_test

# %%
confusion_matrix(balanced_y_test_df.Vote, balanced_y_test_df.prediction)

# %%
# None Weight Confusion Table

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index in None_y_test_df.index:
    curr_vote=list(None_y_test_df[None_y_test_df.index==index].Vote)[0]
    curr_predict=list(None_y_test_df[None_y_test_df.index==index].prediction)[0]
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
            
None_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                                    index=range(2))
None_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
None_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
None_confus_tbl_test['Predict_Single']=0
None_confus_tbl_test['Predict_Double']=0

None_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
None_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
None_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
None_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
None_confus_tbl_test

# %%
print(classification_report(y_test_df.Vote, SVM_classifier_NoneWeight_00_predictions))

# %%
print(classification_report(y_test_df.Vote, SVM_classifier_balanced_00_predictions))

# %%
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models_Oct17/"
filename = model_dir + "SVM_classifier_balanced_SG_" + VI_idx + "_01_Oct17_AUC.sav"
pickle.dump(SVM_classifier_balanced_00, open(filename, 'wb'))
print (filename)

filename = model_dir + "SVM_classifier_NoneWeight_SG_" + VI_idx + "_01_Oct17_AUC.sav"
pickle.dump(SVM_classifier_NoneWeight_00, open(filename, 'wb'))

print (filename)

# %% [markdown]
# #### None weight  Wins 
# in terms of accuracy!

# %%
None_y_test_df_act_1_pred_2=None_y_test_df[None_y_test_df.Vote==1].copy()
None_y_test_df_act_2_pred_1=None_y_test_df[None_y_test_df.Vote==2].copy()

None_y_test_df_act_1_pred_2=None_y_test_df_act_1_pred_2[None_y_test_df_act_1_pred_2.prediction==2].copy()
None_y_test_df_act_2_pred_1=None_y_test_df_act_2_pred_1[None_y_test_df_act_2_pred_1.prediction==1].copy()

None_y_test_df_act_2_pred_1 = pd.merge(None_y_test_df_act_2_pred_1, 
                                       ground_truth_labels_extended, 
                                       on=['ID'], how='left')

None_y_test_df_act_1_pred_2 = pd.merge(None_y_test_df_act_1_pred_2, 
                                       ground_truth_labels_extended, 
                                       on=['ID'], how='left')
print (None_y_test_df_act_2_pred_1.ExctAcr.sum()-None_y_test_df_act_1_pred_2.ExctAcr.sum())

# %% [markdown]
# # balanced weight acreage

# %%
balanced_y_test_df_act_1_pred_2=balanced_y_test_df[balanced_y_test_df.Vote==1].copy()
balanced_y_test_df_act_2_pred_1=balanced_y_test_df[balanced_y_test_df.Vote==2].copy()
                                

balanced_y_test_df_act_1_pred_2=balanced_y_test_df_act_1_pred_2[balanced_y_test_df_act_1_pred_2.prediction==2].copy()
balanced_y_test_df_act_2_pred_1=balanced_y_test_df_act_2_pred_1[balanced_y_test_df_act_2_pred_1.prediction==1].copy()

balanced_y_test_df_act_2_pred_1 = pd.merge(balanced_y_test_df_act_2_pred_1, \
                                           ground_truth_labels_extended, on=['ID'], how='left')
balanced_y_test_df_act_1_pred_2 = pd.merge(balanced_y_test_df_act_1_pred_2, \
                                           ground_truth_labels_extended, on=['ID'], how='left')

print (np.abs(balanced_y_test_df_act_2_pred_1.ExctAcr.sum()-balanced_y_test_df_act_1_pred_2.ExctAcr.sum()).round(1))

# %% [markdown]
# # Different way of weighing classes?

# %%
# svm with class weight on an imbalanced classification dataset
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)

# %%
# define model
model = SVC(gamma='scale', class_weight='balanced')
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))

# %%
# %%time
# define model 
model = SVC(gamma='scale')

# define grid 
balance = [{0:100,1:1}, {0:10,1:1}, {0:1,1:1}, {0:1,1:10}, {0:1,1:100}]
param_grid = dict(class_weight=balance)

# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')

grid.fit(X, y)

# %%
# report the best configuration
print("Best: %f using %s" % (grid.best_score_, grid.best_params_))

# report all configurations
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# %%
set(y_train_df.Vote.values)

# %% [markdown]
# ## Apply similar thing on our data

# %%
# %%time
# [{2:100,1:1}, {2:10,1:1}, {2:1,1:1}, {2:1,1:10}, {2:1,1:100}]
# define grid
param_grid = {'C':[5, 10, 13, 14, 15, 16, 17, 20, 40, 80],
              'kernel':['linear', 'poly', 'rbf', 'sigmoid'], # 'precomputed'
              'class_weight': [{2:100,1:1}, {2:50,1:1}, {2:25, 1:1}, {2:10,1:1}, 
                               {2:1,1:1}, 
                               {2:1,1:100}, {2:1,1:50}, {2:1,1:25}, {2:1, 1:10},
                               'balanced', None]
             }

# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# define grid search
SVM_classifier_balanced_00 = GridSearchCV(estimator=SVC(gamma='scale', random_state=0), 
                             param_grid=param_grid, 
                             n_jobs=-1, cv=cv, scoring='roc_auc',
                             verbose=1)

SVM_classifier_balanced_00.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values)

# report the best configuration
print("Best: %f using %s" % (SVM_classifier_balanced_00.best_score_, 
                             SVM_classifier_balanced_00.best_params_))

# report all configurations
means = SVM_classifier_balanced_00.cv_results_['mean_test_score']
stds = SVM_classifier_balanced_00.cv_results_['std_test_score']
params = SVM_classifier_balanced_00.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# %%
print (SVM_classifier_balanced_00.best_params_)
print (SVM_classifier_balanced_00.best_score_)

# %% [markdown]
# ### Test

# %%
SVM_classifier_NEW_predictions = SVM_classifier_balanced_00.predict(x_test_df.iloc[:, 1:])

NEW_y_test_df=y_test_df.copy()
NEW_y_test_df["prediction"] = list(SVM_classifier_NEW_predictions)

# %%
confusion_matrix(NEW_y_test_df.Vote, NEW_y_test_df.prediction)

# %%
print(classification_report(y_test_df.Vote, SVM_classifier_NEW_predictions))

# %%
# %%time
# [{2:100,1:1}, {2:10,1:1}, {2:1,1:1}, {2:1,1:10}, {2:1,1:100}]
# define grid
param_grid = {'C':[5, 10, 13, 14, 15, 16, 17, 20, 40, 80],
              'kernel':['linear', 'poly', 'rbf', 'sigmoid'], # 'precomputed'
              'class_weight': [None]
              }

# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# define grid search
SVM_classifier_noScore = GridSearchCV(estimator=SVC(gamma='scale', random_state=0), 
                                      param_grid=param_grid, 
                                      n_jobs=-1, cv=cv,
                                      verbose=1, scoring="roc_auc")

SVM_classifier_noScore.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values)

# report the best configuration
print("Best: %f using %s" % (SVM_classifier_noScore.best_score_, 
                             SVM_classifier_noScore.best_params_))

# report all configurations
means = SVM_classifier_noScore.cv_results_['mean_test_score']
stds = SVM_classifier_noScore.cv_results_['std_test_score']
params = SVM_classifier_noScore.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# %%

# %%

# %%

# %%
