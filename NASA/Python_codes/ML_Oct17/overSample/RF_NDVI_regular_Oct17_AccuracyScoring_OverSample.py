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

from datetime import date, datetime
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

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
# # NDVI - SG - SR 0.3

# %%
VI_idx = "NDVI"
smooth_type = "regular"

# %%
ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
overSamples_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/overSamples/"

model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models_Oct17/overSample/"
os.makedirs(model_dir, exist_ok=True)

# %%
# %%time
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))
f_name = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample3.csv"
wide_overSample = pd.read_csv(overSamples_data_folder + f_name) # train80_GT_wide:
print ("train set size of sample ratio 3 is ", wide_overSample.shape)

x_train_df=wide_overSample.copy()
x_train_df.drop(columns=["Vote"], inplace=True)
y_train_df=wide_overSample[["ID", "Vote"]]

#### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order
print ((list(x_train_df.ID)==list(y_train_df.ID)))
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

RF_grid_2 = GridSearchCV(RandomForestClassifier(random_state=0), 
                         parameters, cv=5, verbose=1,
                         error_score='raise')
RF_grid_2.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

filename = model_dir + smooth_type + "_" + VI_idx + "_RF_grid_2_Oct17_AccScoring_Oversample_SR3.sav"
pickle.dump(RF_grid_2, open(filename, 'wb'))

print (RF_grid_2.best_params_)
print (RF_grid_2.best_score_)
##########
##########    Test
##########
x_test_df = pd.read_csv(ML_data_folder + "widen_test_TS/" + VI_idx + "_" + smooth_type + \
                        "_wide_test20_split_2Bconsistent_Oct17.csv")

y_test_df = x_test_df[["ID", "Vote"]].copy()
x_test_df.drop(columns=["Vote"], inplace=True)

RF_grid_2_predictions = RF_grid_2.predict(x_test_df.iloc[:, 1:])
RF_grid_2_y_test_df=y_test_df.copy()
RF_grid_2_y_test_df["prediction"]=list(RF_grid_2_predictions)

print ()
print (RF_grid_2_y_test_df.head(2))
print ()

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in RF_grid_2_y_test_df.index:
    curr_vote=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].Vote)[0]
    curr_predict=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].prediction)[0]
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
            
RF_grid_2_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
RF_grid_2_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
RF_grid_2_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
RF_grid_2_confus_tbl_test['Predict_Single']=0
RF_grid_2_confus_tbl_test['Predict_Double']=0

RF_grid_2_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
RF_grid_2_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
RF_grid_2_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
RF_grid_2_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))
RF_grid_2_confus_tbl_test

# %% [markdown]
# ## SG - NDVI - SR 0.4

# %%
del(x_train_df, wide_overSample, f_name, x_test_df)
del(RF_grid_2, RF_grid_2_confus_tbl_test, RF_grid_2_y_test_df)

# %%
# %%time
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))
f_name = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample4.csv"
wide_overSample = pd.read_csv(overSamples_data_folder + f_name) # train80_GT_wide:
print ("train set size of sample ratio 4 is", wide_overSample.shape)

x_train_df=wide_overSample.copy()
x_train_df.drop(columns=["Vote"], inplace=True)
y_train_df=wide_overSample[["ID", "Vote"]]

#### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order
print ((list(x_train_df.ID)==list(y_train_df.ID)))
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

RF_grid_2 = GridSearchCV(RandomForestClassifier(random_state=0), 
                         parameters, cv=5, verbose=1,
                         error_score='raise')
RF_grid_2.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

filename = model_dir + smooth_type + "_" + VI_idx + "_RF_grid_2_Oct17_AccScoring_Oversample_SR4.sav"
pickle.dump(RF_grid_2, open(filename, 'wb'))

print (RF_grid_2.best_params_)
print (RF_grid_2.best_score_)
##########
##########    Test
##########
x_test_df = pd.read_csv(ML_data_folder + "widen_test_TS/" + VI_idx + "_" + smooth_type + \
                        "_wide_test20_split_2Bconsistent_Oct17.csv")

y_test_df = x_test_df[["ID", "Vote"]].copy()
x_test_df.drop(columns=["Vote"], inplace=True)

RF_grid_2_predictions = RF_grid_2.predict(x_test_df.iloc[:, 1:])
RF_grid_2_y_test_df=y_test_df.copy()
RF_grid_2_y_test_df["prediction"]=list(RF_grid_2_predictions)

print ()
print (RF_grid_2_y_test_df.head(2))
print ()

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in RF_grid_2_y_test_df.index:
    curr_vote=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].Vote)[0]
    curr_predict=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].prediction)[0]
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
            
RF_grid_2_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
RF_grid_2_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
RF_grid_2_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
RF_grid_2_confus_tbl_test['Predict_Single']=0
RF_grid_2_confus_tbl_test['Predict_Double']=0

RF_grid_2_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
RF_grid_2_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
RF_grid_2_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
RF_grid_2_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))
RF_grid_2_confus_tbl_test

# %% [markdown]
# ## SG - NDVI - SR 0.5

# %%
del(x_train_df, wide_overSample, f_name, x_test_df)
del(RF_grid_2, RF_grid_2_confus_tbl_test, RF_grid_2_y_test_df)

# %%
# %%time
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))
f_name = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample5.csv"
wide_overSample = pd.read_csv(overSamples_data_folder + f_name) # train80_GT_wide:
print ("train set size of sample ratio 5 is", wide_overSample.shape)

x_train_df=wide_overSample.copy()
x_train_df.drop(columns=["Vote"], inplace=True)
y_train_df=wide_overSample[["ID", "Vote"]]

#### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order
print ((list(x_train_df.ID)==list(y_train_df.ID)))
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

RF_grid_2 = GridSearchCV(RandomForestClassifier(random_state=0), 
                         parameters, cv=5, verbose=1,
                         error_score='raise')
RF_grid_2.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

filename = model_dir + smooth_type + "_" + VI_idx + "_RF_grid_2_Oct17_AccScoring_Oversample_SR5.sav"
pickle.dump(RF_grid_2, open(filename, 'wb'))

print (RF_grid_2.best_params_)
print (RF_grid_2.best_score_)
##########
##########    Test
##########
x_test_df = pd.read_csv(ML_data_folder + "widen_test_TS/" + VI_idx + "_" + smooth_type + \
                        "_wide_test20_split_2Bconsistent_Oct17.csv")

y_test_df = x_test_df[["ID", "Vote"]].copy()
x_test_df.drop(columns=["Vote"], inplace=True)

RF_grid_2_predictions = RF_grid_2.predict(x_test_df.iloc[:, 1:])
RF_grid_2_y_test_df=y_test_df.copy()
RF_grid_2_y_test_df["prediction"]=list(RF_grid_2_predictions)

print ()
print (RF_grid_2_y_test_df.head(2))
print ()

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in RF_grid_2_y_test_df.index:
    curr_vote=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].Vote)[0]
    curr_predict=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].prediction)[0]
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
            
RF_grid_2_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
RF_grid_2_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
RF_grid_2_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
RF_grid_2_confus_tbl_test['Predict_Single']=0
RF_grid_2_confus_tbl_test['Predict_Double']=0

RF_grid_2_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
RF_grid_2_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
RF_grid_2_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
RF_grid_2_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))
RF_grid_2_confus_tbl_test

# %% [markdown]
# ## SG - NDVI - SR 0.6

# %%
del(x_train_df, wide_overSample, f_name, x_test_df)
del(RF_grid_2, RF_grid_2_confus_tbl_test, RF_grid_2_y_test_df)

# %%
# %%time
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))
f_name = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample6.csv"
wide_overSample = pd.read_csv(overSamples_data_folder + f_name) # train80_GT_wide:
print ("train set size of sample ratio 6 is", wide_overSample.shape)

x_train_df=wide_overSample.copy()
x_train_df.drop(columns=["Vote"], inplace=True)
y_train_df=wide_overSample[["ID", "Vote"]]

#### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order
print ((list(x_train_df.ID)==list(y_train_df.ID)))
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

RF_grid_2 = GridSearchCV(RandomForestClassifier(random_state=0), 
                         parameters, cv=5, verbose=1,
                         error_score='raise')
RF_grid_2.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

filename = model_dir + smooth_type + "_" + VI_idx + "_RF_grid_2_Oct17_AccScoring_Oversample_SR6.sav"
pickle.dump(RF_grid_2, open(filename, 'wb'))

print (RF_grid_2.best_params_)
print (RF_grid_2.best_score_)
##########
##########    Test
##########
x_test_df = pd.read_csv(ML_data_folder + "widen_test_TS/" + VI_idx + "_" + smooth_type + \
                        "_wide_test20_split_2Bconsistent_Oct17.csv")

y_test_df = x_test_df[["ID", "Vote"]].copy()
x_test_df.drop(columns=["Vote"], inplace=True)

RF_grid_2_predictions = RF_grid_2.predict(x_test_df.iloc[:, 1:])
RF_grid_2_y_test_df=y_test_df.copy()
RF_grid_2_y_test_df["prediction"]=list(RF_grid_2_predictions)

print ()
print (RF_grid_2_y_test_df.head(2))
print ()

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in RF_grid_2_y_test_df.index:
    curr_vote=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].Vote)[0]
    curr_predict=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].prediction)[0]
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
            
RF_grid_2_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
RF_grid_2_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
RF_grid_2_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
RF_grid_2_confus_tbl_test['Predict_Single']=0
RF_grid_2_confus_tbl_test['Predict_Double']=0

RF_grid_2_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
RF_grid_2_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
RF_grid_2_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
RF_grid_2_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))
RF_grid_2_confus_tbl_test

# %% [markdown]
# ## SG - NDVI - SR 0.7

# %%
del(x_train_df, wide_overSample, f_name, x_test_df)
del(RF_grid_2, RF_grid_2_confus_tbl_test, RF_grid_2_y_test_df)

# %%
# %%time
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))
f_name = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample7.csv"
wide_overSample = pd.read_csv(overSamples_data_folder + f_name) # train80_GT_wide:
print ("train set size of sample ratio 7 is", wide_overSample.shape)

x_train_df=wide_overSample.copy()
x_train_df.drop(columns=["Vote"], inplace=True)
y_train_df=wide_overSample[["ID", "Vote"]]

#### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order
print ((list(x_train_df.ID)==list(y_train_df.ID)))
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

RF_grid_2 = GridSearchCV(RandomForestClassifier(random_state=0), 
                         parameters, cv=5, verbose=1, 
                         error_score='raise')
RF_grid_2.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

filename = model_dir + smooth_type + "_" + VI_idx + "_RF_grid_2_Oct17_AccScoring_Oversample_SR7.sav"
pickle.dump(RF_grid_2, open(filename, 'wb'))

print (RF_grid_2.best_params_)
print (RF_grid_2.best_score_)
##########
##########    Test
##########
x_test_df = pd.read_csv(ML_data_folder + "widen_test_TS/" + VI_idx + "_" + smooth_type + \
                        "_wide_test20_split_2Bconsistent_Oct17.csv")

y_test_df = x_test_df[["ID", "Vote"]].copy()
x_test_df.drop(columns=["Vote"], inplace=True)

RF_grid_2_predictions = RF_grid_2.predict(x_test_df.iloc[:, 1:])
RF_grid_2_y_test_df=y_test_df.copy()
RF_grid_2_y_test_df["prediction"]=list(RF_grid_2_predictions)

print ()
print (RF_grid_2_y_test_df.head(2))
print ()

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in RF_grid_2_y_test_df.index:
    curr_vote=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].Vote)[0]
    curr_predict=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].prediction)[0]
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
            
RF_grid_2_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
RF_grid_2_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
RF_grid_2_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
RF_grid_2_confus_tbl_test['Predict_Single']=0
RF_grid_2_confus_tbl_test['Predict_Double']=0

RF_grid_2_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
RF_grid_2_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
RF_grid_2_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
RF_grid_2_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
RF_grid_2_confus_tbl_test

# %% [markdown]
# ## SG - NDVI - SR 0.8

# %%
del(x_train_df, wide_overSample, f_name, x_test_df)
del(RF_grid_2, RF_grid_2_confus_tbl_test, RF_grid_2_y_test_df)

# %%
# %%time
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))
f_name = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample8.csv"
wide_overSample = pd.read_csv(overSamples_data_folder + f_name) # train80_GT_wide:
print ("train set size of sample ratio 8 is", wide_overSample.shape)

x_train_df=wide_overSample.copy()
x_train_df.drop(columns=["Vote"], inplace=True)
y_train_df=wide_overSample[["ID", "Vote"]]

#### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order
print ((list(x_train_df.ID)==list(y_train_df.ID)))
parameters = {'n_jobs':[6],
              'criterion': ["gini", "entropy"], # log_loss 
              'max_depth':[2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20],
              'min_samples_split':[2, 3, 4, 5],
              'max_features': ["sqrt", "log2", None],
              'class_weight':['balanced', 'balanced_subsample', None],
              'ccp_alpha':[0.0, 1, 2, 3], 
             # 'min_impurity_decreasefloat':[0, 1, 2], # problem with sqrt stuff?
              'max_samples':[None, 1, 2, 3, 4, 5]
             }

RF_grid_2 = GridSearchCV(RandomForestClassifier(random_state=0), 
                         parameters, cv=5, verbose=1,
                         error_score='raise')
RF_grid_2.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

filename = model_dir + smooth_type + "_" + VI_idx + "_RF_grid_2_Oct17_AccScoring_Oversample_SR8.sav"
pickle.dump(RF_grid_2, open(filename, 'wb'))

print (RF_grid_2.best_params_)
print (RF_grid_2.best_score_)
##########
##########    Test
##########
x_test_df = pd.read_csv(ML_data_folder + "widen_test_TS/" + VI_idx + "_" + smooth_type + \
                        "_wide_test20_split_2Bconsistent_Oct17.csv")

y_test_df = x_test_df[["ID", "Vote"]].copy()
x_test_df.drop(columns=["Vote"], inplace=True)

RF_grid_2_predictions = RF_grid_2.predict(x_test_df.iloc[:, 1:])
RF_grid_2_y_test_df=y_test_df.copy()
RF_grid_2_y_test_df["prediction"]=list(RF_grid_2_predictions)

print ()
print (RF_grid_2_y_test_df.head(2))
print ()

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in RF_grid_2_y_test_df.index:
    curr_vote=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].Vote)[0]
    curr_predict=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].prediction)[0]
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
            
RF_grid_2_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
RF_grid_2_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
RF_grid_2_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
RF_grid_2_confus_tbl_test['Predict_Single']=0
RF_grid_2_confus_tbl_test['Predict_Double']=0

RF_grid_2_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
RF_grid_2_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
RF_grid_2_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
RF_grid_2_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
RF_grid_2_confus_tbl_test

# %%
from datetime import date, datetime
print (date.today(), "-", datetime.now().strftime("%H:%M:%S"))

# %%
