####
#### July 14, 2022
####
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

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


import h5py
import sys

start_time = time.time()

# search path for modules# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

####################################################################################
###
###                      Aeolus Core path
###
####################################################################################

sys.path.append('/home/hnoorazar/NASA/')
import NASA_core as nc
# import NASA_plot_core as ncp

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
meta_dir = "/data/hydro/users/Hossein/NASA/0000_parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]

# Read Training Set Labels
# training_set_dir = "/data/hydro/users/Hossein/NASA/04_regularized_TS/"
ground_truth_labels = pd.read_csv(meta_dir +"train_labels.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))

# Read the data
VI_idx = "EVI"
data_dir = "/data/hydro/users/Hossein/NASA/04_regularized_TS/"
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

ground_truth = data[data.ID.isin(list(ground_truth_labels.ID.unique()))].copy()
len(ground_truth.ID.unique())


# Toss Small

ground_truth_labels_extended = pd.merge(ground_truth_labels, meta, on=['ID'], how='left')
ground_truth_labels = ground_truth_labels_extended[ground_truth_labels_extended.ExctAcr>=10].copy()
ground_truth_labels.reset_index(drop=True, inplace=True)


ground_truth = ground_truth[ground_truth.ID.isin((list(meta_moreThan10Acr.ID)))].copy()
ground_truth_labels = ground_truth_labels[ground_truth_labels.ID.isin((list(meta_moreThan10Acr.ID)))].copy()

ground_truth.reset_index(drop=True, inplace=True)
ground_truth_labels.reset_index(drop=True, inplace=True)


# sort
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

# Widen
EVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + EVI_colnames
ground_truth_wide = pd.DataFrame(columns=columnNames, 
                                index=range(len(ground_truth.ID.unique())))
ground_truth_wide["ID"] = ground_truth.ID.unique()

for an_ID in ground_truth.ID.unique():
    curr_df = ground_truth[ground_truth.ID==an_ID]
    
    ground_truth_wide_indx = ground_truth_wide[ground_truth_wide.ID==an_ID].index
    ground_truth_wide.loc[ground_truth_wide_indx, "EVI_1":"EVI_36"] = curr_df.EVI.values[:36]


ground_truth_labels = ground_truth_labels.set_index('ID')
ground_truth_labels = ground_truth_labels.reindex(index=ground_truth_wide['ID'])
ground_truth_labels = ground_truth_labels.reset_index()

print (ground_truth_labels.ExctAcr.min())
ground_truth_labels.head(2)

ground_truth_labels=ground_truth_labels[["ID", "Vote"]]


x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(ground_truth_wide, 
                                                                ground_truth_labels, 
                                                                test_size=0.2, 
                                                                random_state=0,
                                                                shuffle=True,
                                                                stratify=ground_truth_labels.Vote.values)

print ("x_test_df.shape: ", x_test_df.shape)



parameters = {'n_jobs':[4],
              'criterion': ["gini", "entropy"], # log_loss 
              'max_depth':[2, 4, 6, 8, 9, 10, 11, 12, 14, 16, 18, 20],
              'min_samples_split':[2, 3, 4, 5],
              'max_features': ["sqrt", "log2", None],
              'class_weight':['balanced', 'balanced_subsample', None],
              'ccp_alpha':[0.0, 1, 2, 3], 
             # 'min_impurity_decreasefloat':[0, 1, 2], # roblem with sqrt stuff?
              'max_samples':[None, 1, 2, 3, 4, 5]
             } # , 
forest_grid_1 = GridSearchCV(RandomForestClassifier(random_state=0), 
                             parameters, cv=5, verbose=1,
                             error_score='raise')

forest_grid_1.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

import pickle
model_dir = "/data/hydro/users/Hossein/NASA/zz_ML_models/"
filename = model_dir + 'forest_grid_1_noLogLoss_regular.sav'
pickle.dump(forest_1_default, open(filename, 'wb'))

print (forest_grid_1.best_params_)
print (forest_grid_1.best_score_)

forest_grid_1_predictions = forest_grid_1.predict(x_test_df.iloc[:, 1:])
forest_2_grid_y_test_df=y_test_df.copy()
forest_2_grid_y_test_df["prediction"]=list(forest_grid_1_predictions)
forest_2_grid_y_test_df.head(2)

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in forest_1_default_y_test_df.index:
    curr_vote=list(forest_2_grid_y_test_df[forest_2_grid_y_test_df.index==index_].Vote)[0]
    curr_predict=list(forest_2_grid_y_test_df[forest_2_grid_y_test_df.index==index_].prediction)[0]
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

output_dir = "/data/hydro/users/Hossein/NASA/zz_ML_data/"
out_name = output_dir + "confusion_forest_grid_1_noLogLoss_" + indeks + "_regular.csv"
balanced_confus_tbl_test.to_csv(out_name, index = False)

out_name = output_dir + "testPred_forest_grid_1_noLogLoss_" + indeks + "_regular.csv"
forest_2_grid_y_test_df.to_csv(out_name, index = False)

print (time.time() - start_time )