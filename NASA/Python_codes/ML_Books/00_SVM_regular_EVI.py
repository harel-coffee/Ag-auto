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
import scipy, scipy.signal

from datetime import date
import time
import random
from random import seed
from random import random

import os, os.path
import shutil

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

import h5py
import sys

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core as rcp

# %%
# !pip install numpy==1.23.5

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
print (len(meta.ID.unique()))
meta_lessThan10Acr=meta[meta.ExctAcr<10]
print (meta_lessThan10Acr.shape)

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
# # Read the Data

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

# %%

# %% [markdown]
# # Toss small fields

# %%
len(meta_moreThan10Acr.ID.unique())

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

# %%

# %% [markdown]
# # Sort the order of time-series and experts' labels identically

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

# %%
# mins = ground_truth.groupby("ID")[VI_idx].min()
# maxs = ground_truth.groupby("ID")[VI_idx].max()
# _ranges = maxs-mins
# _ranges = pd.DataFrame(_ranges)
# _ranges.reset_index(inplace=True)

# mins = pd.DataFrame(mins)
# mins.reset_index(inplace=True)


# _ranges.rename(columns = {'EVI':'EVI_range'}, inplace = True)
# mins.rename(columns = {'EVI':'EVI_min'}, inplace = True)

# _ranges.head(2)

# ground_truth = pd.merge(ground_truth, _ranges, on=['ID'], how='left')
# ground_truth = pd.merge(ground_truth, mins, on=['ID'], how='left')
# ground_truth["EVI_ratio"] = (ground_truth["EVI"]-ground_truth["EVI_min"])/ground_truth["EVI_range"]
# ground_truth.head(2)

# %%
# ID_1=ground_truth.ID.unique()[0]
# ID_2=ground_truth.ID.unique()[12]

# # _minimum = ground_truth[ground_truth.ID==ID_1].EVI.values.min()
# # _range = ground_truth[ground_truth.ID==ID_1].EVI.values.max()-_minimum
# # y_1 = (ground_truth[ground_truth.ID==ID_1].EVI.values-_minimum)/_range
# # _minimum = ground_truth[ground_truth.ID==ID_2].EVI.values.min()
# # _range = ground_truth[ground_truth.ID==ID_2].EVI.values.max()-_minimum
# # y_2 = (ground_truth[ground_truth.ID==ID_2].EVI.values-_minimum)/_range

# dtw_score = dtw_metric(ground_truth[ground_truth.ID==ID_1].EVI.values, 
#                 ground_truth[ground_truth.ID==ID_2].EVI.values)

# dtw_score_ratios = dtw_metric(ground_truth[ground_truth.ID==ID_1].EVI_ratio.values, 
#                        ground_truth[ground_truth.ID==ID_2].EVI_ratio.values)

# # print ("dtw score is {:.2f}.".format(dtw_score))
# plt.figure(figsize=(20,4))
# plt.subplot(1, 1, 1)

# # plot EVIs
# plt.plot(range(len(ground_truth[ground_truth.ID==ID_1].human_system_start_time.values)),
#          ground_truth[ground_truth.ID==ID_1].EVI.values,
#         c='k', linewidth=5);

# plt.plot(range(len(ground_truth[ground_truth.ID==ID_2].human_system_start_time.values)),
#          ground_truth[ground_truth.ID==ID_2].EVI.values,
#         c='red', linewidth=5);

# # plot ratios
# plt.plot(range(len(ground_truth[ground_truth.ID==ID_1].human_system_start_time.values)), 
#          ground_truth[ground_truth.ID==ID_1].EVI_ratio.values,
#          ".-.", c='k', linewidth=2, label="ratios");

# plt.plot(range(len(ground_truth[ground_truth.ID==ID_2].human_system_start_time.values)), 
#          ground_truth[ground_truth.ID==ID_2].EVI_ratio.values,
#          ".-.", c='red', linewidth=2, label="ratios");

# title = "dtw score for EVI is [{:.2f}] and for ratios is [{:.2f}].".format(dtw_score, dtw_score_ratios)
# plt.ylim([-0.2, 1.05]);
# plt.title(title , fontsize = 20);

# %%
# s1=ground_truth[ground_truth.ID==ID_1].EVI_ratio.values
# s2=ground_truth[ground_truth.ID==ID_2].EVI_ratio.values
# path = dtw.warping_path(s1, s2)
# dtwvis.plot_warping(s1, s2, path)
# distance = dtw.distance(s1, s2)

# %%

# %%
# d, paths = dtw.warping_paths(s1, s2, window=10, use_pruning=True);
# best_path = dtw.best_path(paths);
# dtwvis.plot_warpingpaths(s1, s2, paths, best_path);

# %% [markdown]
# # Widen Ground Truth Table

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

# %%

# %% [markdown]
# # Split Train and Test Set
#
# #### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order

# %%
ground_truth_labels.head(2)

# %%
ground_truth_labels.CropTyp.unique()

# %%
ground_truth.head(2)

# %%
ground_truth_labels

# %%
ground_truth_labels = ground_truth_labels.set_index('ID')
ground_truth_labels = ground_truth_labels.reindex(index=ground_truth_wide['ID'])
ground_truth_labels = ground_truth_labels.reset_index()

# %%
print (ground_truth_labels.ExctAcr.min())
ground_truth_labels.head(2)

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
x_test_df.shape

# %%
out_name=training_set_dir+"train80_split_expertLabels_2Bconsistent.csv"
y_train_df.to_csv(out_name, index = False)

out_name=training_set_dir+"test20_split_expertLabels_2Bconsistent.csv"
y_test_df.to_csv(out_name, index = False)

# %%

# %% [markdown]
# # Start SVM

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
#
#

# %%
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


# %%
def DTW_prune(ts1, ts2):
    d,_ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True);
    return d


# %%
parameters = {'C':[5, 10, 13, 14, 15, 16, 17, 20, 40, 80],
              'kernel':['linear', 'poly', 'rbf', 'sigmoid'] # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
              } # , 
SVM_classifier_balanced_00 = GridSearchCV(SVC(random_state=0, class_weight='balanced'), 
                                          parameters, cv=5, verbose=1)

SVM_classifier_balanced_00.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values)

print (SVM_classifier_balanced_00.best_params_)
print (SVM_classifier_balanced_00.best_score_)

# %%

# %%
parameters = {'C':[5, 10, 13, 14, 15, 16, 17, 20, 40, 80],
              'kernel':['linear', 'poly', 'rbf', 'sigmoid'] # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
              } # , 
SVM_classifier_NoneWeight_00 = GridSearchCV(SVC(random_state=0), 
                                            parameters, cv=5, verbose=1)

SVM_classifier_NoneWeight_00.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values)

print (SVM_classifier_NoneWeight_00.best_params_)
print (SVM_classifier_NoneWeight_00.best_score_)

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

# %%
print(classification_report(y_test_df.Vote, SVM_classifier_NoneWeight_00_predictions))

# %%
print(classification_report(y_test_df.Vote, SVM_classifier_balanced_00_predictions))

# %%

# %%
import pickle
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models/"
filename = model_dir + 'SVM_classifier_balanced_regularEVI_00.sav'
pickle.dump(SVM_classifier_balanced_00, open(filename, 'wb'))

# %%
filename = model_dir + 'SVM_classifier_NoneWeight_regularEVI_00.sav'
pickle.dump(SVM_classifier_NoneWeight_00, open(filename, 'wb'))

# %%
None_y_test_df_act_1_pred_2=None_y_test_df[None_y_test_df.Vote==1].copy()
None_y_test_df_act_2_pred_1=None_y_test_df[None_y_test_df.Vote==2].copy()

None_y_test_df_act_1_pred_2=None_y_test_df_act_1_pred_2[None_y_test_df_act_1_pred_2.prediction==2].copy()
None_y_test_df_act_2_pred_1=None_y_test_df_act_2_pred_1[None_y_test_df_act_2_pred_1.prediction==1].copy()

# %%
None_y_test_df_act_2_pred_1

# %%
None_y_test_df_act_2_pred_1 = pd.merge(None_y_test_df_act_2_pred_1, ground_truth_labels_extended, on=['ID'], how='left')
None_y_test_df_act_1_pred_2 = pd.merge(None_y_test_df_act_1_pred_2, ground_truth_labels_extended, on=['ID'], how='left')

# %%
print (None_y_test_df_act_2_pred_1.ExctAcr.sum())
print (None_y_test_df_act_1_pred_2.ExctAcr.sum())

# %%
None_y_test_df[None_y_test_df.Vote==2].shape

# %%

# %%

# %%
parameters = {'C':[5, 10, 13, 14, 15, 16, 17, 20, 40, 80],
              'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
              'class_weight':['balanced', None]} # , 
SVM_classifier_gridWeight = GridSearchCV(SVC(random_state=0), 
                                            parameters, cv=5, verbose=1,
                                            error_score='raise')

SVM_classifier_gridWeight.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values)

print (SVM_classifier_gridWeight.best_params_)
print (SVM_classifier_gridWeight.best_score_)

# %%
