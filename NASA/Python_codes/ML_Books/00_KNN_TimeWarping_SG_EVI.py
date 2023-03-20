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

from random import seed
from random import random
import random
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
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core.py as rcp

# %%
from tslearn.metrics import dtw as dtw_metric
# https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

# %%

# %% [markdown]
# # Read Fields Metadata

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta.head(2)

# %%
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)

# %% [markdown]
# # Read Training Set Labels

# %%
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
ground_truth_labels = pd.read_csv(training_set_dir+"train_labels.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))
ground_truth_labels.head(2)

# %%

# %% [markdown]
# # Read the Data

# %%
VI_idx = "EVI"
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
ground_truth = data[data.ID.isin(list(ground_truth_labels.ID.unique()))].copy()

# %% [markdown]
# # Toss small fields

# %%
ground_truth = ground_truth[ground_truth.ID.isin((list(meta_moreThan10Acr.ID)))].copy()
ground_truth_labels = ground_truth_labels[ground_truth_labels.ID.isin((list(meta_moreThan10Acr.ID)))].copy()

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
ground_truth.head(2)

# %%
ground_truth_labels.head(2)

# %%
mins = ground_truth.groupby("ID")[VI_idx].min()
maxs = ground_truth.groupby("ID")[VI_idx].max()
_ranges = maxs-mins
_ranges = pd.DataFrame(_ranges)
_ranges.reset_index(inplace=True)

mins = pd.DataFrame(mins)
mins.reset_index(inplace=True)


_ranges.rename(columns = {'EVI':'EVI_range'}, inplace = True)
mins.rename(columns = {'EVI':'EVI_min'}, inplace = True)

print (_ranges.head(2))

ground_truth = pd.merge(ground_truth, _ranges, on=['ID'], how='left')
ground_truth = pd.merge(ground_truth, mins, on=['ID'], how='left')
ground_truth["EVI_ratio"] = (ground_truth["EVI"]-ground_truth["EVI_min"])/ground_truth["EVI_range"]
ground_truth.head(2)

# %%
ID_1=ground_truth.ID.unique()[0]
ID_2=ground_truth.ID.unique()[12]

# _minimum = ground_truth[ground_truth.ID==ID_1].EVI.values.min()
# _range = ground_truth[ground_truth.ID==ID_1].EVI.values.max()-_minimum
# y_1 = (ground_truth[ground_truth.ID==ID_1].EVI.values-_minimum)/_range
# _minimum = ground_truth[ground_truth.ID==ID_2].EVI.values.min()
# _range = ground_truth[ground_truth.ID==ID_2].EVI.values.max()-_minimum
# y_2 = (ground_truth[ground_truth.ID==ID_2].EVI.values-_minimum)/_range

dtw_score = dtw_metric(ground_truth[ground_truth.ID==ID_1].EVI.values, 
                       ground_truth[ground_truth.ID==ID_2].EVI.values)

dtw_score_ratios = dtw_metric(ground_truth[ground_truth.ID==ID_1].EVI_ratio.values, 
                       ground_truth[ground_truth.ID==ID_2].EVI_ratio.values)

# print ("dtw score is {:.2f}.".format(dtw_score))

plt.figure(figsize=(20,4))
plt.subplot(1, 1, 1)

# plot EVIs
plt.plot(range(len(ground_truth[ground_truth.ID==ID_1].human_system_start_time.values)),
         ground_truth[ground_truth.ID==ID_1].EVI.values,
        c='k', linewidth=5);


plt.plot(range(len(ground_truth[ground_truth.ID==ID_2].human_system_start_time.values)),
         ground_truth[ground_truth.ID==ID_2].EVI.values,
        c='red', linewidth=5);

# plot ratios
plt.plot(range(len(ground_truth[ground_truth.ID==ID_1].human_system_start_time.values)), 
         ground_truth[ground_truth.ID==ID_1].EVI_ratio.values,
         ".-.", c='k', linewidth=2, label="ratios");

plt.plot(range(len(ground_truth[ground_truth.ID==ID_2].human_system_start_time.values)), 
         ground_truth[ground_truth.ID==ID_2].EVI_ratio.values,
         ".-.", c='red', linewidth=2, label="ratios");


title = "dtw score for EVI is [{:.2f}] and for ratios is [{:.2f}].".format(dtw_score, dtw_score_ratios)
plt.ylim([-0.2, 1.05]);
plt.title(title , fontsize = 20);

# %%
s1=ground_truth[ground_truth.ID==ID_1].EVI_ratio.values
s2=ground_truth[ground_truth.ID==ID_2].EVI_ratio.values
path = dtw.warping_path(s1, s2)
dtwvis.plot_warping(s1, s2, path)
distance = dtw.distance(s1, s2)


# %%

# %%
def DTW_prune(ts1, ts2):
    d,_ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True);
    return d
    


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
ground_truth_wide.head(2)

# %%
ground_truth_labels.shape

# %% [markdown]
# # Split Train and Test Set
#
# #### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order

# %%
ground_truth_labels = ground_truth_labels.set_index('ID')
ground_truth_labels = ground_truth_labels.reindex(index=ground_truth_wide['ID'])
ground_truth_labels = ground_truth_labels.reset_index()

# %%
ground_truth_labels.head(2)

# %%
x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(ground_truth_wide, 
                                                                ground_truth_labels, 
                                                                test_size=0.2, 
                                                                random_state=0,
                                                                shuffle=True,
                                                                stratify=ground_truth_labels.Vote.values)

# %% [markdown]
# # Train

# %%
# %%time
parameters = {'n_neighbors':[2, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20],
              "weights":["uniform"]}
KNN_DTW_prune = GridSearchCV(KNeighborsClassifier(metric=DTW_prune), parameters, cv=5, verbose=1)
KNN_DTW_prune.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values)

# %%
print (KNN_DTW_prune.best_params_)

# %%
x_test_df = x_test_df[x_test_df.ID.isin((list(meta_moreThan10Acr.ID)))].copy()
y_test_df = y_test_df[y_test_df.ID.isin((list(meta_moreThan10Acr.ID)))].copy()

# %%
# %%time
KNN_DTW_prune_predictions_uniform = KNN_DTW_prune.predict(x_test_df.iloc[:, 1:])

# %%
(KNN_DTW_prune_predictions_uniform==y_test_df.Vote).sum()

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
y_test_df_copy_uniform=y_test_df.copy()
y_test_df_copy_uniform["KNN_DTW_prune_predictions_uniform"] = list(KNN_DTW_prune_predictions_uniform)
y_test_df_copy_uniform.head(2)

# %%

# %%
true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index in y_test_df_copy.index:
    curr_vote=list(y_test_df_copy[y_test_df_copy.index==index].Vote)[0]
    curr_predict=list(y_test_df_copy_uniform[y_test_df_copy.index==index].KNN_DTW_prune_predictions_uniform)[0]
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

# %%
A = y_test_df_copy_uniform[y_test_df_copy_uniform.Vote==2]
A = A[A.KNN_DTW_prune_predictions_uniform==1]
meta[meta.ID.isin(list(A.ID))].ExctAcr.sum()

# %%
A = y_test_df_copy_uniform[y_test_df_copy_uniform.Vote==1]
A = A[A.KNN_DTW_prune_predictions_uniform==2]
meta[meta.ID.isin(list(A.ID))].ExctAcr.sum()

# %% [markdown]
# # Confusion Table for Test Set

# %%
confus_tbl_test_uniform = pd.DataFrame(columns=['NoName', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
confus_tbl_test_uniform.loc[0, 'NoName'] = 'Actual_Single'
confus_tbl_test_uniform.loc[1, 'NoName'] = 'Actual_Double'
confus_tbl_test_uniform['Predict_Single']=0
confus_tbl_test_uniform['Predict_Double']=0

confus_tbl_test_uniform.loc[0, "Predict_Single"]=true_single_predicted_single
confus_tbl_test_uniform.loc[0, "Predict_Double"]=true_single_predicted_double
confus_tbl_test_uniform.loc[1, "Predict_Single"]=true_double_predicted_single
confus_tbl_test_uniform.loc[1, "Predict_Double"]=true_double_predicted_double
confus_tbl_test_uniform

# %%
print(classification_report(y_test_df.Vote, KNN_DTW_prune_predictions_uniform))

# %%

# %%
print ("IF **Single** is positive and double is negative:")
print ("")

TP = list(confus_tbl_test[confus_tbl_test.NoName=="Actual_Single"]["Predict_Single"])[0]
FP = list(confus_tbl_test[confus_tbl_test.NoName=="Actual_Double"]["Predict_Single"])[0]
FN = list(confus_tbl_test[confus_tbl_test.NoName=="Actual_Single"]["Predict_Double"])[0]
print ("Precision is [{0:.2f}]".format(TP/(TP+FP)))
print ("Recall is [{0:.2f}]".format(TP/(TP+FN)))

print ("_________________________________________________________")
print ("IF **Single** is negative. double is positive:")

TP = list(confus_tbl_test[confus_tbl_test.NoName=="Actual_Double"]["Predict_Double"])[0]
FP = list(confus_tbl_test[confus_tbl_test.NoName=="Actual_Single"]["Predict_Double"])[0]
FN = list(confus_tbl_test[confus_tbl_test.NoName=="Actual_Double"]["Predict_Single"])[0]

print ("")
print ("Precision is [{0:.2f}]".format(TP/(TP+FP)))
print ("Recall is [{0:.2f}]".format(TP/(TP+FN)))

# %% [markdown]
# # Export Trained Model

# %%
import pickle
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models/"

filename = model_dir + "00_KNN_SG_"+ VI_idx + "_DTW_prune_" + KNN_DTW_prune.best_params_["weights"] + "Weight_" + \
                        str(KNN_DTW_prune.best_params_["n_neighbors"]) + "NNisBest.sav"
print (filename)
pickle.dump(KNN_DTW_prune, open(filename, 'wb'))

# %%
# How to load the saved model:
# loaded_model_KNN_DTW_prune = pickle.load(open(filename, 'rb'))
# loaded_model.predict(x_test_df.iloc[0:2, 1:])

# %%
# # %%time
# parameters = {'n_neighbors':[10]}
# KNN_DTW_prune_10NN = GridSearchCV(KNeighborsClassifier(metric=DTW_prune), parameters, cv=5, verbose=1);
# KNN_DTW_prune_10NN.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values);
# # %%time
# KNN_DTW_prune_10NN_predictions = KNN_DTW_prune_10NN.predict(x_test_df.iloc[:, 1:])

# %%

# %% [markdown]
# # Improve

# %% [markdown]
# ### Lets see if putting weights will make a difference

# %%
# %%time
parameters = {'n_neighbors':[2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
              "weights":["distance"]}
KNN_DTW_prune_weightsDistance = GridSearchCV(KNeighborsClassifier(metric=DTW_prune), 
                                             parameters, cv=5, verbose=1)
KNN_DTW_prune_weightsDistance.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values)

# %%
print ("KNN_DTW_prune_weightsDistance.best_params_ is", KNN_DTW_prune_weightsDistance.best_params_)
print ("KNN_DTW_prune.best_params_ is", KNN_DTW_prune.best_params_)

# %%
# %%time
KNN_DTW_prune_weightsDistance_predictions = KNN_DTW_prune_weightsDistance.predict(x_test_df.iloc[:, 1:])

# %%
print ("******* KNN_DTW_prune_predictions *******")
print(classification_report(y_test_df.Vote, KNN_DTW_prune_predictions))
print ("====================================================================================")
print ("******* KNN_DTW_prune_weightsDistance_predictions *******")
print(classification_report(y_test_df.Vote, KNN_DTW_prune_weightsDistance_predictions))

# %%
y_test_df_copy=y_test_df.copy()
y_test_df_copy["weightDist_predictions"] = list(KNN_DTW_prune_weightsDistance_predictions)
y_test_df_copy.head(2)

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

actual_double_predicted_single_IDs=[]
actual_single_predicted_double_IDs=[]

for index in y_test_df_copy.index:
    curr_vote=list(y_test_df_copy[y_test_df_copy.index==index].Vote)[0]
    curr_predict=list(y_test_df_copy[y_test_df_copy.index==index].weightDist_predictions)[0]
    if curr_vote==curr_predict:
        if curr_vote==1:
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
            actual_single_predicted_double_IDs+=list(y_test_df_copy[y_test_df_copy.index==index].ID)
        else:
            true_double_predicted_single+=1
            actual_double_predicted_single_IDs += list(y_test_df_copy[y_test_df_copy.index==index].ID)

confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
confus_tbl_test['Predict_Single']=0
confus_tbl_test['Predict_Double']=0

confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
confus_tbl_test

# %%
import pickle
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models/"

filename = model_dir + "00_KNN_SG_" + VI_idx + "_DTW_prune_" + \
                        KNN_DTW_prune_weightsDistance.best_params_["weights"] + \
                       "Weight_" + str(KNN_DTW_prune_weightsDistance.best_params_["n_neighbors"]) + \
                       "NNisBest.sav"

print (filename)
pickle.dump(KNN_DTW_prune_weightsDistance, open(filename, 'wb'))

# %%
actual_double_predicted_single_IDs

# %%
actual_single_predicted_double_IDs

# %%
plt.figure(figsize=(20,4))
plt.subplot(1, 1, 1)

ID_1 = actual_double_predicted_single_IDs[4]
# plot EVI0
# plt.plot(range(len(ground_truth[ground_truth.ID==ID_1].human_system_start_time.values)),
#          ground_truth[ground_truth.ID==ID_1].EVI.values,
#         c='k', linewidth=5);

plt.plot(ground_truth[ground_truth.ID==ID_1].human_system_start_time.values,
         ground_truth[ground_truth.ID==ID_1].EVI.values,
         c='k', linewidth=5);


title = list(meta[meta.ID==ID_1].CropTyp)[0].replace(",", "")  + ", " + ID_1
plt.ylim([-0.2, 1.05]);
plt.title(title , fontsize=20);

# %%
meta[meta.ID.isin(actual_double_predicted_single_IDs)]

# %%
meta[meta.ID.isin(actual_single_predicted_double_IDs)]

# %%
plt.figure(figsize=(20,4))
plt.subplot(1, 1, 1)

ID_1 = actual_single_predicted_double_IDs[0]
# plot EVIs
# plt.plot(range(len(ground_truth[ground_truth.ID==ID_1].human_system_start_time.values)),
#          ground_truth[ground_truth.ID==ID_1].EVI.values,
#         c='k', linewidth=5);

plt.plot(ground_truth[ground_truth.ID==ID_1].human_system_start_time.values,
         ground_truth[ground_truth.ID==ID_1].EVI.values,
        c='k', linewidth=5);


title = list(meta[meta.ID==ID_1].CropTyp)[0].replace(",", "")  + ", " + ID_1
plt.ylim([-0.2, 1.05]);
plt.title(title , fontsize = 20);

# %%

# %%
