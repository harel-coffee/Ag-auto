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
import scipy, scipy.signal

from datetime import date
import time

import random
from random import seed, random
# import shutil

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

import pickle #, h5py
import sys, os, os.path
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
# # Read the Data

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
# # Toss small fields

# %%
GT_TS = GT_TS[GT_TS.ID.isin((list(meta_moreThan10Acr.ID)))].copy()
GT_labels = GT_labels[GT_labels.ID.isin((list(meta_moreThan10Acr.ID)))].copy()

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

# %% [markdown]
# # Sort the order of time-series and experts' labels identically

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

# %%
GT_TS.head(2)

# %%
GT_labels.head(2)

# %%
mins = GT_TS.groupby("ID")[VI_idx].min()
maxs = GT_TS.groupby("ID")[VI_idx].max()
_ranges = maxs-mins
_ranges = pd.DataFrame(_ranges)
_ranges.reset_index(inplace=True)

mins = pd.DataFrame(mins)
mins.reset_index(inplace=True)


_ranges.rename(columns = {'EVI':'EVI_range'}, inplace = True)
mins.rename(columns = {'EVI':'EVI_min'}, inplace = True)

print (_ranges.head(2))

GT_TS = pd.merge(GT_TS, _ranges, on=['ID'], how='left')
GT_TS = pd.merge(GT_TS, mins, on=['ID'], how='left')
GT_TS[VI_idx+"_ratio"] = (GT_TS[VI_idx]-GT_TS[VI_idx + "_min"])/GT_TS[VI_idx+"_range"]
GT_TS.head(2)

# %%
ID_1=GT_TS.ID.unique()[0]
ID_2=GT_TS.ID.unique()[12]

# _minimum = GT_TS[GT_TS.ID==ID_1].EVI.values.min()
# _range = GT_TS[GT_TS.ID==ID_1].EVI.values.max()-_minimum
# y_1 = (GT_TS[GT_TS.ID==ID_1].EVI.values-_minimum)/_range
# _minimum = GT_TS[GT_TS.ID==ID_2].EVI.values.min()
# _range = GT_TS[GT_TS.ID==ID_2].EVI.values.max()-_minimum
# y_2 = (GT_TS[GT_TS.ID==ID_2].EVI.values-_minimum)/_range

dtw_score = dtw_metric(GT_TS[GT_TS.ID==ID_1].EVI.values, GT_TS[GT_TS.ID==ID_2].EVI.values)

dtw_score_ratios = dtw_metric(GT_TS[GT_TS.ID==ID_1].EVI_ratio.values, 
                              GT_TS[GT_TS.ID==ID_2].EVI_ratio.values)

# print ("dtw score is {:.2f}.".format(dtw_score))

plt.figure(figsize=(20,4))
plt.subplot(1, 1, 1)

# plot EVIs
plt.plot(range(len(GT_TS[GT_TS.ID==ID_1].human_system_start_time.values)),
         GT_TS[GT_TS.ID==ID_1].EVI.values,
         c='k', linewidth=5);


plt.plot(range(len(GT_TS[GT_TS.ID==ID_2].human_system_start_time.values)),
         GT_TS[GT_TS.ID==ID_2].EVI.values,
        c='red', linewidth=5);

# plot ratios
plt.plot(range(len(GT_TS[GT_TS.ID==ID_1].human_system_start_time.values)), 
         GT_TS[GT_TS.ID==ID_1].EVI_ratio.values,
         ".-.", c='k', linewidth=2, label="ratios");

plt.plot(range(len(GT_TS[GT_TS.ID==ID_2].human_system_start_time.values)), 
         GT_TS[GT_TS.ID==ID_2].EVI_ratio.values,
         ".-.", c='red', linewidth=2, label="ratios");


title = "dtw score for EVI is [{:.2f}] and for ratios is [{:.2f}].".format(dtw_score, dtw_score_ratios)
plt.ylim([-0.2, 1.05]);
plt.title(title , fontsize = 20);

# %%
s1=GT_TS[GT_TS.ID==ID_1].EVI_ratio.values
s2=GT_TS[GT_TS.ID==ID_2].EVI_ratio.values
path = dtw.warping_path(s1, s2)
dtwvis.plot_warping(s1, s2, path)
distance = dtw.distance(s1, s2)

# %%
d, paths = dtw.warping_paths(s1, s2, window=10, use_pruning=True);
best_path = dtw.best_path(paths);
dtwvis.plot_warpingpaths(s1, s2, paths, best_path);


# %%
def DTW_prune(ts1, ts2):
    d,_ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True);
    return d


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
    GT_wide.loc[GT_wide_indx, VI_idx+"_1":VI_idx+"_36"] = curr_df.EVI.values[:36]

print (len(GT_wide.ID.unique()))
GT_wide.head(2)

# %% [markdown]
# # Split Train and Test Set
#
# #### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order

# %%
# ground_truth_labels = ground_truth_labels.set_index('ID')
# ground_truth_labels = ground_truth_labels.reindex(index=ground_truth_wide['ID'])
# ground_truth_labels = ground_truth_labels.reset_index()

# x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(ground_truth_wide, 
#                                                                 ground_truth_labels, 
#                                                                 test_size=0.2, 
#                                                                 random_state=0,
#                                                                 shuffle=True,
#                                                                 stratify=ground_truth_labels.Vote.values)

# This cell and some edits after this are new compared to 00_SVM_SG_EVI.ipynb.
# I want to avoid splitting and just use the one I created earlier.

ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
train80 = pd.read_csv(ML_data_folder+"train80_split_2Bconsistent_Oct17.csv")
test20 = pd.read_csv(ML_data_folder+"test20_split_2Bconsistent_Oct17.csv")
GT_labels.head(2)

# %%
GT_labels.head(2)

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
KNN_DTW_prune_predictions = KNN_DTW_prune.predict(x_test_df.iloc[:, 1:])

# %%
(KNN_DTW_prune_predictions==y_test_df.Vote).sum()

# %%
y_test_df_copy=y_test_df.copy()
y_test_df_copy["KNN_DTW_prune_predictions"] = list(KNN_DTW_prune_predictions)
y_test_df_copy.head(2)

# %%
true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index in y_test_df_copy.index:
    curr_vote=list(y_test_df_copy[y_test_df_copy.index==index].Vote)[0]
    curr_predict=list(y_test_df_copy[y_test_df_copy.index==index].KNN_DTW_prune_predictions)[0]
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

# %% [markdown]
# # Confusion Table for Test Set

# %%
confus_tbl_test = pd.DataFrame(columns=['NoName', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
confus_tbl_test.loc[0, 'NoName'] = 'Actual_Single'
confus_tbl_test.loc[1, 'NoName'] = 'Actual_Double'
confus_tbl_test['Predict_Single']=0
confus_tbl_test['Predict_Double']=0

confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
confus_tbl_test

# %%
print(classification_report(y_test_df.Vote, KNN_DTW_prune_predictions))

# %%
# print ("IF **Single** is positive and double is negative:")
# print ("")

# TP = list(confus_tbl_test[confus_tbl_test.NoName=="Actual_Single"]["Predict_Single"])[0]
# FP = list(confus_tbl_test[confus_tbl_test.NoName=="Actual_Double"]["Predict_Single"])[0]
# FN = list(confus_tbl_test[confus_tbl_test.NoName=="Actual_Single"]["Predict_Double"])[0]
# print ("Precision is [{0:.2f}]".format(TP/(TP+FP)))
# print ("Recall is [{0:.2f}]".format(TP/(TP+FN)))

# print ("_________________________________________________________")
# print ("IF **Single** is negative. double is positive:")

# TP = list(confus_tbl_test[confus_tbl_test.NoName=="Actual_Double"]["Predict_Double"])[0]
# FP = list(confus_tbl_test[confus_tbl_test.NoName=="Actual_Single"]["Predict_Double"])[0]
# FN = list(confus_tbl_test[confus_tbl_test.NoName=="Actual_Double"]["Predict_Single"])[0]

# print ("")
# print ("Precision is [{0:.2f}]".format(TP/(TP+FP)))
# print ("Recall is [{0:.2f}]".format(TP/(TP+FN)))

# %% [markdown]
# # Export Trained Model

# %%
filename = model_dir + "01_KNN_regular_" + VI_idx + "_DTW_prune_" + \
           KNN_DTW_prune.best_params_["weights"] + "Weight_" + \
                        str(KNN_DTW_prune.best_params_["n_neighbors"]) + "NNisBest.sav"

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
# Improve?

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

# %% [markdown]
# # Confusion Table for Test Set

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
confusion_matrix(y_test_df_copy.Vote, y_test_df_copy.weightDist_predictions)

# %%
filename = model_dir + "00_KNN_regular_" + VI_idx + "_DTW_prune_" + \
           KNN_DTW_prune_weightsDistance.best_params_["weights"] + \
           "Weight_" + str(KNN_DTW_prune_weightsDistance.best_params_["n_neighbors"]) + \
           "NNisBest.sav"

pickle.dump(KNN_DTW_prune_weightsDistance, open(filename, 'wb'))

# %%
actual_double_predicted_single_IDs

# %%
actual_single_predicted_double_IDs

# %%
plt.figure(figsize=(20,4))
plt.subplot(1, 1, 1)

ID_1 = actual_double_predicted_single_IDs[2]
# plot EVI0
# plt.plot(range(len(ground_truth[ground_truth.ID==ID_1].human_system_start_time.values)),
#          ground_truth[ground_truth.ID==ID_1].EVI.values,
#         c='k', linewidth=5);

plt.plot(GT_TS[GT_TS.ID==ID_1].human_system_start_time.values,
         GT_TS[GT_TS.ID==ID_1].EVI.values,
         c='k', linewidth=5);


title = list(meta[meta.ID==ID_1].CropTyp)[0].replace(",", "")  + ", " + ID_1
plt.ylim([-0.2, 1.05]);
plt.title(title , fontsize=20);

# %%
meta[meta.ID.isin(actual_double_predicted_single_IDs)]["ExctAcr"].sum()

# %%
meta[meta.ID.isin(actual_single_predicted_double_IDs)]["ExctAcr"].sum()

# %%

# %%
plt.figure(figsize=(20,4))
plt.subplot(1, 1, 1)

ID_1 = actual_single_predicted_double_IDs[0]
# plot EVIs
# plt.plot(range(len(ground_truth[ground_truth.ID==ID_1].human_system_start_time.values)),
#          ground_truth[ground_truth.ID==ID_1].EVI.values,
#         c='k', linewidth=5);

plt.plot(GT_TS[GT_TS.ID==ID_1].human_system_start_time.values,
         GT_TS[GT_TS.ID==ID_1].EVI.values,
        c='k', linewidth=5);

title = list(meta[meta.ID==ID_1].CropTyp)[0].replace(",", "")  + ", " + ID_1
plt.ylim([-0.2, 1.05]);
plt.title(title , fontsize = 20);

# %%

# %%

# %% [markdown]
# ## Fast WTD

# %%

# import array
# s1 = array.array('d',[0, 0, 1, 2, 1, 0, 1, 0, 0])
# s2 = array.array('d',[0, 1, 2, 0, 0, 0, 0, 0, 0])
# d = dtw.distance_fast(s1, s2, use_pruning=True)

# %%
# pip install -vvv --upgrade --force-reinstall dtaidistance

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title,
                        X, y, axes=None, ylim=None,
                        cv=None, n_jobs=None,
                        train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


fig, axes = plt.subplots(3, 2, figsize=(10, 15))

X, y = load_digits(return_X_y=True)

title = "Learning Curves (Naive Bayes)"
# Cross validation with 50 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(
    estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4
)

title = r"Learning Curves"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, 
                    X=ground_truth_wide.loc[:, "EVI_1":], 
                    y=ground_truth_labels.Vote.values, 
                    axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4)

plt.show()

# %%

# %%

# %%
