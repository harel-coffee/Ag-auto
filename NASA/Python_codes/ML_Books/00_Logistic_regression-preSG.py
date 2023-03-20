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
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter


from random import seed
from random import random

import os, os.path
import shutil

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

import h5py
import sys
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core.py as rcp

# %% [markdown]
# # Read Training Set Labels

# %%
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
ground_truth_labels = pd.read_csv(training_set_dir+"train_labels.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))
ground_truth_labels.head(2)

# %% [markdown]
# # Start VI images

# %%
VI_idx = "EVI"
data_dir = "//Users/hn/Documents/01_research_data/NASA/VI_TS/04_regularized_TS/"

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

# %%
ground_truth.shape

# %% [markdown]
# # Sort the order of time-series and experts' labels identically

# %%
ground_truth.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
ground_truth_labels.sort_values(by=["ID"], inplace=True)

ground_truth.reset_index(drop=True, inplace=True)
ground_truth_labels.reset_index(drop=True, inplace=True)

# %%
assert (len(ground_truth.ID.unique()) == len(ground_truth_labels.ID.unique()))

# %%
print (list(ground_truth.ID)[0])
print (list(ground_truth_labels.ID)[0])
print ("____________________________________")
print (list(ground_truth.ID)[-1])
print (list(ground_truth_labels.ID)[-1])
print ("____________________________________")
print (list(ground_truth.ID.unique())==list(ground_truth_labels.ID.unique()))

# %% [markdown]
# # Convert the dataframe to numpy array 2D

# %%
# TS_array = np.zeros((len(ground_truth.ID.unique()), 36))
# for count, ID in enumerate(ground_truth.ID.unique()):
#     TS_array[count] = ground_truth[ground_truth.ID==ID][VI_idx].values[:36]

# %%
# plt.figure(figsize=(20,4))
# plt.subplot(1, 1, 1)

# field_number = 0
# plt.scatter(ground_truth[ground_truth.ID==ground_truth.ID.unique()[field_number]][\
#                                                                         "human_system_start_time"].values[:36],
#             ground_truth[ground_truth.ID==ground_truth.ID.unique()[field_number]][VI_idx].values[:36],
#             marker='o', s=65, c='b', label=VI_idx);

# plt.scatter(ground_truth[ground_truth.ID==ground_truth.ID.unique()[field_number]]["human_system_start_time"\
#                                                                                  ].values[:36],
#             TS_array[0], 
#             marker='o', s=25, c='r', label=VI_idx);

# plt.ylim([-.2, 1])

# vote = str(ground_truth_labels[ground_truth_labels.ID==ground_truth.ID.unique()[field_number]]['Vote'].values[0])
# title = ground_truth.ID.unique()[field_number] + ", " + "Vote: " + vote
# plt.title(title , fontsize = 20);

# %% [markdown]
# # Split Train and Test Set

# %%
# # split the train and test set randomly
# x_train, x_test, y_train, y_test = train_test_split(TS_array, ground_truth_labels.Vote.values, 
#                                                     test_size=0.25, 
#                                                     random_state=0,
#                                                     shuffle=True,
#                                                     stratify=ground_truth_labels.Vote.values)

# %%
# Make an instance of the Model
# all parameters not specified are set to their defaults

# logisticRegr = LogisticRegression(max_iter=100000)
# logisticRegr.fit(x_train, y_train)

# %%
# # Returns a NumPy Array
# # Predict for One Observation (image)
# test_index = 0
# prediction = logisticRegr.predict(x_test[test_index].reshape(1,-1))
# print ("prediction is {}.".format(prediction))

# ######################################################
# plt.figure(figsize=(20,4))
# plt.subplot(1, 1, 1)

# field_number = 0
# plt.scatter(range(len(x_test[test_index])),
#             x_test[test_index],
#             marker='o', s=65, c='b', label=VI_idx);

# plt.ylim([-.2, 1])

# title = "prediction is " + str(prediction) + ", and label is " + str(y_test[test_index])
# plt.title(title , fontsize = 20);


# %%
# y_test[test_index]

# %% [markdown]
# # Logistic Regression with Pandas
#
# I want to split pandas dataframes so that we have access to IDs easily.

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
ground_truth_wide.head(2)

# %%

# %%
ground_truth_labels.head(2)

# %%
len(ground_truth_wide.ID)

# %% [markdown]
# # Check if we did the right things:

# %%
(ground_truth_wide.ID==ground_truth_labels.ID).sum()

# %%
plt.figure(figsize=(20,4))
plt.subplot(1, 1, 1)

field_number = 120
ID = ground_truth.ID.unique()[field_number]
plt.scatter(ground_truth[ground_truth.ID==ID]["human_system_start_time"].values[:36],
            ground_truth[ground_truth.ID==ID][VI_idx].values[:36],
            marker='o', s=65, c='b', label=VI_idx);

plt.scatter(ground_truth[ground_truth.ID==ID]["human_system_start_time"].values[:36],
            ground_truth_wide[ground_truth_wide.ID==ID].values[0][1:],
            marker='o', s=25, c='r', label=VI_idx);
plt.ylim([-0.2, 1])

vote = str(ground_truth_labels[ground_truth_labels.ID==ground_truth.ID.unique()[field_number]]['Vote'].values[0])
title = ground_truth.ID.unique()[field_number] + ", " + "Vote: " + vote
plt.title(title , fontsize = 20);

# %%
len(ground_truth_wide.ID.unique())==len(ground_truth_labels.ID.unique())

# %%
ground_truth.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
ground_truth_labels.sort_values(by=["ID"], inplace=True)

ground_truth.reset_index(drop=True, inplace=True)
ground_truth_labels.reset_index(drop=True, inplace=True)

# %%
ground_truth_wide.head(2)

# %%
ground_truth_labels.head(2)

# %%
print ("number of fields in the training set is [{}].".format(len(ground_truth_wide.ID)))
(ground_truth_labels.ID == ground_truth_wide.ID).sum()== len(ground_truth_labels.ID)

# %% [markdown]
# ### Sort the order of time-series and experts' labels identically

# %%
ground_truth.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
ground_truth_labels.sort_values(by=["ID"], inplace=True)

ground_truth.reset_index(drop=True, inplace=True)
ground_truth_labels.reset_index(drop=True, inplace=True)

# %%
# split the train and test set randomly
# split the train and test set randomly
x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(ground_truth_wide, 
                                                                ground_truth_labels, 
                                                                test_size=0.2, 
                                                                random_state=0,
                                                                shuffle=True,
                                                                stratify=ground_truth_labels.Vote.values)

# %%
# Make an instance of the Model
# all parameters not specified are set to their defaults

# random_state is used for solver == ‘sag’, ‘saga’ or ‘liblinear’
logisticRegr = LogisticRegression(random_state=0); 
logisticRegr.fit(x_train_df.loc[:, "EVI_1":"EVI_36"], y_train_df.Vote.values);

# %%
# Returns a NumPy Array
# Predict for One Observation (image)
test_index = 0
ID = x_test_df.iloc[test_index,].ID
prediction = logisticRegr.predict(x_test_df.iloc[test_index, 1:].values.reshape(1,-1))
print ("prediction is {}.".format(prediction))

######################################################
plt.figure(figsize=(20,4))
plt.subplot(1, 1, 1)

plt.scatter(range(len(x_test_df.iloc[test_index, 1:])),
            x_test_df.iloc[test_index, 1:].values,
            marker='o', s=65, c='b', label=VI_idx);
plt.ylim([-0.2, 1])

curr_V = str(y_test_df[y_test_df.ID == ID].Vote.values[0])
title = ID + ", prediction is " + str(prediction) + ", and label is [" + curr_V + "]."
plt.title(title , fontsize = 20);

# %%
# plt.scatter(range(len(x_test_df.iloc[test_index, 1:])),
#             x_test_df.iloc[test_index, 1:].values,
#             marker='o', s=65, c='b', label=VI_idx);
# plt.ylim([-0.2, 1])

# %%
confusion_table = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], index=range(2))
confusion_table.loc[0, 'None'] = 'Actual_Single'
confusion_table.loc[1, 'None'] = 'Actual_Double'
confusion_table['Predict_Single']=0
confusion_table['Predict_Double']=0

actual_double_predicted_single=[]

for idx in x_test_df.index:
    ID = x_test_df.loc[idx,].ID
    prediction = logisticRegr.predict(x_test_df.loc[idx, "EVI_1":].values.reshape(1,-1))
    prediction = prediction[0]
    curr_V = y_test_df[y_test_df.ID == ID].Vote.values[0]
    
    if prediction==curr_V==1:
        confusion_table.loc[0, "Predict_Single"]+=1
    elif prediction==curr_V==2:
        confusion_table.loc[1, "Predict_Double"]+=1
    
    elif prediction!=curr_V:
        if prediction==1:
            confusion_table.loc[1, "Predict_Single"]+=1
            actual_double_predicted_single += [ID]
        elif prediction==2:
            confusion_table.loc[0, "Predict_Double"]+=1
            

confusion_table

# %% [markdown]
# # Read Raw data so we can plot them

# %%
raw_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/data_for_train_individual_counties/"

raw_names = ["L7_T1C2L2_Scaled_AdamBenton2016_2015-01-01_2017-10-14.csv", 
             "L7_T1C2L2_Scaled_FranklinYakima2018_2017-01-01_2019-10-14.csv", 
             "L7_T1C2L2_Scaled_Grant2017_2016-01-01_2018-10-14.csv", 
             "L7_T1C2L2_Scaled_Walla2015_2014-01-01_2016-12-31.csv",
             "L8_T1C2L2_Scaled_AdamBenton2016_2015-01-01_2017-10-14.csv",
             "L8_T1C2L2_Scaled_FranklinYakima2018_2017-01-01_2019-10-14.csv",
             "L8_T1C2L2_Scaled_Grant2017_2016-01-01_2018-10-14.csv",
             "L8_T1C2L2_Scaled_Walla2015_2014-01-01_2016-12-31.csv"]

raw_data=pd.DataFrame()

for file in raw_names:
    curr_file=pd.read_csv(raw_dir + file)
    curr_file=curr_file[["ID", VI_idx, "system_start_time"]]
    curr_file = curr_file[curr_file[VI_idx].notna()]
    curr_file=nc.add_human_start_time_by_system_start_time(curr_file)
    curr_file=curr_file[["ID", VI_idx, "human_system_start_time"]]
    # curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])
    
    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years)==3 or len(all_years)==2:
        proper_year = all_years[1]
    elif len(all_years)==1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
    raw_data=pd.concat([raw_data, curr_file])

raw_data.reset_index(drop=True, inplace=True)
raw_data.head(2)

# %%
multiple_test_size = len(actual_double_predicted_single)
n_cols = 1
n_rows = multiple_test_size

fig, axes = plt.subplots(n_rows, 1, figsize=(18, 40), sharey='row',
                        gridspec_kw={'hspace': 0.5, 'wspace': .5});

actual_double_predicted_single = sorted(actual_double_predicted_single)
for index in range(len(actual_double_predicted_single)):
    ID = actual_double_predicted_single[index]
    prediction = logisticRegr.predict(x_test_df[x_test_df.ID == ID].values[0][1:].reshape(1,-1))
    
    xx = ground_truth[ground_truth.ID==ID]["human_system_start_time"].values[:36]
    yy = ground_truth[ground_truth.ID==ID]["EVI"].values[:36]
    axes[index].plot(xx, yy, c='k', linewidth=3);
    
    raw_xx = raw_data[raw_data.ID==ID]["human_system_start_time"]
    raw_yy = raw_data[raw_data.ID==ID]["EVI"]
    axes[index].scatter(raw_xx, raw_yy, marker='o', s=65, c='dodgerblue', label=VI_idx);

    axes[index].set_ylim([-0.2, 1])
    axes[index].grid(True); 
    
    curr_V = str(y_test_df[y_test_df.ID == ID].Vote.values[0])
    title = ID + ", prediction is " + str(prediction) + ", and label is [" + curr_V + "]."
    # axes[index].legend(loc="best")
    axes[index].set_title(title);
    axes[index].xaxis.set_major_locator(mdates.MonthLocator())
    axes[index].xaxis.set_major_formatter(DateFormatter('%b'))

# plt.tight_layout()

plot_path = "/Users/hn/Desktop/"
os.makedirs(plot_path, exist_ok=True)
fig_name = plot_path + "test_set_ActualDouble_PredictedOne_preSG.png"
plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight', facecolor="w")


# %%
actual_double_predicted_single

# %%
# # This will be predicted Wrong
# fig, ax = plt.subplots(1, 1, figsize=(18, 2), sharex='col', sharey='row',
#                         gridspec_kw={'hspace': 0.1, 'wspace': .1});

# ax.grid(True)
    
# field_number = 120
# ID = "105301_WSDA_SF_2017"
# ax.scatter(ground_truth[ground_truth.ID==ID]["human_system_start_time"].values[:36],
#             ground_truth[ground_truth.ID==ID]["EVI"].values[:36],
#             marker='o', s=65, c='b', label=VI_idx);

# ax.plot(ground_truth[ground_truth.ID==ID]["human_system_start_time"].values[:36], 
#          ground_truth[ground_truth.ID==ID]["EVI"].values[:36], 
#          c='k', linewidth=2);

# plt.ylim([-0.2, 1])

# import matplotlib.dates as mdates
# from matplotlib.dates import MonthLocator, DateFormatter
# ax.xaxis.set_major_locator(mdates.MonthLocator())
# ax.xaxis.set_major_formatter(DateFormatter('%b'))

# vote = str(ground_truth_labels[ground_truth_labels.ID==ground_truth.ID.unique()[field_number]]['Vote'].values[0])
# title = ground_truth.ID.unique()[field_number] + ", " + "Vote: " + vote
# ax.set_title(title , fontsize = 20);

# %%

# %%
# multiple_test_size = 21
# plt.figure(figsize=(20, 50))
# n_cols = 1
# n_rows = multiple_test_size

# for index in range(len(actual_double_predicted_single)):
#     ID = actual_double_predicted_single[index]
#     prediction = logisticRegr.predict(x_test_df[x_test_df.ID == ID].values[0][1:].reshape(1,-1))

#     plt.subplot(n_rows, n_cols, index + 1) # 1 row, 5 columns, 
#     plt.scatter(range(len(x_test_df[x_test_df.ID == ID].values[0][1:])),
#                 x_test_df[x_test_df.ID == ID].values[0][1:],
#                 marker='o', s=65, c='b', label=VI_idx);
#     plt.ylim([-0.1, 1])
    
#     curr_V = str(y_test_df[y_test_df.ID == ID].Vote.values[0])
#     title = ID + ", prediction is " + str(prediction) + ", and label is [" + curr_V + "]."
#     plt.legend(loc="best")
#     plt.title(title , fontsize = 20);

# plt.tight_layout()

# %%

# %% [markdown]
# # Confusion table for training set

# %%
confus_tbl_train = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], index=range(2))
confus_tbl_train.loc[0, 'None'] = 'Actual_Single'
confus_tbl_train.loc[1, 'None'] = 'Actual_Double'
confus_tbl_train['Predict_Single']=0
confus_tbl_train['Predict_Double']=0

actual_double_predicted_single=[]

for idx in x_train_df.index:
    ID = x_train_df.loc[idx,].ID
    prediction = logisticRegr.predict(x_train_df.loc[idx, "EVI_1":].values.reshape(1,-1))
    prediction = prediction[0]
    curr_V = y_train_df[y_train_df.ID == ID].Vote.values[0]
    
    if prediction==curr_V==1:
        confus_tbl_train.loc[0, "Predict_Single"]+=1
    elif prediction==curr_V==2:
        confus_tbl_train.loc[1, "Predict_Double"]+=1
    
    elif prediction!=curr_V:
        if prediction==1:
            confus_tbl_train.loc[1, "Predict_Single"]+=1
            actual_double_predicted_single += [ID]
        elif prediction==2:
            confus_tbl_train.loc[0, "Predict_Double"]+=1
            
confus_tbl_train

# %%

# %%

# %% [markdown]
# # Horrible? on train set.
# We are underfitting?
#    - Logistic Regression with Ratios.
#    - Plot learning curves
#    - Use a larger train set and cross validation!
#    - Diagnose (how?)
#    - Tune parameters (what parameters?)
#    - Look at AIC, BIC? Good reminder.
#    - Exponential (or other types of) smoothing?
#    - Forget about logistic: try
#        - Distance-based (KNN with dynamic time warping)
#        - Interval-based (TimeSeriesForest)
#        - Dictionary-based (BOSS, cBOSS)
#        - Frequency-based (RISE — like TimeSeriesForest but with other features)
#        - Shapelet-based (Shapelet Transform Classifier)

# %%
# Make an instance of the Model
# all parameters not specified are set to their defaults

# random_state is used for solver == ‘sag’, ‘saga’ or ‘liblinear’
logisticRegr = LogisticRegression(random_state=0,
                                  class_weight="balanced");
logisticRegr.fit(x_train_df.loc[:, "EVI_1":"EVI_36"], y_train_df.Vote.values);

# %%
confus_tbl_train = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], index=range(2))
confus_tbl_train.loc[0, 'None'] = 'Actual_Single'
confus_tbl_train.loc[1, 'None'] = 'Actual_Double'
confus_tbl_train['Predict_Single']=0
confus_tbl_train['Predict_Double']=0

actual_double_predicted_single=[]

for idx in x_train_df.index:
    ID = x_train_df.loc[idx,].ID
    prediction = logisticRegr.predict(x_train_df.loc[idx, "EVI_1":].values.reshape(1,-1))
    prediction = prediction[0]
    curr_V = y_train_df[y_train_df.ID == ID].Vote.values[0]
    
    if prediction==curr_V==1:
        confus_tbl_train.loc[0, "Predict_Single"]+=1
    elif prediction==curr_V==2:
        confus_tbl_train.loc[1, "Predict_Double"]+=1
    
    elif prediction!=curr_V:
        if prediction==1:
            confus_tbl_train.loc[1, "Predict_Single"]+=1
            actual_double_predicted_single += [ID]
        elif prediction==2:
            confus_tbl_train.loc[0, "Predict_Double"]+=1
            
confus_tbl_train

# %%
print((ground_truth_labels.Vote==1).sum()/(ground_truth_labels.Vote==2).sum())

# %%
# Make an instance of the Model
# all parameters not specified are set to their defaults
from sklearn.linear_model import LogisticRegressionCV
logisticRegr_CV = LogisticRegressionCV(max_iter=100000, cv=10)
logisticRegr_CV.fit(ground_truth_wide.loc[:, "EVI_1":], ground_truth_labels.Vote.values)

# %%

# %%

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
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
f_name = "/Users/hn/Documents/01_research_data/NASA/" +\
         "VI_TS/moreBands_train_set/" +\
         "L8_T1C2L2_Scaled_FranklinYakima2018_2018-01-01_2019-01-01.csv"
A = pd.read_csv(f_name)

# %%
A.head(2)

# %%
L = len(A['SR_B2'].index[A['SR_B2'].apply(np.isnan)])

# %%
cc="SR_B3"
sum(A['SR_B2'].index[A['SR_B2'].apply(np.isnan)] == A[cc].index[A[cc].apply(np.isnan)])

# %%
len(A['SR_B2'].index[A['SR_B2'].apply(np.isnan)])

# %%
len(A[cc].index[A[cc].apply(np.isnan)])

# %%
V = A['SR_B2'].index[A['SR_B2'].apply(np.isnan)].isin(A[cc].index[A[cc].apply(np.isnan)])

# %%
V = list(V)

# %%
V.index(False)

# %%
A['SR_B2'].index[A['SR_B2'].apply(np.isnan)][334116]

# %%
A.head(2)

# %%
L8_dict = {"SR_B2":"blue",
           "SR_B3":"green",
           "SR_B4":"red",
           "SR_B5":"NIR",
           "SR_B6":"short_I_1",
           "SR_B7":"short_I_2",
           }

# %%

# %%
A.head(2)

# %%
