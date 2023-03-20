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
from matplotlib import pyplot
from pylab import imshow
from matplotlib.image import imread

import h5py
import pickle
import sys

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core as rcp

# %%
from tslearn.metrics import dtw as dtw_metric

# https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

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
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
ground_truth_labels = pd.read_csv(training_set_dir+"train_labels.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))
ground_truth_labels.head(2)

# %% [markdown]
# # Read the Data

# %%
VI_idx = "EVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"

# %%
landsat_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/data_for_train_individual_counties/"

landsat_fNames = [x for x in os.listdir(landsat_dir) if x.endswith(".csv")]

landsat_DF = pd.DataFrame()
for fName in landsat_fNames:
    curr = pd.read_csv(landsat_dir+fName)
    curr.dropna(subset=[VI_idx], inplace=True)
    landsat_DF=pd.concat([landsat_DF, curr])

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
ground_truth_labels_extended = pd.merge(ground_truth_labels, meta, on=['ID'], how='left')
ground_truth_labels = ground_truth_labels_extended[ground_truth_labels_extended.ExctAcr>=10].copy()
ground_truth_labels.reset_index(drop=True, inplace=True)

print ("There are [{:.0f}] fields in total whose area" \
        "adds up to [{:.2f}].".format(len(ground_truth_labels_extended), \
                                          ground_truth_labels_extended.ExctAcr.sum()))


print ("There are [{:.0f}] fields larger than 10 acres whose "\
       "area adds up to [{:.2f}].".format(len(ground_truth_labels), \
                                              ground_truth_labels.ExctAcr.sum()))


# %%
ground_truth = ground_truth[ground_truth.ID.isin((list(meta_moreThan10Acr.ID)))].copy()
ground_truth_labels = ground_truth_labels[ground_truth_labels.ID.isin((list(meta_moreThan10Acr.ID)))].copy()

ground_truth.reset_index(drop=True, inplace=True)
ground_truth_labels.reset_index(drop=True, inplace=True)

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

# %% [markdown]
# # Split Train and Test Set
#
# #### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order

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
landsat_DF = landsat_DF[landsat_DF.ID.isin(list(y_test_df.ID))]
landsat_DF = nc.add_human_start_time_by_system_start_time(landsat_DF)
landsat_DF.reset_index(drop=True, inplace=True)
landsat_DF.head(2)

# %%
x_test_df.iloc[:, 1:].head(2)

# %%
x_test_df.head(2)

# %% [markdown]
# # Read SVM SG From Disk

# %%
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models/"
filename = model_dir + 'SVM_classifier_balanced_SGEVI_00.sav'
SVM_classifier_balanced_00 = pickle.load(open(filename, 'rb'))

# filename = model_dir + 'SVM_classifier_NoneWeight_SGEVI_00.sav'
# SVM_classifier_NoneWeight_00 = pickle.load(open(filename, 'rb'))

# %% [markdown]
# #### Predict SVMs on SG data

# %%
# SVM_classifier_NoneWeight_00_predictions = SVM_classifier_NoneWeight_00.predict(x_test_df.iloc[:, 1:])
SVM_classifier_balanced_00_predictions = SVM_classifier_balanced_00.predict(x_test_df.iloc[:, 1:])

# %% [markdown]
# #### Form Table of Mistakes of SVM

# %%
SVM_balanced_y_test_df=y_test_df.copy()
SVM_balanced_y_test_df["prediction"] = list(SVM_classifier_balanced_00_predictions)
SVM_balanced_y_test_df.head(2)

# %%
# SVM_None_y_test_df=y_test_df.copy()
# SVM_None_y_test_df["prediction"] = list(SVM_classifier_NoneWeight_00_predictions)
# SVM_None_y_test_df.head(2)

# %% [markdown]
# #### Write down the test result on the disk

# %%
test_result_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/test_results/"
os.makedirs(test_result_dir, exist_ok=True)

# %%
out_name=test_result_dir+ "SG_SVM_balancedWeight_Expert20Percent_test"+VI_idx+".csv"
SVM_balanced_y_test_df.to_csv(out_name, index = False)

# %%
# out_name=test_result_dir+ "SG_SVM_NoneWeight_y_test.csv"
# SVM_None_y_test_df.to_csv(out_name, index = False)

# %% [markdown]
# #### Print the mistakes crop types and plot them

# %%
# SVM_None_y_test_df = pd.merge(SVM_None_y_test_df, meta, on=['ID'], how='left')
SVM_balanced_y_test_df = pd.merge(SVM_balanced_y_test_df, meta, on=['ID'], how='left')

# %%

# %%
SVM_balanced_y_test_df_A2_P1 = SVM_balanced_y_test_df[SVM_balanced_y_test_df.Vote==2]
SVM_balanced_y_test_df_A2_P1 = SVM_balanced_y_test_df_A2_P1[SVM_balanced_y_test_df_A2_P1.prediction==1]

SVM_balanced_y_test_df_A1_P2 = SVM_balanced_y_test_df[SVM_balanced_y_test_df.Vote==1]
SVM_balanced_y_test_df_A1_P2 = SVM_balanced_y_test_df_A1_P2[SVM_balanced_y_test_df_A1_P2.prediction==2]

# %%
sorted(SVM_balanced_y_test_df_A1_P2.CropTyp)

# %%
SVM_balanced_y_test_df_A1_P2

# %% [markdown]
# #### Plot SVM mistakes

# %%
size = 15
title_FontSize = 8
legend_FontSize = 8
tick_FontSize = 12
label_FontSize = 14

params = {'legend.fontsize': 15, # medium, large
          # 'figure.figsize': (6, 4),
          'axes.labelsize': size,
          'axes.titlesize': size*1.2,
          'xtick.labelsize': size, #  * 0.75
          'ytick.labelsize': size, #  * 0.75
          'axes.titlepad': 10}

#
#  Once set, you cannot change them, unless restart the notebook
#
plt.rc('font', family = 'Palatino')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update(params)
# pylab.rcParams.update(params)
# plt.rc('text', usetex=True)

def plot_oneColumn_CropTitle(dt, raw_dt, titlee, _label = "raw", idx="EVI", _color="dodgerblue"):
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 4), sharex=False, sharey='col', # sharex=True, sharey=True,
                           gridspec_kw={'hspace': 0.35, 'wspace': .05});
    ax.grid(True);
    ax.plot(dt['human_system_start_time'], dt[idx], linewidth=4, color=_color, label=_label) 

    ax.scatter(raw_dt['human_system_start_time'], raw_dt[idx], s=20, c="r", label="raw")

    ax.set_title(titlee)
    ax.set_ylabel(idx) # , labelpad=20); # fontsize = label_FontSize,
    ax.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
    ax.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
    ax.legend(loc="upper right");
    plt.yticks(np.arange(0, 1.05, 0.2))
    # ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.set_ylim(-0.1, 1.05)


# %%
test_result_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/test_results/"
os.makedirs(test_result_dir, exist_ok=True)

# %%

# %%
for anID in list(SVM_balanced_y_test_df_A1_P2.ID):
    curr_dt = ground_truth[ground_truth.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]
    
    curr_vote = SVM_balanced_y_test_df_A1_P2[SVM_balanced_y_test_df_A1_P2.ID==anID].Vote.values[0]
    curr_pred = SVM_balanced_y_test_df_A1_P2[SVM_balanced_y_test_df_A1_P2.ID==anID].prediction.values[0]
    
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+\
             "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, 
                                        titlee=title_, _label = "EVI (5-step smoothed)")

    plot_path = test_result_dir + "SG_SVM_balanced_20PercentExpertTest_plots_A1_P2/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')
    
    
for anID in list(SVM_balanced_y_test_df_A2_P1.ID):
    curr_dt = ground_truth[ground_truth.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]    
    
    curr_vote = SVM_balanced_y_test_df_A2_P1[SVM_balanced_y_test_df_A2_P1.ID==anID].Vote.values[0]
    curr_pred = SVM_balanced_y_test_df_A2_P1[SVM_balanced_y_test_df_A2_P1.ID==anID].prediction.values[0]
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+\
             "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, titlee=title_, 
                                        _label = "EVI (5-step smoothed)")
    
    plot_path = test_result_dir + "SG_SVM_balanced_20PercentExpertTest_plots_A2_P1/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')


# %% [markdown]
# # Random Forest

# %%
filename = model_dir + 'SGEVI_forest_grid_1.sav'
forest_grid_1_SG = pickle.load(open(filename, 'rb'))

# %%
forest_grid_1_predictions = forest_grid_1_SG.predict(x_test_df.iloc[:, 1:])
forest_grid_1_y_test_df=y_test_df.copy()
forest_grid_1_y_test_df["prediction"]=list(forest_grid_1_predictions)
forest_grid_1_y_test_df.head(2)

# %%

# %%
true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in forest_grid_1_y_test_df.index:
    curr_vote=list(forest_grid_1_y_test_df[forest_grid_1_y_test_df.index==index_].Vote)[0]
    curr_predict=list(forest_grid_1_y_test_df[forest_grid_1_y_test_df.index==index_].prediction)[0]
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
            
forest_grid_1_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
forest_grid_1_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
forest_grid_1_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
forest_grid_1_confus_tbl_test['Predict_Single']=0
forest_grid_1_confus_tbl_test['Predict_Double']=0

forest_grid_1_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
forest_grid_1_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
forest_grid_1_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
forest_grid_1_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
forest_grid_1_confus_tbl_test

# %%
forest_grid_1_y_test_df = pd.merge(forest_grid_1_y_test_df, meta, on=['ID'], how='left')
out_name=test_result_dir+ "SG_RF_y_test.csv"
forest_grid_1_y_test_df.to_csv(out_name, index = False)

# %%
forest_grid_1_yTest_A1P2 = forest_grid_1_y_test_df[forest_grid_1_y_test_df.Vote==1]
forest_grid_1_yTest_A1P2 = forest_grid_1_yTest_A1P2[forest_grid_1_yTest_A1P2.prediction==2]

forest_grid_1_yTest_A2P1 = forest_grid_1_y_test_df[forest_grid_1_y_test_df.Vote==2]
forest_grid_1_yTest_A2P1 = forest_grid_1_yTest_A2P1[forest_grid_1_yTest_A2P1.prediction==1]

# %%
forest_grid_1_yTest_A2P1.groupby(['CropTyp'])['CropTyp'].count()

# %%
forest_grid_1_yTest_A1P2.groupby(['CropTyp'])['CropTyp'].count()

# %%

# %%
for anID in list(forest_grid_1_yTest_A1P2.ID):
    curr_dt = ground_truth[ground_truth.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]
    
    curr_vote = forest_grid_1_yTest_A1P2[forest_grid_1_yTest_A1P2.ID==anID].Vote.values[0]
    curr_pred = forest_grid_1_yTest_A1P2[forest_grid_1_yTest_A1P2.ID==anID].prediction.values[0]
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+\
              "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, titlee=title_, _label = "EVI (5-step smoothed)")
    
    plot_path = test_result_dir + "SG_RF_plots_A1_P2/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')
    
for anID in list(forest_grid_1_yTest_A2P1.ID):
    curr_dt = ground_truth[ground_truth.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]
    
    curr_vote = forest_grid_1_yTest_A2P1[forest_grid_1_yTest_A2P1.ID==anID].Vote.values[0]
    curr_pred = forest_grid_1_yTest_A2P1[forest_grid_1_yTest_A2P1.ID==anID].prediction.values[0]
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+\
              "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, titlee=title_, _label = "EVI (5-step smoothed)")
    
    plot_path = test_result_dir + "SG_RF_plots_A2_P1/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')

# %% [markdown]
# # Neural Nets. Deep Learning. Transfer Learning
#
# ### Need to complete this
# Choose a probability threshold... etc. 

# %%
NN_result = pd.read_csv(training_set_dir + "01_transfer_learning_result/" + \
                        "01_SG_TL_Expert_testSet_predictions_EVI.csv")

NN_result.rename(columns={"filename": "ID"}, inplace=True)
NN_result.rename(columns={"human_predict": "Vote"}, inplace=True) 

NN_result.replace("double", 2, inplace=True)
NN_result.replace("single", 1, inplace=True)

# %%
ID = [x[0] for x in NN_result.ID.str.split(".")]
NN_result.ID=ID

ID = ["_".join(x[1:]) for x in NN_result.ID.str.split("_")]
NN_result.ID=ID

NN_result.head(2)

# %% tags=["hide-cell", "hide_cell"]
NN_result_A1P2=NN_result[NN_result.Vote==1]
NN_result_A1P2=NN_result_A1P2[NN_result_A1P2.prob_point7==2]

NN_result_A2P1=NN_result[NN_result.Vote==2]
NN_result_A2P1=NN_result_A2P1[NN_result_A2P1.prob_point7==1]

# %%
NN_result_A2P1

# %%
for anID in list(NN_result_A1P2.ID):
    curr_dt = ground_truth[ground_truth.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]
    
    curr_vote = NN_result_A1P2[NN_result_A1P2.ID==anID].Vote.values[0]
    curr_pred = NN_result_A1P2[NN_result_A1P2.ID==anID].prob_point7.values[0]
    
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+\
              "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, 
                                        titlee=title_, _label = "EVI (5-step smoothed)")
    
    plot_path = test_result_dir + "SG_NN_plots_A1_P2/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')

for anID in list(NN_result_A2P1.ID):
    curr_dt = ground_truth[ground_truth.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]
    
    curr_vote = NN_result_A2P1[NN_result_A2P1.ID==anID].Vote.values[0]
    curr_pred = NN_result_A2P1[NN_result_A2P1.ID==anID].prob_point7.values[0]
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+\
              "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, 
                                        titlee=title_, _label = "EVI (5-step smoothed)")
    
    plot_path = test_result_dir + "SG_NN_plots_A2_P1/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')


# %%

# %% [markdown]
# # kNN
# Winner is **uniform** weight.

# %%
def DTW_prune(ts1, ts2):
    d,_ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True);
    return d


# %%
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models/"
filename = model_dir + "00_KNN_SG_EVI_DTW_prune_distanceWeight_4NNisBest.sav"
distanceWeight_KNN = pickle.load(open(filename, 'rb'))

# %%
filename = model_dir + "00_KNN_SG_EVI_DTW_prune_uniformWeight_9NNisBest.sav"
uniform_KNN = pickle.load(open(filename, 'rb'))

# %%
# %%time
KNN_DTW_test_predictions_uniform = uniform_KNN.predict(x_test_df.iloc[:, 1:])
KNN_DTW_test_predictions_distanceWeight = distanceWeight_KNN.predict(x_test_df.iloc[:, 1:])

# %%
KNN_y_test=y_test_df.copy()
KNN_y_test["KNN_pred_uniform"]=list(KNN_DTW_test_predictions_uniform)
KNN_y_test["KNN_pred_distance"]=list(KNN_DTW_test_predictions_distanceWeight)
KNN_y_test.head(2)

# %%
KNN_y_test = pd.merge(KNN_y_test, meta, on=['ID'], how='left')
KNN_y_test.head(2)

# %%
KNN_y_test_uniform_A1P2=KNN_y_test[KNN_y_test.Vote==1]
KNN_y_test_uniform_A1P2=KNN_y_test_uniform_A1P2[KNN_y_test_uniform_A1P2.KNN_pred_uniform==2]

KNN_y_test_uniform_A2P1=KNN_y_test[KNN_y_test.Vote==2]
KNN_y_test_uniform_A2P1=KNN_y_test_uniform_A2P1[KNN_y_test_uniform_A2P1.KNN_pred_uniform==1]

KNN_y_test_uniform_A2P1.ExctAcr.sum()-KNN_y_test_uniform_A1P2.ExctAcr.sum()

# %%
KNN_y_test_dist_A1P2=KNN_y_test[KNN_y_test.Vote==1]
KNN_y_test_dist_A1P2=KNN_y_test_dist_A1P2[KNN_y_test_dist_A1P2.KNN_pred_distance==2]

KNN_y_test_dist_A2P1=KNN_y_test[KNN_y_test.Vote==2]
KNN_y_test_dist_A2P1=KNN_y_test_dist_A2P1[KNN_y_test_dist_A2P1.KNN_pred_distance==1]

KNN_y_test_dist_A2P1.ExctAcr.sum()-KNN_y_test_dist_A1P2.ExctAcr.sum()

# %%
out_name=test_result_dir+ "SG_KNN_y_test.csv"
KNN_y_test.to_csv(out_name, index = False)

# %%
for anID in list(KNN_y_test_uniform_A1P2.ID):
    curr_dt = ground_truth[ground_truth.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]
    
    curr_vote = KNN_y_test_uniform_A1P2[KNN_y_test_uniform_A1P2.ID==anID].Vote.values[0]
    curr_pred = KNN_y_test_uniform_A1P2[KNN_y_test_uniform_A1P2.ID==anID].KNN_pred_uniform.values[0]
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+\
              "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, titlee=title_, _label = "EVI (5-step smoothed)")
    
    plot_path = test_result_dir + "SG_KNN_plots_A1_P2/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')
    
for anID in list(KNN_y_test_uniform_A2P1.ID):
    curr_dt = ground_truth[ground_truth.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]
    
    curr_vote = KNN_y_test_uniform_A2P1[KNN_y_test_uniform_A2P1.ID==anID].Vote.values[0]
    curr_pred = KNN_y_test_uniform_A2P1[KNN_y_test_uniform_A2P1.ID==anID].KNN_pred_uniform.values[0]
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+\
              "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, titlee=title_, _label = "EVI (5-step smoothed)")
    
    plot_path = test_result_dir + "SG_KNN_plots_A2_P1/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')


# %%
anID in list(KNN_y_test_uniform_A1P2.ID)

# %%
KNN_y_test_uniform_A2P1.groupby(['CropTyp'])['CropTyp'].count()

# %%
KNN_y_test_uniform_A1P2.groupby(['CropTyp'])['CropTyp'].count()

# %% [markdown]
# # Are mistakes in Common?

# %%
SG_common_mistakes = y_test_df.copy()

# %%
SVM_balanced_y_test_df_A2_P1.rename(columns={"prediction": "SVM_balanced_pred_A2P1"}, inplace=True)
SVM_balanced_y_test_df_A1_P2.rename(columns={"prediction": "SVM_balanced_pred_A1P2"}, inplace=True)

# %%
SG_common_mistakes=pd.merge(SG_common_mistakes, \
                            SVM_balanced_y_test_df_A2_P1[["ID","SVM_balanced_pred_A2P1"]],on=['ID'],how='left')

SG_common_mistakes=pd.merge(SG_common_mistakes, 
                            SVM_balanced_y_test_df_A1_P2[["ID","SVM_balanced_pred_A1P2"]],on=['ID'],how='left')

# %%
forest_grid_1_yTest_A2P1.rename(columns={"prediction": "RF_G1_pred_A2P1"}, inplace=True)
forest_grid_1_yTest_A1P2.rename(columns={"prediction": "RF_G1_pred_A1P2"}, inplace=True)

# %%
SG_common_mistakes=pd.merge(SG_common_mistakes,\
                            forest_grid_1_yTest_A2P1[["ID","RF_G1_pred_A2P1"]],on=['ID'],how='left')

SG_common_mistakes=pd.merge(SG_common_mistakes,\
                            forest_grid_1_yTest_A1P2[["ID","RF_G1_pred_A1P2"]],on=['ID'],how='left')

# %%
KNN_y_test_uniform_A2P1.rename(columns={"KNN_pred_uniform": "KNN_uniform_pred_A2P1"}, inplace=True)
KNN_y_test_uniform_A1P2.rename(columns={"KNN_pred_uniform": "KNN_uniform_pred_A1P2"}, inplace=True)

# %%
SG_common_mistakes=pd.merge(SG_common_mistakes,\
                            KNN_y_test_uniform_A2P1[["ID","KNN_uniform_pred_A2P1"]], on=['ID'], how='left')

SG_common_mistakes=pd.merge(SG_common_mistakes,\
                            KNN_y_test_uniform_A1P2[["ID","KNN_uniform_pred_A1P2"]], on=['ID'], how='left')

# %%
NN_result_A2P1.rename(columns={"prob_point7": "NN_pred_A2P1"}, inplace=True)
NN_result_A1P2.rename(columns={"prob_point7": "NN_pred_A1P2"}, inplace=True)

# %%
SG_common_mistakes=pd.merge(SG_common_mistakes,\
                            NN_result_A2P1[["ID","NN_pred_A2P1"]], on=['ID'], how='left')

SG_common_mistakes=pd.merge(SG_common_mistakes,\
                            NN_result_A1P2[["ID","NN_pred_A1P2"]], on=['ID'], how='left')

# %%
# thresh int, Require that many non-NA values.
SG_common_mistakes_clean=SG_common_mistakes.dropna(thresh=4)
SG_common_mistakes_clean

# %%
SG_common_mistakes_clean=pd.merge(SG_common_mistakes_clean, meta, on=['ID'], how='left')
SG_common_mistakes_clean

# %%

# %%
vote_columns =['SVM_balanced_pred_A2P1', 'SVM_balanced_pred_A1P2',
               'RF_G1_pred_A2P1', 'RF_G1_pred_A1P2', 
               'KNN_uniform_pred_A2P1', 'KNN_uniform_pred_A1P2',
               'NN_pred_A2P1', 'NN_pred_A1P2']

vote_colMethod =['SVM', 'SVM', 'RF', 'RF', 'kNN', 'kNN', 'NN', 'NN']
               
for anID in list(SG_common_mistakes_clean.ID):
    curr_dt = ground_truth[ground_truth.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]    
    
    curr_mistake = SG_common_mistakes_clean[SG_common_mistakes_clean.ID==anID]
    curr_vote=list(curr_mistake.Vote)[0]
    v = (curr_mistake[vote_columns].notna().iloc[0])*1
    mistakeMethods=[vote_colMethod[i] for i in np.where(v.values == 1)[0]]

    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + \
             " acre), Expert Vote: " + str(curr_vote) + ", mistakes: " + str(mistakeMethods)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, titlee=title_, 
                                        _label = "EVI (5-step smoothed)")
    
    if curr_vote==2:
        plot_path = test_result_dir + "commonMistakes/SG/A2P1/"
    else:
        plot_path = test_result_dir + "commonMistakes/SG/A1P2/"
        
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')

# %%
SG_common_mistakes_clean.sort_values(by=['Vote', 'CropTyp'], inplace=True)

# %%
SG_common_mistakes_clean.iloc[12]

# %%

# %%

# %%

# %%
# training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
# ground_truth_labels = pd.read_csv(training_set_dir+"train_labels.csv")

# expert_test_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/SG_train_images_" + VI_idx + "/test20/"

# test_filenames = os.listdir(expert_test_dir)
# expert_test_df = pd.DataFrame({'filename': test_filenames})
# nb_samples = expert_test_df.shape[0]

# expert_test_df["human_predict"] = expert_test_df.filename.str.split("_", expand=True)[0]
# expert_test_df["prob_single"]=-1.0
# print (expert_test_df.shape)
# expert_test_df.head(2)

# def load_image(filename):
#     # load the image
#     img = load_img(filename, target_size=(224, 224))
#     # convert to array
#     img = img_to_array(img)
#     # reshape into a single sample with 3 channels
#     img = img.reshape(1, 224, 224, 3)
#     # center pixel data
#     img = img.astype('float32')
#     img = img - [123.68, 116.779, 103.939]
#     return img

# # load an image and predict the class
# def run_example():
#     # load the image
#     test_dir = experts_test_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/" + \
#                                   "SG_train_images_" + VI_idx + "/test20/"
    
#     img = load_image(test_dir+'double_1624_WSDA_SF_2016.jpg')
#     # load model
#     model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models/"
#     model = load_model(model_dir + '01_TL_SingleDoubleEVI_SG_train80.h5')
#     result = model.predict(img)
#     print("result[0]: ", result[0])

# # entry point, run the example
# run_example()

# file_name = 'double_101163_WSDA_SF_2017.jpg'
# test_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/" + \
#            "/limitCrops_nonExpert_images_" + VI_idx + "/"
# img = load_image(test_dir+file_name)
# result = model.predict(img)
# print ("probability of being single cropped is {}.".format(result[0]))

# pyplot.subplot(111)
# # define filename
# filename = img
# image = imread(test_dir+file_name)
# pyplot.imshow(image)
# pyplot.show()
