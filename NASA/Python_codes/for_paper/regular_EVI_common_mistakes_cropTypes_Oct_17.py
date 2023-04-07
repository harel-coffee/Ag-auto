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
import time# , shutil
import numpy as np
import pandas as pd
from datetime import date

# import random
# from random import seed, random
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report
# import scipy, scipy.signal

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

import pickle #, h5py
import sys, os, os.path

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core.py as rcp

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
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
ground_truth_labels = pd.read_csv(training_set_dir+"groundTruth_labels_Oct17_2022.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))
ground_truth_labels.head(2)

# %%
VI_idx="EVI"

# %% [markdown]
# # Read the Data

# %%
SG_data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"
file_names = ["SG_Walla2015_" + VI_idx + "_JFD.csv", "SG_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              "SG_Grant2017_" + VI_idx + "_JFD.csv", "SG_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

SG_data_4_plot=pd.DataFrame()

for file in file_names:
    curr_file=pd.read_csv(SG_data_dir + file)
    curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])
    
    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years)==3 or len(all_years)==2:
        proper_year = all_years[1]
    elif len(all_years)==1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
    SG_data_4_plot=pd.concat([SG_data_4_plot, curr_file])

SG_data_4_plot.reset_index(drop=True, inplace=True)
SG_data_4_plot.head(2)

# %%
landsat_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/data_for_train_individual_counties/"
landsat_fNames = [x for x in os.listdir(landsat_dir) if x.endswith(".csv")]

landsat_DF = pd.DataFrame()
for fName in landsat_fNames:
    curr = pd.read_csv(landsat_dir+fName)
    curr.dropna(subset=[VI_idx], inplace=True)
    landsat_DF=pd.concat([landsat_DF, curr])

# %%
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/04_regularized_TS/"
file_names = ["regular_Walla2015_" + VI_idx + "_JFD.csv", 
              "regular_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              "regular_Grant2017_" + VI_idx + "_JFD.csv", 
              "regular_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

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
print (f"{data.shape=}")
ground_truth = data[data.ID.isin(list(ground_truth_labels.ID.unique()))].copy()
print (f"{ground_truth.shape=}")

# %% [markdown]
# # Toss small fields

# %%
ground_truth_labels_extended = pd.merge(ground_truth_labels, meta, on=['ID'], how='left')
ground_truth_labels = ground_truth_labels_extended[ground_truth_labels_extended.ExctAcr>=10].copy()
ground_truth_labels.reset_index(drop=True, inplace=True)

print ("There are [{:.0f}] fields in total whose "\
       "area adds up to [{:.2f}].".format(len(ground_truth_labels_extended), \
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
test20_split_2Bconsistent_Oct17 = pd.read_csv(training_set_dir + "test20_split_2Bconsistent_Oct17.csv")
train80_split_2Bconsistent_Oct17 = pd.read_csv(training_set_dir + "train80_split_2Bconsistent_Oct17.csv")

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
test20_split_2Bconsistent_Oct17.head(2)

# %%
ground_truth_wide.head(2)

# %%
# x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(ground_truth_wide, 
#                                                                 ground_truth_labels, 
#                                                                 test_size=0.2, 
#                                                                 random_state=0,
#                                                                 shuffle=True,
#                                                                 stratify=ground_truth_labels.Vote.values)

x_train_df = ground_truth_wide[ground_truth_wide.ID.isin(list(train80_split_2Bconsistent_Oct17.ID))]
x_test_df  = ground_truth_wide[ground_truth_wide.ID.isin(list(test20_split_2Bconsistent_Oct17.ID))]
y_train_df = ground_truth_labels[ground_truth_labels.ID.isin(list(train80_split_2Bconsistent_Oct17.ID))]
y_test_df  = ground_truth_labels[ground_truth_labels.ID.isin(list(test20_split_2Bconsistent_Oct17.ID))]

# %%
landsat_DF = landsat_DF[landsat_DF.ID.isin(list(y_test_df.ID))]
landsat_DF = nc.add_human_start_time_by_system_start_time(landsat_DF)
landsat_DF.reset_index(drop=True, inplace=True)
landsat_DF.head(2)

# %%
y_test_df.head(2)

# %%
""" Throws error. we are trying to unpickle a lower version with a higher version of pickle!
"""
# # !pip3 install scikit-learn==0.19.1 

# %%
# %load_ext autoreload
# %autoreload

# %% [markdown]
# # Read SVM regular From Disk

# %%
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models_Oct17/"

filename = model_dir + 'SVM_classifier_balanced_regular_EVI_01_Oct17.sav'
SVM_classifier_balanced = pickle.load(open(filename, 'rb'))

filename = model_dir + 'SVM_classifier_NoneWeight_regular_EVI_01_Oct17.sav'
SVM_classifier_NoneWeight = pickle.load(open(filename, 'rb'))

# %% [markdown]
# #### Predict SVMs on regular data

# %%
SVM_classifier_NoneWeight_predictions = SVM_classifier_NoneWeight.predict(x_test_df.iloc[:, 1:])
SVM_classifier_balanced_predictions   = SVM_classifier_balanced.predict(x_test_df.iloc[:, 1:])

# %% [markdown]
# #### Form Table of Mistakes of SVM

# %%
SVM_balanced_y_test_df=y_test_df.copy()
SVM_None_y_test_df=y_test_df.copy()
SVM_balanced_y_test_df["prediction"] = list(SVM_classifier_balanced_predictions)
SVM_balanced_y_test_df.head(2)

# %%
SVM_None_y_test_df=y_test_df.copy()
SVM_None_y_test_df=y_test_df.copy()
SVM_None_y_test_df["prediction"] = list(SVM_classifier_NoneWeight_predictions)
SVM_None_y_test_df.head(2)

# %% [markdown]
# #### Write down the test result on the disk

# %%
test_result_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/test_results/"
os.makedirs(test_result_dir, exist_ok=True)

# %%
out_name=test_result_dir+ "regular_SVM_balancedWeight_y_test.csv"
SVM_balanced_y_test_df.to_csv(out_name, index = False)

# %%
out_name=test_result_dir+ "regular_SVM_NoneWeight_y_test.csv"
SVM_None_y_test_df.to_csv(out_name, index = False)

# %%

# %% [markdown]
# #### Print the mistakes crop types and plot them

# %%
SVM_None_y_test_df = pd.merge(SVM_None_y_test_df, meta, on=['ID'], how='left')
SVM_balanced_y_test_df = pd.merge(SVM_balanced_y_test_df, meta, on=['ID'], how='left')

# %%
# balanced_y_test_df_A2_P1 = balanced_y_test_df[balanced_y_test_df.Vote==2]
# balanced_y_test_df_A2_P1 = balanced_y_test_df_A2_P1[balanced_y_test_df_A2_P1.prediction==1]

# balanced_y_test_df_A1_P2 = balanced_y_test_df[balanced_y_test_df.Vote==1]
# balanced_y_test_df_A1_P2 = balanced_y_test_df_A1_P2[balanced_y_test_df_A1_P2.prediction==2]

# %%
SVM_None_y_test_df_A2_P1 = SVM_None_y_test_df[SVM_None_y_test_df.Vote==2]
SVM_None_y_test_df_A2_P1 = SVM_None_y_test_df_A2_P1[SVM_None_y_test_df_A2_P1.prediction==1]

SVM_None_y_test_df_A1_P2 = SVM_None_y_test_df[SVM_None_y_test_df.Vote==1]
SVM_None_y_test_df_A1_P2 = SVM_None_y_test_df_A1_P2[SVM_None_y_test_df_A1_P2.prediction==2]

# %%
SVM_None_y_test_df_A2_P1

# %%
SVM_None_y_test_df_A1_P2

# %%
sorted(SVM_None_y_test_df_A1_P2.CropTyp)

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

# size = 10
# title_FontSize = 5
# tick_legend_FontSize = 10 
# label_FontSize = 14

# params = {'legend.fontsize': tick_legend_FontSize, # medium, large
#           # 'figure.figsize': (6, 4),
#           'axes.labelsize': tick_legend_FontSize*1.2,
#           'axes.titlesize': tick_legend_FontSize*1.5,
#           'xtick.labelsize': tick_legend_FontSize, #  * 0.75
#           'ytick.labelsize': tick_legend_FontSize, #  * 0.75
#           'axes.titlepad': 10}

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
SVM_None_y_test_df_A2_P1.rename(columns={"prediction": "SVM_None_pred_A2P1"}, inplace=True)
SVM_None_y_test_df_A1_P2.rename(columns={"prediction": "SVM_None_pred_A1P2"}, inplace=True)

# %%
test_result_dir

# %%
for anID in list(SVM_None_y_test_df_A1_P2.ID):
    curr_dt = SG_data_4_plot[SG_data_4_plot.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]
        
    curr_vote = SVM_None_y_test_df_A1_P2[SVM_None_y_test_df_A1_P2.ID==anID].Vote.values[0]
    curr_pred = SVM_None_y_test_df_A1_P2[SVM_None_y_test_df_A1_P2.ID==anID].SVM_None_pred_A1P2.values[0]
    
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+\
             "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, 
                                        titlee=title_, _label = "EVI (5-step smoothed)")

    
    plot_path = test_result_dir + "regular_SVM_None_plots_A1_P2/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')
    
for anID in list(SVM_None_y_test_df_A2_P1.ID):
    curr_dt = SG_data_4_plot[SG_data_4_plot.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]    
    
    curr_vote = SVM_None_y_test_df_A2_P1[SVM_None_y_test_df_A2_P1.ID==anID].Vote.values[0]
    curr_pred = SVM_None_y_test_df_A2_P1[SVM_None_y_test_df_A2_P1.ID==anID].SVM_None_pred_A2P1.values[0]
    
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+\
             "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, titlee=title_, 
                                        _label = "EVI (5-step smoothed)")
    
    
    plot_path = test_result_dir + "regular_SVM_None_plots_A2_P1/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')

# %% [markdown]
# # Random Forest

# %%
model_dir

# %%
filename = model_dir + 'regular_EVI_RF1_default_Oct17_accuracyScoring.sav'
regular_forest_default_model = pickle.load(open(filename, 'rb'))

# %%
regular_forest_default_preds = regular_forest_default_model.predict(x_test_df.iloc[:, 1:])
regular_forest_default_y_test_df=y_test_df.copy()
regular_forest_default_y_test_df["prediction"]=list(regular_forest_default_preds)
regular_forest_default_y_test_df.head(2)

# %%

# %%
out_name=test_result_dir+ "regular_RF_default_Oct17_accuracyScoring_y_test.csv"
regular_forest_default_y_test_df.to_csv(out_name, index = False)

# %%
true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in regular_forest_default_y_test_df.index:
    curr_vote=list(regular_forest_default_y_test_df[regular_forest_default_y_test_df.index==index_].Vote)[0]
    curr_predict=list(regular_forest_default_y_test_df[regular_forest_default_y_test_df.index==index_].prediction)[0]
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

# %%

# %%
x_test_df.head(2)

# %%
regular_forest_default_y_test_df = pd.merge(regular_forest_default_y_test_df, meta, on=['ID'], how='left')

# %%
forest_default_yTest_A1P2 = regular_forest_default_y_test_df[regular_forest_default_y_test_df.Vote==1]
forest_default_yTest_A1P2 = forest_default_yTest_A1P2[forest_default_yTest_A1P2.prediction==2]

forest_default_yTest_A2P1 = regular_forest_default_y_test_df[regular_forest_default_y_test_df.Vote==2]
forest_default_yTest_A2P1 = forest_default_yTest_A2P1[forest_default_yTest_A2P1.prediction==1]

# %%
forest_default_yTest_A2P1.groupby(['CropTyp'])['CropTyp'].count()

# %%
forest_default_yTest_A1P2.groupby(['CropTyp'])['CropTyp'].count()

# %%
forest_default_yTest_A2P1.rename(columns={"prediction": "RF_default_pred_A2P1"}, inplace=True)
forest_default_yTest_A1P2.rename(columns={"prediction": "RF_default_pred_A1P2"}, inplace=True)

# %%
forest_default_yTest_A1P2

# %%
for anID in list(forest_default_yTest_A1P2.ID):
    curr_dt = SG_data_4_plot[SG_data_4_plot.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]
    

    curr_vote = forest_default_yTest_A1P2[forest_default_yTest_A1P2.ID==anID].Vote.values[0]
    curr_pred = forest_default_yTest_A1P2[forest_default_yTest_A1P2.ID==anID].RF_default_pred_A1P2.values[0]    
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+\
             "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, titlee=title_, _label = "EVI (5-step smoothed)")
    
    plot_path = test_result_dir + "regular_RF_plots_A1_P2/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    # plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')
    
    
for anID in list(forest_default_yTest_A2P1.ID):
    curr_dt = SG_data_4_plot[SG_data_4_plot.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]
    
    curr_vote = forest_default_yTest_A2P1[forest_default_yTest_A2P1.ID==anID].Vote.values[0]
    curr_pred = forest_default_yTest_A2P1[forest_default_yTest_A2P1.ID==anID].RF_default_pred_A2P1.values[0]    
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+\
             "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)

    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, titlee=title_, _label = "EVI (5-step smoothed)")
    
    plot_path = test_result_dir + "regular_RF_plots_A2_P1/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    # plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')

# %% [markdown]
# # Neural Nets. Deep Learning. Transfer Learning
#
# ### Need to complete this
# Choose a probability threshold... etc. 

# %%
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
import tensorflow as tf
from keras.models import load_model

from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model


# %%
# model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models/"
# model = load_model(model_dir + '01_TL_SingleDoubleEVI_regular_train80.h5')

# %%
# training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
# ground_truth_labels = pd.read_csv(training_set_dir+"train_labels.csv")

# %%
# expert_test_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/regular_train_images_" + VI_idx + "/test20/"

# test_filenames = os.listdir(expert_test_dir)
# expert_test_df = pd.DataFrame({
#                                'filename': test_filenames
#                                     })
# nb_samples = expert_test_df.shape[0]

# expert_test_df["human_predict"] = expert_test_df.filename.str.split("_", expand=True)[0]
# expert_test_df["prob_single"]=-1.0
# print (expert_test_df.shape)
# expert_test_df.head(2)

# %%
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
#                                   "regular_train_images_" + VI_idx + "/test20/"
    
#     img = load_image(test_dir+'double_1624_WSDA_SF_2016.jpg')
#     # load model
#     model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models/"
#     model = load_model(model_dir + '01_TL_SingleDoubleEVI_regular_train80.h5')
#     result = model.predict(img)
#     print("result[0]: ", result[0])

# # entry point, run the example
# run_example()

# %%

# %% [markdown]
# # KNN

# %%
def DTW_prune(ts1, ts2):
    d,_ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True);
    return d


# %%
filename = model_dir + "00_KNN_regular_EVI_DTW_prune_distanceWeight_16NNisBest.sav"
distanceWeight_KNN = pickle.load(open(filename, 'rb'))

# %%
# %%time
KNN_DTW_test_predictions_distanceWeight = distanceWeight_KNN.predict(x_test_df.iloc[:, 1:])

# %%
KNN_y_test=y_test_df.copy()
# KNN_y_test["KNN_pred_uniform"] = list(KNN_DTW_test_predictions_uniform)
KNN_y_test["KNN_pred_distance"] = list(KNN_DTW_test_predictions_distanceWeight)
KNN_y_test.head(2)

# %%
KNN_y_test = pd.merge(KNN_y_test, meta, on=['ID'], how='left')

# %%
KNN_y_test_dist_A1P2=KNN_y_test[KNN_y_test.Vote==1]
KNN_y_test_dist_A1P2=KNN_y_test_dist_A1P2[KNN_y_test_dist_A1P2.KNN_pred_distance==2]

KNN_y_test_dist_A2P1=KNN_y_test[KNN_y_test.Vote==2]
KNN_y_test_dist_A2P1=KNN_y_test_dist_A2P1[KNN_y_test_dist_A2P1.KNN_pred_distance==1]

print (KNN_y_test_dist_A2P1.ExctAcr.sum())
print (KNN_y_test_dist_A1P2.ExctAcr.sum())
abs(KNN_y_test_dist_A2P1.ExctAcr.sum()-KNN_y_test_dist_A1P2.ExctAcr.sum())

# %%
# KNN_y_test_uniform_A1P2=KNN_y_test[KNN_y_test.Vote==1]
# KNN_y_test_uniform_A1P2=KNN_y_test_uniform_A1P2[KNN_y_test_uniform_A1P2.KNN_pred_uniform==2]

# KNN_y_test_uniform_A2P1=KNN_y_test[KNN_y_test.Vote==2]
# KNN_y_test_uniform_A2P1=KNN_y_test_uniform_A2P1[KNN_y_test_uniform_A2P1.KNN_pred_uniform==1]


# print (KNN_y_test_uniform_A2P1.ExctAcr.sum())
# print (KNN_y_test_uniform_A1P2.ExctAcr.sum())

# abs(KNN_y_test_uniform_A2P1.ExctAcr.sum()-KNN_y_test_uniform_A1P2.ExctAcr.sum())

# %%
KNN_y_test_dist_A1P2

# %%
KNN_y_test_dist_A2P1

# %% [markdown]
# # Confusion Tables

# %%
KNN_y_test.head(2)

# %%
# y_test_df_copy=y_test_df.copy()
# y_test_df_copy["weightDist_predictions"] = list(KNN_DTW_prune_weightsDistance_predictions)
# y_test_df_copy.head(2)

####
####   Uniform weights
####

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

actual_double_predicted_single_IDs=[]
actual_single_predicted_double_IDs=[]

for index in KNN_y_test.index:
    curr_vote=list(KNN_y_test[KNN_y_test.index==index].Vote)[0]
    curr_predict=list(KNN_y_test[KNN_y_test.index==index].KNN_pred_uniform)[0]
    if curr_vote==curr_predict:
        if curr_vote==1:
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
            actual_single_predicted_double_IDs+=list(KNN_y_test[KNN_y_test.index==index].ID)
        else:
            true_double_predicted_single+=1
            actual_double_predicted_single_IDs += list(KNN_y_test[KNN_y_test.index==index].ID)

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
# y_test_df_copy=y_test_df.copy()
# y_test_df_copy["weightDist_predictions"] = list(KNN_DTW_prune_weightsDistance_predictions)
# y_test_df_copy.head(2)

####
####   Uniform weights
####

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

actual_double_predicted_single_IDs=[]
actual_single_predicted_double_IDs=[]

for index in KNN_y_test.index:
    curr_vote=list(KNN_y_test[KNN_y_test.index==index].Vote)[0]
    curr_predict=list(KNN_y_test[KNN_y_test.index==index].KNN_pred_distance)[0]
    if curr_vote==curr_predict:
        if curr_vote==1:
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
            actual_single_predicted_double_IDs+=list(KNN_y_test[KNN_y_test.index==index].ID)
        else:
            true_double_predicted_single+=1
            actual_double_predicted_single_IDs += list(KNN_y_test[KNN_y_test.index==index].ID)

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
out_name=test_result_dir+ "regular_KNN_y_test.csv"
# KNN_y_test.to_csv(out_name, index = False)

# %%

# %%
for anID in list(KNN_y_test_uniform_A1P2.ID):
    curr_dt = SG_data_4_plot[SG_data_4_plot.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]
    
    curr_vote = KNN_y_test_uniform_A1P2[KNN_y_test_uniform_A1P2.ID==anID].Vote.values[0]
    curr_pred = KNN_y_test_uniform_A1P2[KNN_y_test_uniform_A1P2.ID==anID].KNN_pred_uniform.values[0]    
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+\
             "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, titlee=title_, 
                                        _label = "EVI (5-step smoothed)")

    plot_path = test_result_dir + "regular_KNN_plots_A1_P2/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    # plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')
    
for anID in list(KNN_y_test_uniform_A2P1.ID):
    curr_dt = SG_data_4_plot[SG_data_4_plot.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]
    
    curr_vote = KNN_y_test_uniform_A2P1[KNN_y_test_uniform_A2P1.ID==anID].Vote.values[0]
    curr_pred = KNN_y_test_uniform_A2P1[KNN_y_test_uniform_A2P1.ID==anID].KNN_pred_uniform.values[0]    
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+\
             "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, titlee=title_, 
                                        _label = "EVI (5-step smoothed)")
    
    plot_path = test_result_dir + "regular_KNN_plots_A2_P1/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    # plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')


# %%
KNN_y_test_uniform_A2P1.groupby(['CropTyp'])['CropTyp'].count()

# %%
KNN_y_test_uniform_A1P2.groupby(['CropTyp'])['CropTyp'].count()

# %%

# %%

# %% [markdown]
# # Are mistakes in Common?

# %%
common_mistakes = y_test_df.copy()

# %%
SVM_None_y_test_df_A2_P1.rename(columns={"prediction": "SVM_None_pred_A2P1"}, inplace=True)
SVM_None_y_test_df_A1_P2.rename(columns={"prediction": "SVM_None_pred_A1P2"}, inplace=True)

# %%
common_mistakes=pd.merge(common_mistakes, SVM_None_y_test_df_A2_P1[["ID","SVM_None_pred_A2P1"]],on=['ID'],how='left')
common_mistakes=pd.merge(common_mistakes, SVM_None_y_test_df_A1_P2[["ID","SVM_None_pred_A1P2"]],on=['ID'],how='left')

# %%
forest_default_yTest_A2P1.rename(columns={"prediction": "RF_default_pred_A2P1"}, inplace=True)
forest_default_yTest_A1P2.rename(columns={"prediction": "RF_default_pred_A1P2"}, inplace=True)

# %%
common_mistakes=pd.merge(common_mistakes,\
                        forest_default_yTest_A2P1[["ID","RF_default_pred_A2P1"]],on=['ID'],how='left')
common_mistakes=pd.merge(common_mistakes,\
                         forest_default_yTest_A1P2[["ID","RF_default_pred_A1P2"]],on=['ID'],how='left')

# %%
KNN_y_test_uniform_A2P1.rename(columns={"KNN_pred_uniform": "KNN_uniform_pred_A2P1"}, inplace=True)
KNN_y_test_uniform_A1P2.rename(columns={"KNN_pred_uniform": "KNN_uniform_pred_A1P2"}, inplace=True)

# %%
common_mistakes=pd.merge(common_mistakes,\
                         KNN_y_test_uniform_A2P1[["ID","KNN_uniform_pred_A2P1"]],on=['ID'],how='left')

common_mistakes=pd.merge(common_mistakes,\
                         KNN_y_test_uniform_A1P2[["ID","KNN_uniform_pred_A1P2"]], on=['ID'],how='left')

# %%
common_mistakes_clean=pd.DataFrame() # = common_mistakes.copy() # common_mistakes.dropna(thresh=4)

# vote_columns =['SVM_None_pred_A2P1', 'SVM_None_pred_A1P2',
#                'RF_default_pred_A2P1', 'RF_default_pred_A1P2', 
#                'KNN_uniform_pred_A2P1', 'KNN_uniform_pred_A1P2']
# for indeks_ in common_mistakes.index:
#     curr_row = common_mistakes[common_mistakes.index==indeks_].copy()
#     if list(curr_row.isnull().sum(axis=1))[0]<5:
#         common_mistakes_clean = pd.concat([common_mistakes_clean, curr_row])

# %%
common_mistakes_clean=common_mistakes.dropna(thresh=4)
common_mistakes_clean

# %%
common_mistakes_clean=pd.merge(common_mistakes_clean, meta, on=['ID'], how='left')
common_mistakes_clean

# %%

# %%
common_mistakes_clean.iloc[5]

# %%
vote_columns =['SVM_None_pred_A2P1', 'SVM_None_pred_A1P2',
               'RF_default_pred_A2P1', 'RF_default_pred_A1P2', 
               'KNN_uniform_pred_A2P1', 'KNN_uniform_pred_A1P2']

vote_colMethod =['SVM', 'SVM', 'RF', 'RF', 'kNN', 'kNN']
               
for anID in list(common_mistakes_clean.ID):
    curr_dt = SG_data_4_plot[SG_data_4_plot.ID==anID].copy()
    curr_meta = meta[meta.ID==anID].copy()
    
    curr_year=curr_dt.human_system_start_time.dt.year.unique()[0]
    curr_raw = landsat_DF[landsat_DF.ID==anID].copy()
    curr_raw=curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]    
    
    curr_mistake = common_mistakes_clean[common_mistakes_clean.ID==anID]
    curr_vote=list(curr_mistake.Vote)[0]
    v = (curr_mistake[vote_columns].notna().iloc[0])*1
    mistakeMethods=[vote_colMethod[i] for i in np.where(v.values == 1)[0]]

    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + \
             " acre), Expert Vote: " + str(curr_vote) + ", mistakes: " + str(mistakeMethods)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_dt, raw_dt=curr_raw, titlee=title_, 
                                        _label = "EVI (5-step smoothed)")
    
    if curr_vote==2:
        plot_path = test_result_dir + "commonMistakes/regular/A2P1/"
    else:
        plot_path = test_result_dir + "commonMistakes/regular/A1P2/"
        
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + anID + '.pdf'
    # plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')

# %%
