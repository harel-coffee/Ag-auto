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

# %% [markdown]
# This notebook is bulit by copying the EVI version. 
#
# But that does not matter because 
#    - ```NDVI``` works better than ```EVI```. So, we will go with this version. 
#    - In this notebook we will read predictions that are already pre-computed using best models

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
dir_base = "/Users/hn/Documents/01_research_data/NASA/"
meta_dir = dir_base + "/parameters/"
SF_data_dir = dir_base + "/data_part_of_shapefile/"
pred_dir_base = dir_base + "/RegionalStatData/"
# pred_dir = pred_dir_base + "02_ML_preds/"

# %%

# %%
meta_6000 = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_6000_moreThan10Acr=meta_6000[meta_6000.ExctAcr>10]

print (meta_6000.shape)
print (meta_6000_moreThan10Acr.shape)
meta_6000.head(2)

# %%
out_name = SF_data_dir + "all_SF_data_concatenated.csv"
all_SF_data = pd.read_csv(SF_data_dir + "all_SF_data_concatenated.csv")
all_SF_data.head(2)

# %% [markdown]
# # Read groundTruth Set Labels

# %%
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
GT_labels = pd.read_csv(training_set_dir+"groundTruth_labels_Oct17_2022.csv")
print ("Unique Votes: ", GT_labels.Vote.unique())
print (len(GT_labels.ID.unique()))
GT_labels.head(2)

# %%
all_preds = pd.read_csv(pred_dir_base + "all_preds.csv")
print(f'{all_preds.shape}')
all_preds.head(2)

# %% [markdown]
# ## Read test set and then figure which ones are labeled by experts

# %%
test20 = pd.read_csv(training_set_dir+"test20_split_2Bconsistent_Oct17.csv")
print(f'{test20.shape}')
test20.head(2)

# %%
train_dir_preOct = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
test20_expertLabels_2Bconsistent = pd.read_csv(train_dir_preOct + "test20_split_expertLabels_2Bconsistent.csv")
train80_expertLabels_2Bconsistent = pd.read_csv(train_dir_preOct + "train80_split_expertLabels_2Bconsistent.csv")
expertLabels = pd.concat([test20_expertLabels_2Bconsistent, train80_expertLabels_2Bconsistent])
expertLabels.head(2)

# %%
mistake_searchSpace = test20[test20.ID.isin(list(expertLabels.ID))]
mistake_searchSpace.reset_index(drop=True, inplace=True)
mistake_searchSpace = pd.merge(mistake_searchSpace, meta_6000, on=['ID'], how='left')

print (f"{mistake_searchSpace.shape=}")
print (f"{mistake_searchSpace.ExctAcr.min()=}")
mistake_searchSpace.head(2)

# %%

# %%
mistake_space_preds = all_preds[all_preds.ID.isin(list(mistake_searchSpace.ID))]
mistake_space_preds.reset_index(drop=True, inplace=True)
print (mistake_space_preds.shape)
mistake_space_preds.head(2)

# %%

# %%
regular_preds = mistake_space_preds[["ID", 
                                     "SVM_NDVI_regular_preds", "KNN_NDVI_regular_preds", 
                                     "DL_NDVI_regular_prob_point9", "RF_NDVI_regular_preds"]].copy()

SG_preds = mistake_space_preds[["ID", 
                                "SVM_NDVI_SG_preds", "KNN_NDVI_SG_preds", 
                                "DL_NDVI_SG_prob_point9",  "RF_NDVI_SG_preds"]].copy()

SG_preds.head(2)

# %% [markdown]
# # Read the TS Data

# %%
VI_idx = "NDVI"

SG_data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"
file_names = ["SG_Walla2015_" + VI_idx + "_JFD.csv", "SG_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              "SG_Grant2017_" + VI_idx + "_JFD.csv", "SG_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

SG_TS=pd.DataFrame()

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
    SG_TS = pd.concat([SG_TS, curr_file])

SG_TS.reset_index(drop=True, inplace=True)
SG_TS.head(2)

# %%
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/04_regularized_TS/"
file_names = ["regular_Walla2015_" + VI_idx + "_JFD.csv", 
              "regular_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              "regular_Grant2017_" + VI_idx + "_JFD.csv", 
              "regular_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

regular_TS = pd.DataFrame()

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
    regular_TS=pd.concat([regular_TS, curr_file])

regular_TS.reset_index(drop=True, inplace=True)
regular_TS.head(2)

# %%
landsat_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/data_for_train_individual_counties/"
landsat_fNames = [x for x in os.listdir(landsat_dir) if x.endswith(".csv")]

landsat_raw = pd.DataFrame()
for fName in landsat_fNames:
    curr = pd.read_csv(landsat_dir+fName)
    curr.dropna(subset=[VI_idx], inplace=True)
    landsat_raw=pd.concat([landsat_raw, curr])
    
landsat_raw.head(2)

# %% [markdown]
# ### Subset the TSs to mistake search space

# %%
SG_TS = SG_TS[SG_TS.ID.isin(list(mistake_searchSpace.ID))]
regular_TS = regular_TS[regular_TS.ID.isin(list(mistake_searchSpace.ID))]
landsat_raw = landsat_raw[landsat_raw.ID.isin(list(mistake_searchSpace.ID))]

SG_TS.reset_index(drop=True, inplace=True)
regular_TS.reset_index(drop=True, inplace=True)
landsat_raw.reset_index(drop=True, inplace=True)

landsat_raw = nc.add_human_start_time_by_system_start_time(landsat_raw)

print (f"{SG_TS.shape=}")
print (f"{regular_TS.shape=}")
print (f"{landsat_raw.shape=}")

# %%

# %% [markdown]
# # Are mistakes in Common?

# %%
print (mistake_searchSpace.shape)
print (expertLabels.shape)

# %%
SG_preds = pd.merge(SG_preds, expertLabels, on=['ID'],how='left')
regular_preds = pd.merge(regular_preds, expertLabels, on=['ID'],how='left')

# %% [markdown]
# #### See where all predictions agree.

# %%
regular_preds['matching_preds'] = regular_preds.apply(lambda x: x.SVM_NDVI_regular_preds == \
                                                                x.KNN_NDVI_regular_preds == \
                                                                x.DL_NDVI_regular_prob_point9 == \
                                                                x.RF_NDVI_regular_preds, \
                                                                 axis=1)

SG_preds['matching_preds'] = SG_preds.apply(lambda x: x.SVM_NDVI_SG_preds == \
                                                      x.KNN_NDVI_SG_preds == \
                                                      x.DL_NDVI_SG_prob_point9 == \
                                                      x.RF_NDVI_SG_preds, \
                                                      axis=1)

# %%
SG_preds.head(2)

# %%
regular_preds = regular_preds[regular_preds.matching_preds==True]
SG_preds = SG_preds[SG_preds.matching_preds==True]

# %%
SG_preds[~(SG_preds.RF_NDVI_SG_preds == SG_preds.Vote)]

# %%
print (regular_preds.shape)
print (regular_preds[~(regular_preds.RF_NDVI_regular_preds == regular_preds.Vote)].shape)

# %%
reg_common_mistake_fields = regular_preds[~(regular_preds.RF_NDVI_regular_preds == \
                                                 regular_preds.Vote)]

reg_common_mistake_fields = reg_common_mistake_fields.ID.unique()
reg_common_mistake_fields

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


# %%
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
plot_dir = "/Users/hn/Documents/01_research_data/NASA/for_paper/plots/common_mistakes/"
os.makedirs(plot_dir, exist_ok=True)

# %%
for anID in list(reg_common_mistake_fields):
    curr_smooth = regular_TS[regular_TS.ID == anID]
    
    curr_raw = landsat_raw[landsat_raw.ID == anID]
    curr_year = curr_smooth.human_system_start_time.dt.year.unique()[0]
    curr_raw = curr_raw[curr_raw.human_system_start_time.dt.year==curr_year]
    
    curr_meta = meta_6000[meta_6000.ID==anID].copy()
    curr_vote = regular_preds[regular_preds.ID==anID].Vote.values[0]
    curr_pred =  regular_preds[regular_preds.ID==anID].RF_NDVI_regular_preds.values[0]
    title_ = list(curr_meta.CropTyp)[0] + " (" + str(list(curr_meta.Acres)[0]) + " acre), "+ \
            "Experts' vote: " + str(curr_vote) + ", prediction: " + str(curr_pred)
    
    curr_plt = plot_oneColumn_CropTitle(dt=curr_smooth, raw_dt=curr_raw, idx="NDVI",
                                        titlee=title_, _label = VI_idx + " (4-step smoothed)")

    final_plot_path = plot_dir + "NDVI_regular_expert/"
    os.makedirs(final_plot_path, exist_ok=True)
    fig_name = final_plot_path + anID + '_reg_Expert.pdf'
    plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')

# %%
anID

# %%

# %%
final_plot_path

# %%
