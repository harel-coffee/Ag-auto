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
from datetime import date
import time
import scipy
import scipy.signal
import os, os.path
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core as rcp

# %% [markdown]
# # Set up Directories

# %%
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"
ML_data_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"

# %% [markdown]
# # Read Train Labels and IDs

# %%
train_labels = pd.read_csv(ML_data_dir + "train_labels.csv")
train_labels.head(2)

# %%
train_labels.shape

# %% [markdown]
# # Toss small fields.

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta=meta[meta.ExctAcr>10]
print (meta.shape)
print (len(train_labels.ID.unique()))
meta.head(2)

# %%
train_labels=train_labels[train_labels.ID.isin(list(meta.ID.unique()))].copy()
train_labels.shape

# %% [markdown]
# # Read TS files

# %%
VI_idx="EVI"
file_names = ["SG_Walla2015_"+VI_idx+"_JFD.csv", "SG_AdamBenton2016_"+VI_idx+"_JFD.csv", 
              "SG_Grant2017_"+VI_idx+"_JFD.csv", "SG_FranklinYakima2018_"+VI_idx+"_JFD.csv"]

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
data.loc[data[VI_idx]<0, VI_idx]=0
data.head(2)

# %% [markdown]
# # Filter the train fields TS

# %%
trainIDs = list(train_labels.ID.unique())
data = data[data.ID.isin(trainIDs)]
data.reset_index(drop=True, inplace=True)

print (len(data.ID.unique()))
len(trainIDs)

# %%
for curr_ID in data.ID.unique():
    crr_fld=data[data.ID==curr_ID].copy()
    crr_fld.reset_index(drop=True, inplace=True)
    # crr_fld['human_system_start_time'] = pd.to_datetime(crr_fld['human_system_start_time'])
    SFYr = crr_fld.human_system_start_time.dt.year.unique()[0]
    fig, ax = plt.subplots();
    fig.set_size_inches(10, 2.5)
    ax.grid(False);
    ax.plot(crr_fld['human_system_start_time'], crr_fld[VI_idx], 
            c ='dodgerblue', linewidth=5)

    ax.axis("off")
    # ax.set_xlabel('time'); # , labelpad = 15
    # ax.set_ylabel(VI_idx, fontsize=12); # , labelpad = 15
    # ax.tick_params(axis = 'y', which = 'major');
    # ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)

    left = crr_fld['human_system_start_time'][0]
    right = crr_fld['human_system_start_time'].values[-1]
    ax.set_xlim([left, right]); # the following line alsow orks
    ax.set_ylim([-0.005, 1]); # the following line alsow orks


    crop_count = train_labels[train_labels.ID==curr_ID]["Vote"].values[0]
    if crop_count==1:
        crop_count_letter="single"
    else:
        crop_count_letter="double"
    
    # train_images is the same as expert labels!
    plot_path = "/Users/hn/Documents/01_research_data/NASA/ML_data/SG_train_images_" + VI_idx + "/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + crop_count_letter + "_" + curr_ID +'.jpg'
    plt.savefig(fname = fig_name, dpi=200, bbox_inches='tight', facecolor="w")
    plt.close('all')
print (plot_path)

# %% [markdown]
# ### Sample EVI plots for Paper:

# %%
samples_for_paper_dir="/Users/hn/Documents/01_research_data/NASA/for_paper/plots/SG_"+VI_idx+"_CleanDP_samples/"

sample_IDs=["489_WSDA_SF_2016", "1746_WSDA_SF_2016", "8856_WSDA_SF_2016", "150461_WSDA_SF_2017"]

for curr_ID in sample_IDs:
    crr_fld=data[data.ID==curr_ID].copy()
    crr_fld.reset_index(drop=True, inplace=True)
    SFYr = crr_fld.human_system_start_time.dt.year.unique()[0]
    fig, ax = plt.subplots();
    fig.set_size_inches(10, 2.5)
    ax.grid(False);
    ax.plot(crr_fld['human_system_start_time'], crr_fld[VI_idx], 
            c ='dodgerblue', linewidth=5)

    ax.axis("off")

    left = crr_fld['human_system_start_time'][0]
    right = crr_fld['human_system_start_time'].values[-1]
    ax.set_xlim([left, right]); # the following line alsow orks
    ax.set_ylim([-0.005, 1]); # the following line alsow orks


    crop_count = train_labels[train_labels.ID==curr_ID]["Vote"].values[0]
    if crop_count==1:
        crop_count_letter="single"
    else:
        crop_count_letter="double"
    
    # train_images is the same as expert labels!
    plot_path = samples_for_paper_dir
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + crop_count_letter + "_" + curr_ID +'.pdf'
    plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight', facecolor="w")
    plt.close('all')
print (plot_path)

# %% [markdown]
# # NDVI

# %%
VI_idx="NDVI"
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
data.loc[data[VI_idx]<0, VI_idx]=0
data.head(2)

# %%
trainIDs = list(train_labels.ID.unique())
data = data[data.ID.isin(trainIDs)]
data.reset_index(drop=True, inplace=True)
print (len(data.ID.unique()))
len(trainIDs)

# %%
for curr_ID in data.ID.unique():
    crr_fld=data[data.ID==curr_ID].copy()
    crr_fld.reset_index(drop=True, inplace=True)
    # crr_fld['human_system_start_time'] = pd.to_datetime(crr_fld['human_system_start_time'])
    SFYr = crr_fld.human_system_start_time.dt.year.unique()[0]
    fig, ax = plt.subplots();
    fig.set_size_inches(10, 2.5)
    ax.grid(False);
    ax.plot(crr_fld['human_system_start_time'], crr_fld[VI_idx], 
            c ='dodgerblue', linewidth=5)

    ax.axis("off")
    left = crr_fld['human_system_start_time'][0]
    right = crr_fld['human_system_start_time'].values[-1]
    ax.set_xlim([left, right]); # the following line alsow orks
    ax.set_ylim([-0.005, 1]); # the following line alsow orks

    crop_count = train_labels[train_labels.ID==curr_ID]["Vote"].values[0]
    if crop_count==1:
        crop_count_letter="single"
    else:
        crop_count_letter="double"
    
    # train_images is the same as expert labels!
    plot_path = "/Users/hn/Documents/01_research_data/NASA/ML_data/SG_train_images_" + VI_idx + "/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + crop_count_letter + "_" + curr_ID +'.jpg'
    plt.savefig(fname = fig_name, dpi=200, bbox_inches='tight', facecolor="w")
    plt.close('all')
    
plot_path

# %% [markdown]
# # Regular EVI

# %%
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/04_regularized_TS/"

# %%
VI_idx = "EVI"
file_names = ["regular_Walla2015_"+VI_idx +"_JFD.csv", "regular_AdamBenton2016_"+VI_idx +"_JFD.csv", 
              "regular_Grant2017_"+VI_idx +"_JFD.csv", "regular_FranklinYakima2018_"+VI_idx +"_JFD.csv"]

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
data.loc[data[VI_idx]<0, VI_idx]=0
data.head(2)

# %%
trainIDs = list(train_labels.ID.unique())
data = data[data.ID.isin(trainIDs)]
data.reset_index(drop=True, inplace=True)
print (len(data.ID.unique()))
len(trainIDs)

# %%
for curr_ID in data.ID.unique():
    crr_fld=data[data.ID==curr_ID].copy()
    crr_fld.reset_index(drop=True, inplace=True)
    # crr_fld['human_system_start_time'] = pd.to_datetime(crr_fld['human_system_start_time'])
    SFYr = crr_fld.human_system_start_time.dt.year.unique()[0]
    fig, ax = plt.subplots();
    fig.set_size_inches(10, 2.5)
    ax.grid(False);
    ax.plot(crr_fld['human_system_start_time'], crr_fld[VI_idx], 
            c ='dodgerblue', linewidth=5)

    ax.axis("off")

    left = crr_fld['human_system_start_time'][0]
    right = crr_fld['human_system_start_time'].values[-1]
    ax.set_xlim([left, right]); # the following line alsow orks
    ax.set_ylim([-0.005, 1]); # the following line alsow orks


    crop_count = train_labels[train_labels.ID==curr_ID]["Vote"].values[0]
    if crop_count==1:
        crop_count_letter="single"
    else:
        crop_count_letter="double"
    
    # train_images is the same as expert labels!
    plot_path = "/Users/hn/Documents/01_research_data/NASA/ML_data/regular_train_images_" + VI_idx + "/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + crop_count_letter + "_" + curr_ID +'.jpg'
    plt.savefig(fname = fig_name, dpi=200, bbox_inches='tight', facecolor="w")
    plt.close('all')
    # ax.legend(loc = "upper left");
print (plot_path)

# %% [markdown]
# # Regular NDVI

# %%
VI_idx="NDVI"
file_names = ["regular_Walla2015_"+VI_idx+"_JFD.csv", "regular_AdamBenton2016_"+VI_idx+"_JFD.csv", 
              "regular_Grant2017_"+VI_idx+"_JFD.csv", "regular_FranklinYakima2018_"+VI_idx+"_JFD.csv"]

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
data.loc[data[VI_idx]<0, VI_idx]=0
data.head(2)

# %%
trainIDs = list(train_labels.ID.unique())
data = data[data.ID.isin(trainIDs)]
data.reset_index(drop=True, inplace=True)
len(trainIDs)

# %%
for curr_ID in data.ID.unique():
    crr_fld=data[data.ID==curr_ID].copy()
    crr_fld.reset_index(drop=True, inplace=True)
    # crr_fld['human_system_start_time'] = pd.to_datetime(crr_fld['human_system_start_time'])
    SFYr = crr_fld.human_system_start_time.dt.year.unique()[0]
    fig, ax = plt.subplots();
    fig.set_size_inches(10, 2.5)
    ax.grid(False);
    ax.plot(crr_fld['human_system_start_time'], crr_fld[VI_idx], 
            c ='dodgerblue', linewidth=5)

    ax.axis("off")
    left = crr_fld['human_system_start_time'][0]
    right = crr_fld['human_system_start_time'].values[-1]
    ax.set_xlim([left, right]); # the following line alsow orks
    ax.set_ylim([-0.005, 1]); # the following line alsow orks

    crop_count = train_labels[train_labels.ID==curr_ID]["Vote"].values[0]
    if crop_count==1:
        crop_count_letter="single"
    else:
        crop_count_letter="double"
    
    # train_images is the same as expert labels!
    plot_path = "/Users/hn/Documents/01_research_data/NASA/ML_data/regular_train_images_" + VI_idx + "/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + crop_count_letter + "_" + curr_ID +'.jpg'
    plt.savefig(fname = fig_name, dpi=200, bbox_inches='tight', facecolor="w")
    plt.close('all')

print(plot_path)

# %%
