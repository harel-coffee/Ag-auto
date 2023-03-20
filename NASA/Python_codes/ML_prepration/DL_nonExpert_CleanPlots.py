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
# # Toss small fields.

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
print (meta.shape)
meta=meta[meta.ExctAcr>10]
print (meta.shape)
meta.head(2)

# %%
nonExpert_V = meta_dir + "nonExpert_2605_votes.csv"
nonExpert_V=pd.read_csv(nonExpert_2605_votes)
nonExpert_V.shape

# %% [markdown]
# # Read TS files

# %%
VI_idx="EVI"

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

# %% [markdown]
# # Filter the non-expert fields TS

# %%
nonExpert_V.head(2)

# %%
print (len(data.ID.unique()))
nonExpert_IDs = list(nonExpert_V.ID.unique())
data = data[data.ID.isin(nonExpert_IDs)]
data.reset_index(drop=True, inplace=True)
print (len(data.ID.unique()))

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


    crop_count = nonExpert_V[nonExpert_V.ID==curr_ID]["Vote"].values[0]
    if crop_count==1:
        crop_count_letter="single"
    else:
        crop_count_letter="double"
    
    # nonExpert_V is the same as expert labels!
    plot_path = "/Users/hn/Documents/01_research_data/NASA/ML_data/SG_nonExpert_" + VI_idx + "/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + crop_count_letter + "_" + curr_ID +'.jpg'
    plt.savefig(fname = fig_name, dpi=200, bbox_inches='tight', facecolor="w")
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

nonExpert_V_IDs = list(nonExpert_V.ID.unique())
data = data[data.ID.isin(nonExpert_V_IDs)]
data.reset_index(drop=True, inplace=True)
print (len(data.ID.unique()))
data.head(2)

# %%
for curr_ID in data.ID.unique():
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

    crop_count = nonExpert_V[nonExpert_V.ID==curr_ID]["Vote"].values[0]
    if crop_count==1:
        crop_count_letter="single"
    else:
        crop_count_letter="double"
    
    # nonExpert_V is the same as expert labels!
    plot_path = "/Users/hn/Documents/01_research_data/NASA/ML_data/SG_nonExpert_" + VI_idx + "/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + crop_count_letter + "_" + curr_ID +'.jpg'
    plt.savefig(fname = fig_name, dpi=200, bbox_inches='tight', facecolor="w")
    plt.close('all')
    # ax.legend(loc = "upper left");
    
plot_path

# %% [markdown]
# # Regular EVI

# %%
VI_idx="EVI"

regular_data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/04_regularized_TS/"
file_names = ["regular_Walla2015_" + VI_idx +"_JFD.csv", "regular_AdamBenton2016_" + VI_idx +"_JFD.csv", 
              "regular_Grant2017_" + VI_idx +"_JFD.csv", "regular_FranklinYakima2018_" + VI_idx +"_JFD.csv"]

data=pd.DataFrame()

for file in file_names:
    curr_file=pd.read_csv(regular_data_dir + file)
    curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])
    
    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years)==3 or len(all_years)==2:
        proper_year=all_years[1]
    elif len(all_years)==1:
        proper_year=all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
    data=pd.concat([data, curr_file])

data.reset_index(drop=True, inplace=True)
data.loc[data[VI_idx]<0, VI_idx]=0

nonExpert_V_IDs = list(nonExpert_V.ID.unique())
data = data[data.ID.isin(nonExpert_V_IDs)]
data.reset_index(drop=True, inplace=True)

print (len(data.ID.unique()))
data.head(2)

# %%
for curr_ID in data.ID.unique():
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

    crop_count = nonExpert_V[nonExpert_V.ID==curr_ID]["Vote"].values[0]
    if crop_count==1:
        crop_count_letter="single"
    else:
        crop_count_letter="double"
    
    # nonExpert_V is the same as expert labels!
    plot_path = "/Users/hn/Documents/01_research_data/NASA/ML_data/regular_nonExpert_" + VI_idx + "/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + crop_count_letter + "_" + curr_ID +'.jpg'
    plt.savefig(fname = fig_name, dpi=200, bbox_inches='tight', facecolor="w")
    plt.close('all')
    # ax.legend(loc = "upper left");
    
plot_path

# %% [markdown]
# # Regular NDVI

# %%
VI_idx="NDVI"

regular_data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/04_regularized_TS/"
file_names = ["regular_Walla2015_"+VI_idx +"_JFD.csv", "regular_AdamBenton2016_"+VI_idx +"_JFD.csv", 
              "regular_Grant2017_"+VI_idx +"_JFD.csv", "regular_FranklinYakima2018_"+VI_idx +"_JFD.csv"]

data=pd.DataFrame()

for file in file_names:
    curr_file=pd.read_csv(regular_data_dir + file)
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

nonExpert_V_IDs = list(nonExpert_V.ID.unique())
data = data[data.ID.isin(nonExpert_V_IDs)]
data.reset_index(drop=True, inplace=True)

print (len(data.ID.unique()))
data.head(2)

# %%
for curr_ID in data.ID.unique():
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

    crop_count = nonExpert_V[nonExpert_V.ID==curr_ID]["Vote"].values[0]
    if crop_count==1:
        crop_count_letter="single"
    else:
        crop_count_letter="double"
    
    # nonExpert_V is the same as expert labels!
    plot_path = "/Users/hn/Documents/01_research_data/NASA/ML_data/regular_nonExpert_" + VI_idx + "/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + crop_count_letter + "_" + curr_ID +'.jpg'
    plt.savefig(fname = fig_name, dpi=200, bbox_inches='tight', facecolor="w")
    plt.close('all')

plot_path

# %%
