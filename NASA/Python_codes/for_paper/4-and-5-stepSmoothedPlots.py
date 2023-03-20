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
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")

training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data/"
ground_truth_labels = pd.read_csv(training_set_dir+"train_labels.csv")

# %% [markdown]
# # Read SG data

# %%
VI_idx = "EVI"

# %%
SG_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"
file_names = ["SG_Walla2015_" + VI_idx + "_JFD.csv", "SG_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              "SG_Grant2017_" + VI_idx + "_JFD.csv", "SG_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

SG_data=pd.DataFrame()

for file in file_names:
    curr_file=pd.read_csv(SG_dir + file)
    curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])
    
    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years)==3 or len(all_years)==2:
        proper_year = all_years[1]
    elif len(all_years)==1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
    SG_data=pd.concat([SG_data, curr_file])

SG_data.reset_index(drop=True, inplace=True)
SG_data.head(2)

# %% [markdown]
# # Read Raw Data

# %%
landsat_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/data_for_train_individual_counties/"

landsat_fNames = [x for x in os.listdir(landsat_dir) if x.endswith(".csv")]

landsat_DF = pd.DataFrame()
for fName in landsat_fNames:
    curr = pd.read_csv(landsat_dir+fName)
    curr.dropna(subset=[VI_idx], inplace=True)
    landsat_DF=pd.concat([landsat_DF, curr])

# %% [markdown]
# # Read Regular data

# %%
regular_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/04_regularized_TS/"
file_names = ["regular_Walla2015_" + VI_idx + "_JFD.csv", 
              "regular_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              "regular_Grant2017_" + VI_idx + "_JFD.csv", 
              "regular_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

regular_data=pd.DataFrame()

for file in file_names:
    curr_file=pd.read_csv(regular_dir + file)
    curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])
    
    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years)==3 or len(all_years)==2:
        proper_year = all_years[1]
    elif len(all_years)==1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
    regular_data=pd.concat([regular_data, curr_file])

regular_data.reset_index(drop=True, inplace=True)
regular_data.head(2)


# %%

# %%
def plot_45step(SG_dt, regular_dt, raw_dt, titlee, idx="EVI"):
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 4), sharex=False, sharey='col', # sharex=True, sharey=True,
                           gridspec_kw={'hspace': 0.35, 'wspace': .05});
    ax.grid(True);
    ax.plot(SG_dt['human_system_start_time'], SG_dt[idx], linewidth=4, \
            color="dodgerblue", label="5-step smooth",
           linestyle="solid") 

    ax.plot(regular_dt['human_system_start_time'], regular_dt[idx], linewidth=4, \
            color="#2ca02c", label="4-step smooth",
            linestyle="dotted") 

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
an_ID = "106509_WSDA_SF_2017"
a_meta = meta[meta.ID==an_ID]

a_SG = SG_data[SG_data.ID==an_ID].copy()
a_regular = regular_data[regular_data.ID==an_ID].copy()

a_landsat = landsat_DF[landsat_DF.ID==an_ID].copy()
a_landsat = nc.add_human_start_time_by_system_start_time(a_landsat)
a_landsat.dropna(subset=[VI_idx], inplace=True)

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
curr_year=a_SG.human_system_start_time.dt.year.unique()[0]

a_landsat=a_landsat[a_landsat.human_system_start_time.dt.year==curr_year]

curr_vote = ground_truth_labels[ground_truth_labels.ID==an_ID]
curr_vote = list(curr_vote.Vote)[0]

title_ = list(a_meta.CropTyp)[0] + " (" + \
         str(list(a_meta.Acres)[0]) + " acre), "+ \
         "Experts' vote: " + str(curr_vote)

curr_plt = plot_45step(SG_dt=a_SG, regular_dt=a_regular, raw_dt=a_landsat, titlee=title_, idx="EVI")


test_result_dir = "/Users/hn/Documents/01_research_data/NASA/for_paper/plots/"
os.makedirs(test_result_dir, exist_ok=True)

plot_path = test_result_dir 
os.makedirs(plot_path, exist_ok=True)
fig_name = plot_path + an_ID + '_45_smooth.pdf'
# plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
# plt.close('all')

# %%

# %%

# %%
