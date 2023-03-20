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

# %% [markdown]
# We had a meeting on Sept. 19 to go through easy fields and add them to the ground truth table.
#
# Kirti wanted to add some plots to the paper.

# %%
import pandas as pd
import numpy as np
import datetime
from datetime import date
import datetime
import time

import os, os.path
from os import listdir
from os.path import isfile, join

import re
# from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys

# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as npc

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
VI_idx="EVI"

landsat_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/data_for_train_individual_counties/"
landsat_fNames = [x for x in os.listdir(landsat_dir) if x.endswith(".csv")]

landsat_DF = pd.DataFrame()
for fName in landsat_fNames:
    curr = pd.read_csv(landsat_dir+fName)
    curr.dropna(subset=[VI_idx], inplace=True)
    landsat_DF=pd.concat([landsat_DF, curr])

landsat_DF.reset_index(drop=True, inplace=True)

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
field_IDs = ["8479_WSDA_SF_2016", 
             "102070_WSDA_SF_2017", 
             "106585_WSDA_SF_2018",
             "103236_WSDA_SF_2018",
             "104485_WSDA_SF_2018",
             "188425_WSDA_SF_2015",
             "108868_WSDA_SF_2017"]

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
for curr_ID in field_IDs:
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 3), sharex=False, sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
    ax1.grid(True)
    curr_SG_data=SG_data_4_plot[SG_data_4_plot.ID==curr_ID].copy()
    
    curr_landsat_DF=landsat_DF[landsat_DF.ID==curr_ID].copy()
    curr_landsat_DF=nc.add_human_start_time_by_system_start_time(curr_landsat_DF)

    curr_landsat_DF.sort_values(by=['human_system_start_time'], inplace=True)
    curr_SG_data.sort_values(by=['human_system_start_time'], inplace=True)

    curr_year=curr_SG_data.human_system_start_time.dt.year.unique()[0]
    curr_landsat_DF=curr_landsat_DF[curr_landsat_DF.human_system_start_time.dt.year==curr_year]

    ax1.plot(curr_SG_data['human_system_start_time'], curr_SG_data[VI_idx], 
            linewidth=4, color="dodgerblue", label="smoothed") 

    ax1.scatter(curr_landsat_DF['human_system_start_time'], 
               curr_landsat_DF[VI_idx], 
               s=20, c="r", label="raw")


    ######## get meta for title
    curr_crop = list(meta_moreThan10Acr[meta_moreThan10Acr.ID==curr_ID].CropTyp)[0]
    titlee=curr_crop

    ax1.set_title(titlee)
    ax1.set_ylabel(VI_idx) # , labelpad=20); # fontsize = label_FontSize,
    ax1.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
    ax1.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
    ax1.legend(loc="upper right");
    ####################################
    plt.yticks(np.arange(0, 1.05, 0.2));
    # ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax1.set_ylim(-0.1, 1.05);

    plot_dir = "/Users/hn/Documents/01_research_data/NASA/for_paper/plots/kirtis_finalSurvey/"
    os.makedirs(plot_dir, exist_ok=True)

    file_name = plot_dir + curr_ID + "_" + VI_idx + ".pdf"
    if curr_ID=="108868_WSDA_SF_2017":
        file_name = plot_dir + curr_ID + "_" + VI_idx + "_Michael.pdf"
        
    plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);
    plt.close('all')


# %%

# %%

# %%
