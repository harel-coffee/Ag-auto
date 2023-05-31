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
# # Plot Mistakes
#
# We had a short meeting with Kirti (Last week of May 2023). Plot all mistakes for result section.
# This notebook is created on May 30, 2023.
#

# %%
import pandas as pd
import numpy as np
import datetime, time, re
from datetime import date

import os, os.path, sys
from os import listdir
from os.path import isfile, join

# from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as npc

# %%
database_dir = "/Users/hn/Documents/01_research_data/NASA/"
ML_data_dir  = database_dir + "ML_data_Oct17/"
pred_dir = database_dir + "/RegionalStatData/"
SG_data_dir = database_dir + "/VI_TS/05_SG_TS/"
landsat_dir = database_dir + "/VI_TS/data_for_train_individual_counties/"
params_dir   = database_dir + "parameters/"
plot_dir_base = database_dir + "for_paper/plots/all_mistake_plot_May302023/"

SF_data_dir   = database_dir + "/data_part_of_shapefile/"

# %%
f_names=["AdamBenton2016.csv", "FranklinYakima2018.csv", "Grant2017.csv", "Walla2015.csv"]
SF_data=pd.DataFrame()

for file in f_names:
    curr_file=pd.read_csv(SF_data_dir + file)
    SF_data=pd.concat([SF_data, curr_file])

print (f"{SF_data.shape}")
print (f"{SF_data.county.unique()}")

# Toss non-irrigated fields.
SF_data = nc.filter_out_nonIrrigated(SF_data)
SF_data.reset_index(inplace=True, drop=True)

print (f"{SF_data.shape}")

SF_data.head(2)

# %%
all_preds = pd.read_csv(pred_dir + "all_preds_overSample.csv")

# %%
# For non-oversample
# NDVI_reg_cols = ["SVM_NDVI_regular_preds", "KNN_NDVI_regular_preds", 
#                  "DL_NDVI_regular_prob_point3", "RF_NDVI_regular_preds"]
# EVI_reg_cols  = ["SVM_EVI_regular_preds",   "KNN_EVI_regular_preds",  
#                  "DL_EVI_regular_prob_point9",  "RF_EVI_regular_preds"]
# EVI_SG_cols   = ["SVM_EVI_SG_preds",             "KNN_EVI_SG_preds",       
#                  "DL_EVI_SG_prob_point9",       "RF_EVI_SG_preds"]

NDVI_SG_cols  = ["SVM_NDVI_SG_preds",      "KNN_NDVI_SG_preds",      
                 "DL_NDVI_SG_prob_point3", "RF_NDVI_SG_preds"]

all_preds = all_preds[["ID"] + NDVI_SG_cols]

all_preds = pd.merge(all_preds, SF_data[["ID", "CropTyp", "Acres", "LstSrvD",
                                         "Irrigtn", "county", "ExctAcr"]], on=['ID'], how='left')

all_preds.head(2)

# %%
AnnualPerennialToss = pd.read_csv(params_dir + "AnnualPerennialTossMay122023.csv")
print (f"{AnnualPerennialToss.shape=}")

AnnualPerennialToss.rename(columns={"Crop_Type": "CropTyp"}, inplace=True)
print (AnnualPerennialToss.potential.unique())
AnnualPerennialToss.head(2)

# %%
toss_crops = list(AnnualPerennialToss[AnnualPerennialToss.potential=="toss"].CropTyp)

# %%
# perennial_crops = ["apple", "apricot", "blueberry", 
#                    "cherry", "caneberry",
#                    "grape, table", "grape, wine", 'grape, unknown', 'grape, wine', "nectarine/peach",
#                    "orchard, unknown", "pear", "plum", "strawberry", "walnut"]

# 'alkali bee bed', 'apple', 'apricot', 'asparagus', 'blueberry',
#        'bluegrass seed', 'caneberry', 'cherry', 'chestnut', 'fescue seed',
#        'filbert', 'grape, juice', 'grape, table', 'grape, unknown',
#        'grape, wine', 'green manure', 'hops', 'mint', 'nectarine/peach',
#        'orchard, unknown', 'pasture', 'pear', 'plum', 'rhubarb',
#        'strawberry', 'walnut', 'wheat fallow'


# %%
perennial_crops = list(AnnualPerennialToss[AnnualPerennialToss.potential=="n"].CropTyp.unique())

# %% [markdown]
# # Parameters

# %%
VI_idx = "NDVI"

# %%
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
landsat_fNames = [x for x in os.listdir(landsat_dir) if x.endswith(".csv")]

landsat_DF = pd.DataFrame()
for fName in landsat_fNames:
    curr = pd.read_csv(landsat_dir+fName)
    curr.dropna(subset=[VI_idx], inplace=True)
    landsat_DF=pd.concat([landsat_DF, curr])

# %%
landsat_DF = landsat_DF[landsat_DF.ID.isin(list(all_preds.ID.unique()))].copy()
landsat_DF = nc.add_human_start_time_by_system_start_time(landsat_DF)
landsat_DF = landsat_DF[["ID", VI_idx, "human_system_start_time"]]
landsat_DF.reset_index(drop=True, inplace=True)
len(landsat_DF.ID.unique())

# %%
landsat_DF.head(2)

# %%
SG_data_4_plot.head(2)

# %%
all_preds = all_preds[all_preds.CropTyp.isin(perennial_crops)]
SF_data = SF_data[SF_data.CropTyp.isin(perennial_crops)]

SG_data_4_plot = SG_data_4_plot[SG_data_4_plot.ID.isin(list(all_preds.ID.unique()))]
landsat_DF = landsat_DF[landsat_DF.ID.isin(list(all_preds.ID.unique()))]

# %% [markdown]
# # Subset perennial crops

# %%
all_preds[all_preds.KNN_NDVI_SG_preds==2].CropTyp.unique()
all_preds.head(2)

# %%

# %%
size = 15
params = {'legend.fontsize': 15, # medium, large
          # 'figure.figsize': (6, 4),
          'axes.labelsize': size,
          'axes.titlesize': size*1.2,
          'xtick.labelsize': size, #  * 0.75
          'ytick.labelsize': size, #  * 0.75
          'axes.titlepad': 10}

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
# all perennials even if out of test set

reg_plot_path = plot_dir_base + "RegionalSummary_perennials/"

for a_col in NDVI_SG_cols:
    print (a_col)
    ML_model = a_col.split("_")[0]
    sub_dir = "_".join(a_col.split("_")[0:3])
    
    curr_preds = all_preds[["ID", a_col]]
    curr_mistakes = curr_preds[curr_preds[a_col]==2]
    
    for an_ID in curr_mistakes.ID.unique():
        curr_smooth = SG_data_4_plot[SG_data_4_plot.ID==an_ID]
        
        curr_raw = landsat_DF[landsat_DF.ID==an_ID]
        # pick proper year.
        curr_yr = int(an_ID.split("_")[-1])
        curr_raw = curr_raw[curr_raw.human_system_start_time.dt.year == curr_yr]
        
        curr_meta = all_preds[all_preds.ID==an_ID]
        curr_crop = list(curr_meta.CropTyp)[0]
        
        if not(curr_crop in toss_crops):
        
            curr_pred = list(curr_mistakes[curr_mistakes.ID==an_ID][a_col])[0]

            title_ = f"{ML_model}: {curr_pred}, {curr_crop}, {an_ID} "

            curr_plt = plot_oneColumn_CropTitle(dt=curr_smooth, raw_dt=curr_raw, titlee=title_,
                                        _label= f"5-step {VI_idx}", idx=VI_idx)

            cropType_4_dir = "".join("_".join(curr_crop.split(",")).split(" "))
            plot_path = reg_plot_path + sub_dir + "/" + cropType_4_dir + "/"
            os.makedirs(plot_path, exist_ok=True)
            fig_name = plot_path + an_ID + '.pdf'
            plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')
            plt.close('all')

# %%
all_preds[(all_preds.CropTyp=="apple") & (all_preds.KNN_NDVI_SG_preds==2)]

# %%
a_col = "KNN_NDVI_SG_preds"
an_ID ="44370_WSDA_SF_2018"
ML_model = a_col.split("_")[0]
sub_dir = "_".join(a_col.split("_")[0:3])

curr_preds = all_preds[["ID", a_col]]
curr_mistakes = curr_preds[curr_preds[a_col]==2]

# %%
curr_smooth = SG_data_4_plot[SG_data_4_plot.ID==an_ID]

curr_raw = landsat_DF[landsat_DF.ID==an_ID]
# pick proper year.
curr_yr = int(an_ID.split("_")[-1])
curr_raw = curr_raw[curr_raw.human_system_start_time.dt.year == curr_yr]

curr_meta = all_preds[all_preds.ID==an_ID]
curr_crop = list(curr_meta.CropTyp)[0]

# %%
