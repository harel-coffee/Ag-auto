# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# This is a copy of ```overSamplePreds_4SummaryStat_2PotensNumerator_AllirrDenom.ipynb``` for Aug 02, 2023 presentation.
#
# ______________________________________________________________________
# The textbelow is from ```overSamplePreds_4SummaryStat_2PotensNumerator_AllirrDenom.ipynb```
#
# ## All irrigated fields in denominator!
#
# Here we will only include double-crop potentials
# that are listed: [here](https://docs.google.com/spreadsheets/d/1ncCDrVImqVFO7bl_YlcbxjG8P_1UT0ls/edit?usp=sharing&ouid=110224975301518346252&rtpof=true&sd=true)

# %%
import numpy as np
import pandas as pd

from datetime import date
import time

import random
from random import seed, random

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

# import pickle #, h5py
import sys, os, os.path
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc

# %%
dir_base = "/Users/hn/Documents/01_research_data/NASA/"
meta_dir = dir_base + "/parameters/"
SF_data_dir   = dir_base + "/data_part_of_shapefile/"
pred_dir_base = dir_base + "/RegionalStatData/"
pred_dir = pred_dir_base + "02_ML_preds_oversampled/"

plot_dir = dir_base + "/for_paper/plots/overSampleRegionalStats/cropWise/seedsLumped/"
os.makedirs(plot_dir, exist_ok=True)

# %% [markdown]
# # Read Fields Metadata

# %%
# meta_6000 = pd.read_csv(meta_dir + "evaluation_set.csv")
# meta_6000_moreThan10Acr = meta_6000[meta_6000.ExctAcr>10]

# print (meta_6000.shape)
# print (meta_6000_moreThan10Acr.shape)
# meta_6000.head(2)

# %%
AnnualPerennialToss = pd.read_csv(meta_dir + "AnnualPerennialTossMay122023.csv")
AnnualPerennialToss.rename(columns={"Crop_Type": "CropTyp"}, inplace=True)
print (f"{AnnualPerennialToss.shape=}")
print (AnnualPerennialToss.potential.unique())

# %%
# badCrops = ["christmas tree", "crp/conservation", "developed", "driving range", "golf course", "green manure",
#             "nursery, caneberry", "nursery, greenhouse", "nursery, lavender", "nursery, orchard/vineyard",
#             "nursery, ornamental", "nursery, silviculture", "reclamation seed", "research station", "unknown"]

badCrops = list(AnnualPerennialToss.loc[AnnualPerennialToss.potential=="toss", "CropTyp"])

# %%
AnnualPerennialToss = AnnualPerennialToss[AnnualPerennialToss.potential!="toss"]
AnnualPerennialToss.reset_index(drop=True, inplace=True)
print (f"{AnnualPerennialToss.shape=}")
print (AnnualPerennialToss.potential.unique())
AnnualPerennialToss.head(2)

# %%
AnnualPerennialToss.potential.unique()

# %%
len(AnnualPerennialToss[AnnualPerennialToss.potential != "n"].CropTyp.unique())

# %%
only_potentialCrops = AnnualPerennialToss[AnnualPerennialToss.potential=="y"]
only_potentialCrops.reset_index(drop=True, inplace=True)
print (f"{only_potentialCrops.shape=}")
print (f"{only_potentialCrops.potential.unique()=}")
only_potentialCrops.head(2)

# %%
f_names=["AdamBenton2016.csv", "FranklinYakima2018.csv", "Grant2017.csv", "Walla2015.csv"]

SF_data=pd.DataFrame()

for file in f_names:
    curr_file=pd.read_csv(SF_data_dir + file)
    SF_data=pd.concat([SF_data, curr_file])

print (f"{SF_data.shape}")
print (f"{SF_data.county.unique()}")

# Toss the crops that we wanted to toss
SF_data = SF_data[~SF_data.CropTyp.isin(badCrops)]

irr_nonIrr_Acr = SF_data['ExctAcr'].sum().round()
SF_data_irr_nonIrr = SF_data.copy()
#
# Toss non-irrigated fields.
#
SF_data = nc.filter_out_nonIrrigated(SF_data)

SF_data = SF_data[SF_data.ExctAcr>10]
SF_data.reset_index(inplace=True, drop=True)

print (f"{SF_data.shape}")

SF_data.head(2)

# %%
# sorted(list(SF_data.CropTyp.unique()))

# %% [markdown]
# ### Crop-wise need to be filtered by last survey date

# %%
### Filter by last survey date 
county_year_dict = {"Adams" : 2016, 
                    "Benton":2016,
                    "Franklin":2018,
                    "Grant": 2017, 
                    "Walla Walla":2015,
                    "Yakima":2018}

SF_data_LSD = pd.DataFrame()
for a_county in county_year_dict.keys():
    curr_county_DF = SF_data[SF_data.county==a_county].copy()
    curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
    SF_data_LSD = pd.concat([SF_data_LSD, curr_county_DF])

SF_data_LSD.reset_index(drop=True, inplace=True)
print (SF_data_LSD.shape)

# %%
SF_data_LSD.ExctAcr.min()

# %%
SF_data_LSD.Irrigtn.unique()

# %%
sorted(list(SF_data_LSD.CropTyp.unique()))[:10]

# %%
SF_data_LSD.head(2)

# %%
annuals = AnnualPerennialToss[AnnualPerennialToss.potential.isin(["y", "yn"])]
SF_data_LSD = SF_data_LSD[SF_data_LSD.CropTyp.isin(list(annuals.CropTyp))]

# %%
SF_data_LSD_large_irr_includeHay = SF_data_LSD.copy()
SF_data_LSD_large_irr_noHay = SF_data_LSD.copy()

# %%
aaa = sorted(list(only_potentialCrops.CropTyp.unique()))
SF_data_LSD_large_irr_noHay = SF_data_LSD_large_irr_noHay[SF_data_LSD_large_irr_noHay.CropTyp.isin(aaa)]

# %%
SF_data.groupby(["county"])['ExctAcr'].sum()

# %%
# to prevent mistakes: usin SF_data from 
# old code since this is a copy of another notebook
all_SF_data = SF_data.copy()
del(SF_data)

# %%
Adams = SF_data_LSD_large_irr_includeHay[SF_data_LSD_large_irr_includeHay.county=="Adams"]
print (Adams.LstSrvD.unique().min())
print (Adams.LstSrvD.unique().max())
sorted(list(SF_data_LSD_large_irr_includeHay.county.unique()))

# %% [markdown]
# # Read predictions
#
# **Predictions are only irrigated**

# %%
# out_name = pred_dir_base + "NDVI_regular_preds_overSample.csv"
# NDVI_regular_preds = pd.read_csv(out_name)

# out_name = pred_dir_base + "EVI_regular_preds_overSample.csv"
# EVI_regular_preds = pd.read_csv(out_name)

# out_name = pred_dir_base + "NDVI_SG_preds_overSample.csv"
# NDVI_SG_preds = pd.read_csv(out_name)

# out_name = pred_dir_base + "EVI_SG_preds_overSample.csv"
# EVI_SG_preds = pd.read_csv(out_name)

# %%
all_preds_overSample = pd.read_csv(pred_dir_base + "all_preds_overSample.csv")
all_preds_overSample = all_preds_overSample[all_preds_overSample.ExctAcr > 10]
# sorted(list(all_preds_overSample.Irrigtn.unique()))

# %% [markdown]
# # Lump together the seeds

# %%
len(all_preds_overSample.CropTyp.unique())

# %%
# sorted(list(all_preds_overSample.CropTyp.unique()))

# %%
seed_idx = all_preds_overSample.loc[all_preds_overSample['CropTyp'].str.contains("seed")].index
all_preds_overSample.loc[seed_idx, "CropTyp"] = "seed crops"
len(all_preds_overSample.CropTyp.unique())

# %%

# %%
preds_LSD_large_irr_noHay = all_preds_overSample[all_preds_overSample.ID.isin(\
                                                                list(SF_data_LSD_large_irr_noHay.ID))].copy()

preds_LSD_large_irr_includeHay = all_preds_overSample[all_preds_overSample.ID.isin(\
                                                                list(SF_data_LSD_large_irr_includeHay.ID))].copy()

# %%
print (f"{all_preds_overSample.shape = }")
print (f"{preds_LSD_large_irr_includeHay.shape = }")
print (f"{preds_LSD_large_irr_noHay.shape = }")

# %%
preds_LSD_large_irr_noHay.head(2)

# %%
sorted(list(preds_LSD_large_irr_includeHay.CropTyp.unique()))[:4]

# %%
preds_LSD_large_irr_noHay.columns

# %%
toss_cols = ["SVM_EVI_regular_preds", "KNN_EVI_regular_preds",
             "DL_EVI_regular_prob_point9", "RF_EVI_regular_preds",
             "SVM_EVI_SG_preds", "KNN_EVI_SG_preds", 
             "DL_EVI_SG_prob_point9", "RF_EVI_SG_preds",
             "SVM_NDVI_regular_preds", "KNN_NDVI_regular_preds",
             "DL_NDVI_regular_prob_point3", "RF_NDVI_regular_preds"]

preds_LSD_large_irr_noHay.drop(labels = toss_cols,
                                axis="columns", inplace=True)

preds_LSD_large_irr_includeHay.drop(labels = toss_cols,
                                             axis="columns", inplace=True)

preds_LSD_large_irr_noHay.reset_index(drop=True, inplace=True)
preds_LSD_large_irr_includeHay.reset_index(drop=True, inplace=True)

# %%
preds_LSD_large_irr_includeHay.head(2)

# %% [markdown]
# # Include Hay
#
# ### Crop-Wise Acreage

# %%
# sorted(list(all_preds_overSample.CropTyp.unique()))

# %%
NDVI_SG_summary_cropAcr = pd.DataFrame(columns=list(preds_LSD_large_irr_includeHay.columns[1:5]))

NDVI_SG_summary_cropAcr[NDVI_SG_summary_cropAcr.columns[0]] = \
    preds_LSD_large_irr_includeHay.groupby([NDVI_SG_summary_cropAcr.columns[0], "CropTyp"])['ExctAcr'].sum()

NDVI_SG_summary_cropAcr[NDVI_SG_summary_cropAcr.columns[1]] = \
            preds_LSD_large_irr_includeHay.groupby([NDVI_SG_summary_cropAcr.columns[1], "CropTyp"])['ExctAcr'].sum()

NDVI_SG_summary_cropAcr[NDVI_SG_summary_cropAcr.columns[2]] = \
            preds_LSD_large_irr_includeHay.groupby([NDVI_SG_summary_cropAcr.columns[2], "CropTyp"])['ExctAcr'].sum()

NDVI_SG_summary_cropAcr[NDVI_SG_summary_cropAcr.columns[3]] = \
            preds_LSD_large_irr_includeHay.groupby([NDVI_SG_summary_cropAcr.columns[3], "CropTyp"])['ExctAcr'].sum()

NDVI_SG_summary_cropAcr.index.rename(['label', 'CropTyp'], inplace=True)
NDVI_SG_summary_cropAcr.round()

# %%
tick_legend_FontSize = 10

params = {'legend.fontsize': tick_legend_FontSize, # medium, large
          # 'figure.figsize': (6, 4),
          'axes.labelsize': tick_legend_FontSize*1.2,
          'axes.titlesize': tick_legend_FontSize*1.3,
          'xtick.labelsize': tick_legend_FontSize, #  * 0.75
          'ytick.labelsize': tick_legend_FontSize, #  * 0.75
          'axes.titlepad': 10}

plt.rc('font', family = 'Palatino')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update(params)

# %%
# "#332288"
color_dict = {"SVM": "#DDCC77",
              "kNN": "#E69F00",
              "RF": "#332288", # "#6699CC",
              "DL":'#0072B2'
             }

# %%

# %%
df = NDVI_SG_summary_cropAcr.copy()
df.reset_index(inplace=True)
df = df[df.label==2]
################################################################
fig, axs = plt.subplots(1, 1, figsize=(12, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

X_axis = np.arange(len(df.CropTyp))

bar_width_ = 1
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_NDVI_SG_preds, color = color_dict["SVM"], width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_NDVI_SG_preds, color =color_dict["kNN"], width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_NDVI_SG_prob_point3, color = color_dict["DL"], width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_NDVI_SG_preds, color = color_dict["RF"], width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, df.CropTyp)

axs.set_ylabel("double-cropped acreage")


# axs.set_xlabel("crop type")
# axs.set_title("5-step NDVI")
# axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')

ymin, ymax = axs.get_ylim()
axs.set(ylim=(ymin-1, ymax+25), axisbelow=True);

file_name = plot_dir + "NDVI_SG_acreage_cropWise_wHays_lumpedSeed.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

file_name = plot_dir + "NDVI_SG_acreage_cropWise_wHays_lumpedSeed.png"
plt.savefig(fname = file_name, dpi=200, bbox_inches='tight', transparent=False);
plt.show()
del(df)

# %%
plot_dir

# %% [markdown]
# # Crop-wise Acreage %
#
# ### Compute total area per county
# This is not true in this notebook: and remember small fields are out on predictions but not ```SF_data```.
#
# 100 fields in total. 10 are potatoes. 1 potato field is double-cropped. 
#
# - Scenario 1: 10% of potatoes are double-cropped among them selves
# - Scenario 2: 1% of potatoes are double-cropped among all fields.
#
# I will go with Scenario 1.

# %%
sorted(list(SF_data_LSD_large_irr_includeHay.CropTyp.unique()))[:4]

# %%
seed_idx = SF_data_LSD_large_irr_includeHay.loc[SF_data_LSD_large_irr_includeHay\
                                                ['CropTyp'].str.contains("seed")].index
SF_data_LSD_large_irr_includeHay.loc[seed_idx, "CropTyp"] = "seed crops"

# %%
SF_data_grp_area = SF_data_LSD_large_irr_includeHay.groupby(['CropTyp'])['ExctAcr'].sum()
SF_data_grp_area = pd.DataFrame(SF_data_grp_area)
SF_data_grp_area.reset_index(drop=False, inplace=True)
SF_data_grp_area.head(3)

# %%

# %%
df = NDVI_SG_summary_cropAcr.copy()
df.reset_index(inplace=True)
df = df[df.label==2]

df = pd.merge(df, SF_data_grp_area, on=(["CropTyp"]), how='left')
df.iloc[:,2:6] = 100 * (df.iloc[:,2:6].div(df.ExctAcr, axis=0))

################################################################
fig, axs = plt.subplots(1, 1, figsize=(12, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

X_axis = np.arange(len(df.CropTyp))

bar_width_ = 1
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_NDVI_SG_preds, color = color_dict["SVM"], width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_NDVI_SG_preds, color = color_dict["kNN"], width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_NDVI_SG_prob_point3, color = color_dict["DL"], width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_NDVI_SG_preds, color = color_dict["RF"], width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, df.CropTyp)

axs.set_ylabel("double-cropped acreage (%)")
# axs.set_xlabel("crop type")
# axs.set_title("5-step NDVI")
# axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')

# send the guidelines to the back
ymin, ymax = axs.get_ylim()
axs.set(ylim=(ymin-1, ymax+25), axisbelow=True);

file_name = plot_dir + "NDVI_SG_acreagePerc_cropWise_wHays_lumpedSeed.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

file_name = plot_dir + "NDVI_SG_acreagePerc_cropWise_wHays_lumpedSeed.png"
plt.savefig(fname = file_name, dpi=200, bbox_inches='tight', transparent=False);

plt.show()
# del(df)

# %%
df[df.CropTyp == "seed crops"]

# %% [markdown]
# # Crop-wise Counts

# %%
NDVI_SG_summary_countPerCrop = pd.DataFrame(columns=list(preds_LSD_large_irr_includeHay.columns[1:5]))
NDVI_SG_summary_countPerCrop[NDVI_SG_summary_countPerCrop.columns[0]] = \
      preds_LSD_large_irr_includeHay.groupby([NDVI_SG_summary_countPerCrop.columns[0], "CropTyp"])['ID'].count()

NDVI_SG_summary_countPerCrop[NDVI_SG_summary_countPerCrop.columns[1]] = \
      preds_LSD_large_irr_includeHay.groupby([NDVI_SG_summary_countPerCrop.columns[1], "CropTyp"])['ID'].count()
    
NDVI_SG_summary_countPerCrop[NDVI_SG_summary_countPerCrop.columns[2]] = \
      preds_LSD_large_irr_includeHay.groupby([NDVI_SG_summary_countPerCrop.columns[2], "CropTyp"])['ID'].count()
    
NDVI_SG_summary_countPerCrop[NDVI_SG_summary_countPerCrop.columns[3]] = \
      preds_LSD_large_irr_includeHay.groupby([NDVI_SG_summary_countPerCrop.columns[3], "CropTyp"])['ID'].count()

NDVI_SG_summary_countPerCrop.index.rename(['label', 'CropTyp'], inplace=True)
NDVI_SG_summary_countPerCrop.round().head(2)

# %% [markdown]
# ## Crop-wise Count %
#
# #### Compute total count per crop
# and remember small fields are out of study.

# %%
# All irrigated fields must be there (orchards, alfalfa category, and 2-potens)

# %%

# %%
SF_data_grp_count = SF_data_LSD_large_irr_includeHay.groupby(['CropTyp'])['ID'].count()
SF_data_grp_count = pd.DataFrame(SF_data_grp_count)
SF_data_grp_count.reset_index(drop=False, inplace=True)
SF_data_grp_count.rename(columns={"ID": "field_count"}, inplace=True)
SF_data_grp_count.head(5)

# %%
df = NDVI_SG_summary_countPerCrop.copy()
df.reset_index(inplace=True)
df = df[df.label==2]
df = pd.merge(df, SF_data_grp_count, on=(["CropTyp"]), how='left')
df.iloc[:,2:6] = 100 * (df.iloc[:,2:6].div(df.field_count, axis=0))

##################################################################
fig, axs = plt.subplots(1, 1, figsize=(12, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

X_axis = np.arange(len(df.CropTyp))

bar_width_ = 1
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_NDVI_SG_preds, color = color_dict["SVM"], width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_NDVI_SG_preds, color = color_dict["kNN"], width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_NDVI_SG_prob_point3, color = color_dict["DL"], width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_NDVI_SG_preds, color = color_dict["RF"], width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, df.CropTyp)

axs.set_ylabel("double-cropped field-count (%)")
# axs.set_xlabel("crop type")
# axs.set_title("5-step NDVI")
# axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')

# send the guidelines to the back
ymin, ymax = axs.get_ylim()
axs.set(ylim=(ymin-1, ymax+25), axisbelow=True);

file_name = plot_dir + "NDVI_SG_countPerc_CropTypWise_wHays_lumpedSeed.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

file_name = plot_dir + "NDVI_SG_countPerc_CropTypWise_wHays_lumpedSeed.png"
plt.savefig(fname = file_name, dpi=200, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%

# %%

# %%
