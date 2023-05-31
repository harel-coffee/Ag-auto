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

plot_dir = dir_base + "/for_paper/plots/overSampleRegionalStats/Perennials_MistakeStats/"
os.makedirs(plot_dir, exist_ok=True)

# %% [markdown]
# # Read Fields Metadata

# %%
AnnualPerennialToss = pd.read_csv(meta_dir + "AnnualPerennialTossMay122023.csv")
print (f"{AnnualPerennialToss.shape=}")
print (AnnualPerennialToss.potential.unique())

# %%
badCrops = list(AnnualPerennialToss.loc[AnnualPerennialToss.potential=="toss", "Crop_Type"])
annuals_CropType_df = AnnualPerennialToss[AnnualPerennialToss.potential=="n"]
annuals_Crops = list(annuals_CropType_df.Crop_Type.unique())

# %%
# annuals_Crops = ['alkali bee bed', 'apple', 'apricot',
#                  'asparagus', 'blueberry', 'bluegrass seed',
#                  'caneberry', 'cherry', 'chestnut', 'fescue seed',
#                  'filbert', 'grape, juice', 'grape, table', 'grape, unknown',
#                  'grape, wine', 'green manure', 'hops', 'mint', 'nectarine/peach',
#                  'orchard, unknown', 'pasture', 'pear', 'plum', 'rhubarb', 'strawberry', 'walnut', 
#                  'wheat fallow']

# %%
len(annuals_Crops)

# %%

# %%
AnnualPerennialToss = AnnualPerennialToss[AnnualPerennialToss.potential!="toss"]
AnnualPerennialToss.reset_index(drop=True, inplace=True)
print (f"{AnnualPerennialToss.shape=}")
print (AnnualPerennialToss.potential.unique())
AnnualPerennialToss.head(2)

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
# Toss non-irrigated fields.
SF_data = nc.filter_out_nonIrrigated(SF_data)
SF_data.reset_index(inplace=True, drop=True)

print (f"{SF_data.shape}")

SF_data.head(2)

# %%
print (f"{irr_nonIrr_Acr=}")
print (f"{SF_data['ExctAcr'].sum()=}")

# %%
SF_data.groupby(["county"])['ExctAcr'].sum()

# %%
# SF_data_irr_nonIrr
perennial_crops = ["apple", "apricot", "blueberry", 
                   "cherry", "caneberry",
                   "grape, table", "grape, wine", 'grape, unknown', 'grape, wine', "nectarine/peach",
                   "orchard, unknown", "pear", "plum", "strawberry", "walnut"]

SF_data_nonPotential = SF_data_irr_nonIrr[SF_data_irr_nonIrr.CropTyp.isin(perennial_crops)].copy()
print (f"{SF_data_nonPotential['ExctAcr'].sum()=:.3f}")
SF_data_nonPotential.Irrigtn.unique()

# %%
Adams = SF_data[SF_data.county=="Adams"]
print (Adams.LstSrvD.unique().min())
print (Adams.LstSrvD.unique().max())
sorted(list(SF_data.county.unique()))

# %%
print (f"{len(SF_data.CropTyp.unique())=}")
print (f"{len(SF_data.Irrigtn.unique())=}")
# sorted(list(SF_data.CropTyp.unique()))

# %%
print (f"{SF_data.ExctAcr.sum().round()=}")
print (f"{nc.filter_out_nonIrrigated(SF_data).ExctAcr.sum().round()=}")

# %%
print (f"{len(SF_data[SF_data.ExctAcr>10].CropTyp.unique())=}")
print (f"{len(SF_data[SF_data.ExctAcr>10].Irrigtn.unique())=}")
print (f"{SF_data[SF_data.ExctAcr>10].ExctAcr.sum()=}")
# sorted(list(SF_data_irr.CropTyp.unique()))

# %% [markdown]
# # Read predictions
#
# **Predictions are only irrigated**

# %%
out_name = pred_dir_base + "NDVI_regular_preds_overSample.csv"
NDVI_regular_preds = pd.read_csv(out_name)

out_name = pred_dir_base + "EVI_regular_preds_overSample.csv"
EVI_regular_preds = pd.read_csv(out_name)

out_name = pred_dir_base + "NDVI_SG_preds_overSample.csv"
NDVI_SG_preds = pd.read_csv(out_name)

out_name = pred_dir_base + "EVI_SG_preds_overSample.csv"
EVI_SG_preds = pd.read_csv(out_name)

# %%
annuals_Crops[:4]

# %% [markdown]
# # Toss bad crops, non-irrigated, and filter large fields in predictions but not in denominator.
#
# (**Predictions are only irrigated**)
#
# Clean from this point on. These were not dropped in non-overSample

# %%
EVI_SG_preds = EVI_SG_preds[EVI_SG_preds.CropTyp.isin(annuals_Crops)]
EVI_regular_preds = EVI_regular_preds[EVI_regular_preds.CropTyp.isin(annuals_Crops)]

NDVI_SG_preds = NDVI_SG_preds[NDVI_SG_preds.CropTyp.isin(annuals_Crops)]
NDVI_regular_preds = NDVI_regular_preds[NDVI_regular_preds.CropTyp.isin(annuals_Crops)]

print (f"{len(EVI_SG_preds.CropTyp.unique())=}")
print (f"{len(EVI_regular_preds.CropTyp.unique())=}")
print (f"{len(NDVI_SG_preds.CropTyp.unique())=}")
print (f"{len(NDVI_regular_preds.CropTyp.unique())=}")

EVI_SG_preds = EVI_SG_preds[EVI_SG_preds.ExctAcr>10]
EVI_regular_preds = EVI_regular_preds[EVI_regular_preds.ExctAcr>10]

NDVI_SG_preds = NDVI_SG_preds[NDVI_SG_preds.ExctAcr>10]
NDVI_regular_preds = NDVI_regular_preds[NDVI_regular_preds.ExctAcr>10]

# SF_data = SF_data[SF_data.CropTyp.isin(potentialCrops)]
# SF_data = SF_data[SF_data.ExctAcr>10]
SF_data.reset_index(inplace=True, drop=True)
SF_data.head(2)

# %%
# sorted(list(SF_data.CropTyp.unique()))
NDVI_regular_preds.CropTyp.unique()

# %%
SF_data.CropTyp.unique()

# %%
sorted(list(EVI_SG_preds.CropTyp.unique()))

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

# %% [markdown]
# # Filter LastSurveyDate 
#
# as we want to look at each crop and we want the labels to be correct. Or, since these are perennials we do not need that!

# %%
# ### Filter by last survey date 
# county_year_dict = {"Adams":2016, 
#                     "Benton":2016,
#                     "Franklin":2018,
#                     "Grant": 2017, 
#                     "Walla Walla":2015,
#                     "Yakima":2018}

# NDVI_SG_preds_LSD = pd.DataFrame()
# EVI_SG_preds_LSD = pd.DataFrame()
# NDVI_regular_preds_LSD = pd.DataFrame()
# EVI_regular_preds_LSD = pd.DataFrame()

# SF_data_LSD = pd.DataFrame()

# for a_county in county_year_dict.keys():
#     curr_county_DF = NDVI_SG_preds[NDVI_SG_preds.county==a_county].copy()
#     curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
#     NDVI_SG_preds_LSD = pd.concat([NDVI_SG_preds_LSD, curr_county_DF])
# NDVI_SG_preds_LSD.reset_index(drop=True, inplace=True)


# for a_county in county_year_dict.keys():
#     curr_county_DF = EVI_SG_preds[EVI_SG_preds.county==a_county].copy()
#     curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
#     EVI_SG_preds_LSD = pd.concat([EVI_SG_preds_LSD, curr_county_DF])
# EVI_SG_preds_LSD.reset_index(drop=True, inplace=True)

# for a_county in county_year_dict.keys():
#     curr_county_DF = NDVI_regular_preds[NDVI_regular_preds.county==a_county].copy()
#     curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
#     NDVI_regular_preds_LSD = pd.concat([NDVI_regular_preds_LSD, curr_county_DF])
# NDVI_regular_preds_LSD.reset_index(drop=True, inplace=True)


# for a_county in county_year_dict.keys():
#     curr_county_DF = EVI_regular_preds[EVI_regular_preds.county==a_county].copy()
#     curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
#     EVI_regular_preds_LSD = pd.concat([EVI_regular_preds_LSD, curr_county_DF])
# EVI_regular_preds_LSD.reset_index(drop=True, inplace=True)

# for a_county in county_year_dict.keys():
#     curr_county_DF = SF_data[SF_data.county==a_county].copy()
#     curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
#     SF_data_LSD = pd.concat([SF_data_LSD, curr_county_DF])

# print (SF_data_LSD.shape)
# SF_data_LSD = SF_data_LSD[SF_data_LSD.ExctAcr>10].copy()
# SF_data_LSD.reset_index(drop=True, inplace=True)
# # print (SF_data_LSD.shape)

# %% [markdown]
# # Crop-Wise Stats
#
#    - **For this we need to filter by last survey date so that labels are correct.**
#
# DFs we have to use here are
#    - ```NDVI_regular_preds```
#    - ```EVI_regular_preds```
#    - ```NDVI_SG_preds```
#    - ```EVI_SG_preds```

# %%
print (SF_data.shape)
Adams = SF_data[SF_data.county=="Adams"]
print (Adams.LstSrvD.unique().min())
print (Adams.LstSrvD.unique().max())
sorted(list(SF_data.county.unique()))

# %%
assert len(NDVI_SG_preds.ID.unique()) == len(EVI_SG_preds.ID.unique()) == \
       len(NDVI_regular_preds.ID.unique()) == len(EVI_regular_preds.ID.unique())

print (f"{SF_data.shape=}")

# %%

# %% [markdown]
# # Crop-Wise Count
#
# ### Compute total count per crop
# and remember small fields are out of study.

# %%
SF_data_grp_cropCount = SF_data.groupby(['CropTyp'])['ID'].count()
SF_data_grp_cropCount = pd.DataFrame(SF_data_grp_cropCount)
SF_data_grp_cropCount.reset_index(drop=False, inplace=True)
SF_data_grp_cropCount.rename(columns={"ID": "field_count"}, inplace=True)
SF_data_grp_cropCount.head(5)


# %%
def group_countFields_perCrop(df, group_cols):
    """ groups by two columns given by group_cols.
    The second column in group_cols must be "CropTyp"
    and the first one is something like
                  SVM_NDVI_SG_preds
                  SVM_NDVI_regular_preds
                  SVM_EVI_SG_preds
                  SVM_EVI_regular_preds
    """
    col = df.groupby([group_cols[0], group_cols[1]])["CropTyp"].count().reset_index(
        name=group_cols[0]+'_fieldCount')
    col.rename(columns={group_cols[0]: "label",
                        group_cols[0]+'_fieldCount':group_cols[0]}, inplace=True)
    return (col)


# %% [markdown]
# # Crop-Wise Count

# %%
NDVI_regular_crop_summary_LSD = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_countFields_perCrop(NDVI_regular_preds, [NDVI_regular_preds.columns[1], "CropTyp"])
col2 = group_countFields_perCrop(NDVI_regular_preds, [NDVI_regular_preds.columns[2], "CropTyp"])
col3 = group_countFields_perCrop(NDVI_regular_preds, [NDVI_regular_preds.columns[3], "CropTyp"])
col4 = group_countFields_perCrop(NDVI_regular_preds, [NDVI_regular_preds.columns[4], "CropTyp"])

NDVI_regular_crop_summary_LSD = pd.concat([NDVI_regular_crop_summary_LSD, col1])
NDVI_regular_crop_summary_LSD = pd.merge(
    NDVI_regular_crop_summary_LSD, col2, on=(["label", "CropTyp"]), how="left"
)
NDVI_regular_crop_summary_LSD = pd.merge(
    NDVI_regular_crop_summary_LSD, col3, on=(["label", "CropTyp"]), how="left"
)
NDVI_regular_crop_summary_LSD = pd.merge(
    NDVI_regular_crop_summary_LSD, col4, on=(["label", "CropTyp"]), how="left"
)

NDVI_regular_crop_summary_LSD = pd.merge(
    NDVI_regular_crop_summary_LSD, SF_data_grp_cropCount, on=(["CropTyp"]), how="left"
)
################################################
df = NDVI_regular_crop_summary_LSD.copy()
# df.reset_index(inplace=True)
df = df[df.label==2]
df.head(2)

fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

axs.grid(axis='y', which='both')
X_axis = np.arange(len(df.CropTyp))

bar_width_ = 2
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_NDVI_regular_preds, color ='dodgerblue', width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_NDVI_regular_preds, color ='green', width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_NDVI_regular_prob_point3, color ='orange', width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_NDVI_regular_preds, color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, df.CropTyp)

axs.set_ylabel("double-cropped field-count")
axs.set_xlabel("crop")
axs.set_title("4-step NDVI")
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "NDVI_regular_count_cropWise_annual.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%
EVI_regular_crop_summary_LSD = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_countFields_perCrop(EVI_regular_preds, [EVI_regular_preds.columns[1], "CropTyp"])
col2 = group_countFields_perCrop(EVI_regular_preds, [EVI_regular_preds.columns[2], "CropTyp"])
col3 = group_countFields_perCrop(EVI_regular_preds, [EVI_regular_preds.columns[3], "CropTyp"])
col4 = group_countFields_perCrop(EVI_regular_preds, [EVI_regular_preds.columns[4], "CropTyp"])

EVI_regular_crop_summary_LSD = pd.concat([EVI_regular_crop_summary_LSD, col1])
EVI_regular_crop_summary_LSD = pd.merge(
    EVI_regular_crop_summary_LSD, col2, on=(["label", "CropTyp"]), how="left"
)
EVI_regular_crop_summary_LSD = pd.merge(
    EVI_regular_crop_summary_LSD, col3, on=(["label", "CropTyp"]), how="left"
)
EVI_regular_crop_summary_LSD = pd.merge(
    EVI_regular_crop_summary_LSD, col4, on=(["label", "CropTyp"]), how="left"
)

EVI_regular_crop_summary_LSD = pd.merge(
    EVI_regular_crop_summary_LSD, SF_data_grp_cropCount, on=(["CropTyp"]), how="left"
)
################################################
df = EVI_regular_crop_summary_LSD.copy()
# df.reset_index(inplace=True)
df = df[df.label==2]
df.head(2)

fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

axs.grid(axis='y', which='both')
X_axis = np.arange(len(df.CropTyp))

bar_width_ = 2
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_EVI_regular_preds, color ='dodgerblue', width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_EVI_regular_preds, color ='green', width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_EVI_regular_prob_point9, color ='orange', width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_EVI_regular_preds, color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, df.CropTyp)

axs.set_ylabel("double-cropped field-count")
axs.set_xlabel("crop")
axs.set_title("4-step EVI")
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "EVI_regular_count_cropWise_annual.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%
NDVI_SG_crop_summary_LSD = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_countFields_perCrop(NDVI_SG_preds, [NDVI_SG_preds.columns[1], "CropTyp"])
col2 = group_countFields_perCrop(NDVI_SG_preds, [NDVI_SG_preds.columns[2], "CropTyp"])
col3 = group_countFields_perCrop(NDVI_SG_preds, [NDVI_SG_preds.columns[3], "CropTyp"])
col4 = group_countFields_perCrop(NDVI_SG_preds, [NDVI_SG_preds.columns[4], "CropTyp"])

NDVI_SG_crop_summary_LSD = pd.concat([NDVI_SG_crop_summary_LSD, col1])
NDVI_SG_crop_summary_LSD = pd.merge(
    NDVI_SG_crop_summary_LSD, col2, on=(["label", "CropTyp"]), how="left"
)
NDVI_SG_crop_summary_LSD = pd.merge(
    NDVI_SG_crop_summary_LSD, col3, on=(["label", "CropTyp"]), how="left"
)
NDVI_SG_crop_summary_LSD = pd.merge(
    NDVI_SG_crop_summary_LSD, col4, on=(["label", "CropTyp"]), how="left"
)

NDVI_SG_crop_summary_LSD = pd.merge(
    NDVI_SG_crop_summary_LSD, SF_data_grp_cropCount, on=(["CropTyp"]), how="left"
)
################################################
df = NDVI_SG_crop_summary_LSD.copy()
# df.reset_index(inplace=True)
df = df[df.label==2]
df.head(2)

fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

axs.grid(axis='y', which='both')
X_axis = np.arange(len(df.CropTyp))

bar_width_ = 2
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_NDVI_SG_preds, color ='dodgerblue', width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_NDVI_SG_preds, color ='green', width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_NDVI_SG_prob_point3, color ='orange', width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_NDVI_SG_preds, color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, df.CropTyp)

axs.set_ylabel("double-cropped field-count")
axs.set_xlabel("crop")
axs.set_title("5-step NDVI")
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "NDVI_SG_count_cropWise_annual.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%
EVI_SG_crop_summary_LSD = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_countFields_perCrop(EVI_SG_preds, [EVI_SG_preds.columns[1], "CropTyp"])
col2 = group_countFields_perCrop(EVI_SG_preds, [EVI_SG_preds.columns[2], "CropTyp"])
col3 = group_countFields_perCrop(EVI_SG_preds, [EVI_SG_preds.columns[3], "CropTyp"])
col4 = group_countFields_perCrop(EVI_SG_preds, [EVI_SG_preds.columns[4], "CropTyp"])

EVI_SG_crop_summary_LSD = pd.concat([EVI_SG_crop_summary_LSD, col1])
EVI_SG_crop_summary_LSD = pd.merge(
    EVI_SG_crop_summary_LSD, col2, on=(["label", "CropTyp"]), how="left"
)
EVI_SG_crop_summary_LSD = pd.merge(
    EVI_SG_crop_summary_LSD, col3, on=(["label", "CropTyp"]), how="left"
)
EVI_SG_crop_summary_LSD = pd.merge(
    EVI_SG_crop_summary_LSD, col4, on=(["label", "CropTyp"]), how="left"
)

EVI_SG_crop_summary_LSD = pd.merge(
    EVI_SG_crop_summary_LSD, SF_data_grp_cropCount, on=(["CropTyp"]), how="left"
)
################################################
df = EVI_SG_crop_summary_LSD.copy()
# df.reset_index(inplace=True)
df = df[df.label==2]
df.head(2)

fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

axs.grid(axis='y', which='both')
X_axis = np.arange(len(df.CropTyp))

bar_width_ = 2
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_EVI_SG_preds, color ='dodgerblue', width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_EVI_SG_preds, color ='green', width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_EVI_SG_prob_point9, color ='orange', width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_EVI_SG_preds, color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, df.CropTyp)

axs.set_ylabel("double-cropped field-count")
axs.set_xlabel("crop")
axs.set_title("5-step EVI")
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "EVI_SG_count_cropWise_annual.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%

# %% [markdown]
# # Crop-Wise Count %

# %%
NDVI_regular_crop_summary_LSD = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_countFields_perCrop(NDVI_regular_preds, [NDVI_regular_preds.columns[1], "CropTyp"])
col2 = group_countFields_perCrop(NDVI_regular_preds, [NDVI_regular_preds.columns[2], "CropTyp"])
col3 = group_countFields_perCrop(NDVI_regular_preds, [NDVI_regular_preds.columns[3], "CropTyp"])
col4 = group_countFields_perCrop(NDVI_regular_preds, [NDVI_regular_preds.columns[4], "CropTyp"])

NDVI_regular_crop_summary_LSD = pd.concat([NDVI_regular_crop_summary_LSD, col1])
NDVI_regular_crop_summary_LSD = pd.merge(
    NDVI_regular_crop_summary_LSD, col2, on=(["label", "CropTyp"]), how="left"
)
NDVI_regular_crop_summary_LSD = pd.merge(
    NDVI_regular_crop_summary_LSD, col3, on=(["label", "CropTyp"]), how="left"
)
NDVI_regular_crop_summary_LSD = pd.merge(
    NDVI_regular_crop_summary_LSD, col4, on=(["label", "CropTyp"]), how="left"
)

NDVI_regular_crop_summary_LSD = pd.merge(
    NDVI_regular_crop_summary_LSD, SF_data_grp_cropCount, on=(["CropTyp"]), how="left"
)

NDVI_regular_crop_summary_LSD.iloc[:,2:6] = 100 * (NDVI_regular_crop_summary_LSD.iloc[:,2:6].div(
                                                  NDVI_regular_crop_summary_LSD.field_count, axis=0))
################################################
df = NDVI_regular_crop_summary_LSD.copy()
# df.reset_index(inplace=True)
df = df[df.label==2]
df.head(2)

fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

axs.grid(axis='y', which='both')
X_axis = np.arange(len(df.CropTyp))

bar_width_ = 2
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_NDVI_regular_preds, color ='dodgerblue', width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_NDVI_regular_preds, color ='green', width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_NDVI_regular_prob_point3, color ='orange', width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_NDVI_regular_preds, color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, df.CropTyp)

axs.set_ylabel("double-cropped field-count (%)")
axs.set_xlabel("crop")
axs.set_title("4-step NDVI")
# axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "NDVI_regular_countPerc_cropWise_annual_cropWiseDenom.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%
NDVI_SG_crop_summary_LSD = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_countFields_perCrop(NDVI_SG_preds, [NDVI_SG_preds.columns[1], "CropTyp"])
col2 = group_countFields_perCrop(NDVI_SG_preds, [NDVI_SG_preds.columns[2], "CropTyp"])
col3 = group_countFields_perCrop(NDVI_SG_preds, [NDVI_SG_preds.columns[3], "CropTyp"])
col4 = group_countFields_perCrop(NDVI_SG_preds, [NDVI_SG_preds.columns[4], "CropTyp"])

NDVI_SG_crop_summary_LSD = pd.concat([NDVI_SG_crop_summary_LSD, col1])
NDVI_SG_crop_summary_LSD = pd.merge(
    NDVI_SG_crop_summary_LSD, col2, on=(["label", "CropTyp"]), how="left"
)
NDVI_SG_crop_summary_LSD = pd.merge(
    NDVI_SG_crop_summary_LSD, col3, on=(["label", "CropTyp"]), how="left"
)
NDVI_SG_crop_summary_LSD = pd.merge(
    NDVI_SG_crop_summary_LSD, col4, on=(["label", "CropTyp"]), how="left"
)

NDVI_SG_crop_summary_LSD = pd.merge(
    NDVI_SG_crop_summary_LSD, SF_data_grp_cropCount, on=(["CropTyp"]), how="left"
)

NDVI_SG_crop_summary_LSD.iloc[:,2:6] = 100 * (NDVI_SG_crop_summary_LSD.iloc[:,2:6].div(
                                                  NDVI_SG_crop_summary_LSD.field_count, axis=0))
################################################
df = NDVI_SG_crop_summary_LSD.copy()
# df.reset_index(inplace=True)
df = df[df.label==2]
df.head(2)

fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

axs.grid(axis='y', which='both')
X_axis = np.arange(len(df.CropTyp))

bar_width_ = 2
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_NDVI_SG_preds, color ='dodgerblue', width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_NDVI_SG_preds, color ='green', width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_NDVI_SG_prob_point3, color ='orange', width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_NDVI_SG_preds, color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, df.CropTyp)

axs.set_ylabel("double-cropped field-count (%)")
axs.set_xlabel("crop")
axs.set_title("5-step NDVI")
# axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "NDVI_SG_countPerc_cropWise_annual_cropWiseDenom.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%
EVI_SG_crop_summary_LSD = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_countFields_perCrop(EVI_SG_preds, [EVI_SG_preds.columns[1], "CropTyp"])
col2 = group_countFields_perCrop(EVI_SG_preds, [EVI_SG_preds.columns[2], "CropTyp"])
col3 = group_countFields_perCrop(EVI_SG_preds, [EVI_SG_preds.columns[3], "CropTyp"])
col4 = group_countFields_perCrop(EVI_SG_preds, [EVI_SG_preds.columns[4], "CropTyp"])

EVI_SG_crop_summary_LSD = pd.concat([EVI_SG_crop_summary_LSD, col1])
EVI_SG_crop_summary_LSD = pd.merge(
    EVI_SG_crop_summary_LSD, col2, on=(["label", "CropTyp"]), how="left"
)
EVI_SG_crop_summary_LSD = pd.merge(
    EVI_SG_crop_summary_LSD, col3, on=(["label", "CropTyp"]), how="left"
)
EVI_SG_crop_summary_LSD = pd.merge(
    EVI_SG_crop_summary_LSD, col4, on=(["label", "CropTyp"]), how="left"
)

EVI_SG_crop_summary_LSD = pd.merge(
    EVI_SG_crop_summary_LSD, SF_data_grp_cropCount, on=(["CropTyp"]), how="left"
)

EVI_SG_crop_summary_LSD.iloc[:,2:6] = 100 * (EVI_SG_crop_summary_LSD.iloc[:,2:6].div(
                                                  EVI_SG_crop_summary_LSD.field_count, axis=0))
################################################
df = EVI_SG_crop_summary_LSD.copy()
# df.reset_index(inplace=True)
df = df[df.label==2]
df.head(2)

fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

axs.grid(axis='y', which='both')
X_axis = np.arange(len(df.CropTyp))

bar_width_ = 2
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_EVI_SG_preds, color ='dodgerblue', width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_EVI_SG_preds, color ='green', width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_EVI_SG_prob_point9, color ='orange', width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_EVI_SG_preds, color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, df.CropTyp)

axs.set_ylabel("double-cropped field-count (%)")
axs.set_xlabel("crop")
axs.set_title("5-step EVI")
# axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "EVI_SG_countPerc_cropWise_annual_cropWiseDenom.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%
EVI_regular_crop_summary_LSD = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_countFields_perCrop(EVI_regular_preds, [EVI_regular_preds.columns[1], "CropTyp"])
col2 = group_countFields_perCrop(EVI_regular_preds, [EVI_regular_preds.columns[2], "CropTyp"])
col3 = group_countFields_perCrop(EVI_regular_preds, [EVI_regular_preds.columns[3], "CropTyp"])
col4 = group_countFields_perCrop(EVI_regular_preds, [EVI_regular_preds.columns[4], "CropTyp"])

EVI_regular_crop_summary_LSD = pd.concat([EVI_regular_crop_summary_LSD, col1])
EVI_regular_crop_summary_LSD = pd.merge(
    EVI_regular_crop_summary_LSD, col2, on=(["label", "CropTyp"]), how="left"
)
EVI_regular_crop_summary_LSD = pd.merge(
    EVI_regular_crop_summary_LSD, col3, on=(["label", "CropTyp"]), how="left"
)
EVI_regular_crop_summary_LSD = pd.merge(
    EVI_regular_crop_summary_LSD, col4, on=(["label", "CropTyp"]), how="left"
)

EVI_regular_crop_summary_LSD = pd.merge(
    EVI_regular_crop_summary_LSD, SF_data_grp_cropCount, on=(["CropTyp"]), how="left"
)

EVI_regular_crop_summary_LSD.iloc[:,2:6] = 100 * (EVI_regular_crop_summary_LSD.iloc[:,2:6].div(
                                                  EVI_regular_crop_summary_LSD.field_count, axis=0))
################################################
df = EVI_regular_crop_summary_LSD.copy()
# df.reset_index(inplace=True)
df = df[df.label==2]
df.head(2)

fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});

axs.grid(axis='y', which='both')
X_axis = np.arange(len(df.CropTyp))

bar_width_ = 2
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.CropTyp), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_EVI_regular_preds, color ='dodgerblue', width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_EVI_regular_preds, color ='green', width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_EVI_regular_prob_point9, color ='orange', width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_EVI_regular_preds, color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 90)
axs.set_xticks(X_axis, df.CropTyp)

axs.set_ylabel("double-cropped field-count (%)")
axs.set_xlabel("crop")
axs.set_title("4-step EVI")
# axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "EVI_regular_countPerc_cropWise_annual_cropWiseDenom.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%

# %%
