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

plot_dir = dir_base + "/for_paper/plots/overSampleRegionalStats/AllirrDenom/only2PotensNumerator/"
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
print (f"{AnnualPerennialToss.shape=}")
print (AnnualPerennialToss.potential.unique())

# %%
# badCrops = ["christmas tree", 
#             "crp/conservation",
#             "developed", 
#             "driving range", 
#             "golf course",
#             "green manure",
#             "nursery, caneberry",
#             "nursery, greenhouse",
#             "nursery, lavender",
#             "nursery, orchard/vineyard",
#             "nursery, ornamental",
#             "nursery, silviculture",
#             "reclamation seed",
#             "research station",
#             "unknown"]

badCrops = list(AnnualPerennialToss.loc[AnnualPerennialToss.potential=="toss", "Crop_Type"])

# %%
AnnualPerennialToss = AnnualPerennialToss[AnnualPerennialToss.potential!="toss"]
AnnualPerennialToss.reset_index(drop=True, inplace=True)
print (f"{AnnualPerennialToss.shape=}")
print (AnnualPerennialToss.potential.unique())
AnnualPerennialToss.head(2)

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
SF_data_nonPotential['ExctAcr'].sum()

# %%
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
SF_data = nc.filter_out_nonIrrigated(SF_data)

# %%
Adams = SF_data[SF_data.county=="Adams"]
Adams.LstSrvD.unique().min()

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

# %% [markdown]
# # Toss bad crops, non-irrigated, and filter large fields in predictions but not in denominator.
#
# (**Predictions are only irrigated**)
#
# Clean from this point on. These were not dropped in non-overSample

# %%
potentialCrops = list(only_potentialCrops.Crop_Type.unique())

# %%
EVI_SG_preds = EVI_SG_preds[EVI_SG_preds.CropTyp.isin(potentialCrops)]
EVI_regular_preds = EVI_regular_preds[EVI_regular_preds.CropTyp.isin(potentialCrops)]

NDVI_SG_preds = NDVI_SG_preds[NDVI_SG_preds.CropTyp.isin(potentialCrops)]
NDVI_regular_preds = NDVI_regular_preds[NDVI_regular_preds.CropTyp.isin(potentialCrops)]

print (f"{len(EVI_SG_preds.CropTyp.unique())=}")
print (f"{len(EVI_regular_preds.CropTyp.unique())=}")
print (f"{len(NDVI_SG_preds.CropTyp.unique())=}")
print (f"{len(NDVI_regular_preds.CropTyp.unique())=}")

EVI_SG_preds = EVI_SG_preds[EVI_SG_preds.ExctAcr>10]
EVI_regular_preds = EVI_regular_preds[EVI_regular_preds.ExctAcr>10]

NDVI_SG_preds = NDVI_SG_preds[NDVI_SG_preds.ExctAcr>10]
NDVI_regular_preds = NDVI_regular_preds[NDVI_regular_preds.ExctAcr>10]


SF_data = nc.filter_out_nonIrrigated(SF_data)
# SF_data = SF_data[SF_data.CropTyp.isin(potentialCrops)]
# SF_data = SF_data[SF_data.ExctAcr>10]
SF_data.reset_index(inplace=True, drop=True)
SF_data.head(2)

# %%
# sorted(list(SF_data.CropTyp.unique()))

# %% [markdown]
# # County-Wise Acreage

# %%
# sorted(list(EVI_SG_preds.CropTyp.unique()))

# %%
EVI_SG_preds.head(2)

# %%
EVI_SG_summary_countyAcr = pd.DataFrame(columns=list(EVI_SG_preds.columns[1:5]))
EVI_SG_summary_countyAcr

# %%
EVI_SG_summary_countyAcr[EVI_SG_summary_countyAcr.columns[0]] = EVI_SG_preds.groupby(\
                                                 [EVI_SG_summary_countyAcr.columns[0], "county"])['ExctAcr'].sum()
EVI_SG_summary_countyAcr

# %%
EVI_SG_summary_countyAcr = pd.DataFrame(columns=list(EVI_SG_preds.columns[1:5]))
EVI_SG_summary_countyAcr[EVI_SG_summary_countyAcr.columns[0]] = EVI_SG_preds.groupby(\
                                                 [EVI_SG_summary_countyAcr.columns[0], "county"])['ExctAcr'].sum()

EVI_SG_summary_countyAcr[EVI_SG_summary_countyAcr.columns[1]] = EVI_SG_preds.groupby(\
             [EVI_SG_summary_countyAcr.columns[1], "county"])['ExctAcr'].sum()

EVI_SG_summary_countyAcr[EVI_SG_summary_countyAcr.columns[2]] = EVI_SG_preds.groupby(\
              [EVI_SG_summary_countyAcr.columns[2], "county"])['ExctAcr'].sum()

EVI_SG_summary_countyAcr[EVI_SG_summary_countyAcr.columns[3]] = EVI_SG_preds.groupby(\
                [EVI_SG_summary_countyAcr.columns[3], "county"])['ExctAcr'].sum()

EVI_SG_summary_countyAcr.index.rename(['label', 'county'], inplace=True)
EVI_SG_summary_countyAcr.round()

# %%

# %%
NDVI_SG_summary_countyAcr = pd.DataFrame(columns=list(NDVI_SG_preds.columns[1:5]))
NDVI_SG_summary_countyAcr[NDVI_SG_summary_countyAcr.columns[0]] = \
             NDVI_SG_preds.groupby([NDVI_SG_summary_countyAcr.columns[0], "county"])['ExctAcr'].sum()

NDVI_SG_summary_countyAcr[NDVI_SG_summary_countyAcr.columns[1]] = \
            NDVI_SG_preds.groupby([NDVI_SG_summary_countyAcr.columns[1], "county"])['ExctAcr'].sum()

NDVI_SG_summary_countyAcr[NDVI_SG_summary_countyAcr.columns[2]] = \
            NDVI_SG_preds.groupby([NDVI_SG_summary_countyAcr.columns[2], "county"])['ExctAcr'].sum()

NDVI_SG_summary_countyAcr[NDVI_SG_summary_countyAcr.columns[3]] = \
            NDVI_SG_preds.groupby([NDVI_SG_summary_countyAcr.columns[3], "county"])['ExctAcr'].sum()

NDVI_SG_summary_countyAcr.index.rename(['label', 'county'], inplace=True)
NDVI_SG_summary_countyAcr.round()

# %%
NDVI_regular_summary_countyAcr = pd.DataFrame(columns=list(NDVI_regular_preds.columns[1:5]))
NDVI_regular_summary_countyAcr[NDVI_regular_summary_countyAcr.columns[0]] = \
             NDVI_regular_preds.groupby([NDVI_regular_summary_countyAcr.columns[0], "county"])['ExctAcr'].sum()
    
NDVI_regular_summary_countyAcr[NDVI_regular_summary_countyAcr.columns[1]] = \
           NDVI_regular_preds.groupby([NDVI_regular_summary_countyAcr.columns[1], "county"])['ExctAcr'].sum()

NDVI_regular_summary_countyAcr[NDVI_regular_summary_countyAcr.columns[2]] = \
     NDVI_regular_preds.groupby([NDVI_regular_summary_countyAcr.columns[2], "county"])['ExctAcr'].sum()

NDVI_regular_summary_countyAcr[NDVI_regular_summary_countyAcr.columns[3]] = \
      NDVI_regular_preds.groupby([NDVI_regular_summary_countyAcr.columns[3], "county"])['ExctAcr'].sum()

NDVI_regular_summary_countyAcr.index.rename(['label', 'county'], inplace=True)
NDVI_regular_summary_countyAcr.round()

# %%
EVI_regular_summary_countyAcr = pd.DataFrame(columns=list(EVI_regular_preds.columns[1:5]))

EVI_regular_summary_countyAcr[EVI_regular_summary_countyAcr.columns[0]] = \
               EVI_regular_preds.groupby([EVI_regular_summary_countyAcr.columns[0], "county"])['ExctAcr'].sum()

EVI_regular_summary_countyAcr[EVI_regular_summary_countyAcr.columns[1]] = \
             EVI_regular_preds.groupby([EVI_regular_summary_countyAcr.columns[1], "county"])['ExctAcr'].sum()
    
EVI_regular_summary_countyAcr[EVI_regular_summary_countyAcr.columns[2]] = \
              EVI_regular_preds.groupby([EVI_regular_summary_countyAcr.columns[2], "county"])['ExctAcr'].sum()
    
EVI_regular_summary_countyAcr[EVI_regular_summary_countyAcr.columns[3]] = \
              EVI_regular_preds.groupby([EVI_regular_summary_countyAcr.columns[3], "county"])['ExctAcr'].sum()
# EVI_regular_summary_countyAcr.reset_index(drop=False, inplace=True, col_fill='', names="predicted_label")
EVI_regular_summary_countyAcr.index.rename(['label', 'county'], inplace=True)
EVI_regular_summary_countyAcr.round()

# %% [markdown]
# # County-wise Acreage %
#
# ### Compute total area per county
# and remember small fields are out on predictions but not ```SF_data```.

# %%
Adams = SF_data[SF_data.county=="Adams"]
print (Adams.LstSrvD.unique().min())
print (Adams.LstSrvD.unique().max())
sorted(list(SF_data.county.unique()))

# %%
print (f"{SF_data.ExctAcr.min()=:.4f}")
SF_data.head(2)

# %%
# sorted(list(SF_data.CropTyp.unique()))

# %%
SF_data_grp_area = SF_data.groupby(['county'])['ExctAcr'].sum()
SF_data_grp_area = pd.DataFrame(SF_data_grp_area)
SF_data_grp_area.reset_index(drop=False, inplace=True)
SF_data_grp_area

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

color_dict = {"SVM": "#DDCC77",
              "kNN": "#E69F00",
              "DL": "#332288", # "#6699CC",
              "RF":'#0072B2'
             }

# %%
plot_dir

# %%
df = EVI_regular_summary_countyAcr.copy()
df.reset_index(inplace=True)
df = df[df.label==2]

df = pd.merge(df, SF_data_grp_area, on=(["county"]), how='left')
df.iloc[:,2:6] = 100 * (df.iloc[:,2:6].div(df.ExctAcr, axis=0))

################################################################
fig, axs = plt.subplots(1, 1, figsize=(7, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

X_axis = np.arange(len(df.county))

bar_width_ = 1
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.county), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_EVI_regular_preds, color = color_dict["SVM"], width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_EVI_regular_preds, color = color_dict["kNN"], width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_EVI_regular_prob_point9, color = color_dict["DL"], width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_EVI_regular_preds, color = color_dict["RF"], width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 0)
axs.set_xticks(X_axis, df.county)

axs.set_ylabel("double-cropped acreage (%)")
# axs.set_xlabel("county")
axs.set_title("4-step EVI")
# axs.set_ylim([0, 105])

# send the guidelines to the back
ymin, ymax = axs.get_ylim()
axs.set(ylim=(ymin-1, ymax+1), axisbelow=True);

axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "EVI_regular_acreagePerc_countyWise_potens_AllIrrDenom.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%

# %%
df = EVI_SG_summary_countyAcr.copy()
df.reset_index(inplace=True)
df = df[df.label==2]

df = pd.merge(df, SF_data_grp_area, on=(["county"]), how='left')
df.iloc[:,2:6] = 100 * (df.iloc[:,2:6].div(df.ExctAcr, axis=0))

################################################################
fig, axs = plt.subplots(1, 1, figsize=(7, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

X_axis = np.arange(len(df.county))

bar_width_ = 1
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.county), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_EVI_SG_preds, color = color_dict["SVM"], width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_EVI_SG_preds, color = color_dict["kNN"], width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_EVI_SG_prob_point9, color = color_dict["DL"], width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_EVI_SG_preds, color = color_dict["RF"], width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 0)
axs.set_xticks(X_axis, df.county)

axs.set_ylabel("double-cropped acreage (%)")
# axs.set_xlabel("county")
axs.set_title("5-step EVI")
# axs.set_ylim([0, 105])

# send the guidelines to the back
ymin, ymax = axs.get_ylim()
axs.set(ylim=(ymin-1, ymax+1), axisbelow=True);

axs.legend(loc="best");
axs.xaxis.set_ticks_position('none') 
file_name = plot_dir + "EVI_SG_acreagePerc_countyWise_potens_AllIrrDenom.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%

# %%
df = NDVI_regular_summary_countyAcr.copy()
df.reset_index(inplace=True)
df = df[df.label==2]
df = pd.merge(df, SF_data_grp_area, on=(["county"]), how='left')
df.iloc[:,2:6] = 100 * (df.iloc[:,2:6].div(df.ExctAcr, axis=0))

################################################################
fig, axs = plt.subplots(1, 1, figsize=(7, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

X_axis = np.arange(len(df.county))

bar_width_ = 1
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.county), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_NDVI_regular_preds, color = color_dict["SVM"], width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_NDVI_regular_preds, color = color_dict["kNN"], width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_NDVI_regular_prob_point3, color = color_dict["DL"], width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_NDVI_regular_preds, color = color_dict["RF"], width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 0)
axs.set_xticks(X_axis, df.county)

axs.set_ylabel("double-cropped acreage (%)")
# axs.set_xlabel("county")
axs.set_title("4-step NDVI")
# axs.set_ylim([0, 105])
axs.legend(loc="best");

# send the guidelines to the back
ymin, ymax = axs.get_ylim()
axs.set(ylim=(ymin-1, ymax+1), axisbelow=True);


axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "NDVI_regular_acreagePerc_countyWise_potens_AllIrrDenom.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%

# %%
df = NDVI_SG_summary_countyAcr.copy()
df.reset_index(inplace=True)
df = df[df.label==2]

df = pd.merge(df, SF_data_grp_area, on=(["county"]), how='left')
df.iloc[:,2:6] = 100 * (df.iloc[:,2:6].div(df.ExctAcr, axis=0))

################################################################
fig, axs = plt.subplots(1, 1, figsize=(7, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

X_axis = np.arange(len(df.county))

bar_width_ = 1
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.county), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_NDVI_SG_preds, color = color_dict["SVM"], width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_NDVI_SG_preds, color = color_dict["kNN"], width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_NDVI_SG_prob_point3, color = color_dict["DL"], width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_NDVI_SG_preds, color = color_dict["RF"], width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 0)
axs.set_xticks(X_axis, df.county)

axs.set_ylabel("double-cropped acreage (%)")
# axs.set_xlabel("county")
# axs.set_title("NDVI")
# axs.set_ylim([0, 105])
# send the guidelines to the back

ymin, ymax = axs.get_ylim()
axs.set(ylim=(ymin-1, ymax+1), axisbelow=True);

axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "NDVI_SG_acreagePerc_countyWise_potens_AllIrrDenom.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
# del(df)

# %%
SF_data_grp_area

# %%
df = NDVI_SG_summary_countyAcr.copy()
df.reset_index(inplace=True)
df = df[df.label==2]

df = pd.merge(df, SF_data_grp_area, on=(["county"]), how='left')
df_copy = df.copy()
df_copy["county"] = "all counties"
df_copy.sum()[2:].values
df.loc[len(df.index)] = ["2", "all counties"] +  list(df_copy.sum()[2:].values)
df

# %%
df = NDVI_SG_summary_countyAcr.copy()
df.reset_index(inplace=True)
df = df[df.label==2]

df = pd.merge(df, SF_data_grp_area, on=(["county"]), how='left')
df_copy = df.copy()
df_copy["county"] = "all counties"
df_copy.sum()[2:].values
df.loc[len(df.index)] = ["2", "all counties"] +  list(df_copy.sum()[2:].values)


df.iloc[:,2:6] = 100 * (df.iloc[:,2:6].div(df.ExctAcr, axis=0))

################################################################
fig, axs = plt.subplots(1, 1, figsize=(7, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

X_axis = np.arange(len(df.county))

bar_width_ = 1
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.county), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_NDVI_SG_preds, color = color_dict["SVM"], width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_NDVI_SG_preds, color = color_dict["kNN"], width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_NDVI_SG_prob_point3, color = color_dict["DL"], width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_NDVI_SG_preds, color = color_dict["RF"], width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 0)
axs.set_xticks(X_axis, df.county)

axs.set_ylabel("double-cropped acreage (%)")
# axs.set_xlabel("county")
# axs.set_title("NDVI")
# axs.set_ylim([0, 105])
# send the guidelines to the back
ymin, ymax = axs.get_ylim()
axs.set(ylim=(ymin-1, ymax+1), axisbelow=True);

axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "NDVI_SG_acreagePerc_countyWise_potens_AllIrrDenom_acllCounties.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
# del(df)

# %%

# %%

# %%

# %%

# %%
# Adams = SF_data[SF_data.county=="Adams"]
# print (Adams.LstSrvD.unique().min())
# print (Adams.LstSrvD.unique().max())
# sorted(list(SF_data.county.unique()))

# %% [markdown]
# # County-wise Counts

# %%
Adams = SF_data[SF_data.county=="Adams"]
Adams.LstSrvD.unique().min()

# %%
NDVI_SG_summary_countPerCounty = pd.DataFrame(columns=list(NDVI_SG_preds.columns[1:5]))

NDVI_SG_summary_countPerCounty[NDVI_SG_summary_countPerCounty.columns[0]] = \
             NDVI_SG_preds.groupby([NDVI_SG_summary_countPerCounty.columns[0], "county"])['ID'].count()

NDVI_SG_summary_countPerCounty[NDVI_SG_summary_countPerCounty.columns[1]] = \
             NDVI_SG_preds.groupby([NDVI_SG_summary_countPerCounty.columns[1], "county"])['ID'].count()
    
NDVI_SG_summary_countPerCounty[NDVI_SG_summary_countPerCounty.columns[2]] = \
             NDVI_SG_preds.groupby([NDVI_SG_summary_countPerCounty.columns[2], "county"])['ID'].count()
    
NDVI_SG_summary_countPerCounty[NDVI_SG_summary_countPerCounty.columns[3]] = \
             NDVI_SG_preds.groupby([NDVI_SG_summary_countPerCounty.columns[3], "county"])['ID'].count()
# NDVI_SG_summary_countPerCounty.reset_index(drop=False, inplace=True, col_fill='', names="predictedabel")

NDVI_SG_summary_countPerCounty.index.rename(['label', 'county'], inplace=True)
NDVI_SG_summary_countPerCounty.round().head(2)

# %%
NDVI_reg_summary_countPerCounty = pd.DataFrame(columns=list(NDVI_regular_preds.columns[1:5]))

NDVI_reg_summary_countPerCounty[NDVI_reg_summary_countPerCounty.columns[0]] = \
             NDVI_regular_preds.groupby([NDVI_reg_summary_countPerCounty.columns[0], "county"])['ID'].count()

NDVI_reg_summary_countPerCounty[NDVI_reg_summary_countPerCounty.columns[1]] = \
             NDVI_regular_preds.groupby([NDVI_reg_summary_countPerCounty.columns[1], "county"])['ID'].count()
    
NDVI_reg_summary_countPerCounty[NDVI_reg_summary_countPerCounty.columns[2]] = \
             NDVI_regular_preds.groupby([NDVI_reg_summary_countPerCounty.columns[2], "county"])['ID'].count()
    
NDVI_reg_summary_countPerCounty[NDVI_reg_summary_countPerCounty.columns[3]] = \
             NDVI_regular_preds.groupby([NDVI_reg_summary_countPerCounty.columns[3], "county"])['ID'].count()
# NDVI_reg_summary_countPerCounty.reset_index(drop=False, inplace=True, col_fill='', names="predictedabel")

NDVI_reg_summary_countPerCounty.index.rename(['label', 'county'], inplace=True)
NDVI_reg_summary_countPerCounty.round().head(2)

# %%
EVI_SG_summary_countPerCounty = pd.DataFrame(columns=list(EVI_SG_preds.columns[1:5]))

EVI_SG_summary_countPerCounty[EVI_SG_summary_countPerCounty.columns[0]] = \
             EVI_SG_preds.groupby([EVI_SG_summary_countPerCounty.columns[0], "county"])['ID'].count()

EVI_SG_summary_countPerCounty[EVI_SG_summary_countPerCounty.columns[1]] = \
             EVI_SG_preds.groupby([EVI_SG_summary_countPerCounty.columns[1], "county"])['ID'].count()
    
EVI_SG_summary_countPerCounty[EVI_SG_summary_countPerCounty.columns[2]] = \
             EVI_SG_preds.groupby([EVI_SG_summary_countPerCounty.columns[2], "county"])['ID'].count()
    
EVI_SG_summary_countPerCounty[EVI_SG_summary_countPerCounty.columns[3]] = \
             EVI_SG_preds.groupby([EVI_SG_summary_countPerCounty.columns[3], "county"])['ID'].count()
# EVI_SG_summary_countPerCounty.reset_index(drop=False, inplace=True, col_fill='', names="predictedabel")

EVI_SG_summary_countPerCounty.index.rename(['label', 'county'], inplace=True)
EVI_SG_summary_countPerCounty.round().head(2)

# %%
EVI_reg_summary_countPerCounty = pd.DataFrame(columns=list(EVI_regular_preds.columns[1:5]))

EVI_reg_summary_countPerCounty[EVI_reg_summary_countPerCounty.columns[0]] = \
             EVI_regular_preds.groupby([EVI_reg_summary_countPerCounty.columns[0], "county"])['ID'].count()

EVI_reg_summary_countPerCounty[EVI_reg_summary_countPerCounty.columns[1]] = \
             EVI_regular_preds.groupby([EVI_reg_summary_countPerCounty.columns[1], "county"])['ID'].count()
    
EVI_reg_summary_countPerCounty[EVI_reg_summary_countPerCounty.columns[2]] = \
             EVI_regular_preds.groupby([EVI_reg_summary_countPerCounty.columns[2], "county"])['ID'].count()
    
EVI_reg_summary_countPerCounty[EVI_reg_summary_countPerCounty.columns[3]] = \
             EVI_regular_preds.groupby([EVI_reg_summary_countPerCounty.columns[3], "county"])['ID'].count()
# EVI_reg_summary_countPerCounty.reset_index(drop=False, inplace=True, col_fill='', names="predictedabel")

EVI_reg_summary_countPerCounty.index.rename(['label', 'county'], inplace=True)
EVI_reg_summary_countPerCounty.round().head(2)

# %%
# Adams = SF_data[SF_data.county=="Adams"]
# print (Adams.LstSrvD.unique().min())
# print (Adams.LstSrvD.unique().max())
# sorted(list(SF_data.county.unique()))

# %% [markdown]
# # County-wise Count %
#
# ### Compute total count per county
# and remember small fields are out of study.

# %%
# All irrigated fields must be there (orchards, alfalfa category, and 2-potens)
sorted(list(SF_data.CropTyp.unique()))

# %%
SF_data_grp_count = SF_data.groupby(['county'])['ID'].count()
SF_data_grp_count = pd.DataFrame(SF_data_grp_count)
SF_data_grp_count.reset_index(drop=False, inplace=True)
SF_data_grp_count.rename(columns={"ID": "field_count"}, inplace=True)
SF_data_grp_count

# %%
df = EVI_reg_summary_countPerCounty.copy()
df.reset_index(inplace=True)
df = df[df.label==2]
df = pd.merge(df, SF_data_grp_count, on=(["county"]), how='left')
df.iloc[:,2:6] = 100 * (df.iloc[:,2:6].div(df.field_count, axis=0))

##################################################################
fig, axs = plt.subplots(1, 1, figsize=(7, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

X_axis = np.arange(len(df.county))

bar_width_ = 1
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.county), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_EVI_regular_preds, color ='dodgerblue', width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_EVI_regular_preds, color ='green', width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_EVI_regular_prob_point9, color ='orange', width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_EVI_regular_preds, color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 0)
axs.set_xticks(X_axis, df.county)

axs.set_ylabel("double-cropped field-count (%)")
# axs.set_xlabel("county")
axs.set_title("4-step EVI")
# axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "EVI_regular_countPerc_countyWise_potens_AllIrrDenom.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%

# %%
df = EVI_SG_summary_countPerCounty.copy()
df.reset_index(inplace=True)
df = df[df.label==2]
df = pd.merge(df, SF_data_grp_count, on=(["county"]), how='left')
df.iloc[:,2:6] = 100 * (df.iloc[:,2:6].div(df.field_count, axis=0))

##################################################################
fig, axs = plt.subplots(1, 1, figsize=(7, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

X_axis = np.arange(len(df.county))

bar_width_ = 1
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.county), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_EVI_SG_preds, color ='dodgerblue', width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_EVI_SG_preds, color ='green', width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_EVI_SG_prob_point9, color ='orange', width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_EVI_SG_preds, color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 0)
axs.set_xticks(X_axis, df.county)

axs.set_ylabel("double-cropped field-count (%)")
# axs.set_xlabel("county")
axs.set_title("5-step EVI")
# axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "EVI_SG_countPerc_countyWise_potens_AllIrrDenom.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%

# %%
df = NDVI_SG_summary_countPerCounty.copy()
df.reset_index(inplace=True)
df = df[df.label==2]
df = pd.merge(df, SF_data_grp_count, on=(["county"]), how='left')
df.iloc[:,2:6] = 100 * (df.iloc[:,2:6].div(df.field_count, axis=0))

##################################################################
fig, axs = plt.subplots(1, 1, figsize=(7, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

X_axis = np.arange(len(df.county))

bar_width_ = 1
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.county), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_NDVI_SG_preds, color ='dodgerblue', width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_NDVI_SG_preds, color ='green', width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_NDVI_SG_prob_point3, color ='orange', width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_NDVI_SG_preds, color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 0)
axs.set_xticks(X_axis, df.county)

axs.set_ylabel("double-cropped field-count (%)")
# axs.set_xlabel("county")
# axs.set_title("NDVI")
# axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "NDVI_SG_countPerc_countyWise_potens_AllIrrDenom.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%

# %%
df = NDVI_reg_summary_countPerCounty.copy()
df.reset_index(inplace=True)
df = df[df.label==2]
df = pd.merge(df, SF_data_grp_count, on=(["county"]), how='left')
df.iloc[:,2:6] = 100 * (df.iloc[:,2:6].div(df.field_count, axis=0))

##################################################################
fig, axs = plt.subplots(1, 1, figsize=(7, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

X_axis = np.arange(len(df.county))

bar_width_ = 1
step_size_ = 5*bar_width_
X_axis = np.array(range(0, step_size_*len(df.county), step_size_))

axs.bar(X_axis - bar_width_*2, df.SVM_NDVI_regular_preds, color ='dodgerblue', width = bar_width_, label="SVM")
axs.bar(X_axis - bar_width_, df.KNN_NDVI_regular_preds, color ='green', width = bar_width_, label="kNN")
axs.bar(X_axis, df.DL_NDVI_regular_prob_point3, color ='orange', width = bar_width_, label="DL")
axs.bar(X_axis + bar_width_, df.RF_NDVI_regular_preds, color ='c', width = bar_width_, label="RF")

axs.tick_params(axis='x', labelrotation = 0)
axs.set_xticks(X_axis, df.county)

axs.set_ylabel("double-cropped field-count (%)")
# axs.set_xlabel("county")
axs.set_title("4-step NDVI")
# axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "NDVI_regular_countPerc_countyWise_potens_AllIrrDenom.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %% [markdown]
# # Crop-Wise Won't make much sense, would it?

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# Adams = SF_data[SF_data.county=="Adams"]
# print (Adams.LstSrvD.unique().min())
# print (Adams.LstSrvD.unique().max())
# sorted(list(SF_data.county.unique()))

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
### Filter by last survey date 
county_year_dict = {"Adams":2016, 
                    "Benton":2016,
                    "Franklin":2018,
                    "Grant": 2017, 
                    "Walla Walla":2015,
                    "Yakima":2018}

NDVI_SG_preds_LSD = pd.DataFrame()
EVI_SG_preds_LSD = pd.DataFrame()
NDVI_regular_preds_LSD = pd.DataFrame()
EVI_regular_preds_LSD = pd.DataFrame()

SF_data_LSD = pd.DataFrame()

for a_county in county_year_dict.keys():
    curr_county_DF = NDVI_SG_preds[NDVI_SG_preds.county==a_county].copy()
    curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
    NDVI_SG_preds_LSD = pd.concat([NDVI_SG_preds_LSD, curr_county_DF])
NDVI_SG_preds_LSD.reset_index(drop=True, inplace=True)


for a_county in county_year_dict.keys():
    curr_county_DF = EVI_SG_preds[EVI_SG_preds.county==a_county].copy()
    curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
    EVI_SG_preds_LSD = pd.concat([EVI_SG_preds_LSD, curr_county_DF])
EVI_SG_preds_LSD.reset_index(drop=True, inplace=True)

for a_county in county_year_dict.keys():
    curr_county_DF = NDVI_regular_preds[NDVI_regular_preds.county==a_county].copy()
    curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
    NDVI_regular_preds_LSD = pd.concat([NDVI_regular_preds_LSD, curr_county_DF])
NDVI_regular_preds_LSD.reset_index(drop=True, inplace=True)


for a_county in county_year_dict.keys():
    curr_county_DF = EVI_regular_preds[EVI_regular_preds.county==a_county].copy()
    curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
    EVI_regular_preds_LSD = pd.concat([EVI_regular_preds_LSD, curr_county_DF])
EVI_regular_preds_LSD.reset_index(drop=True, inplace=True)

for a_county in county_year_dict.keys():
    curr_county_DF = SF_data[SF_data.county==a_county].copy()
    curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
    SF_data_LSD = pd.concat([SF_data_LSD, curr_county_DF])

print (SF_data_LSD.shape)
SF_data_LSD = SF_data_LSD[SF_data_LSD.ExctAcr>10].copy()
SF_data_LSD.reset_index(drop=True, inplace=True)
print (SF_data_LSD.shape)

# %%
print (SF_data.shape)
Adams = SF_data[SF_data.county=="Adams"]
print (Adams.LstSrvD.unique().min())
print (Adams.LstSrvD.unique().max())
sorted(list(SF_data.county.unique()))

# %%
print (f"{SF_data_LSD.shape=}")
print (f"{SF_data.shape=}")

# %%
NDVI_SG_preds = NDVI_SG_preds_LSD.copy()
EVI_SG_preds = EVI_SG_preds_LSD.copy()
NDVI_regular_preds = NDVI_regular_preds_LSD.copy()
EVI_regular_preds = EVI_regular_preds_LSD.copy()
SF_data = SF_data_LSD.copy()

# %%
# This must be false as we are considering denominator to be all irrigated fields
sorted(list(NDVI_SG_preds.ID.unique())) == sorted(list(SF_data_LSD.ID.unique()))

# %%

# %%
assert len(NDVI_SG_preds.ID.unique()) == len(EVI_SG_preds.ID.unique()) == \
       len(NDVI_regular_preds.ID.unique()) == len(EVI_regular_preds.ID.unique())

del(NDVI_SG_preds_LSD, EVI_SG_preds_LSD, NDVI_regular_preds_LSD, EVI_regular_preds_LSD, SF_data_LSD)

# %%
Adams = SF_data[SF_data.county=="Adams"]
print (Adams.LstSrvD.unique().min())
print (Adams.LstSrvD.unique().max())
sorted(list(SF_data.county.unique()))

# %%
Adams = SF_data[SF_data.county=="Franklin"]
print (Adams.LstSrvD.unique().min())
print (Adams.LstSrvD.unique().max())


# %%
def group_sum_area(df, group_cols):
    """ groups by two columns given by group_cols.
    The second column in group_cols must be "CropTyp"
    and the first one is something like
                  SVM_NDVI_SG_preds
                  SVM_NDVI_regular_preds
                  SVM_EVI_SG_preds
                  SVM_EVI_regular_preds
    """
    col = df.groupby([group_cols[0], group_cols[1]])['ExctAcr'].sum().reset_index(
                            name=group_cols[0]+'_acr_sum')
    col.rename(columns={group_cols[0]: "label",
                        group_cols[0]+'_acr_sum':group_cols[0]}, inplace=True)
    return (col)


# %% [markdown]
# # Crop-Wise Acreage

# %%
NDVI_SG_preds_Adams = NDVI_SG_preds[NDVI_SG_preds.county=="Adams"]
print (NDVI_SG_preds_Adams.LstSrvD.unique().min())
print (NDVI_SG_preds_Adams.LstSrvD.unique().max())

# %%
NDVI_SG_crop_summary = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_sum_area(NDVI_SG_preds, [NDVI_SG_preds.columns[1], "CropTyp"])
col2 = group_sum_area(NDVI_SG_preds, [NDVI_SG_preds.columns[2], "CropTyp"])
col3 = group_sum_area(NDVI_SG_preds, [NDVI_SG_preds.columns[3], "CropTyp"])
col4 = group_sum_area(NDVI_SG_preds, [NDVI_SG_preds.columns[4], "CropTyp"])

NDVI_SG_crop_summary = pd.concat([NDVI_SG_crop_summary, col1])
NDVI_SG_crop_summary = pd.merge(NDVI_SG_crop_summary, col2, on=(["label", "CropTyp"]), how='left')
NDVI_SG_crop_summary = pd.merge(NDVI_SG_crop_summary, col3, on=(["label", "CropTyp"]), how='left')
NDVI_SG_crop_summary = pd.merge(NDVI_SG_crop_summary, col4, on=(["label", "CropTyp"]), how='left')

NDVI_SG_crop_summary.head(2)

# %%
EVI_SG_crop_summary = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_sum_area(EVI_SG_preds, [EVI_SG_preds.columns[1], "CropTyp"])
col2 = group_sum_area(EVI_SG_preds, [EVI_SG_preds.columns[2], "CropTyp"])
col3 = group_sum_area(EVI_SG_preds, [EVI_SG_preds.columns[3], "CropTyp"])
col4 = group_sum_area(EVI_SG_preds, [EVI_SG_preds.columns[4], "CropTyp"])

EVI_SG_crop_summary = pd.concat([EVI_SG_crop_summary, col1])
EVI_SG_crop_summary = pd.merge(EVI_SG_crop_summary, col2, on=(["label", "CropTyp"]), how='left')
EVI_SG_crop_summary = pd.merge(EVI_SG_crop_summary, col3, on=(["label", "CropTyp"]), how='left')
EVI_SG_crop_summary = pd.merge(EVI_SG_crop_summary, col4, on=(["label", "CropTyp"]), how='left')

EVI_SG_crop_summary.head(2)

# %%
EVI_regular_crop_summary = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_sum_area(EVI_regular_preds, [EVI_regular_preds.columns[1], "CropTyp"])
col2 = group_sum_area(EVI_regular_preds, [EVI_regular_preds.columns[2], "CropTyp"])
col3 = group_sum_area(EVI_regular_preds, [EVI_regular_preds.columns[3], "CropTyp"])
col4 = group_sum_area(EVI_regular_preds, [EVI_regular_preds.columns[4], "CropTyp"])

EVI_regular_crop_summary = pd.concat([EVI_regular_crop_summary, col1])
EVI_regular_crop_summary = pd.merge(EVI_regular_crop_summary, col2, on=(["label", "CropTyp"]), how='left')
EVI_regular_crop_summary = pd.merge(EVI_regular_crop_summary, col3, on=(["label", "CropTyp"]), how='left')
EVI_regular_crop_summary = pd.merge(EVI_regular_crop_summary, col4, on=(["label", "CropTyp"]), how='left')

EVI_regular_crop_summary.head(2)

# %%
NDVI_regular_crop_summary = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_sum_area(NDVI_regular_preds, [NDVI_regular_preds.columns[1], "CropTyp"])
col2 = group_sum_area(NDVI_regular_preds, [NDVI_regular_preds.columns[2], "CropTyp"])
col3 = group_sum_area(NDVI_regular_preds, [NDVI_regular_preds.columns[3], "CropTyp"])
col4 = group_sum_area(NDVI_regular_preds, [NDVI_regular_preds.columns[4], "CropTyp"])

NDVI_regular_crop_summary = pd.concat([NDVI_regular_crop_summary, col1])
NDVI_regular_crop_summary = pd.merge(NDVI_regular_crop_summary, col2, on=(["label", "CropTyp"]), how='left')
NDVI_regular_crop_summary = pd.merge(NDVI_regular_crop_summary, col3, on=(["label", "CropTyp"]), how='left')
NDVI_regular_crop_summary = pd.merge(NDVI_regular_crop_summary, col4, on=(["label", "CropTyp"]), how='left')

NDVI_regular_crop_summary.head(2)

# %% [markdown]
# # Crop-Wise Acreage %

# %%
NDVI_SG_crop_summary.head(2)

# %%
EVI_regular_preds.head(2)

# %% [markdown]
# ### Compute Total area per crop 

# %%
print (f"{SF_data.ExctAcr.min()=:0.4f}")

# %%
SF_data_grp_area = SF_data.groupby(['CropTyp'])['ExctAcr'].sum()
SF_data_grp_area = pd.DataFrame(SF_data_grp_area)
SF_data_grp_area.reset_index(drop=False, inplace=True)
SF_data_grp_area.head(2)

# %%
NDVI_SG_crop_summary_LSD = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_sum_area(NDVI_SG_preds, [NDVI_SG_preds.columns[1], "CropTyp"])
col2 = group_sum_area(NDVI_SG_preds, [NDVI_SG_preds.columns[2], "CropTyp"])
col3 = group_sum_area(NDVI_SG_preds, [NDVI_SG_preds.columns[3], "CropTyp"])
col4 = group_sum_area(NDVI_SG_preds, [NDVI_SG_preds.columns[4], "CropTyp"])

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
    NDVI_SG_crop_summary_LSD, SF_data_grp_area, on=(["CropTyp"]), how="left"
)

NDVI_SG_crop_summary_LSD.iloc[:,2:6] = 100* (NDVI_SG_crop_summary_LSD.iloc[:,2:6].div(
                                                       NDVI_SG_crop_summary_LSD.ExctAcr, axis=0))

NDVI_SG_crop_summary_LSD.head(2)

# %%
print (f"{NDVI_SG_crop_summary.SVM_NDVI_SG_preds.max()=}")
print (f"{NDVI_SG_crop_summary.SVM_NDVI_SG_preds.idxmax()=}")
print ("")
print (NDVI_SG_crop_summary.iloc[NDVI_SG_crop_summary.SVM_NDVI_SG_preds.idxmax()])

# %%
EVI_SG_crop_summary_LSD = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_sum_area(EVI_SG_preds, [EVI_SG_preds.columns[1], "CropTyp"])
col2 = group_sum_area(EVI_SG_preds, [EVI_SG_preds.columns[2], "CropTyp"])
col3 = group_sum_area(EVI_SG_preds, [EVI_SG_preds.columns[3], "CropTyp"])
col4 = group_sum_area(EVI_SG_preds, [EVI_SG_preds.columns[4], "CropTyp"])

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
    EVI_SG_crop_summary_LSD, SF_data_grp_area, on=(["CropTyp"]), how="left"
)

EVI_SG_crop_summary_LSD.iloc[:,2:6] = 100 * (EVI_SG_crop_summary_LSD.iloc[:,2:6].div(
                                                       EVI_SG_crop_summary_LSD.ExctAcr, axis=0))

EVI_SG_crop_summary_LSD.head(2)

# %%
NDVI_regular_crop_summary_LSD = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_sum_area(NDVI_regular_preds, [NDVI_regular_preds.columns[1], "CropTyp"])
col2 = group_sum_area(NDVI_regular_preds, [NDVI_regular_preds.columns[2], "CropTyp"])
col3 = group_sum_area(NDVI_regular_preds, [NDVI_regular_preds.columns[3], "CropTyp"])
col4 = group_sum_area(NDVI_regular_preds, [NDVI_regular_preds.columns[4], "CropTyp"])

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
    NDVI_regular_crop_summary_LSD, SF_data_grp_area, on=(["CropTyp"]), how="left"
)

NDVI_regular_crop_summary_LSD.iloc[:,2:6] = 100 * (NDVI_regular_crop_summary_LSD.iloc[:,2:6].div(
                                                             NDVI_regular_crop_summary_LSD.ExctAcr, axis=0))

NDVI_regular_crop_summary_LSD.head(2)

# %%
EVI_regular_crop_summary_LSD = pd.DataFrame(columns=["label", "CropTyp"])

col1 = group_sum_area(EVI_regular_preds, [EVI_regular_preds.columns[1], "CropTyp"])
col2 = group_sum_area(EVI_regular_preds, [EVI_regular_preds.columns[2], "CropTyp"])
col3 = group_sum_area(EVI_regular_preds, [EVI_regular_preds.columns[3], "CropTyp"])
col4 = group_sum_area(EVI_regular_preds, [EVI_regular_preds.columns[4], "CropTyp"])

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
    EVI_regular_crop_summary_LSD, SF_data_grp_area, on=(["CropTyp"]), how="left"
)


EVI_regular_crop_summary_LSD.iloc[:,2:6] = 100 * (EVI_regular_crop_summary_LSD.iloc[:,2:6].div(
                                                                 EVI_regular_crop_summary_LSD.ExctAcr, axis=0))

EVI_regular_crop_summary_LSD.head(2)

# %%
EVI_regular_crop_summary_LSD_2cropped = EVI_regular_crop_summary_LSD[
                                                EVI_regular_crop_summary_LSD.label==2].copy()

NDVI_regular_crop_summary_LSD_2cropped = NDVI_regular_crop_summary_LSD[
                                                NDVI_regular_crop_summary_LSD.label==2].copy()

EVI_SG_crop_summary_LSD_2cropped = EVI_SG_crop_summary_LSD[
                                                EVI_SG_crop_summary_LSD.label==2].copy()

NDVI_SG_crop_summary_LSD_2cropped = NDVI_SG_crop_summary_LSD[
                                                NDVI_SG_crop_summary_LSD.label==2].copy()

# %%
NDVI_SG_crop_summary_LSD_2cropped.reset_index(inplace=True, drop=True)
EVI_SG_crop_summary_LSD_2cropped.reset_index(inplace=True, drop=True)

EVI_regular_crop_summary_LSD_2cropped.reset_index(inplace=True, drop=True)
NDVI_regular_crop_summary_LSD_2cropped.reset_index(inplace=True, drop=True)

# %%
print (NDVI_SG_crop_summary_LSD_2cropped.shape)
print (EVI_SG_crop_summary_LSD_2cropped.shape)

print (NDVI_regular_crop_summary_LSD_2cropped.shape)
print (EVI_regular_crop_summary_LSD_2cropped.shape)

NDVI_SG_crop_summary_LSD_2cropped.head(2)

# %%
NDVI_SG_crop_summary_LSD_2cropped.sort_values(by=["SVM_NDVI_SG_preds"], ascending=[False]).head(10)

# %%
print (f"{NDVI_SG_crop_summary_LSD_2cropped.SVM_NDVI_SG_preds.max()=}")
print (f"{NDVI_SG_crop_summary_LSD_2cropped.SVM_NDVI_SG_preds.idxmax()=}")
print ("")
print (NDVI_SG_crop_summary_LSD_2cropped.iloc[NDVI_SG_crop_summary_LSD_2cropped.SVM_NDVI_SG_preds.idxmax()])

# %%
NDVI_SG_crop_summary_LSD_2cropped.fillna(0, inplace=True)
EVI_SG_crop_summary_LSD_2cropped.fillna(0, inplace=True)
NDVI_regular_crop_summary_LSD_2cropped.fillna(0, inplace=True)
EVI_regular_crop_summary_LSD_2cropped.fillna(0, inplace=True)

# %%
# fig = plt.figure(figsize = (10, 5))
 
# # creating the bar plot
# plt.bar(NDVI_SG_crop_summary_LSD_2cropped.CropTyp, 
#         NDVI_SG_crop_summary_LSD_2cropped.SVM_NDVI_SG_preds, color ='dodgerblue',
#         width = 0.4)

# plt.xticks(rotation = 90)
# plt.xlabel("crop type")
# plt.ylabel("double-cropped acreage (%)")
# plt.title("SVM (5-step NDVI) predictions")
# plt.show()

# %%
NDVI_SG_crop_summary_LSD_2cropped = NDVI_SG_crop_summary_LSD_2cropped[\
                                          NDVI_SG_crop_summary_LSD_2cropped.CropTyp.isin(potentialCrops)]

EVI_SG_crop_summary_LSD_2cropped = EVI_SG_crop_summary_LSD_2cropped[\
                                            EVI_SG_crop_summary_LSD_2cropped.CropTyp.isin(potentialCrops)]

NDVI_regular_crop_summary_LSD_2cropped = NDVI_regular_crop_summary_LSD_2cropped[\
                                            NDVI_regular_crop_summary_LSD_2cropped.CropTyp.isin(potentialCrops)]

EVI_regular_crop_summary_LSD_2cropped = EVI_regular_crop_summary_LSD_2cropped[\
                                            EVI_regular_crop_summary_LSD_2cropped.CropTyp.isin(potentialCrops)]

# %%

# %%
df = NDVI_SG_crop_summary_LSD_2cropped.copy()
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

axs.set_ylabel("double-cropped acreage (%)")
axs.set_xlabel("crop")
axs.set_title("NDVI")
axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "NDVI_SG_AcreagePrecent_cropWise_potens_AllIrrDenom.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%

# %%
df = EVI_SG_crop_summary_LSD_2cropped.copy()
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

axs.set_ylabel("double-cropped acreage (%)")
axs.set_xlabel("crop")
axs.set_title("5-step EVI")
axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "EVI_SG_AcreagePrecent_cropWise_potens_AllIrrDenom.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%

# %%
df = EVI_regular_crop_summary_LSD_2cropped.copy()

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

axs.set_ylabel("double-cropped acreage (%)")
axs.set_xlabel("crop")
axs.set_title("4-step EVI")
axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "EVI_regular_AcreagePrecent_cropWise_potens_AllIrrDenom.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%

# %%
df = NDVI_regular_crop_summary_LSD_2cropped.copy()

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

axs.set_ylabel("double-cropped acreage (%)")
axs.set_xlabel("crop")
axs.set_title("4-step NDVI")
axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "NDVI_regular_AcreagePrecent_cropWise_potens_AllIrrDenom.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%

# %% [markdown]
# # Crop-Wise Count
#
# ### Compute total count per crop
# and remember small fields are out of study.

# %%
Adams = SF_data[SF_data.county=="Adams"]
print (Adams.LstSrvD.unique().min())
print (Adams.LstSrvD.unique().max())
sorted(list(SF_data.county.unique()))

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


# %%
def group_sum_area(df, group_cols):
    """ groups by two columns given by group_cols.
    The second column in group_cols must be "CropTyp"
    and the first one is something like
                  SVM_NDVI_SG_preds
                  SVM_NDVI_regular_preds
                  SVM_EVI_SG_preds
                  SVM_EVI_regular_preds
    """
    col = df.groupby([group_cols[0], group_cols[1]])['ExctAcr'].sum().reset_index(
                            name=group_cols[0]+'_acr_sum')
    col.rename(columns={group_cols[0]: "label",
                        group_cols[0]+'_acr_sum':group_cols[0]}, inplace=True)
    return (col)


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

NDVI_regular_crop_summary_LSD.head(2)

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
axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "NDVI_regular_countPerc_cropWise_potens_AllIrrDenom.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

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

NDVI_SG_crop_summary_LSD.head(2)

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
# axs.set_title("NDVI")
axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "NDVI_SG_countPerc_cropWise_potens_AllIrrDenom.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

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

EVI_SG_crop_summary_LSD.head(2)

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
axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "EVI_SG_countPerc_cropWise_potens_AllIrrDenom.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

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

EVI_regular_crop_summary_LSD.head(2)

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
axs.set_ylim([0, 105])
axs.legend(loc="best");
axs.xaxis.set_ticks_position('none')
file_name = plot_dir + "EVI_regular_countPerc_cropWise_potens_AllIrrDenom.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

plt.show()
del(df)

# %%

# %%

# %%

# %%
