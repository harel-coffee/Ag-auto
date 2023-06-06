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
# In this notebook we create stats to go to [Google Sheet](https://bit.ly/43AWu3b).
#
# Here annuals are filtered by last survey date. Perennials are not filtered by last survey date. And alfalfa category will be done both ways. 
#
# We also need (exclude small fields altogether as well)
#
#   - Total irrigated acres (year-by-year)
#   - Total irrigated acres excluding alfalfa and other perennials (year-by-year)
#   - Total irrigated acres excluding only perennials (year-by-year)
#   
#   - DC all
#   - DC hay
#   - DC other Perennials

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

import pickle #, h5py
import sys, os, os.path
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc

# %%
dir_base = "/Users/hn/Documents/01_research_data/NASA/"
meta_dir = dir_base + "/parameters/"
SF_data_dir   = dir_base + "/data_part_of_shapefile/"
pred_dir_base = dir_base + "/RegionalStatData/"
pred_dir = pred_dir_base + "02_ML_preds_oversampled/"

plot_dir = dir_base + "for_paper/plots/ultimate/"
os.makedirs(plot_dir, exist_ok=True)

# %%
AnnualPerennialToss = pd.read_csv(meta_dir + "AnnualPerennialTossMay122023.csv")
AnnualPerennialToss.rename(columns={"Crop_Type": "CropTyp"}, inplace=True)
print (f"{AnnualPerennialToss.shape=}")
print (AnnualPerennialToss.potential.unique())

badCrops = list(AnnualPerennialToss.loc[AnnualPerennialToss.potential=="toss", "CropTyp"])
annual_crops = list(AnnualPerennialToss.loc[AnnualPerennialToss.potential=="y", "CropTyp"])
perennial_crops = list(AnnualPerennialToss.loc[AnnualPerennialToss.potential=="n", "CropTyp"])
hay_crops = list(AnnualPerennialToss.loc[AnnualPerennialToss.potential=="yn", "CropTyp"])

# %%

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
SF_data = SF_data[SF_data.ExctAcr>10].copy()
SF_data.reset_index(inplace=True, drop=True)

print (f"{SF_data.shape}")

SF_data.head(2)

# %%
### Filter by last survey date where most of fields are surveyed.
county_year_dict = {"Adams":2016, 
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
# print (SF_data_LSD.shape)

# %%
print (f"{SF_data.ExctAcr.min()}")
print ("")
(sorted(list(SF_data.Irrigtn.unique())))

# %% [markdown]
# ## Total Irrigated Acre in all counties

# %%
print (f"{SF_data.ExctAcr.sum()=}")
SF_data.groupby(["county"])['ExctAcr'].sum().round(2)

# %%
print (f"{SF_data_LSD.ExctAcr.sum()=}")
SF_data_LSD.groupby(["county"])['ExctAcr'].sum().round(2)

# %% [markdown]
# ### year by year total irrigated acre

# %%
print (f"{SF_data[SF_data['LstSrvD'].str.contains(str(2015))].ExctAcr.sum()=}")
print (f"{SF_data[SF_data['LstSrvD'].str.contains(str(2016))].ExctAcr.sum()=}")
print (f"{SF_data[SF_data['LstSrvD'].str.contains(str(2017))].ExctAcr.sum()=}")
print (f"{SF_data[SF_data['LstSrvD'].str.contains(str(2018))].ExctAcr.sum()=}")

# %% [markdown]
# # Excluding hay and perennials

# %%
annual_SF_data = SF_data[SF_data.CropTyp.isin(list(annual_crops))]
print (f"{annual_SF_data.ExctAcr.sum()=}")
annual_SF_data.groupby(["county"])['ExctAcr'].sum().round(2)

# %%
annual_SF_data_LSD = SF_data_LSD[SF_data_LSD.CropTyp.isin(list(annual_crops))]
print (f"{annual_SF_data_LSD.ExctAcr.sum()=}")
annual_SF_data_LSD.groupby(["county"])['ExctAcr'].sum().round(2)

# %%
print (f"{annual_SF_data[annual_SF_data['LstSrvD'].str.contains(str(2015))].ExctAcr.sum().round(2)= }")
print (f"{annual_SF_data[annual_SF_data['LstSrvD'].str.contains(str(2016))].ExctAcr.sum().round(2)= }")
print (f"{annual_SF_data[annual_SF_data['LstSrvD'].str.contains(str(2017))].ExctAcr.sum().round(2)= }")
print (f"{annual_SF_data[annual_SF_data['LstSrvD'].str.contains(str(2018))].ExctAcr.sum().round(2)= }")

# %%
print (f"{annual_SF_data_LSD[annual_SF_data_LSD['LstSrvD'].str.contains(str(2015))].ExctAcr.sum().round(2)= }")
print (f"{annual_SF_data_LSD[annual_SF_data_LSD['LstSrvD'].str.contains(str(2016))].ExctAcr.sum().round(2)= }")
print (f"{annual_SF_data_LSD[annual_SF_data_LSD['LstSrvD'].str.contains(str(2017))].ExctAcr.sum().round(2)= }")
print (f"{annual_SF_data_LSD[annual_SF_data_LSD['LstSrvD'].str.contains(str(2018))].ExctAcr.sum().round(2)= }")

# %%
B = list(annual_SF_data_LSD[annual_SF_data_LSD.county=="Benton"].ID.unique())

# %%
BB = list(annual_SF_data[annual_SF_data.county=="Benton"].ID.unique())

# %%
missing_fields = [x for x in BB if not(x in B)]

# %%
annual_SF_data[annual_SF_data.ID.isin(missing_fields)]

# %%

# %% [markdown]
# # Exclude perennials

# %%
noPerennial_SF_data = SF_data[~SF_data.CropTyp.isin(perennial_crops)]
print (f"{noPerennial_SF_data.ExctAcr.sum().round(2) = }")
print ()
noPerennial_SF_data.groupby(["county"])['ExctAcr'].sum().round(2)

# %%
noPerennial_SF_data_LSD = SF_data_LSD[~SF_data_LSD.CropTyp.isin(perennial_crops)]
print (f"{noPerennial_SF_data_LSD.ExctAcr.sum().round(2)=}")
print ()
noPerennial_SF_data_LSD.groupby(["county"])['ExctAcr'].sum().round(2)

# %%
print (f"{noPerennial_SF_data[noPerennial_SF_data['LstSrvD'].str.contains(str(2015))].ExctAcr.sum().round(2) = }")
print (f"{noPerennial_SF_data[noPerennial_SF_data['LstSrvD'].str.contains(str(2016))].ExctAcr.sum().round(2) = }")
print (f"{noPerennial_SF_data[noPerennial_SF_data['LstSrvD'].str.contains(str(2017))].ExctAcr.sum().round(2) = }")
print (f"{noPerennial_SF_data[noPerennial_SF_data['LstSrvD'].str.contains(str(2018))].ExctAcr.sum().round(2) = }")

# %%
print (f"{noPerennial_SF_data_LSD[noPerennial_SF_data_LSD['LstSrvD'].str.contains(str(2015))].ExctAcr.sum().round(2) = }")
print (f"{noPerennial_SF_data_LSD[noPerennial_SF_data_LSD['LstSrvD'].str.contains(str(2016))].ExctAcr.sum().round(2) = }")
print (f"{noPerennial_SF_data_LSD[noPerennial_SF_data_LSD['LstSrvD'].str.contains(str(2017))].ExctAcr.sum().round(2) = }")
print (f"{noPerennial_SF_data_LSD[noPerennial_SF_data_LSD['LstSrvD'].str.contains(str(2018))].ExctAcr.sum().round(2) = }")

# %% [markdown]
# # Crop-wise Acre and Count
#
# Here I will filter by last survey date because I am putting them in a table with ML predictions.
#
# So, labels must be correct. Moreover, if 2010 was an outlier year for Walla Walla in a sense that everyone grew alfalfa, then the carried over labels to 2015 will skew the stats. So, I want to keep the pools identical.

# %%
print (f"{SF_data.shape}")
print ()
print (f"{sorted(list(SF_data.Irrigtn.unique()))}")
print ()
SF_data.head(2)

# %%
A = (SF_data.groupby(["CropTyp"])['ExctAcr'].sum().round(2))

out_name = "/Users/hn/Desktop/cropWise_acr_googleSheet_irr_LSD.csv"
A = pd.DataFrame(A)
A.reset_index(inplace=True)
A = pd.merge(AnnualPerennialToss, A, on="CropTyp", how='left')
A.to_csv(out_name, index = False)

# %%
A = SF_data.groupby(["CropTyp"])['ID'].count()

out_name = "/Users/hn/Desktop/cropWise_count_googleSheet_irr_LSD.csv"
A = pd.DataFrame(A)
A.reset_index(inplace=True)
A = pd.merge(AnnualPerennialToss, A, on="CropTyp", how='left')
A.fillna(0, inplace=True)
A.ID = A.ID.astype(int)
A.to_csv(out_name, index = False)
print (A.shape)

# %%
len(A.CropTyp)

# %%

# %% [markdown]
# # Read predictions
#
# **Predictions are only irrigated**

# %%
# out_name = pred_dir_base + "NDVI_regular_preds_overSample.csv"
# NDVI_regular_preds = pd.read_csv(out_name)

# out_name = pred_dir_base + "EVI_regular_preds_overSample.csv"
# EVI_regular_preds = pd.read_csv(out_name)

# out_name = pred_dir_base + "EVI_SG_preds_overSample.csv"
# EVI_SG_preds = pd.read_csv(out_name)

out_name = pred_dir_base + "NDVI_SG_preds_overSample.csv"
NDVI_SG_preds = pd.read_csv(out_name)
print ("after reading: ", NDVI_SG_preds.shape)
NDVI_SG_preds = nc.filter_out_nonIrrigated(NDVI_SG_preds)
print ("after irrigation filtering: ", NDVI_SG_preds.shape)
NDVI_SG_preds = NDVI_SG_preds[NDVI_SG_preds.ExctAcr>10]
print ("after field size filtering: ", NDVI_SG_preds.shape)

# sorted(list(NDVI_SG_preds.Irrigtn.unique()))

# %%
print (len(sorted(list(NDVI_SG_preds.CropTyp.unique()))))
NDVI_SG_preds.head(2)

# %%
NDVI_SG_preds_apple = NDVI_SG_preds[NDVI_SG_preds.CropTyp=="apple"].copy()
NDVI_SG_preds_apple.shape

# %%
A = (NDVI_SG_preds[NDVI_SG_preds.DL_NDVI_SG_prob_point3==2].groupby(["CropTyp"])['ExctAcr'].sum().round(2))
A = pd.DataFrame(A)
A.rename(columns={"ExctAcr": "ExctAcrSum_DL2"}, inplace=True)
A.reset_index(inplace=True)
A = pd.merge(AnnualPerennialToss, A, on="CropTyp", how='left')
A = A[["CropTyp", "ExctAcrSum_DL2"]]

B = (NDVI_SG_preds[NDVI_SG_preds.SVM_NDVI_SG_preds==2].groupby(["CropTyp"])['ExctAcr'].sum().round(2))
B = pd.DataFrame(B)
B.rename(columns={"ExctAcr": "ExctAcrSum_SVM2"}, inplace=True)
B.reset_index(inplace=True)
A = pd.merge(A, B, on="CropTyp", how='left')

B = (NDVI_SG_preds[NDVI_SG_preds.RF_NDVI_SG_preds==2].groupby(["CropTyp"])['ExctAcr'].sum().round(2))
B = pd.DataFrame(B)
B.rename(columns={"ExctAcr": "ExctAcrSum_RF2"}, inplace=True)
B.reset_index(inplace=True)
A = pd.merge(A, B, on="CropTyp", how='left')

B = (NDVI_SG_preds[NDVI_SG_preds.KNN_NDVI_SG_preds==2].groupby(["CropTyp"])['ExctAcr'].sum().round(2))
B = pd.DataFrame(B)
B.rename(columns={"ExctAcr": "ExctAcrSum_KNN2"}, inplace=True)
B.reset_index(inplace=True)
A = pd.merge(A, B, on="CropTyp", how='left')

A.fillna(0, inplace=True)
out_name = "/Users/hn/Desktop/cropWise_acr_googleSheet_irr_LSD_NDVI_SG.csv"
A.to_csv(out_name, index = False)

# %%
A = NDVI_SG_preds[NDVI_SG_preds.DL_NDVI_SG_prob_point3==2].groupby(["CropTyp"])['ID'].count()
A = pd.DataFrame(A)
A.rename(columns={"ID": "fieldCount_DL2"}, inplace=True)
A.reset_index(inplace=True)
A = pd.merge(AnnualPerennialToss, A, on="CropTyp", how='left')
A = A[["CropTyp", "fieldCount_DL2"]]

B = NDVI_SG_preds[NDVI_SG_preds.SVM_NDVI_SG_preds==2].groupby(["CropTyp"])['ID'].count()
B = pd.DataFrame(B)
B.rename(columns={"ID": "fieldCount_SVM2"}, inplace=True)
B.reset_index(inplace=True)
B = B[["CropTyp", "fieldCount_SVM2"]]
A = pd.merge(A, B, on="CropTyp", how='left')

B = NDVI_SG_preds[NDVI_SG_preds.RF_NDVI_SG_preds==2].groupby(["CropTyp"])['ID'].count()
B = pd.DataFrame(B)
B.rename(columns={"ID": "fieldCount_RF2"}, inplace=True)
B.reset_index(inplace=True)
B = B[["CropTyp", "fieldCount_RF2"]]
A = pd.merge(A, B, on="CropTyp", how='left')

B = NDVI_SG_preds[NDVI_SG_preds.KNN_NDVI_SG_preds==2].groupby(["CropTyp"])['ID'].count()
B = pd.DataFrame(B)
B.rename(columns={"ID": "fieldCount_KNN2"}, inplace=True)
B.reset_index(inplace=True)
B = B[["CropTyp", "fieldCount_KNN2"]]
A = pd.merge(A, B, on="CropTyp", how='left')

A.fillna(0, inplace=True)
out_name = "/Users/hn/Desktop/cropWise_count_googleSheet_irr_LSD_NDVI_SG.csv"
A.to_csv(out_name, index = False)

# %% [markdown]
# # Question
#
# Of all double-cropped fields that are labeled by ML, how much of them are hay-crops:

# %%
NDVI_SG_preds = NDVI_SG_preds[~NDVI_SG_preds.CropTyp.isin(badCrops)]

# %%
NDVI_SG_preds_hay = NDVI_SG_preds[NDVI_SG_preds.CropTyp.isin(hay_crops)].copy()
NDVI_SG_preds_hay.head(2)

# %%
# SF_data_LSD.groupby(["county"])['ExctAcr'].sum().round(2)
print (f"{NDVI_SG_preds[NDVI_SG_preds.DL_NDVI_SG_prob_point3==2].ExctAcr.sum().round(1)=}")
print (f"{NDVI_SG_preds[NDVI_SG_preds.SVM_NDVI_SG_preds==2].ExctAcr.sum().round(1)=}")
print (f"{NDVI_SG_preds[NDVI_SG_preds.RF_NDVI_SG_preds==2].ExctAcr.sum().round(1)=}")
print (f"{NDVI_SG_preds[NDVI_SG_preds.KNN_NDVI_SG_preds==2].ExctAcr.sum().round(1)=}")

# %%
# SF_data_LSD.groupby(["county"])['ExctAcr'].sum().round(2)
print (f"{NDVI_SG_preds_hay[NDVI_SG_preds_hay.DL_NDVI_SG_prob_point3==2].ExctAcr.sum().round(1)=}")
print (f"{NDVI_SG_preds_hay[NDVI_SG_preds_hay.SVM_NDVI_SG_preds==2].ExctAcr.sum().round(1)=}")
print (f"{NDVI_SG_preds_hay[NDVI_SG_preds_hay.RF_NDVI_SG_preds==2].ExctAcr.sum().round(1)=}")
print (f"{NDVI_SG_preds_hay[NDVI_SG_preds_hay.KNN_NDVI_SG_preds==2].ExctAcr.sum().round(1)=}")

# %% [markdown]
# ### count

# %%
# SF_data_LSD.groupby(["county"])['ExctAcr'].sum().round(2)
print (f"{NDVI_SG_preds[NDVI_SG_preds.DL_NDVI_SG_prob_point3==2].ID.count()=}")
print (f"{NDVI_SG_preds[NDVI_SG_preds.SVM_NDVI_SG_preds==2].ID.count()=}")
print (f"{NDVI_SG_preds[NDVI_SG_preds.RF_NDVI_SG_preds==2].ID.count()=}")
print (f"{NDVI_SG_preds[NDVI_SG_preds.KNN_NDVI_SG_preds==2].ID.count()=}")

# %%
# SF_data_LSD.groupby(["county"])['ExctAcr'].sum().round(2)
print (f"{NDVI_SG_preds_hay[NDVI_SG_preds_hay.DL_NDVI_SG_prob_point3==2].ID.count()=}")
print (f"{NDVI_SG_preds_hay[NDVI_SG_preds_hay.SVM_NDVI_SG_preds==2].ID.count()=}")
print (f"{NDVI_SG_preds_hay[NDVI_SG_preds_hay.RF_NDVI_SG_preds==2].ID.count()=}")
print (f"{NDVI_SG_preds_hay[NDVI_SG_preds_hay.KNN_NDVI_SG_preds==2].ID.count()=}")

# %% [markdown]
# ### Perennials

# %%
NDVI_SG_preds_perennials = NDVI_SG_preds[NDVI_SG_preds.CropTyp.isin(perennial_crops)].copy()
NDVI_SG_preds_perennials.head(2)

# %%
# SF_data_LSD.groupby(["county"])['ExctAcr'].sum().round(2)
print (f"{NDVI_SG_preds_perennials[NDVI_SG_preds_perennials.DL_NDVI_SG_prob_point3==2].ExctAcr.sum().round(1)=}")
print (f"{NDVI_SG_preds_perennials[NDVI_SG_preds_perennials.SVM_NDVI_SG_preds==2].ExctAcr.sum().round(1)=}")
print (f"{NDVI_SG_preds_perennials[NDVI_SG_preds_perennials.RF_NDVI_SG_preds==2].ExctAcr.sum().round(1)=}")
print (f"{NDVI_SG_preds_perennials[NDVI_SG_preds_perennials.KNN_NDVI_SG_preds==2].ExctAcr.sum().round(1)=}")

# %%
# SF_data_LSD.groupby(["county"])['ExctAcr'].sum().round(2)
print (f"{NDVI_SG_preds_perennials[NDVI_SG_preds_perennials.DL_NDVI_SG_prob_point3==2].ID.count()=}")
print (f"{NDVI_SG_preds_perennials[NDVI_SG_preds_perennials.SVM_NDVI_SG_preds==2].ID.count()=}")
print (f"{NDVI_SG_preds_perennials[NDVI_SG_preds_perennials.RF_NDVI_SG_preds==2].ID.count()=}")
print (f"{NDVI_SG_preds_perennials[NDVI_SG_preds_perennials.KNN_NDVI_SG_preds==2].ID.count()=}")

# %% [markdown]
# ### Annuals

# %%
NDVI_SG_preds_annuals = NDVI_SG_preds[NDVI_SG_preds.CropTyp.isin(annual_crops)].copy()
NDVI_SG_preds_annuals.head(2)

# %%
print (NDVI_SG_preds.shape)
print ()
NDVI_SG_preds_annuals.shape[0] + NDVI_SG_preds_perennials.shape[0] + NDVI_SG_preds_hay.shape[0]

# %%
# SF_data_LSD.groupby(["county"])['ExctAcr'].sum().round(2)
print (f"{NDVI_SG_preds_annuals[NDVI_SG_preds_annuals.DL_NDVI_SG_prob_point3==2].ExctAcr.sum().round(1)=}")
print (f"{NDVI_SG_preds_annuals[NDVI_SG_preds_annuals.SVM_NDVI_SG_preds==2].ExctAcr.sum().round(1)=}")
print (f"{NDVI_SG_preds_annuals[NDVI_SG_preds_annuals.RF_NDVI_SG_preds==2].ExctAcr.sum().round(1)=}")
print (f"{NDVI_SG_preds_annuals[NDVI_SG_preds_annuals.KNN_NDVI_SG_preds==2].ExctAcr.sum().round(1)=}")

# %%
# SF_data_LSD.groupby(["county"])['ExctAcr'].sum().round(2)
print (f"{NDVI_SG_preds_annuals[NDVI_SG_preds_annuals.DL_NDVI_SG_prob_point3==2].ID.count()=}")
print (f"{NDVI_SG_preds_annuals[NDVI_SG_preds_annuals.SVM_NDVI_SG_preds==2].ID.count()=}")
print (f"{NDVI_SG_preds_annuals[NDVI_SG_preds_annuals.RF_NDVI_SG_preds==2].ID.count()=}")
print (f"{NDVI_SG_preds_annuals[NDVI_SG_preds_annuals.KNN_NDVI_SG_preds==2].ID.count()=}")

# %%
