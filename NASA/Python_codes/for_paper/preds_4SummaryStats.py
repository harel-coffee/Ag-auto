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

# %%
import numpy as np
import pandas as pd
import scipy, scipy.signal

from datetime import date
import time

import random
from random import seed, random
# import shutil

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

import pickle #, h5py
import sys, os, os.path
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core as rcp

# %%
from tslearn.metrics import dtw as dtw_metric
# https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

# %% [markdown]
# # Read Fields Metadata

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]

print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
SF_data_dir = "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/"
f_names=["AdamBenton2016.csv", "FranklinYakima2018.csv", "Grant2017.csv", "Walla2015.csv"]

SF_data=pd.DataFrame()

for file in f_names:
    curr_file=pd.read_csv(SF_data_dir + file)
    SF_data=pd.concat([SF_data, curr_file])

# %%
len(SF_data.ID.unique())

# %% [markdown]
# # Read Ground-Truth Labels

# %%
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models_Oct17/"
os.makedirs(model_dir, exist_ok=True)

training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"

# %%

# %% [markdown]
# # Read the Data and Widen

# %%
# %%time
VI_idx = "EVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/04_regularized_TS/"

file_names = ["regular_Walla2015_" + VI_idx + "_JFD.csv", "regular_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              "regular_Grant2017_" + VI_idx + "_JFD.csv", "regular_FranklinYakima2018_" + VI_idx + "_JFD.csv"]

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

EVI_4step = data.copy()
del(data)
EVI_4step.head(2)

EVI_4step.head(2)

# %%
SF_data.head(2)

# %%
EVI_4step.head(2)

# %%
EVI_4step = pd.merge(EVI_4step, SF_data[["ID", "CropTyp", "Irrigtn", "DataSrc", "Acres", "LstSrvD", "county"]], 
                     on=['ID'], how='left')

# %%
EVI_4step_Grant = EVI_4step[EVI_4step.county=="Grant"].copy()

# %%
sorted(EVI_4step_Grant.Irrigtn.unique())

# %%
sorted(EVI_4step.Irrigtn.unique())

# %%
sorted(EVI_4step.DataSrc.unique())

# %%
print (len(EVI_4step.ID.unique()))
print (len(SF_data.ID.unique()))

# %%
large_fields = EVI_4step[EVI_4step.Acres>10]
len(large_fields.ID.unique())

# %%
len(EVI_4step[EVI_4step.Irrigtn=="center pivot/none"].ID.unique())+\
len(EVI_4step[EVI_4step.Irrigtn=="none"].ID.unique())+\
len(EVI_4step[EVI_4step.Irrigtn=="none/rill"].ID.unique())+\
len(EVI_4step[EVI_4step.Irrigtn=="none/sprinkler"].ID.unique())+\
len(EVI_4step[EVI_4step.Irrigtn=="unknown"].ID.unique())

# %%
len(large_fields[large_fields.Irrigtn=="center pivot/none"].ID.unique())+\
len(large_fields[large_fields.Irrigtn=="none"].ID.unique())+\
len(large_fields[large_fields.Irrigtn=="none/rill"].ID.unique())+\
len(large_fields[large_fields.Irrigtn=="none/sprinkler"].ID.unique())+\
len(large_fields[large_fields.Irrigtn=="unknown"].ID.unique())

# %%
print (len(EVI_4step[EVI_4step.DataSrc=="nass"].ID.unique()))
print (len(large_fields[large_fields.DataSrc=="nass"].ID.unique()))

# %%
EVI_4step.head(2)

# %%
SF_data_Grant = SF_data[SF_data.county=="Grant"].copy()

SF_data_Grant_irr = nc.filter_out_nonIrrigated(SF_data_Grant)
SF_data_Grant_irr_large = SF_data_Grant_irr[SF_data_Grant_irr.Acres>10].copy()

print (f"{SF_data_Grant.ExctAcr.sum().round() = }")
print (f"{SF_data_Grant_irr.ExctAcr.sum().round() = }")
print (f"{SF_data_Grant_irr_large.ExctAcr.sum().round() = }")

print ("---------------------------------------------------------------------")
SF_data_Grant = SF_data[SF_data.county=="Grant"].copy()
SF_data_Grant_srv = nc.filter_by_lastSurvey(SF_data_Grant, 2017)
SF_data_Grant_irr = nc.filter_out_nonIrrigated(SF_data_Grant_srv)
SF_data_Grant_irr_large = SF_data_Grant_irr[SF_data_Grant_irr.Acres>10].copy()

print (f"{SF_data_Grant.ExctAcr.sum().round() = }")
print (f"{SF_data_Grant_srv.ExctAcr.sum().round() = }")
print (f"{SF_data_Grant_irr.ExctAcr.sum().round() = }")
print (f"{SF_data_Grant_irr_large.ExctAcr.sum().round() = }")

# %%
county_ = "Grant"
year_ = 2017

SF_data_Grant = SF_data[SF_data.county==county_].copy()
print (f"{len(SF_data_Grant.ID.unique() )= }")
print (f"{SF_data_Grant.ExctAcr.sum().round() = }")

SF_data_Grant_irr = nc.filter_out_nonIrrigated(SF_data_Grant)
print (f"{SF_data_Grant_irr.ExctAcr.sum().round() = }")

SF_data_Grant_irr_large = SF_data_Grant_irr[SF_data_Grant_irr.Acres>10].copy()
print (f"{SF_data_Grant_irr_large.ExctAcr.sum().round() = }")

SF_data_Grant_irr_large_year = nc.filter_by_lastSurvey(SF_data_Grant_irr_large, year_) 
print (f"{SF_data_Grant_irr_large_year.ExctAcr.sum().round() = }")

print ()
print ("---------------------------------------------------------------------")
SF_data_Grant = SF_data[SF_data.county==county_].copy()
SF_data_Grant_srv = nc.filter_by_lastSurvey(SF_data_Grant, year_)
SF_data_Grant_irr = nc.filter_out_nonIrrigated(SF_data_Grant_srv)
SF_data_Grant_irr_large = SF_data_Grant_irr[SF_data_Grant_irr.Acres>10].copy()

print (f"{SF_data_Grant.ExctAcr.sum().round() = }")
print (f"{SF_data_Grant_srv.ExctAcr.sum().round() = }")
print (f"{SF_data_Grant_irr.ExctAcr.sum().round() = }")
print (f"{SF_data_Grant_irr_large.ExctAcr.sum().round() = }")

# %%
SF_data.county.unique()

# %%
EVI_4step_irr = nc.filter_out_nonIrrigated(EVI_4step)
sorted(EVI_4step_irr.Irrigtn.unique())

# %%
(1 - (153721/193585))*100

# %%
(1 - (142527/240360))*100

# %%

# %%

# %%

#
#     Widen
#
VI_idx = "EVI"
EVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + EVI_colnames
EVI_4step_wide = pd.DataFrame(columns=columnNames, index=range(len(EVI_4step.ID.unique())))
EVI_4step_wide["ID"] = EVI_4step.ID.unique()

for an_ID in EVI_4step.ID.unique():
    curr_df = EVI_4step[EVI_4step.ID==an_ID]
    
    EVI_4step_wide_indx = EVI_4step_wide[EVI_4step_wide.ID==an_ID].index
    EVI_4step_wide.loc[EVI_4step_wide_indx, "EVI_1":"EVI_36"] = curr_df.EVI.values[:36]

# %%
# %%time
VI_idx = "EVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"

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
EVI_5step = data.copy()
del(data)

EVI_5step.head(2)
#
#     Widen
#
EVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + EVI_colnames
EVI_5step_wide = pd.DataFrame(columns=columnNames, index=range(len(EVI_5step.ID.unique())))
EVI_5step_wide["ID"] = EVI_5step.ID.unique()

for an_ID in EVI_5step.ID.unique():
    curr_df = EVI_5step[EVI_5step.ID==an_ID]
    
    EVI_5step_wide_indx = EVI_5step_wide[EVI_5step_wide.ID==an_ID].index
    EVI_5step_wide.loc[EVI_5step_wide_indx, "EVI_1":"EVI_36"] = curr_df.EVI.values[:36]

# %%

# %%
# %%time

VI_idx = "NDVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/04_regularized_TS/"
file_names = ["regular_Walla2015_" + VI_idx + "_JFD.csv", "regular_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              "regular_Grant2017_" + VI_idx + "_JFD.csv", "regular_FranklinYakima2018_" + VI_idx + "_JFD.csv"]

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

NDVI_4step = data.copy()
del(data)

NDVI_4step.head(2)
#
#     Widen
#
NDVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + NDVI_colnames
NDVI_4step_wide = pd.DataFrame(columns=columnNames, index=range(len(NDVI_4step.ID.unique())))
NDVI_4step_wide["ID"] = NDVI_4step.ID.unique()

for an_ID in NDVI_4step.ID.unique():
    curr_df = NDVI_4step[NDVI_4step.ID==an_ID]
    
    NDVI_4step_wide_indx = NDVI_4step_wide[NDVI_4step_wide.ID==an_ID].index
    NDVI_4step_wide.loc[NDVI_4step_wide_indx, "NDVI_1":"NDVI_36"] = curr_df.NDVI.values[:36]

# %%
VI_idx = "NDVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"
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
NDVI_5step = data.copy()
del(data)

NDVI_5step.head(2)
#
#     Widen
#
NDVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + NDVI_colnames
NDVI_5step_wide = pd.DataFrame(columns=columnNames, index=range(len(NDVI_5step.ID.unique())))
NDVI_5step_wide["ID"] = NDVI_5step.ID.unique()

for an_ID in NDVI_5step.ID.unique():
    curr_df = NDVI_5step[NDVI_5step.ID==an_ID]
    
    NDVI_5step_wide_indx = NDVI_5step_wide[NDVI_5step_wide.ID==an_ID].index
    NDVI_5step_wide.loc[NDVI_5step_wide_indx, "NDVI_1":"NDVI_36"] = curr_df.NDVI.values[:36]

# %%

# %%
len(NDVI_5step.ID.unique())

# %%
sorted(SF_data.ID.unique())==sorted(NDVI_5step.ID.unique())

# %%
print (f"{len(SF_data.CropTyp.unique())=}")
print (f"{len(meta.CropTyp.unique())=}")

# %%

# %%
