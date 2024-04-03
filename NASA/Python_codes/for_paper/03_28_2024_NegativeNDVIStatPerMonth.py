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

# %%

# %%
import numpy as np
import pandas as pd
import time, datetime
import sys, os, os.path

from datetime import date, datetime
import matplotlib.pyplot as plt

# from patsy import cr
# from pprint import pprint
# from statsmodels.sandbox.regression.predstd import wls_prediction_std

sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as ncp

# %%
ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
train_TS_dir_base = "/Users/hn/Documents/01_research_data/NASA/VI_TS/"

# %%
meta = pd.read_csv(meta_dir + "evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
ground_truth_labels = pd.read_csv(ML_data_folder+"groundTruth_labels_Oct17_2022.csv")
ground_truth_labels = ground_truth_labels[ground_truth_labels.ID.isin(list(meta.ID.unique()))].copy()
ground_truth_labels = pd.merge(ground_truth_labels, meta, on=['ID'], how='left')
print ("ground_truth_labels shape is [{}].".format(ground_truth_labels.shape))
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print ("Minimum size is [{}].".format(round(ground_truth_labels.ExctAcr.min(), 3)))
print ("Number of unique fields are [{}].".format( len(ground_truth_labels.ID.unique())))
ground_truth_labels.head(2)

# %%
# %%time
######## Read Raw files
VI_idx = "NDVI"
raw_dir = train_TS_dir_base + "/data_for_train_individual_counties/"
list_files = os.listdir(raw_dir)
list_files = [x for x in list_files if x.endswith(".csv")]
list_files = [x for x in list_files if not("Monterey" in x)]

raw_TS = pd.DataFrame()
for a_file in list_files:
    df = pd.read_csv(raw_dir + a_file)
    df = df[df[VI_idx].notna()]
    df.drop(["EVI"], axis=1, inplace=True)
    raw_TS = pd.concat([raw_TS, df])

raw_TS["ID"] = raw_TS["ID"].astype(str)
raw_TS = nc.add_human_start_time_by_system_start_time(raw_TS)
raw_TS = nc.initial_clean(df=raw_TS, column_to_be_cleaned=VI_idx)
raw_TS.drop(columns=["system_start_time"], inplace=True)

raw_TS.reset_index(drop=True, inplace=True)
raw_TS.head(2)

# %% [markdown]
# ## pick proper years
#
# For each field we have 3 years worth of data

# %%
ext_split = raw_TS.ID.str.split("_", expand=True)
raw_TS["SF_year"] = ext_split[3]
raw_TS["SF_year"] = raw_TS["SF_year"].astype(int)

raw_TS["image_year"] = raw_TS.human_system_start_time.dt.year

# %%
print(len(raw_TS.ID.unique()))

# %%
raw_TS = raw_TS[raw_TS.image_year == raw_TS.SF_year]
raw_TS.reset_index(drop=True, inplace=True)

print(len(raw_TS.ID.unique()))
raw_TS.head(2)

# %%
raw_TS = raw_TS[raw_TS.NDVI < 0]
raw_TS.reset_index(drop=True, inplace=True)

print(len(raw_TS.ID.unique()))
raw_TS.head(2)

# %%
raw_TS["month"] = raw_TS.human_system_start_time.dt.month
raw_TS.head(2)

# %%
raw_TS.groupby("month").count().reset_index()

# %%

# %%

# %%
