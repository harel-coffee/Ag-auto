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
import pandas as pd
import numpy as np
import os, sys
from datetime import date, datetime


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

import scipy, scipy.signal
import pickle, h5py

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc


# %%
# We need this for KNN
def DTW_prune(ts1, ts2):
    d, _ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True)
    return d


# We need this for DL
# load and prepare the image
def load_image(filename):
    img = load_img(filename, target_size=(224, 224))  # load the image
    img = img_to_array(img)  # convert to array
    img = img.reshape(1, 224, 224, 3)  # reshape into a single sample with 3 channels
    img = img.astype("float32")  # center pixel data
    img = img - [123.68, 116.779, 103.939]
    return img



# %%
VI_TS_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/09_WSU_students2022Survey/"

# %%
file_names = ["L7_T1C2L2_WSUStudents2022_2022-01-01_2023-01-01.csv",
              "L8_T1C2L2_WSUStudents2022_2022-01-01_2023-01-01.csv"]

# %%
VI_TS_df = pd.DataFrame()
for f_ in file_names:
    df = pd.read_csv(VI_TS_dir+f_)
    VI_TS_df = pd.concat([VI_TS_df, df])

# %%
VI_TS_df.rename(columns={"GlobalID": "ID"}, inplace=True)

IDs = VI_TS_df.ID.unique()

# %%
EVI_TS_df = VI_TS_df.copy()
NDVI_TS_df = VI_TS_df.copy()

EVI_TS_df = EVI_TS_df[['ID', 'EVI', 'system_start_time']]
NDVI_TS_df = NDVI_TS_df[['ID', 'NDVI', 'system_start_time']]

# %%
EVI_TS_df.dropna(subset=["EVI"], inplace=True)
NDVI_TS_df.dropna(subset=["NDVI"], inplace=True)

EVI_TS_df.sort_values(by=["ID", "system_start_time"], inplace=True)
NDVI_TS_df.sort_values(by=["ID", "system_start_time"], inplace=True)

EVI_TS_df.reset_index(drop=True, inplace=True)
NDVI_TS_df.reset_index(drop=True, inplace=True)

# %%
NDVI_TS_df

# %%

# %%
