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
# trend_widen_pre2008 has thrown an error on Kamiak. Lets see what is wrong.

# %%
import shutup, random, time

shutup.please()

import numpy as np
import pandas as pd

from datetime import date, datetime
from random import seed, random

import sys, os, os.path, shutil


# %%
# VI_idx = sys.argv[1]
# smooth = sys.argv[2]
# batch_no = str(sys.argv[3])

VI_idx = "NDVI"
smooth = "SG"
batch_no = "9"


# %%
data_base = "/data/project/agaid/h.noorazar/NASA/"
data_base = "/Users/hn/Documents/01_research_data/NASA/"
if smooth == "regular":
    in_dir = data_base + "VI_TS/04_regularized_TS/"
else:
    in_dir = data_base + "VI_TS/05_SG_TS/"

# %%
f_name = VI_idx + "_" + smooth + "_" + "intersect_batchNumber" + batch_no + "_JFD_pre2008.csv"
data = pd.read_csv(in_dir + f_name)
data["human_system_start_time"] = pd.to_datetime(data["human_system_start_time"])


# %%
# %%time

VI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37)]
columnNames = ["ID", "year"] + VI_colnames

years = data.human_system_start_time.dt.year.unique()
IDs = data.ID.unique()
no_rows = len(IDs) * len(years)

data_wide = pd.DataFrame(columns=columnNames, index=range(no_rows))
data_wide.ID = list(IDs) * len(years)
data_wide.sort_values(by=["ID"], inplace=True)
data_wide.reset_index(drop=True, inplace=True)
data_wide.year = list(years) * len(IDs)


for an_ID in IDs:
    curr_field = data[data.ID == an_ID]
    curr_years = curr_field.human_system_start_time.dt.year.unique()
    for a_year in curr_years:
        curr_field_year = curr_field[curr_field.human_system_start_time.dt.year == a_year]
        data_wide_indx = data_wide[(data_wide.ID == an_ID) & (data_wide.year == a_year)].index

        if VI_idx == "EVI":
            V = curr_field_year.EVI.values
            if len(V) >= 36:
                data_wide.loc[data_wide_indx, "EVI_1":"EVI_36"] = V[:36]
            else:
                currV = list(V) + [V[-1]] * (36 - len(V))
                data_wide.loc[data_wide_indx, "EVI_1":"EVI_36"] = currV
        elif VI_idx == "NDVI":
            V = curr_field_year.NDVI.values
            if len(V) >= 36:
                data_wide.loc[data_wide_indx, "NDVI_1":"NDVI_36"] = V[:36]
            else:
                currV = list(V) + [V[-1]] * (36 - len(V))
                data_wide.loc[data_wide_indx, "NDVI_1":"NDVI_36"] = currV

# %%
a_year

# %%
curr_field_year = curr_field[curr_field.human_system_start_time.dt.year == a_year]

# %%
curr_field_year.shape

# %%
A = curr_field[curr_field.human_system_start_time.dt.year == 1990]

# %%
curr_field_year.head(4)

# %%
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pylab import imshow
from matplotlib.image import imread



fig, ax = plt.subplots(1, 1, figsize=(15, 4), sharex=False, sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
ax.grid(True);
ax.plot(curr_field_year['human_system_start_time'], curr_field_year["NDVI"], linewidth=4, \
        color="dodgerblue", linestyle="solid") 

# %%
currV = curr_field_year.NDVI.values

# %%

# %%
currV

# %%
data_wide.loc[data_wide_indx, "NDVI_1":"NDVI_36"] = currV

# %%
data_wide

# %%
type(V)

# %%
