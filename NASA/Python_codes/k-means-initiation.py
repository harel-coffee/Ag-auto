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
# initiate the k-means

# %%
import csv
import numpy as np
import pandas as pd

import datetime, time
from datetime import date

import sys, os, os.path
from os import listdir
from os.path import isfile, join

import re
# from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as npc

# %%
size = 20
title_FontSize = 10
legend_FontSize = 14
tick_FontSize = 18
label_FontSize = 14

params = {'legend.fontsize': 17,
          'figure.figsize': (6, 4),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size * 0.75,
          'ytick.labelsize': size * 0.75,
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
data_dir = "/Users/hn/Documents/01_research_data/NASA/data_deBug/05_SG_TS/"

# %%
df = pd.read_csv(data_dir + "SG_AdamBenton2016_EVI_JFD.csv")
df['human_system_start_time'] = pd.to_datetime(df['human_system_start_time'])

# %%
df.head(2)

# %% [markdown]
# # Harmonize the x-values!!!
#
# In the regularization step we bin the date range by looking at the range of images. Therefore, for a given field
# we might have a 10-day bin that goes from Jan 1 to Jan 10 and for another field it goes from Jan 8 to Jan 17. Therefore, let us just to assume the dates are the same as opposed to changing the code to fit the damn clustering step! Anyway, we are picking maximum value of a VI in a 10-day window. The date/x-values are not exact anyway!

# %%
fields_IDs = df.ID.unique()

# %%
field_0 = df[df.ID == fields_IDs[0]].copy()
field_0 = field_0[field_0['human_system_start_time'].dt.year == 2016].copy()
print (field_0.shape)
field_0.head(2)

# %%

# %%
field_1 = df[df.ID == fields_IDs[20]].copy()
field_1 = field_1[field_1['human_system_start_time'].dt.year == 2016].copy()
print (field_1.shape)

# %%
field_1 = df[df.ID == fields_IDs[21]]
field_1 = field_1[field_1['human_system_start_time'].dt.year == 2016].copy()
print (field_1.shape)

# %%
print (df.shape)
df.drop(field_1.index[35:], inplace=True)
print (df.shape)

# %%
print ("it took {:.0f} seconds to run this code.".format(0.1234))

# %%
a = 1201026676960 / 1000
time.strftime('%Y-%m-%d', time.localtime(a))

# %%
A = pd.read_csv("/Users/hn/Documents/01_research_data/NASA/" + \
                "VI_TS/07_2crop_acr/doubleAcr_perCounty_perCrop_EVI_JFD.csv")
A.head(2)

# %%
A.head(2)

# %%
A.shape

# %%
A.CropTyp.unique

# %%
B = A[A.CropTyp != "alfalfa hay"]

# %%
B.CropTyp.unique

# %%
