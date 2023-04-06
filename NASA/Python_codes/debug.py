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
import csv
import numpy as np
import pandas as pd

import datetime, time
from datetime import date

import os, os.path
from os import listdir
from os.path import isfile, join
import sys

import re
# from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sb

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
data_dir = "/Users/hn/Documents/01_research_data/NASA/data_deBug/"

# %%
# df = pd.read_csv(data_dir + "00_noOutlier_int_Grant_Irr_2008_2018_EVI_500randomfields.csv")
# df.dropna(inplace=True)

# %% [markdown]
# ### Check SG result

# %%
df = pd.read_csv(data_dir + "04_SG_int_Grant_Irr_2008_2018_EVI_100randomfields.csv")
df['human_system_start_time'] = pd.to_datetime(df['human_system_start_time'])

# %%
IDs = df.ID.unique()
len(IDs)

# %%
IDs[0]

# %%
curr_field = df[df['ID']==IDs[0]].copy()

# %%
curr_field.EVI.max()

# %%
regularized_TS.tail(0)

# %%
fig, ax = plt.subplots(1, 1, figsize=(30, 4),
                        sharex='col', sharey='row',
                        # sharex=True, sharey=True,
                        gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(curr_field['human_system_start_time'], 
        curr_field['EVI'], 
        '-', linewidth=5, color='dodgerblue', label = "SG")
ax.xaxis.set_major_locator(mdates.YearLocator(1)) # every year.
ax.legend(loc="upper left");
ax.set_ylim(-1, 1)

plot_dir = "/Users/hn/Desktop/"
file_name = plot_dir + "i17302.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False)

# %%
curr_field.loc[403:408, ]

# %%
regular = pd.read_csv(data_dir + "03_regular_int_Grant_Irr_2008_2018_EVI_100randomfields.csv")
regular['human_system_start_time'] = pd.to_datetime(regular['human_system_start_time'])

# %%
SF_data = pd.read_csv("/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/Grant2017.csv")

# %%
Grant = pd.read_csv("/Users/hn/Documents/01_research_data/NASA/data_deBug/SG_Grant2017_NDVI.csv")

# %%
Grant.head(2)

# %%
SF_data.head(2)

# %%
# check =  all(item in list(Grant.ID) for item in list(SF_data.ID))
# check

# %%
print (len(Grant.ID))
print (len(Grant.ID.unique()))
print (Grant.shape)
Grant.head(2)

# %%
SF_data.shape

# %%
SF_data = nc.filter_by_lastSurvey(SF_data, year = "2017") 
SF_data = nc.filter_out_NASS(SF_data)         # Toss NASS
SF_data = nc.filter_out_nonIrrigated(SF_data) # keep only irrigated lands

# %%
SF_data.shape

# %%
fuck = list(SF_data.ID)

# %%
Grant.shape

# %%
Grant = Grant[Grant.ID.isin(fuck)]

# %%
df = pd.read_csv("/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/Monterey2014.csv")

# %%
df.head(2)

# %%
df.Crop2014.unique()

# %% [markdown]
# # Scratch for double-crop acreage

# %%
SF_data_dir = "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/"
data_dir = "/Users/hn/Documents/01_research_data/NASA/data_deBug/"

# %%
county = "AdamBenton2016"
indeks = "EVI"
thresh = 3

acr_df = pd.DataFrame(data = None)

# %%
SF_data = pd.read_csv(SF_data_dir + county + ".csv")
SF_data["ID"] = SF_data["ID"].astype(str)
SF_data = SF_data[["ID", "CropTyp", "ExctAcr", "county"]]
SF_data.head(2)

# %%
SG_df = pd.read_csv(data_dir + "SC_train_" + county + "_" + indeks + str(thresh) + ".csv")
SG_df['human_system_start_time'] = pd.to_datetime(SG_df['human_system_start_time'])
SG_df["ID"] = SG_df["ID"].astype(str) # Monterays ID will be read as integer, convert to string
SG_df.head(2)

# %%
SG_df = SG_df[SG_df.human_system_start_time.dt.year == int(county[-4:])]
SG_df['threshod'] = thresh/10
SG_df['year'] = SG_df.human_system_start_time.dt.year

# %%
SG_df = SG_df[["ID", "season_count", "year", "threshod"]]
SG_df.head(2)

# %%
SG_df = pd.merge(SG_df, SF_data, on=['ID'], how='left')
SG_df['season_count'] = np.where(SG_df['season_count']>=2, 2, 1)
SG_df.head(2)

# %%
acr_df = pd.concat([acr_df, SG_df])

# %%
acr_df.head(2)

# %%
acr_df = acr_df.groupby(['county', 'CropTyp', 'threshod', 'year', 'season_count']).sum()
acr_df.reset_index(inplace=True)
acr_df.sort_values(by=['threshod', 'county', 'CropTyp', 'year', 'season_count'], inplace=True)

# %%
acr_df.head(5)

# %%
out_name = data_dir + "doubleAcr_perCrop_" + indeks + ".csv"
acr_df.to_csv(out_name, index = False)

# %% [markdown]
# # Check the steps of the smoothing... 
#
# Why some acres are missing in the final double-crop intensity

# %%
SF_data_part_dir = "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/"
data_dir_base = "/Users/hn/Documents/01_research_data/NASA/data_deBug/"

# %%
AdamBenton2016_SF = pd.read_csv(SF_data_part_dir + "AdamBenton2016.csv")

# %%
noOutlier_AdamBenton2016_EVI = pd.read_csv(data_dir_base + "02_outliers_removed/" + \
                                           "noOutlier_AdamBenton2016_EVI.csv")

noOutlier_AdamBenton2016_NDVI = pd.read_csv(data_dir_base + "02_outliers_removed/" + \
                                           "noOutlier_AdamBenton2016_NDVI.csv")

# %%
print (len(AdamBenton2016_SF.ID.unique()))
print (len(noOutlier_AdamBenton2016_EVI.ID.unique()))
print (len(noOutlier_AdamBenton2016_NDVI.ID.unique()))

# %%
NoJump_AdamBenton2016_EVI = pd.read_csv(data_dir_base + "03_jumps_removed/" + \
                                           "NoJump_AdamBenton2016_EVI_JFD.csv")

NoJump_AdamBenton2016_NDVI = pd.read_csv(data_dir_base + "03_jumps_removed/" + \
                                           "NoJump_AdamBenton2016_NDVI_JFD.csv")

# %%
print (len(AdamBenton2016_SF.ID.unique()))
print (len(NoJump_AdamBenton2016_EVI.ID.unique()))
print (len(NoJump_AdamBenton2016_NDVI.ID.unique()))

# %%
regular_AdamBenton2016_EVI_JFD = pd.read_csv(data_dir_base + "04_regularized_TS/" + \
                                           "regular_AdamBenton2016_EVI_JFD.csv")

regular_AdamBenton2016_NDVI_JFD = pd.read_csv(data_dir_base + "04_regularized_TS/" + \
                                           "regular_AdamBenton2016_NDVI_JFD.csv")

# %%
print (len(AdamBenton2016_SF.ID.unique()))
print (len(regular_AdamBenton2016_EVI_JFD.ID.unique()))
print (len(regular_AdamBenton2016_NDVI_JFD.ID.unique()))

# %%
SG_AdamBenton2016_EVI = pd.read_csv(data_dir_base + "05_SG_TS/" + \
                                           "SG_AdamBenton2016_EVI.csv")

SG_AdamBenton2016_NDVI = pd.read_csv(data_dir_base + "05_SG_TS/" + \
                                           "SG_AdamBenton2016_NDVI.csv")

# %%
print (len(AdamBenton2016_SF.ID.unique()))
print (len(SG_AdamBenton2016_EVI.ID.unique()))
print (len(SG_AdamBenton2016_NDVI.ID.unique()))

# %%

# %% [markdown]
# ### Season count for training table is for irrigated fields, surveyed on a given year and NASS is out.
# So, we need to filter the data part in the same way
#

# %%
SC_train_AdamBenton2016_EVI3 = pd.read_csv(data_dir_base + "06_SOS_tables/" + \
                                           "SC_train_AdamBenton2016_EVI3.csv")

# %%
print (len(SC_train_AdamBenton2016_EVI3.ID.unique()))

# %%
AdamBenton2016_SF_Survey = nc.filter_by_lastSurvey(AdamBenton2016_SF, year = 2016)
print("No. of fields in SF_data after survey year is {}.".format(len(AdamBenton2016_SF_Survey.ID.unique())))

# %%
AdamBenton2016_SF_Survey_NassOut = nc.filter_out_NASS(AdamBenton2016_SF_Survey)  # Toss NASS
print("No. of fields in SF_data after NASS is {}.".format(len(AdamBenton2016_SF_Survey_NassOut.ID.unique())))


# %%
# keep only irrigated lands
AdamBenton2016_SF_Survey_NassOut_Irr = nc.filter_out_nonIrrigated(AdamBenton2016_SF_Survey_NassOut)
L = len(AdamBenton2016_SF_Survey_NassOut_Irr.ID.unique())
print("No. of fields in SF_data after Irrigation is {}.".format(L))

# %%
AdamBenton2016_SF_Survey_NassOut_Irr.head(2)

# %%
AdamBenton2016_SF_Survey_NassOut_Irr.ExctAcr.sum()

# %%
AdamBenton2016_SF_Survey_NassOut_Irr.groupby(['county']).sum()

# %%
len(SC_train_AdamBenton2016_EVI3.ID.unique())

# %%
len(AdamBenton2016_SF_Survey_NassOut_Irr.ID.unique())

# %%
all(item in list(AdamBenton2016_SF_Survey_NassOut_Irr.ID.unique()) for \
    item in list(SC_train_AdamBenton2016_EVI3.ID.unique()))

# %%
all(item in list(SC_train_AdamBenton2016_EVI3.ID.unique()) for \
    item in list(AdamBenton2016_SF_Survey_NassOut_Irr.ID.unique()))

# %%
SC_train_AdamBenton2016_EVI3.head(2)

# %%
county = "AdamBenton2016"
SC_df = SC_train_AdamBenton2016_EVI3.copy()
SC_df['human_system_start_time'] = pd.to_datetime(SC_df['human_system_start_time'])

# %%
print(SC_df.shape)
SC_df = SC_df[SC_df.human_system_start_time.dt.year == int(county[-4:])]
print(SC_df.shape)

# %%
SC_df['threshold'] = 3/10
SC_df['year'] = SC_df.human_system_start_time.dt.year

# %%
SC_df.head(2)

# %%
SC_df = SC_df[["ID", "season_count", "year", "threshold"]]

# %%
len(SC_df.ID.unique())

# %%
print (all(item in list(AdamBenton2016_SF_Survey_NassOut_Irr.ID.unique()) for \
       item in list(SC_train_AdamBenton2016_EVI3.ID.unique())))

all(item in list(SC_train_AdamBenton2016_EVI3.ID.unique()) for \
    item in list(AdamBenton2016_SF_Survey_NassOut_Irr.ID.unique()))

# %%
AdamBenton2016_SF_Survey_NassOut_Irr.shape

# %%
SC_df.shape

# %%
A = SC_df.copy()

# %%
print (A.shape)
A.drop_duplicates(inplace=True)
print (A.shape)

# %%
SC_df = pd.merge(SC_df, AdamBenton2016_SF_Survey_NassOut_Irr, on=['ID'], how='left')
A = pd.merge(A, AdamBenton2016_SF_Survey_NassOut_Irr, on=['ID'], how='left')

# %%
SC_df.shape

# %%
SC_df.head(2)

# %%
A.head(2)

# %%
A.shape

# %%
A.groupby(['county', 'CropTyp', 'season_count']).sum()

# %%
A.head(2)

# %%
12958 + 1229.81 + 237

# %%
