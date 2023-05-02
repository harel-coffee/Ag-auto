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
# ### Crop Types 
#
# List of crop types in 2008-2018 shapefiles.
#  - First, I will list everything in those shapefiles.
#  - Then, I will do the same for the shapefiles that was actually used to do intersection. I need to wait for Supriya to see what she did.

# %%
import shutup;
shutup.please()
import pandas as pd
import numpy as np

import os, sys, os.path

# %%

# %%
Eastern_WA = ['Adams',    'Asotin',     'Benton',       'Chelan',
              'Columbia', 'Douglas',    'Ferry',        'Franklin',
              'Garfield', 'Grant',       'Kittitas',    'Klickitat',
              'Lincoln',  'Okanogan',    'Pend Oreille', 'Spokane',
              'Stevens',  'Walla Walla', 'Whitman',      'Yakima']

# %% [markdown]
# ### directories

# %%
SF_dir_ = "/Users/hn/Documents/01_research_data/NASA/shapefiles/00_WSDA_separateYears/"

# %%
p_filenames = os.listdir(SF_dir_)
csv_filenames = []

for a_file in p_filenames:
    if a_file.endswith(".csv"):
        csv_filenames += [a_file]

csv_filenames = sorted(csv_filenames)

# %%
SF_data = pd.DataFrame()

for a_file in csv_filenames:
    curr_dt = pd.read_csv(SF_dir_ + a_file)
    curr_dt.OBJECTID = curr_dt.OBJECTID.astype(int).astype(str)
    if "ExactAcres" in curr_dt.columns:
        curr_dt.rename(columns = {"ExactAcres": "Exact_Acre"}, inplace = True)
        
    if "Irrigation" in curr_dt.columns:
        curr_dt.rename(columns = {"Irrigation": "Irrigtn"}, inplace = True)
   
    # print (curr_dt.columns)
    if "Acres" in curr_dt.columns:
        curr_dt.rename(columns = {"Acres": "TotalAcres"}, inplace = True)
    curr_dt = curr_dt[['OBJECTID', 'County', 'CropType', 'Exact_Acre', 'TotalAcres', "Irrigtn"]]
    curr_dt["OBJECTID"] = curr_dt["OBJECTID"] + "_" + a_file.split("_")[-1].split(".csv")[0]
    SF_data=pd.concat([SF_data, curr_dt])

values = {"Irrigtn": "Empty", "CropType": "Empty"}
SF_data.fillna(value=values, inplace=True)

SF_data.head(2)

# %%
SF_data_crops = sorted((list(SF_data.CropType.unique())))
SF_data_crops

# %%
SF_data[SF_data.CropType=="Empty"]

# %%
SF_data[SF_data.CropType=="Empty"]

# %%
SF_data[SF_data.County == "Pacific"]


# %%
def filter_out_nonIrrigated(dt_df_irr):
    dt_irrig = dt_df_irr.copy()
    #
    # drop NA rows in irrigation column
    #
    dt_irrig.dropna(subset=["Irrigtn"], inplace=True)

    dt_irrig["Irrigtn"] = dt_irrig["Irrigtn"].astype(str)

    dt_irrig["Irrigtn"] = dt_irrig["Irrigtn"].str.lower()
    dt_irrig = dt_irrig[~dt_irrig["Irrigtn"].str.contains("none")]
    dt_irrig = dt_irrig[~dt_irrig["Irrigtn"].str.contains("unknown")]
    dt_irrig = dt_irrig[~dt_irrig["Irrigtn"].str.contains("empty")]

    return dt_irrig


# %%
SF_data_irrigated = filter_out_nonIrrigated(SF_data)

# %%
SF_data_irrigated_crops = (sorted((list(SF_data_irrigated.CropType.unique()))))

# %%
[x if x not in SF_data_irrigated_crops for x in SF_data_crops]

# %%
[x for x in SF_data_crops if x not in SF_data_irrigated_crops]

# %%
