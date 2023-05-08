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

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc

# %%

# %%
dir_base = "/Users/hn/Documents/01_research_data/NASA/VI_TS/"
data_dir = dir_base + "/05_SG_TS/"
SF_data_dir = "/Users/hn/Documents/01_research_data/NASA/shapefiles/10_intersect_East_Irr_2008_2018_2cols/"

# %%
VI_idx = "EVI"
batch_no = "2"
smooth = "SG"
print(f"Passed Args. are: {VI_idx=:}, {smooth=:}, and {batch_no=:}!")

# %%
f_name = VI_idx + "_" + smooth + "_" + "intersect_batchNumber" + batch_no + "_JFD.csv"
in_dir = data_dir
data = pd.read_csv(in_dir + f_name)
data["human_system_start_time"] = pd.to_datetime(data["human_system_start_time"])

# %%
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

data_wide.head(2)

# %%
# %%time

for an_ID in IDs:
    curr_field = data[data.ID==an_ID]
    curr_years = curr_field.human_system_start_time.dt.year.unique()
    for a_year in curr_years:
        curr_field_year = curr_field[curr_field.human_system_start_time.dt.year == a_year]
        
        data_wide_indx = data_wide[(data_wide.ID == an_ID) & (data_wide.year == a_year)].index
        if VI_idx == "EVI":
            data_wide.loc[data_wide_indx, "EVI_1":"EVI_36"] = curr_field_year.EVI.values[:36]
        elif VI_idx == "NDVI":
            data_wide.loc[data_wide_indx, "NDVI_1":"NDVI_36"] = curr_field_year.NDVI.values[:36]

# %%
data_wide.head(2)

# %%
wide_TS = data_wide.copy()

# %%
data_wide.iloc[:, 2:].head(2)

# %%
ML_model = "kNN"

# %%

# %%

# %%
