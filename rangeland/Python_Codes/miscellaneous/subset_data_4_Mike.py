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

# %% [markdown]
# # May 2
#
# Mike wants to run stuff with some software.
#
#
# Mail said:
#
# inventories by state by year and annual state rangeland productivity (NDVI I guess) in a csv file

# %%
import shutup

shutup.please()

import pandas as pd
import numpy as np
import os, os.path, pickle, sys
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

from sklearn import preprocessing
import statistics
import statsmodels.api as sm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

from datetime import datetime, date
from scipy.linalg import inv

current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
param_dir = data_dir_base + "parameters/"

Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
mike_dir = data_dir_base + "Mike/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_name_fips = pd.DataFrame({"state_full" : list(abb_dict["full_2_abb"].keys()),
                                "state" : list(abb_dict["full_2_abb"].values())})
state_name_fips = pd.merge(state_name_fips, abb_dict["state_fips"], on=["state"], how="left")
state_name_fips.head(2)

# %%
state_fips_SoI = state_name_fips[state_name_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
state_fips_SoI.head(2)

# %%

# %%
filename = reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
all_data_dict = pd.read_pickle(filename)
print (all_data_dict["Date"])
list(all_data_dict.keys())

# %%
all_df = all_data_dict["all_df_outerjoined"]
all_df.head(2)

# %%
test_inventory_yr = all_df[["year", "unit_matt_npp"]]
test_inventory_yr.dropna(how="any", inplace=True)
print (test_inventory_yr.year.min())
print (test_inventory_yr.year.max())

# %%
test_inventory_yr = all_df[["year", "inventory"]]
test_inventory_yr.dropna(how="any", inplace=True)
print (test_inventory_yr.year.min())
print (test_inventory_yr.year.max())

# %%
test_inventory_yr = all_df[["year", "max_ndvi_in_year_modis"]]
test_inventory_yr.dropna(how="any", inplace=True)
print (test_inventory_yr.year.min())
print (test_inventory_yr.year.max())

# %%
dummy_cols = [x for x in all_df.columns if "dumm" in x]
all_df.drop(columns = dummy_cols, inplace=True)

# %%
keep_cols = ['year', 'inventory', 'state_fips',
             'unit_matt_npp', 'total_matt_npp',
             'unit_matt_npp_std', 'total_matt_npp_std',
             'hay_price_at_1982', 'beef_price_at_1982',
             'rangeland_acre', 'max_ndvi_in_year_modis',
             'EW_meridian', 'herb_avg', 'herb_std']

print (all_df.year.min())
print (all_df.year.max())

all_df = all_df[keep_cols]
# all_df.dropna(subset = ["total_matt_npp"], inplace=True)

print (all_df.year.min())
print (all_df.year.max())

all_df = all_df[all_df.state_fips.isin(list(state_fips_SoI.state_fips))]
all_df.reset_index(drop=True, inplace=True)
all_df.head(2)

# %%
print (all_df.year.min())
print (all_df.year.max())

# %%
test_inventory_yr = all_df[["year", "inventory"]]
test_inventory_yr.dropna(how="any", inplace=True)
print (test_inventory_yr.year.min())
print (test_inventory_yr.year.max())

# %%

# %%
all_df = rc.convert_lb_2_kg(df=all_df, 
                            matt_total_npp_col="total_matt_npp", 
                            new_col_name="metric_total_matt_npp")

# %%
all_df = pd.merge(all_df, state_name_fips, on=["state_fips"], how="left")
all_df.head(2)

# %%
all_df.total_matt_npp[0]/2.2046226218

# %%
all_df.metric_total_matt_npp[0]

# %%
15979574020

# %%
all_df.tail(2)

# %%

# %%
# converting to CSV file
all_df.to_csv(reOrganized_dir + "NPP_NDVI_Invent_Mike_2May2024.csv")

# %%
reOrganized_dir

# %%
all_df.year.max()

# %%
