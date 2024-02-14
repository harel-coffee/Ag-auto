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
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
USDA_data_dir = data_dir_base + "/NASS_downloads/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

import sys
sys.path.append("/Users/hn/Documents/00_GitHub/Rangeland/Python_Codes/")
import rangeland_core as rc

# %%
cattle_inventory = pd.read_csv(USDA_data_dir + "/cow_inventory_Qs/"+ "Q4.csv")
cattle_inventory.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True) 
cattle_inventory.head(2)

# %%
bad_cols  = ["watershed", "watershed_code", 
             "domain", "domain_category", 
             "region", "period",
             "week_ending", "zip_code", "program", "geo_level"]


meta_cols = ["state", "county", "county_ansi", "state_ansi", "ag_district_code"]

# %%
cattle_inventory.drop(bad_cols, axis="columns", inplace=True)

# %%
cattle_inventory['county_ansi'].fillna(666, inplace=True)

cattle_inventory["state_ansi"] = cattle_inventory["state_ansi"].astype('int32')
cattle_inventory["county_ansi"] = cattle_inventory["county_ansi"].astype('int32')

cattle_inventory["state_ansi"] = cattle_inventory["state_ansi"].astype('str')
cattle_inventory["county_ansi"] = cattle_inventory["county_ansi"].astype('str')

cattle_inventory.state = cattle_inventory.state.str.title()
cattle_inventory.county = cattle_inventory.county.str.title()

for idx in cattle_inventory.index:
    if len(cattle_inventory.loc[idx, "state_ansi"]) == 1:
        cattle_inventory.loc[idx, "state_ansi"] = "0" + cattle_inventory.loc[idx, "state_ansi"]
        
    if len(cattle_inventory.loc[idx, "county_ansi"]) == 1:
        cattle_inventory.loc[idx, "county_ansi"] = "00" + cattle_inventory.loc[idx, "county_ansi"]
    elif len(cattle_inventory.loc[idx, "county_ansi"]) == 2:
        cattle_inventory.loc[idx, "county_ansi"] = "0" + cattle_inventory.loc[idx, "county_ansi"]
        
        
cattle_inventory["county_fips"] = cattle_inventory["state_ansi"] + cattle_inventory["county_ansi"]
cattle_inventory[["state", "county", "state_ansi", "county_ansi", "county_fips"]].head(5)

# %%
cattle_inventory.head(2)

# %%
len(cattle_inventory.state.unique())

# %%
len(cattle_inventory.county_fips.unique())

# %%
cattle_inventory.rename(
    columns={"value": "cattle_cow_beef_inventory", "cv_(%)": "cattle_cow_beef_inventory_cv_(%)"},
    inplace=True,
)

print (cattle_inventory.shape)
cattle_inventory = rc.clean_census(df=cattle_inventory, col_="cattle_cow_beef_inventory")
print (cattle_inventory.shape)

# %%
len(cattle_inventory.county_fips.unique())

# %%
NPP = pd.read_csv(Min_data_dir_base + "county_annual_MODIS_NPP.csv")
NPP.rename(columns={"NPP": "modis_npp"}, inplace=True)

NPP = rc.correct_Mins_FIPS(df=NPP, col_="county")
NPP.rename(columns={"county": "county_fips"}, inplace=True)

NPP.head(2)

# %%
len(NPP.county_fips.unique())

# %%
NPP.year.min()

# %%
sorted(cattle_inventory.year.unique())

# %%
NPP = NPP[NPP.year >= 2002]
cattle_inventory = cattle_inventory[cattle_inventory.year >= 2002]

# %%
len(cattle_inventory.county_fips.unique())

# %%
NPP_counties = set(NPP.county_fips)
cattle_inventory_counties = set(cattle_inventory.county_fips)

# %% [markdown]
# # intersetion of counties in common between NPP and cow_inventory

# %%
intersected_counties = NPP_counties.intersection(cattle_inventory_counties)
len(intersected_counties)

# %% [markdown]
# # Counties for which we have data for all years

# %%
NPP = NPP[NPP.county_fips.isin(intersected_counties)]
cattle_inventory = cattle_inventory[cattle_inventory.county_fips.isin(intersected_counties)]
cattle_inventory = cattle_inventory[["year", "county_fips", "cattle_cow_beef_inventory"]]

# %%
NPP.reset_index(drop=True, inplace=True)
cattle_inventory.reset_index(drop=True, inplace=True)
cattle_inventory.head(2)

# %%
NPP.head(2)

# %%
NPP_inventory = pd.merge(NPP, cattle_inventory, on = ["county_fips", "year"], how = "left")

print (len(NPP_inventory.county_fips.unique()))
NPP_inventory.dropna(how="any", inplace=True)

print (len(NPP_inventory.county_fips.unique()))

# %%
full_counties = []
for a_county in NPP_inventory.county_fips.unique():
    df = NPP_inventory[NPP_inventory.county_fips == a_county]
    if len(df.year.unique()) == 4:
        full_counties = full_counties + [a_county]


# %%
len(full_counties)

# %%
