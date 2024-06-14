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
# When we met on June 10, 2024 Mike mentioned it would be good to look into how much and what kind of interaction/dynamic exist between inventory and slaughter. Lets see what we can do!

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

reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_fips = abb_dict["state_fips"]
# state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
# state_fips_SoI.reset_index(drop=True, inplace=True)
# print (len(state_fips_SoI))
print (state_fips.shape)
state_fips.head(2)

# %%
state_fips = abb_dict["state_fips"]
# state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
# state_fips_SoI.reset_index(drop=True, inplace=True)
# print(len(state_fips_SoI))
# state_fips_SoI.head(2)

# %%
filename = reOrganized_dir + "monthly_NDVI_beef_slaughter.sav"

monthly_NDVI_beef_slaughter = pd.read_pickle(filename)
monthly_NDVI_beef_slaughter = monthly_NDVI_beef_slaughter["monthly_NDVI_beef_slaughter"]
monthly_NDVI_beef_slaughter.head(2)

# %% [markdown]
# ### Compute annual slaughter

# %%
monthly_slaughter = monthly_NDVI_beef_slaughter[["year", "region", "slaughter_count"]].copy()
monthly_slaughter.dropna(subset=["slaughter_count"], inplace=True)
monthly_slaughter.reset_index(drop=True, inplace=True)
monthly_slaughter.head(2)

# %%
annual_slaughter = monthly_slaughter.groupby(["region", "year"])["slaughter_count"].sum().reset_index()
annual_slaughter.head(2)

# %% [markdown]
# ### Compute regional inventory

# %%
filename = reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
all_data_dict = pd.read_pickle(filename)
print (all_data_dict["Date"])
list(all_data_dict.keys())

# %%
all_data = all_data_dict["all_df_outerjoined"].copy()
inventory_df = all_data[["year", "inventory", "state_fips"]].copy()

inventory_df.dropna(subset=["inventory"], inplace=True)

# do NOT subset to states of interest as slaughter data ignores that
# inventory_df = inventory_df[inventory_df["state_fips"].isin(list(state_fips_SoI["state_fips"].unique()))]
inventory_df.reset_index(drop=True, inplace=True)

inventory_df.head(2)

# %%

# %%
inventory_df = pd.merge(inventory_df, state_fips[['state', 'state_fips', 'state_full']], 
                        on=["state_fips"], how="left")

inventory_df.head(2)

# %%
shannon_regions_dict_abbr = {"region_1_region_2" : ['CT', 'ME', 'NH', 'VT', 'MA', 'RI', 'NY', 'NJ'], 
                             "region_3" : ['DE', 'MD', 'PA', 'WV', 'VA'],
                             "region_4" : ['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN'],
                             "region_5" : ['IL', 'IN', 'MI', 'MN', 'OH', 'WI'],
                             "region_6" : ['AR', 'LA', 'NM', 'OK', 'TX'],
                             "region_7" : ['IA', 'KS', 'MO', 'NE'],
                             "region_8" : ['CO', 'MT', 'ND', 'SD', 'UT', 'WY'],
                             "region_9" : ['AZ', 'CA', 'HI', 'NV'],
                             "region_10": ['AK', 'ID', 'OR', 'WA']}

# %%
inventory_df["region"] = "NA"
for a_key in shannon_regions_dict_abbr.keys():
    inventory_df.loc[inventory_df["state"].isin(shannon_regions_dict_abbr[a_key]), 'region'] = a_key
inventory_df.head(2)

# %% [markdown]
# ### compute inventory in each region

# %%
region_inventory = inventory_df.copy()
region_inventory = region_inventory[["year", "region", "inventory"]]

region_inventory = region_inventory.groupby(["region", "year"])["inventory"].sum().reset_index()
region_inventory.head(2)

# %%
annual_slaughter.head(2)

# %%
region_slaughter_inventory = pd.merge(region_inventory, annual_slaughter, 
                                      on=["region", "year"], how="outer")

print (f"{region_inventory.shape = }")
print (f"{annual_slaughter.shape = }")
print (f"{region_slaughter_inventory.shape = }")
region_slaughter_inventory.head(2)

# %%

# %%
#### it seems in some years some of the data are not available
region_slaughter_inventory.dropna(how="any", inplace=True)
region_slaughter_inventory.reset_index(drop=True, inplace=True)
print (region_slaughter_inventory.shape)

# %%
region_slaughter_inventory.head(2)

# %%

# %%

# %%

# %%

# %%
