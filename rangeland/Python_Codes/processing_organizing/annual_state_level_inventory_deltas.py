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
import shutup
shutup.please()

import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

sys.path.append("/Users/hn/Documents/00_GitHub/Rangeland/Python_Codes/")
import rangeland_core as rc

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %%
county_id_name_fips = pd.read_csv(Min_data_base + "county_id_name_fips.csv")
county_id_name_fips.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)

county_id_name_fips.sort_values(by=["state", "county"], inplace=True)

county_id_name_fips = rc.correct_Mins_FIPS(df=county_id_name_fips, col_="county")
county_id_name_fips.rename(columns={"county": "county_fips"}, inplace=True)

print (len(county_id_name_fips.state.unique()))
county_id_name_fips.head(2)

# %%
county_id_name_fips["state_fip"] = county_id_name_fips.county_fips.str.slice(0, 2)
county_id_name_fips = county_id_name_fips.drop(columns=["county_name", "county_fips", "fips"])
county_id_name_fips.drop_duplicates(inplace=True)
county_id_name_fips.reset_index(drop=True, inplace=True)
county_id_name_fips.head(2)

# %% [markdown]
# ### read cattle inventory

# %%
shannon_annual = pd.read_csv(reOrganized_dir + "Shannon_Beef_Cows_fromCATINV.csv")
shannon_annual.reset_index(drop=True, inplace=True)
shannon_annual.head(2)

# %%
# form deltas: inventort(t+1) - inventory (t)
inv_deltas = shannon_annual[list(shannon_annual.columns)[2:]].values - \
               shannon_annual[list(shannon_annual.columns)[1:-1]].values

delta_columns = [(str(x) + "-" + str(x-1)) for x in np.arange(1921, 2022)]
# form deltas dataframe
shannon_annual_deltas = pd.DataFrame(data = inv_deltas, columns=delta_columns)
shannon_annual_deltas["state"] = shannon_annual["state"]

# re-order columns
shannon_annual_deltas = shannon_annual_deltas[["state"]+delta_columns]
shannon_annual_deltas.head(2)

# %%
shannon_annual_deltas.head(2)

# %%

# %% [markdown]
# ### convert to tall format

# %%
years = list(shannon_annual_deltas.columns[1:])
num_years = len(years)

inventory_delta_tall = pd.DataFrame(data=None, index=range(num_years*len(shannon_annual_deltas.state.unique())), 
                                    columns=["state", "year", "inventory_delta"], 
                                    dtype=None, copy=False)

idx_ = 0 
for a_state in shannon_annual_deltas.state.unique():
    curr = shannon_annual_deltas[shannon_annual_deltas.state == a_state]
    inventory_delta_tall.loc[idx_: idx_ + num_years - 1 , "inventory_delta"] = curr[years].values[0]
    inventory_delta_tall.loc[idx_: idx_ + num_years - 1 , "state"] = a_state
    inventory_delta_tall.loc[idx_: idx_ + num_years - 1 , "year"] = years
    idx_ = idx_ + num_years
    
inventory_delta_tall.head(2)


# %%
inventory_delta_tall.tail(2)

# %%
shannon_annual_deltas = pd.merge(shannon_annual_deltas, county_id_name_fips, on = ["state"], how = "left")
shannon_annual_deltas.head(2)

# %%
inventory_delta_tall = pd.merge(inventory_delta_tall, county_id_name_fips, on = ["state"], how = "left")
inventory_delta_tall.head(2)

# %%
from datetime import datetime
import pickle

filename = reOrganized_dir + "Shannon_Beef_Cows_fromCATINV_deltas.sav"

export_ = {"shannon_annual_inventory_deltas": shannon_annual_deltas, 
           "shannon_annual_inventory_deltas_tall": inventory_delta_tall, 
           "source_code" : "annual_state_level_inventory_deltas",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
