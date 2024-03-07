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
# In the group meeting it seemed we want to model changes/deltas as a function of max of NDVI and dummy seasons.
#
# But at the time of coding questions rise up:
#   - dummy season changes will be 16 cases: season 1 to season 2, season 1 to season 3, and so on.
#   - Time-Invariant variables should not be differenced.
#
# Then on March 1st, 2024, Mike and I met on zoom and decided that we want:
#   - state-level modeling of changes/deltas. If not a significant discovery is here, then we will switch to county-level and perhaps high frequency data such as slaughter.
#
#   - We want independent variables to be NPP/NDVI, as well as beef and feed prices. Other variables are secondary.
#   Questions will be there for Matt and Shannon about beef and feed prices.

# %% [markdown]
# <p style="font-family: Arial; font-size:1.4em;color:gold;"> Golden </p>

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

# %% [markdown]
# ### Read beef and hay prices

# %%
# filename = reOrganized_dir + "beef_hay_cost_fromMikeLinkandFile.sav"
# beef_hay_cost = pd.read_pickle(filename)
# beef_hay_cost.keys()

# beef_hay_cost_deltas = beef_hay_cost["beef_hay_costDeltas_MikeLinkandFile"]
# beef_hay_cost = beef_hay_cost["beef_hay_cost_MikeLinkandFile"]

# %%

# %%
state_USDA_ShannonCattle = pd.read_pickle(reOrganized_dir + "state_USDA_ShannonCattle.sav")
list(state_USDA_ShannonCattle.keys())

# %% [markdown]
# ## Read inventory deltas

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%

# %%
shannon_invt_tall = state_USDA_ShannonCattle["shannon_invt_tall"]

## pick up states of interest
shannon_invt_tall = shannon_invt_tall[shannon_invt_tall.state.isin(SoI_abb)]
shannon_invt_tall.reset_index(drop=True, inplace=True)
shannon_invt_tall.inventory = shannon_invt_tall.inventory.astype(np.float32)
print(f"{shannon_invt_tall.shape = }")
shannon_invt_tall.head(2)

# %%
beef_price = state_USDA_ShannonCattle["beef_price_at_1982"]
hay_price  = state_USDA_ShannonCattle["HayPrice_Q1_at_1982"]
beef_hay_price_at_1982 = state_USDA_ShannonCattle["beef_hay_price_at_1982"]
print (beef_hay_price_at_1982.shape)
beef_hay_price_at_1982.head(2)

# %%
beef_hay_price_at_1982.dropna(how="any", inplace=False).shape

# %% [markdown]
# ## Pick the years that they intersect

# %%
min_yr = max(beef_hay_price_at_1982.year.min(), shannon_invt_tall.year.min())
max_yr = min(beef_hay_price_at_1982.year.max(), shannon_invt_tall.year.max())
print(f"{min_yr = }")
print(f"{max_yr = }")

# %%
shannon_invt_tall = shannon_invt_tall[(shannon_invt_tall.year >= min_yr) & (shannon_invt_tall.year <= max_yr)]

beef_hay_price_at_1982 = beef_hay_price_at_1982[
    (beef_hay_price_at_1982.year >= min_yr) & (beef_hay_price_at_1982.year <= max_yr)]

# %%
invt_BH_cost = pd.merge(shannon_invt_tall, beef_hay_price_at_1982, on=["year", "state_fips"], how="left")

print(f"{invt_BH_cost.shape=}")
invt_BH_cost.head(3)

# %%

# %%
filename = reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
all_data_dict = pd.read_pickle(filename)
list(all_data_dict.keys())

# %%
all_df_normalized = all_data_dict["all_df_outerjoined_normalized"]
print (all_df_normalized.shape)
all_df_normalized.head(2)

# %%
all_df_normalized.columns

# %%
ndvi_columns = [x for x in all_df_normalized.columns if "ndvi" in x.lower()]
ndvi_columns

# %%
all_df_normalized_needed = all_df_normalized[['year', 'state_fips', 'inventory', 
                                              "max_ndvi_in_year_modis",
                                              "beef_price_at_1982", "hay_price_at_1982"]].copy()
print (all_df_normalized_needed.shape)
all_df_normalized_needed.dropna(how="any", inplace=True)
all_df_normalized_needed.reset_index(drop=True, inplace=True)
print (all_df_normalized_needed.shape)
all_df_normalized_needed.head(3)

# %%
print (all_df_normalized_needed.year.min())
print (all_df_normalized_needed.year.max())

# %% [markdown]
# # Model
#
# We are using national beef and hay prices for all states! if it's the same for all states, what's the diff between including and excluding them.

# %%
list(inventoryRatio_beefHayCostDeltas.columns)

# %%
indp_vars = [
    "state_total_npp",
    "allHay_wtAvg_2_$perTon_calendar_yr_normal",
    "steers_medium_large_600_650lbs_$_cwt_yrAvg_normal",
]
y_var = "inventory_ratio"

#################################################################
X = all_data[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = np.log(all_data[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%
indp_vars = ["state_total_npp"]
y_var = "inventory_ratio"

#################################################################
X = all_data[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = np.log(all_data[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%
