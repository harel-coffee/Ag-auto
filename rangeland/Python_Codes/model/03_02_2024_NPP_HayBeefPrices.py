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
# **The reason I separated NPP and NDVI is that missing values are not identical. Samples count is different. they differ only 29 samples. So. If it gets painful, merge them together**
#
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
state_fips = abb_dict["state_fips"]
state_fips = state_fips[state_fips.state.isin(SoI_abb)].copy()
state_fips.reset_index(drop=True, inplace=True)
print (len(state_fips))
state_fips.head(2)

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
chicken_price = state_USDA_ShannonCattle["chicken_price_at_1982"]
hay_price  = state_USDA_ShannonCattle["HayPrice_Q1_at_1982"]

beef_chicken_hay_price_at_1982 = state_USDA_ShannonCattle["beef_chicken_hay_price_at_1982"]
print (beef_chicken_hay_price_at_1982.shape)
beef_chicken_hay_price_at_1982.head(2)

# %%
beef_chicken_hay_price_at_1982.dropna(how="any", inplace=False).shape

# %% [markdown]
# ## Pick the years that they intersect

# %%
min_yr = max(beef_chicken_hay_price_at_1982.year.min(), shannon_invt_tall.year.min())
max_yr = min(beef_chicken_hay_price_at_1982.year.max(), shannon_invt_tall.year.max())
print(f"{min_yr = }")
print(f"{max_yr = }")

# %%
shannon_invt_tall = shannon_invt_tall[(shannon_invt_tall.year >= min_yr) & (shannon_invt_tall.year <= max_yr)]

beef_chicken_hay_price_at_1982 = beef_chicken_hay_price_at_1982[
    (beef_chicken_hay_price_at_1982.year >= min_yr) & (beef_chicken_hay_price_at_1982.year <= max_yr)]

# %%
invt_BH_cost = pd.merge(shannon_invt_tall, beef_chicken_hay_price_at_1982, 
                        on=["year", "state_fips"], how="left")
print(f"{len(invt_BH_cost.state_fips.unique())=}")
print(f"{invt_BH_cost.shape=}")
invt_BH_cost.head(3)

# %%

# %%
filename = reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
all_data_dict = pd.read_pickle(filename)
list(all_data_dict.keys())

# %%
all_df_normalized = all_data_dict["all_df_outerjoined_normalized"]
all_df_normalized = all_df_normalized[all_df_normalized.state_fips.isin(state_fips.state_fips)].copy()
all_df_normalized.reset_index(drop=True, inplace=True)
print (len(all_df_normalized.state_fips.unique()))
print (all_df_normalized.shape)
all_df_normalized.head(2)

# %%
ndvi_columns = [x for x in all_df_normalized.columns if "ndvi" in x.lower()]
ndvi_columns

# %% [markdown]
# # Add state-dummy variables

# %%
all_df_normalized["state_dummy_int"] = all_df_normalized["state_fips"].astype(int)
all_df_normalized["state_dummy_str"] = all_df_normalized["state_fips"] + "_" + "dummy"

L = list(all_df_normalized.columns)
L.pop()
L

all_df_normalized = all_df_normalized.pivot(index=L, columns = "state_dummy_str", values = "state_dummy_int")

all_df_normalized.reset_index(drop=False, inplace=True)
all_df_normalized.columns = all_df_normalized.columns.values

# replace NAs with zeros and others (state_fips integers) with 1s.

dummy_cols = [x for x in all_df_normalized.columns if "_dummy" in x]
dummy_cols = [x for x in dummy_cols if not("int" in x)]

for a_col in dummy_cols:
    all_df_normalized[a_col].fillna(0, inplace=True)
    all_df_normalized[a_col] = all_df_normalized[a_col].astype(int)
    
all_df_normalized[dummy_cols] = np.where(all_df_normalized[dummy_cols]!= 0, 1, 0)

all_df_normalized.head(2)

# %%
all_df_normalized[["state_fips"]+ dummy_cols].head(5)

# %%
all_df_normalized.rename(columns={"state_unit_npp": "npp_u",
                                  "state_total_npp": "npp_t"}, 
                         inplace=True)

# %%
needed_cols = ['year', 'state_fips', 'inventory', 
               "npp_u", "npp_t", "rangeland_acre", "ndvi_std_modis",
               "beef_price_at_1982", "hay_price_at_1982", "chicken_price_at_1982"] + \
               dummy_cols + ["state_dummy_int"]\


all_df_normalized_needed = all_df_normalized[needed_cols].copy()
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
# list(inventoryRatio_beefHayCostDeltas.columns)

# %%
indp_vars = ["npp_t", "beef_price_at_1982", "hay_price_at_1982"]
y_var = "inventory"

#################################################################
X = all_df_normalized_needed[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = np.log(all_df_normalized_needed[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%

# %%
indp_vars = ["npp_u", "rangeland_acre", "beef_price_at_1982", "hay_price_at_1982"]
y_var = "inventory"

#################################################################
X = all_df_normalized_needed[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = np.log(all_df_normalized_needed[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%

# %% [markdown]
# ### Form Lags of independent variables

# %%
all_df_normalized_needed.head(2)

# %%
print (all_df_normalized_needed.dropna(how="any", inplace=False).shape)
print (all_df_normalized_needed.shape)

# %%
# 1 year prior
yr_lag = 1
cc_ = ['year', 'state_fips', 'inventory', 'npp_t', 'npp_u',
       'beef_price_at_1982', 'hay_price_at_1982', 'chicken_price_at_1982']

all_df_normalized_needed_yrbefore = all_df_normalized_needed[cc_].copy()
all_df_normalized_needed_yrbefore.drop("inventory", axis=1, inplace=True)
all_df_normalized_needed_yrbefore["year"] = all_df_normalized_needed_yrbefore["year"] + yr_lag

all_df_normalized_needed_yrbefore.rename(columns={"npp_t": "npp_t" + str(yr_lag),
                                                  "npp_u": "npp_u" + str(yr_lag),
                                                  "beef_price_at_1982": "B" + str(yr_lag),
                                                  "hay_price_at_1982" : "H" + str(yr_lag),
                                                  "chicken_price_at_1982": "C" + str(yr_lag)
                                                 }, inplace=True)

all_df_normalized_needed = pd.merge(all_df_normalized_needed, all_df_normalized_needed_yrbefore, 
                                    on=["year", "state_fips"], how="left")

all_df_normalized_needed.head(2)

# %%
# 2 year prior
yr_lag = 2
cc_ = ['year', 'state_fips', 'inventory', 'npp_t', 'npp_u',
       'beef_price_at_1982', 'hay_price_at_1982', 'chicken_price_at_1982']

all_df_normalized_needed_yrbefore = all_df_normalized_needed[cc_].copy()
all_df_normalized_needed_yrbefore.drop("inventory", axis=1, inplace=True)
all_df_normalized_needed_yrbefore["year"] = all_df_normalized_needed_yrbefore["year"] + yr_lag

all_df_normalized_needed_yrbefore.rename(columns={"npp_t": "npp_t" + str(yr_lag),
                                                  "npp_u": "npp_u" + str(yr_lag),
                                                  "beef_price_at_1982": "B" + str(yr_lag),
                                                  "hay_price_at_1982" : "H" + str(yr_lag),
                                                  "chicken_price_at_1982": "C" + str(yr_lag)
                                                 }, inplace=True)

all_df_normalized_needed = pd.merge(all_df_normalized_needed, all_df_normalized_needed_yrbefore, 
                                    on=["year", "state_fips"], how="left")

all_df_normalized_needed.head(2)

# %%
# 3 year prior
yr_lag = 3
cc_ = ['year', 'state_fips', 'inventory', 'npp_t', 'npp_u',
       'beef_price_at_1982', 'hay_price_at_1982', 'chicken_price_at_1982']

all_df_normalized_needed_yrbefore = all_df_normalized_needed[cc_].copy()
all_df_normalized_needed_yrbefore.drop("inventory", axis=1, inplace=True)
all_df_normalized_needed_yrbefore["year"] = all_df_normalized_needed_yrbefore["year"] + yr_lag

all_df_normalized_needed_yrbefore.rename(columns={"npp_t": "npp_t" + str(yr_lag),
                                                  "npp_u": "npp_u" + str(yr_lag),
                                                  "beef_price_at_1982": "B" + str(yr_lag),
                                                  "hay_price_at_1982" : "H" + str(yr_lag),
                                                  "chicken_price_at_1982": "C" + str(yr_lag)
                                                 }, inplace=True)

all_df_normalized_needed = pd.merge(all_df_normalized_needed, all_df_normalized_needed_yrbefore, 
                                    on=["year", "state_fips"], how="left")

all_df_normalized_needed.head(2)

# %%
check_cols = ["year", 
              "npp_t", "npp_t1", "npp_t2", "npp_t3",
              "npp_u", "npp_u1", "npp_u2", "npp_u3",
              "beef_price_at_1982", "B1", "B2", "B3",
              "hay_price_at_1982", "H1", "H2", "H3",
             ]

all_df_normalized_needed.loc[all_df_normalized_needed.state_fips == "01", check_cols].head(5)

# %%

# %% [markdown]
# # Model with 1 year lag

# %%
all_df_normalized.head(2)

# %%
all_df_normalized_needed.head(2)

# %%
indp_vars = ["npp_t", "npp_t1",
             "beef_price_at_1982", "B1",
             "hay_price_at_1982",  "H1",]

y_var = "inventory"

#################################################################
df = all_df_normalized_needed[[y_var] + indp_vars].copy()
df.dropna(how="any", inplace=True)
X = df[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = np.log(df[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%
# indp_vars = ["npp_u", "npp_u1", 'rangeland_acre',
#              "beef_price_at_1982", "B1",
#              "hay_price_at_1982",  "H1",]

# y_var = "inventory"

# #################################################################
# df = all_df_normalized_needed[[y_var] + indp_vars].copy()
# df.dropna(how="any", inplace=True)
# X = df[indp_vars]
# for col_ in indp_vars:
#     X[col_] = X[col_].astype("float")

# X = sm.add_constant(X)
# Y = np.log(df[y_var].astype(float))
# model = sm.OLS(Y, X)
# model_result = model.fit()
# model_result.summary()

# %%

# %%
indp_vars = ["npp_t", "npp_t1", "npp_t2",
             "beef_price_at_1982", "B1", "B2",
             "hay_price_at_1982",  "H1", "H2"]

y_var = "inventory"

#################################################################
df = all_df_normalized_needed[[y_var] + indp_vars].copy()
df.dropna(how="any", inplace=True)
X = df[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = np.log(df[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%

# %%
indp_vars = ["npp_t"]

y_var = "inventory"

#################################################################
df = all_df_normalized_needed[[y_var] + indp_vars].copy()
df.dropna(how="any", inplace=True)
X = df[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = np.log(df[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%

# %%
indp_vars = ["npp_t", "npp_t1", "npp_t2"]

y_var = "inventory"

#################################################################
df = all_df_normalized_needed[[y_var] + indp_vars].copy()
df.dropna(how="any", inplace=True)
X = df[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = np.log(df[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%

# %%
indp_vars = ["npp_t", "npp_t1",
             "beef_price_at_1982", "hay_price_at_1982"]

y_var = "inventory"

#################################################################
df = all_df_normalized_needed[[y_var] + indp_vars].copy()
df.dropna(how="any", inplace=True)
X = df[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = np.log(df[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%

# %%
indp_vars = ["npp_t", "npp_t1", "npp_t2",
             "beef_price_at_1982", "hay_price_at_1982"]

y_var = "inventory"

#################################################################
df = all_df_normalized_needed[[y_var] + indp_vars].copy()
df.dropna(how="any", inplace=True)
X = df[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = np.log(df[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%

# %%
indp_vars = ["npp_t", "npp_t1", "npp_t2", "npp_t3",
             "beef_price_at_1982", "hay_price_at_1982"]

y_var = "inventory"

#################################################################
df = all_df_normalized_needed[[y_var] + indp_vars].copy()
df.dropna(how="any", inplace=True)
X = df[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = np.log(df[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%
list(all_df_normalized.columns)

# %%

# %%

# %%

# %%

# %%

# %%
