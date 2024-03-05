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
# This is a copy of 01_12_2024_county_inven_LongAvgInde_Cleaner.ipynb, obviously.
# Here I want to read all the data from one file rather than each variable in a separate file!
#
# ------------------------------------------------------------------
# ### I called this notebook cleaner.
#
# But, I am not sure if it will be cleaner. Micheal and I met on Jan 24 and he mentioned to use total_rangeland area
# as opposed to rangeland percetange. Same for irrigated hay.
#
# Here let us do
#
# - NPP vs. log(inventory) (NPP is representative of RA and herb_ratio)
# - NPP and human population vs. log(inventory)
#
# - SW vs. log(inventory)
# - SW and RA and herb_ratio vs. log(inventory)
#
#
# Extras.
#
# - NPP and human population and slaughter vs. log(inventory)
# - NPP and human population and slaughter and irr_hay vs. log(inventory)
# - SW and RA and herb_ratio and irr_hay vs. log(inventory)
#
# On Jan 12, Friday, HN, KR, MB had a meeting.
#
# - Inventory as a function of average of ```NPP``` or ```SW``` over 30 (long run, I made 30 up) years.
# - Add some other variable such as heat index and see the effect. F-test.
# - All models on different levels (county, state) all the time!

# %% [markdown]
# # ATTENTION!!!
#
# Inventory on county level comes from CENSUS. Thus, data for this is not annual.
#
# ```NPP``` and ```SW``` on county level comes from us. Thus, we have annual data for these.
#
# Hence, do not (left) merge inventory with ```NPP``` and ```SW```, otherwise, we miss a lot in
# the same data table! Keep them goddamn separate.

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
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %%
# for bold print
start_b = "\033[1m"
end_b = "\033[0;0m"
print("This is " + start_b + "a_bold_text" + end_b + "!")

# %%
abb_dict = pd.read_pickle(param_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
df_OuterJoined = pd.read_pickle(reOrganized_dir + "county_data_OuterJoined.sav")
df_OuterJoined = df_OuterJoined["all_df"]
df_OuterJoined.head(2)

# %% [markdown]
# ## County Fips

# %%
county_fips = pd.read_pickle(reOrganized_dir + "county_fips.sav")
county_fips = county_fips["county_fips"]

print(f"{len(county_fips.state.unique()) = }")
county_fips = county_fips[county_fips.state.isin(SoI_abb)].copy()
county_fips.drop_duplicates(inplace=True)
county_fips.reset_index(drop=True, inplace=True)
county_fips = county_fips[["county_fips", "county_name", "state", "EW"]]
print(f"{len(county_fips.state.unique()) = }")

county_fips.head(2)

# %%
del county_fips

# %% [markdown]
# ## NPP exist only after 2001!
# So let us use subset of cattle inventory from census

# %%
df_OuterJoined = df_OuterJoined[df_OuterJoined.Pallavi == "Y"]
df_OuterJoined = df_OuterJoined[df_OuterJoined.year >= 2001]
df_OuterJoined = df_OuterJoined[df_OuterJoined.year <= 2017]

df_OuterJoined.reset_index(drop=True, inplace=True)

df_OuterJoined.head(2)

# %%
df_OuterJoined.drop(
    labels=[
        "normal",
        "alert",
        "danger",
        "emergency",
        "s1_normal",
        "s1_alert",
        "s1_danger",
        "s1_emergency",
        "s2_normal",
        "s2_alert",
        "s2_danger",
        "s2_emergency",
        "s3_normal",
        "s3_alert",
        "s3_danger",
        "s3_emergency",
        "s4_normal",
        "s4_alert",
        "s4_danger",
        "s4_emergency",
    ],
    axis=1,
    inplace=True,
)

# %% [markdown]
# ## Inventory

# %%
inventory_2017 = df_OuterJoined[df_OuterJoined.year == 2017].copy()
inventory_2017 = inventory_2017[["year", "county_fips", "inventory"]]
print(inventory_2017.shape)
inventory_2017.dropna(how="any", inplace=True)
print(inventory_2017.shape)
inventory_2017.reset_index(drop=True, inplace=True)
inventory_2017.head(2)

inv_2017_Pallavi_cnty_list = list(inventory_2017.county_fips.unique())

# %%
print(inventory_2017.shape)
inventory_2017.head(3)

# %%
df_OuterJoined = df_OuterJoined[
    df_OuterJoined.county_fips.isin(inv_2017_Pallavi_cnty_list)
]

# %% [markdown]
# # WARNING.
#
# **Pallavi's filter shrunk 29 states to 22.**
#
#
# ### Since there are too many incomlete counties, lets just keep them!
#
# Let us keep it simple. inventory as a function of long average ```NPP``` and long average ```SW```.
# And then add heat stress and see what happens.
#
# Then we can add
#  - rangeland acre
#  - herb ratio
#  - irrigated hay %
#  - feed expense
#  - population
#  - slaughter

# %% [markdown]
# # Model Snapshot
# **for 2017 Inventory and long run avg of independent variables**

# %%
inventory_2017.head(2)

# %% [markdown]
# ## Compute long run averages
#
# Since ```NPP``` exist after 2001, we filter ```SW``` from 2001 as well. Otherwise, there is no other reason.

# %%
npp_cols = ["county_total_npp", "unit_npp"]

sw_cols = [
    "S1_countyMean_total_precip",
    "S2_countyMean_total_precip",
    "S3_countyMean_total_precip",
    "S4_countyMean_total_precip",
    "S1_countyMean_avg_Tavg",
    "S2_countyMean_avg_Tavg",
    "S3_countyMean_avg_Tavg",
    "S4_countyMean_avg_Tavg",
]

# %%
common_cols = ["county_fips", "year"]

# %%
NPP_SW_heat = df_OuterJoined[common_cols + npp_cols + sw_cols].copy()
# do not drop NA as it will not affect taking mean
# NPP_SW_heat2.dropna(how="any", inplace=True)
NPP_SW_heat.sort_values(by=["year", "county_fips"], inplace=True)
NPP_SW_heat.reset_index(drop=True, inplace=True)
NPP_SW_heat.head(2)

# %%
print(len(NPP_SW_heat.county_fips.unique()))

# %%
NPP_SW_heat_avg = NPP_SW_heat.groupby("county_fips").mean()
NPP_SW_heat_avg.reset_index(drop=False, inplace=True)
NPP_SW_heat_avg.drop(labels=["year"], axis=1, inplace=True)
NPP_SW_heat_avg = NPP_SW_heat_avg.round(3)
NPP_SW_heat_avg.head(3)

# %% [markdown]
# # Model (time, finally)

# %%
NPP_SW_heat.sort_values(by=["county_fips", "year"], inplace=True)
NPP_SW_heat_avg.sort_values(by=["county_fips"], inplace=True)
inventory_2017.sort_values(by=["county_fips"], inplace=True)
inventory_2017.head(2)

# %%
print(NPP_SW_heat_avg.shape)
print(inventory_2017.shape)

NPP_SW_heat_avg.head(2)

# %% [markdown]
# # CAREFUL
#
# Let us merge average data and inventory data. Keep in mind that the year will be 2017 but the data are averaged over the years (except for the inventory).
#
# Merge them so that counties are in the same order. My sick mind!

# %%
inventory_2017.head(2)

# %%
inv_2017_NPP_SW_heat_avg = pd.merge(
    inventory_2017, NPP_SW_heat_avg, on=["county_fips"], how="left"
)
inv_2017_NPP_SW_heat_avg.head(2)

# %% [markdown]
# ### Normalize (unit length too for VIF)

# %%
all_indp_vars = list(set(sw_cols + npp_cols))  # AW_vars
all_indp_vars = sorted(all_indp_vars)
all_indp_vars

# %% [markdown]
# # Others (Controls)
#
#  - rangeland acre
#  - herb ratio
#  - irrigated hay %
#  - feed expense
#  - population
#  - slaughter

# %%
RA_cols = ["county_fips", "rangeland_fraction", "rangeland_acre"]
herb_cols = ["county_fips", "herb_avg", "herb_area_acr"]
irr_hay_cols = ["county_fips", "irr_hay_area", "irr_hay_as_perc"]
feed_cols = ["county_fips", "year", "feed_expense"]
pop_cols = ["county_fips", "year", "population"]
slaughter_cols = ["year", "county_fips", "slaughter"]


control_cols = (
    ["county_fips"]
    + ["year"]
    + ["rangeland_fraction", "rangeland_acre"]
    + ["herb_avg", "herb_area_acr"]
    + ["irr_hay_area", "irr_hay_as_perc"]
    + ["population"]
    + ["feed_expense", "slaughter"]
)

# %%
controls = df_OuterJoined[control_cols].copy()
controls.head(2)

# %% [markdown]
# ## RA is already satisfies Pallavi condition
#
# ## One more layer of filter according to 2017 inventory

# %%
print(controls[~(controls.population.isna())].year.unique())
print(controls.dropna(how="any", inplace=False).shape)
print(len(controls.dropna(how="any", inplace=False).county_fips.unique()))

a = controls.dropna(how="any", inplace=False).county_fips.unique()
main_cnties = inv_2017_NPP_SW_heat_avg.county_fips.unique()
len([x for x in main_cnties if x in a])

# %%
controls.head(2)

# %% [markdown]
# ## Take Averages

# %%
variable_controls = ["population", "slaughter", "feed_expense"]

constant_controls = [
    "herb_avg",
    "herb_area_acr",
    "rangeland_fraction",
    "rangeland_acre",
    "irr_hay_area",
    "irr_hay_as_perc",
]

constant_control_df = controls[["county_fips"] + constant_controls].drop_duplicates()
constant_control_df.reset_index(drop=True, inplace=True)
constant_control_df.head(2)

# %%
controls_avg = (
    controls[["county_fips"] + variable_controls].groupby("county_fips").mean()
)
controls_avg.reset_index(drop=False, inplace=True)
controls_avg = controls_avg.round(3)
controls_avg.head(2)

# %%
controls_avg = pd.merge(
    controls_avg, constant_control_df, on=["county_fips"], how="left"
)
print(controls_avg.shape)
controls_avg.head(2)

# %% [markdown]
# ### Join - Drop Na. then normalize to unit vector.
#
# Otherwise, when we drop NAs in different cases differently, colums will not have norm 1.

# %%
all_df = pd.merge(
    inv_2017_NPP_SW_heat_avg, controls_avg, on=["county_fips"], how="outer"
)
print(all_df.shape)
all_df.head(2)

# %%
normalize_cols = [
    "population",
    "slaughter",
    "feed_expense",
    "herb_avg",
    "herb_area_acr",
    "rangeland_fraction",
    "rangeland_acre",
    "irr_hay_area",
    "irr_hay_as_perc",
]

normalize_cols = all_indp_vars + normalize_cols

# %%
all_df = all_df[["year", "county_fips", "inventory"] + normalize_cols]

# %%
all_df.dropna(how="any", inplace=True)
all_df.reset_index(drop=True, inplace=True)
print(all_df.shape)

# %% [markdown]
# ### Export and check
#
# ```all_df``` here and ```curr_all``` from ```01_12_2024_county_inven_LongAvgInd_Cleaner_II``` notebook
# and see why ```curr_all``` has 497 rows in it while ```all_df``` has 440.
#
# ```indp_vars = sw_cols + ["rangeland_acre", "herb_area_acr", "irr_hay_area", "population"]```
#

# %%
all_indp_vars

# %%
all_df_normal = all_df.copy()
all_df_normal[normalize_cols] = (
    all_df_normal[normalize_cols] - all_df_normal[normalize_cols].mean()
)

col_norms = all_df_normal[normalize_cols].apply(np.linalg.norm, axis=0)
all_df_normal[normalize_cols] = all_df_normal[normalize_cols] / col_norms

print((np.linalg.norm(all_df_normal.county_total_npp)))
print((np.linalg.norm(all_df_normal.S1_countyMean_total_precip)))
all_df_normal.head(2)

# %%
X = all_df_normal[normalize_cols].values
X

# %%

# %%

# %%
normalize_cols

# %%
print(round(VIFs[normalize_cols.index("county_total_npp")], 1))
print(round(VIFs[normalize_cols.index("unit_npp")], 1))
print(round(VIFs[normalize_cols.index("rangeland_acre")], 1))

# %%
cc = [
    "county_total_npp",
    # 'unit_npp',
    "population",
    "herb_avg",
    # 'herb_area_acr',
    # 'rangeland_fraction',
    "rangeland_acre",
    "irr_hay_area",
]


X = all_df_normal[cc].values
XT_X = np.dot(np.transpose(X), X)
inv_Xt_X = np.linalg.inv(XT_X)
VIFs = np.diagonal(inv_Xt_X)
[round(x, 1) for x in VIFs]

# %%
cc = [  # 'county_total_npp',
    "unit_npp",
    "population",
    "herb_avg",
    # 'herb_area_acr',
    # 'rangeland_fraction',
    "rangeland_acre",
    "irr_hay_area",
]


X = all_df_normal[cc].values
XT_X = np.dot(np.transpose(X), X)
inv_Xt_X = np.linalg.inv(XT_X)
VIFs = np.diagonal(inv_Xt_X)
[round(x, 1) for x in VIFs]

# %%

# %%
