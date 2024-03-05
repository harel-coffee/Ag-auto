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
df_OuterJoined["dangerEncy"] = df_OuterJoined["danger"] + df_OuterJoined["emergency"]

df_OuterJoined["s1_dangerEncy"] = (
    df_OuterJoined["s1_danger"] + df_OuterJoined["s1_emergency"]
)
df_OuterJoined["s2_dangerEncy"] = (
    df_OuterJoined["s2_danger"] + df_OuterJoined["s2_emergency"]
)
df_OuterJoined["s3_dangerEncy"] = (
    df_OuterJoined["s3_danger"] + df_OuterJoined["s3_emergency"]
)
df_OuterJoined["s4_dangerEncy"] = (
    df_OuterJoined["s4_danger"] + df_OuterJoined["s4_emergency"]
)
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

df_OuterJoined.head(2)

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
# ### NPP & RA
#
# We need RA to convert unit NPP to total NPP.

# %%
# RA = pd.read_pickle(param_dir + "filtered_counties_RAsizePallavi.sav")
# RA = RA["filtered_counties_29States"]
# print (f"{len(RA.county_fips.unique()) = }")
# print (f"{len(RA.state.unique()) = }")
# RA.head(2)

# cty_yr_npp = pd.read_csv(reOrganized_dir + "county_annual_GPP_NPP_productivity.csv")
# cty_yr_npp.head(2)

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
npp_cols = ["county_total_npp", "unit_npp", "unit_npp_std", "county_total_npp_std"]

sw_cols = [
    "s1_countymean_total_precip",
    "s2_countymean_total_precip",
    "s3_countymean_total_precip",
    "s4_countymean_total_precip",
    "s1_countymean_avg_tavg",
    "s2_countymean_avg_tavg",
    "s3_countymean_avg_tavg",
    "s4_countymean_avg_tavg",
]


heat_cols = ["dangerEncy"]

# %%
common_cols = ["county_fips", "year"]
# NPP = df_OuterJoined[common_cols + npp_cols].copy()
# SW = df_OuterJoined[common_cols + sw_cols].copy()
# heat = df_OuterJoined[common_cols + heat_cols].copy()

# NPP.dropna(how="any", inplace=True)
# SW.dropna(how="any", inplace=True)
# heat.dropna(how="any", inplace=True)

# NPP.reset_index(drop=False, inplace=True)
# SW.reset_index(drop=False, inplace=True)
# heat.reset_index(drop=False, inplace=True)

# %%
NPP_SW_heat = df_OuterJoined[common_cols + npp_cols + sw_cols + heat_cols].copy()
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
NPP_SW_heat_avg.head(2)

# %%
print(NPP_SW_heat_avg.shape)
print(inventory_2017.shape)

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
# ### Normalize

# %%
HS_var = ["dangerEncy"]

all_indp_vars = list(set(HS_var + sw_cols + npp_cols))  # AW_vars
# all_indp_vars = sorted(all_indp_vars)
all_indp_vars

# %%
print(inv_2017_NPP_SW_heat_avg.unit_npp_std.min())
print(inv_2017_NPP_SW_heat_avg.unit_npp_std.max())
print()
print(inv_2017_NPP_SW_heat_avg.unit_npp.min())
print(inv_2017_NPP_SW_heat_avg.unit_npp.max())
print()
print(inv_2017_NPP_SW_heat_avg.county_total_npp_std.min())
print(inv_2017_NPP_SW_heat_avg.county_total_npp_std.max())
print()
print(inv_2017_NPP_SW_heat_avg.county_total_npp.min())
print(inv_2017_NPP_SW_heat_avg.county_total_npp.max())

# %%
inv_2017_NPP_SW_heat_avg_normal = inv_2017_NPP_SW_heat_avg.copy()
inv_2017_NPP_SW_heat_avg_normal[all_indp_vars] = (
    inv_2017_NPP_SW_heat_avg_normal[all_indp_vars]
    - inv_2017_NPP_SW_heat_avg_normal[all_indp_vars].mean()
) / inv_2017_NPP_SW_heat_avg_normal[all_indp_vars].std(ddof=1)
inv_2017_NPP_SW_heat_avg_normal.head(2)

# %%
print(inv_2017_NPP_SW_heat_avg_normal.unit_npp_std.min())
print(inv_2017_NPP_SW_heat_avg_normal.unit_npp_std.max())
print()
print(inv_2017_NPP_SW_heat_avg_normal.unit_npp.min())
print(inv_2017_NPP_SW_heat_avg_normal.unit_npp.max())
print()
print(inv_2017_NPP_SW_heat_avg_normal.county_total_npp_std.min())
print(inv_2017_NPP_SW_heat_avg_normal.county_total_npp_std.max())
print()
print(inv_2017_NPP_SW_heat_avg_normal.county_total_npp.min())
print(inv_2017_NPP_SW_heat_avg_normal.county_total_npp.max())

# %% [markdown]
# # Model
#
# ### Inventory vs normal ```NPP``` averaged over 2001-2017

# %%
inv_2017_NPP_SW_heat_avg_normal.head(2)

# %%
indp_vars = ["county_total_npp"]
y_var = "inventory"

#################################################################
X_npp = inv_2017_NPP_SW_heat_avg_normal[indp_vars]
X_npp = sm.add_constant(X_npp)
Y = np.log(inv_2017_NPP_SW_heat_avg_normal[y_var].astype(float))
npp_inv_model = sm.OLS(Y, X_npp)
npp_inv_model_result = npp_inv_model.fit()
npp_inv_model_result.summary()

# %%
del (X_npp, npp_inv_model, npp_inv_model_result)

# %%
indp_vars = ["county_total_npp", "county_total_npp_std"]
y_var = "inventory"

#################################################################
X_npp = inv_2017_NPP_SW_heat_avg_normal[indp_vars]
X_npp = sm.add_constant(X_npp)
Y = np.log(inv_2017_NPP_SW_heat_avg_normal[y_var].astype(float))
npp_inv_model = sm.OLS(Y, X_npp)
npp_inv_model_result = npp_inv_model.fit()
npp_inv_model_result.summary()

# %%
del (X_npp, npp_inv_model, npp_inv_model_result)

# %% [markdown]
# ### Inventory vs normal ```SW``` averaged over 2001-2017

# %%
indp_vars = sw_cols
y_var = "inventory"

#################################################################
X_SW = inv_2017_NPP_SW_heat_avg_normal[indp_vars]
X_SW = sm.add_constant(X_SW)
Y = np.log(inv_2017_NPP_SW_heat_avg_normal[y_var].astype(float))
SW_inv_model = sm.OLS(Y, X_SW)
SW_inv_model_result = SW_inv_model.fit()
SW_inv_model_result.summary()

# %%
SW_inv_model_result.params

# %%
del (X_SW, SW_inv_model, SW_inv_model_result)

# %% [markdown]
# ### Inventory vs normal ```NPP``` AND ```dangerEncy``` averaged over 2001-2017

# %%
indp_vars = ["county_total_npp"] + HS_var
y_var = "inventory"

#################################################################
X_npp_dangerEncy = inv_2017_NPP_SW_heat_avg_normal[indp_vars]
X_npp_dangerEncy = sm.add_constant(X_npp_dangerEncy)
Y = np.log(inv_2017_NPP_SW_heat_avg_normal[y_var].astype(float))
npp_dangerEncy_inv_model = sm.OLS(Y, X_npp_dangerEncy)
npp_dangerEncy_inv_model_result = npp_dangerEncy_inv_model.fit()
npp_dangerEncy_inv_model_result.summary()

# %%
del (X_npp_dangerEncy, npp_dangerEncy_inv_model, npp_dangerEncy_inv_model_result)

# %% [markdown]
# ### Inventory vs normal ```SW``` AND ```dangerEncy``` averaged over 2001-2017

# %%
indp_vars = sw_cols + HS_var
y_var = "inventory"

#################################################################
X_SW_dangerEncy = inv_2017_NPP_SW_heat_avg_normal[indp_vars]
X_SW_dangerEncy = sm.add_constant(X_SW_dangerEncy)
Y = np.log(inv_2017_NPP_SW_heat_avg_normal[y_var].astype(float))
SW_dangerEncy_inv_model = sm.OLS(Y, X_SW_dangerEncy)
SW_dangerEncy_inv_model_result = SW_dangerEncy_inv_model.fit()
SW_dangerEncy_inv_model_result.summary()

# %%
SW_dangerEncy_inv_model_result.params

# %%
del (X_SW_dangerEncy, SW_dangerEncy_inv_model, SW_dangerEncy_inv_model_result)

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
# irr_hay = df_OuterJoined[irr_hay_cols].copy()
# print (irr_hay.shape)
# irr_hay.drop_duplicates(inplace=True)
# irr_hay.reset_index(drop=True, inplace=True)
# print (irr_hay.shape)
# irr_hay.head(2)

# %%
# USDA_data = pd.read_pickle(reOrganized_dir + "USDA_data.sav")

# human_population = pd.read_pickle(reOrganized_dir + "human_population.sav")
# human_population = human_population["human_population"]
# human_population.head(2)

# feed_expense = USDA_data["feed_expense"]
# feed_expense = feed_expense[["year", "county_fips", "feed_expense"]]

# human_population = pd.read_pickle(reOrganized_dir + "human_population.sav")
# human_population = human_population["human_population"]

# slaughter_Q1 = pd.read_pickle(reOrganized_dir + "slaughter_Q1.sav")
# slaughter_Q1 = slaughter_Q1["slaughter_Q1"]
# slaughter_Q1.rename(columns={"cattle_on_feed_sale_4_slaughter": "slaughter"}, inplace=True)
# slaughter_Q1 = slaughter_Q1[["year", "county_fips", "slaughter"]]
# print ("max slaughter sale is [{}]".format(slaughter_Q1.slaughter.max()))

# controls = pd.merge(human_population, slaughter_Q1, on=["county_fips", "year"], how="outer")
# controls = pd.merge(controls, feed_expense, on=["county_fips", "year"], how="outer")
# controls = pd.merge(controls, irr_hay, on=["county_fips"], how="outer")
# controls = pd.merge(controls, herb, on=["county_fips"], how="outer")

# controls.head(2)

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
main_cnties = inv_2017_NPP_SW_heat_avg_normal.county_fips.unique()
len([x for x in main_cnties if x in a])

# %%
controls.head(2)

# %% [markdown]
# # Take Averages then Normalize

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

# %%
controls_avg["irr_hay_as_perc_categ"] = controls_avg["irr_hay_as_perc"]

controls_avg.loc[(controls_avg.irr_hay_as_perc <= 6), "irr_hay_as_perc_categ"] = 0

controls_avg.loc[
    (controls_avg.irr_hay_as_perc > 6) & (controls_avg.irr_hay_as_perc <= 96),
    "irr_hay_as_perc_categ",
] = 1

controls_avg.loc[(controls_avg.irr_hay_as_perc > 96), "irr_hay_as_perc_categ"] = 2

controls_avg.head(2)

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

controls_avg_normal = controls_avg.copy()

controls_avg_normal[normalize_cols] = (
    controls_avg_normal[normalize_cols] - controls_avg_normal[normalize_cols].mean()
) / controls_avg_normal[normalize_cols].std(ddof=1)
controls_avg_normal.head(3)

# %%
inv_2017_NPP_SW_heat_avg_normal.head(2)

# %%
controls_avg_normal_NATossed = controls_avg_normal.dropna(how="any", inplace=False)
control_counties = sorted(list(controls_avg_normal_NATossed.county_fips.unique()))
main_counties = sorted(list(controls_avg_normal.county_fips.unique()))
control_counties == main_counties

A = [x for x in main_counties if not (x in control_counties)]
B = [x for x in control_counties if not (x in main_counties)]
print(f"{len(A)=}, {len(B)=}")

# %%
controls_avg_normal.head(2)

# %% [markdown]
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
#
# ## NPP and Control

# %%
inv_2017_NPP_SW_heat_avg_normal.head(2)

# %%
controls_avg_normal.head(2)

# %%
inv_2017_NPP_SW_heat_avg_normal.shape

# %%
controls_avg_normal.head(2)

# %%
del A
L = inv_2017_NPP_SW_heat_avg_normal.county_fips.unique()
A = controls_avg_normal[controls_avg_normal.county_fips.isin(L)]
A.shape

unique_counties = A.county_fips.unique()
repeated_counties = []

for cnty in unique_counties:
    if len(A[A.county_fips == cnty]) > 1:
        repeated_counties = repeated_counties + [cnty]

repeated_counties


# %%
controls_avg_normal[controls_avg_normal.county_fips == "04009"]

# %%
print(inv_2017_NPP_SW_heat_avg_normal.shape)
all_df = pd.merge(
    inv_2017_NPP_SW_heat_avg_normal,
    controls_avg_normal,
    on=["county_fips"],
    how="outer",
)
print(all_df.shape)
all_df.head(2)

# %%
indp_vars = ["county_total_npp", "rangeland_acre"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["county_total_npp", "rangeland_acre", "county_total_npp_std"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["unit_npp", "rangeland_acre"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["unit_npp", "rangeland_acre", "unit_npp_std"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["county_total_npp", "rangeland_acre", "herb_area_acr"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["county_total_npp", "rangeland_acre", "herb_area_acr", "irr_hay_area"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["unit_npp"] + ["rangeland_acre", "herb_area_acr", "irr_hay_area"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["county_total_npp"] + ["rangeland_acre", "irr_hay_area"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%

# %%
indp_vars = ["unit_npp"] + ["rangeland_acre", "herb_area_acr"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["county_total_npp", "population"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["county_total_npp"] + ["rangeland_acre", "population"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["county_total_npp"] + ["rangeland_acre", "herb_area_acr", "population"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()


# %%
model_result.params

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["county_total_npp"] + ["rangeland_acre", "irr_hay_area", "population"]
y_var = "inventory"
#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %% [markdown]
# # SW and Controls

# %%
sw_cols

# %%
all_df.head(2)

# %%
indp_vars = sw_cols + ["rangeland_acre"]
y_var = "inventory"
#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = sw_cols + ["rangeland_acre", "herb_area_acr"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = sw_cols + ["rangeland_acre", "herb_area_acr", "irr_hay_area"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%

# %%
indp_vars = ["county_total_npp"] + [
    "rangeland_acre",
    "herb_area_acr",
    "irr_hay_area",
    "population",
]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
model_result.params.round(2)

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = sw_cols + ["rangeland_acre", "herb_area_acr", "irr_hay_area", "population"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = sw_cols + ["rangeland_acre", "population"]
y_var = "inventory"
#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = sw_cols + ["rangeland_acre", "herb_area_acr", "population"]
y_var = "inventory"
#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["rangeland_acre"]
y_var = "inventory"
#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
all_df.columns

# %%
indp_vars = ["county_total_npp"] + [
    "rangeland_acre",
    "herb_avg",
    "irr_hay_area",
    "population",
]
y_var = "inventory"
#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
indp_vars = ["unit_npp"] + ["rangeland_acre", "herb_avg", "irr_hay_area", "population"]
y_var = "inventory"
#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
indp_vars = ["unit_npp"] + [
    "rangeland_acre",
    "herb_avg",
    "irr_hay_area",
    "population",
    "dangerEncy",
]
y_var = "inventory"
#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
indp_vars = sw_cols + ["rangeland_acre", "herb_avg", "irr_hay_area", "population"]
y_var = "inventory"
#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
indp_vars = ["unit_npp"] + [
    "rangeland_acre",
    "herb_area_acr",
    "irr_hay_area",
    "population",
]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%

# %% [markdown]
# # Export the dataframe above and in the other notebook and see the diff.

# %%
indp_vars = ["unit_npp"] + sw_cols + ["rangeland_acre"]
y_var = "inventory"
#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%

# %%

# %%

# %%

# %%

# %%
