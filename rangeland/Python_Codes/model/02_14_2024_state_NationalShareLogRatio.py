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
# <font color='red'>red</font>, <span style='color:green'>green</span>, $\color{blue}{\text{blue}}$

# %%
import shutup

shutup.please()

import pandas as pd
import numpy as np
import os, os.path, pickle, sys
from datetime import datetime, date

from sklearn import preprocessing
import statistics
import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)
datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

# %%
filename = reOrganized_dir + "state_data_OuterJoined.sav"
state_data = pd.read_pickle(filename)
state_data.keys()

# %%
state_data["NOTE"]

# %%
state_data = state_data["all_df"]
print(f"{len(state_data.columns) = }")
state_data.head(2)

# %%
print(state_data.dropna(subset=["state_unit_npp"])["year"].min())
print(state_data.dropna(subset=["state_unit_npp"])["year"].max())

# %%
state_data = state_data[state_data.year < 2021]

# %% [markdown]
# ### drop not-needed columns

# %%
not_needed_cols = [x for x in state_data.columns if (("gimms" in x) or ("avhrr" in x))]
state_data.drop(labels=not_needed_cols, axis=1, inplace=True)
state_data.head(2)

# %%

# %% [markdown]
# **Subset to states of interest**

# %%
state_fips_df = abb_dict["county_fips"]
state_fips_df = state_fips_df[["state", "state_fips"]].copy()

state_fips_df.drop_duplicates(inplace=True)
state_fips_df = state_fips_df[state_fips_df.state.isin(SoI_abb)]
state_fips_df.reset_index(drop=True, inplace=True)

print(f"{len(state_fips_df) = }")
state_fips_df.head(2)

# %%
state_data = state_data[state_data.state_fips.isin(state_fips_df.state_fips)]
state_data.reset_index(drop=True, inplace=True)

print(f"{len(state_data.state_fips.unique()) = }")
state_data.head(2)

# %% [markdown]
# ## Compute national share of each state

# %%
total_inv = state_data[["year", "inventory"]].copy()
total_inv = total_inv.groupby(["year"]).sum().reset_index()
total_inv.rename(columns={"inventory": "total_inventory"}, inplace=True)

state_data = pd.merge(state_data, total_inv, on=["year"], how="left")
state_data.head(2)

# %%
state_data["inventory_share"] = (
    state_data["inventory"] / state_data["total_inventory"]
) * 100
state_data.head(2)

# %%
var_2_plot = "inventory_share"
year_2_plot = 1920

df = state_data[state_data.year == year_2_plot]

fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharey=False)
axes.grid(axis="y", which="both")
axes.set_axisbelow(True)  # sends the grids underneath the plot


plt.hist(df[var_2_plot], bins=100)
x_max = df[var_2_plot].max()
x_limit = x_max + x_max * 0.05
plt.xticks(np.arange(0, x_limit, 2), rotation="horizontal")

axes.title.set_text(f"{var_2_plot, str(year_2_plot)}")
axes.set_xlabel(var_2_plot)
axes.set_ylabel("count")
fig.tight_layout()

# %%
var_2_plot = "inventory_share"
year_2_plot = 2001

df = state_data[state_data.year == year_2_plot]

fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharey=False)
axes.grid(axis="y", which="both")
axes.set_axisbelow(True)  # sends the grids underneath the plot

plt.hist(df[var_2_plot], bins=100)
x_max = df[var_2_plot].max()
x_limit = x_max + x_max * 0.05
plt.xticks(np.arange(0, x_limit, 2), rotation="horizontal")

axes.title.set_text(f"{var_2_plot, str(year_2_plot)}")
axes.set_xlabel(var_2_plot)
axes.set_ylabel("count")
fig.tight_layout()

# %%
var_2_plot = "inventory_share"
year_2_plot = 2020

df = state_data[state_data.year == year_2_plot]

fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharey=False)
axes.grid(axis="y", which="both")
axes.set_axisbelow(True)  # sends the grids underneath the plot

plt.hist(df[var_2_plot], bins=100)
x_max = df[var_2_plot].max()
x_limit = x_max + x_max * 0.05
plt.xticks(np.arange(0, x_limit, 2), rotation="horizontal")

axes.title.set_text(f"{var_2_plot, str(year_2_plot)}")
axes.set_xlabel(var_2_plot)
axes.set_ylabel("count")
fig.tight_layout()

# %% [markdown]
# ### Subset: 2001 and 2020
#
# Since ```NPP``` exist only between 2001 and 2020.

# %%
state_data = state_data[(state_data.year >= 2001) & (state_data.year <= 2020)].copy()

state_data.reset_index(drop=True, inplace=True)
state_data.head(2)

# %%
### Sort values so we can be sure the ratios are correct
### we need a for-loop or sth. we cannot just do all of the df
### since the row at which state changes will be messed up.
state_data.sort_values(by=["state_fips", "year"], inplace=True)

# %%
cc = ["year", "state_fips", "log_ratio_of_shares_Y2PrevY"]
log_ratios_df = pd.DataFrame(columns=cc)

for a_state in state_data.state_fips.unique():
    curr_df = state_data[state_data.state_fips == a_state].copy()
    curr_ratios = (
        curr_df.inventory_share[1:].values / curr_df.inventory_share[:-1].values
    ).astype(float)
    curr_ratios_log = np.log(curr_ratios)

    curr_ratio_df = pd.DataFrame(columns=cc)
    curr_ratio_df["year"] = curr_df.year[1:].values
    curr_ratio_df["log_ratio_of_shares_Y2PrevY"] = curr_ratios_log
    curr_ratio_df["state_fips"] = a_state
    log_ratios_df = pd.concat([log_ratios_df, curr_ratio_df])
    del curr_ratio_df

# %%
log_ratios_df.head(2)

# %%
state_data = pd.merge(state_data, log_ratios_df, on=["state_fips", "year"], how="left")
state_data.head(2)

# %%
# state_data = rc.clean_census(df=state_data, col_="number_of_farm_operation")

# %%
state_data.columns

# %%
mean_ndvi_cols = [
    x for x in state_data.columns if (("ndvi" in x) & ("modis" in x) & ("mean" in x))
]
max_ndvi_cols = [
    x for x in state_data.columns if (("ndvi" in x) & ("modis" in x) & ("max" in x))
]
sum_ndvi_cols = [
    x for x in state_data.columns if (("ndvi" in x) & ("modis" in x) & ("sum" in x))
]

# %%
W_columns = [x for x in state_data.columns if (("precip" in x) or ("Tavg" in x))]
SW_columns = [x for x in W_columns if not ("annual" in x)]
AW_columns = [x for x in W_columns if "annual" in x]

# %% [markdown]
# ### log ratio and unit NPP model

# %%
indp_vars = ["state_unit_npp"]
y_var = "log_ratio_of_shares_Y2PrevY"
#################################################################
curr_all = state_data[indp_vars + [y_var] + ["state_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = curr_all[y_var].astype(float)
#################################################################
model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %% [markdown]
# ### log ratio and total NPP model

# %%
indp_vars = ["state_total_npp"]
y_var = "log_ratio_of_shares_Y2PrevY"
#################################################################
curr_all = state_data[indp_vars + [y_var] + ["state_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = curr_all[y_var].astype(float)
#################################################################
model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %% [markdown]
# ## Normalize columns

# %%
print(state_data.columns)

# %%
normalize_cols = [
    "state_unit_npp",
    "state_total_npp",
    "sale_4_slaughter_head",
    "population",
    "feed_expense",
    "number_of_farm_operation",
    "crp_wetland_acr",
    "s1_statemean_total_precip",
    "s2_statemean_total_precip",
    "s3_statemean_total_precip",
    "s4_statemean_total_precip",
    "s1_statemean_avg_tavg",
    "s2_statemean_avg_tavg",
    "s3_statemean_avg_tavg",
    "s4_statemean_avg_tavg",
    "s1_statemean_total_danger",
    "s2_statemean_total_danger",
    "s3_statemean_total_danger",
    "s4_statemean_total_danger",
    "s1_statemean_total_emergency",
    "s2_statemean_total_emergency",
    "s3_statemean_total_emergency",
    "s4_statemean_total_emergency",
    "annual_statemean_total_precip",
    "annual_avg_tavg",
    "rangeland_acre",
    "state_area_acre",
    "rangeland_fraction",
    "herb_avg",
    "herb_std",
    "maxNDVI_DoY_stateMean",
    "maxNDVI_DoY_stateMedian",
    "maxNDVI_DoY_stateStd",
    "maxNDVI_DoY_stateMin",
    "maxNDVI_DoY_stateMax",
    "herb_area_acr",
    "irr_hay_area",
    "total_area_irrHayRelated",
    "irr_hay_as_perc",
    "s1_sum_modis_ndvi",
    "s2_sum_modis_ndvi",
    "s3_sum_modis_ndvi",
    "s4_sum_modis_ndvi",
    "s1_mean_modis_ndvi",
    "s2_mean_modis_ndvi",
    "s3_mean_modis_ndvi",
    "s4_mean_modis_ndvi",
    "s1_max_modis_ndvi",
    "s2_max_modis_ndvi",
    "s3_max_modis_ndvi",
    "s4_max_modis_ndvi",
]


# %%
all_df_normalIndp = state_data.copy()

all_df_normalIndp[normalize_cols] = (
    all_df_normalIndp[normalize_cols] - all_df_normalIndp[normalize_cols].mean()
) / all_df_normalIndp[normalize_cols].std(ddof=1)
all_df_normalIndp.head(3)

# %%
del (curr_all, X, Y)

# %% [markdown]
# ### log ratio and unit NPP model (normalized independent)

# %%
indp_vars = ["state_unit_npp"]
y_var = "log_ratio_of_shares_Y2PrevY"
#################################################################
curr_all = all_df_normalIndp[indp_vars + [y_var] + ["state_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = curr_all[y_var].astype(float)
#################################################################
model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %% [markdown]
# ### log ratio and total NPP model (normalized independent)

# %%
indp_vars = ["state_total_npp"]
y_var = "log_ratio_of_shares_Y2PrevY"
#################################################################
curr_all = all_df_normalIndp[indp_vars + [y_var] + ["state_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = curr_all[y_var].astype(float)
#################################################################
model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (X, Y, curr_all, model_)

# %% [markdown]
# ### log ratio and unit NPP and RA model (normalized independent)

# %%
indp_vars = ["state_unit_npp"] + ["rangeland_acre"]
y_var = "log_ratio_of_shares_Y2PrevY"
#################################################################
curr_all = all_df_normalIndp[indp_vars + [y_var] + ["state_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = curr_all[y_var].astype(float)
#################################################################
model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (X, Y, curr_all, model_)

# %% [markdown]
# ### inventory and unit NPP and RA model (normalized independent)

# %%
indp_vars = ["state_unit_npp"] + ["rangeland_acre"]
y_var = "inventory"
#################################################################
curr_all = all_df_normalIndp[indp_vars + [y_var] + ["state_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))
#################################################################
model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (X, Y, curr_all, model_)

# %% [markdown]
# ### inventory and total NPP model (normalized independent)

# %%
indp_vars = ["state_total_npp"]
y_var = "inventory"
#################################################################
curr_all = all_df_normalIndp[indp_vars + [y_var] + ["state_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))
#################################################################
model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %% [markdown]
# ### inventory and total NPP and RA model (normalized independent)

# %%
indp_vars = ["state_total_npp"] + ["rangeland_acre"]
y_var = "inventory"
#################################################################
curr_all = all_df_normalIndp[indp_vars + [y_var] + ["state_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))
#################################################################
model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (X, Y, curr_all, model_)

# %%
all_df_normalIndp.head(2)

# %% [markdown]
# # Idea
#
# Perhaps, we have to take differences of independent variables in modeling the log ratios of inventory.
#
# ### inventory  share and total NPP and RA model (normalized independent)

# %%
indp_vars = ["state_total_npp"] + ["rangeland_acre"]
y_var = "inventory_share"
#################################################################
curr_all = all_df_normalIndp[indp_vars + [y_var] + ["state_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = curr_all[y_var].astype(float)
#################################################################
model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (X, Y, curr_all, model_)

indp_vars = ["state_unit_npp"] + ["rangeland_acre"]
y_var = "inventory_share"
#################################################################
curr_all = all_df_normalIndp[indp_vars + [y_var] + ["state_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = curr_all[y_var].astype(float)
#################################################################
model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
print(sorted(state_data["maxNDVI_DoY_stateMean"].unique()))

# %%
state_data.head(2)

# %%
cc = [
    "year",
    "state_fips",
    "state_unit_npp",
    "state_total_npp",
    "annual_statemean_total_precip",
    "annual_avg_tavg",
    "s1_sum_modis_ndvi",
    "s2_sum_modis_ndvi",
    "s3_sum_modis_ndvi",
    "s4_sum_modis_ndvi",
    "s1_mean_modis_ndvi",
    "s2_mean_modis_ndvi",
    "s3_mean_modis_ndvi",
    "s4_mean_modis_ndvi",
    "s1_max_modis_ndvi",
    "s2_max_modis_ndvi",
    "s3_max_modis_ndvi",
    "s4_max_modis_ndvi",
    "s1_statemean_avg_tavg",
    "s2_statemean_avg_tavg",
    "s3_statemean_avg_tavg",
    "s4_statemean_avg_tavg",
    "s1_statemean_total_precip",
    "s2_statemean_total_precip",
    "s3_statemean_total_precip",
    "s4_statemean_total_precip",
]

ind_deltas_df = pd.DataFrame(columns=cc)
ind_deltas_df.head(2)

for a_state in state_data.state_fips.unique():
    curr_df = state_data[state_data.state_fips == a_state].copy()
    curr_diff = (
        curr_df.loc[curr_df.index[1:], cc[2:]].values
        - curr_df.loc[curr_df.index[:-1], cc[2:]].values
    )
    curr_diff = pd.DataFrame(curr_diff, columns=cc[2:])
    curr_diff["year"] = curr_df.year[1:].values
    curr_diff["state_fips"] = a_state
    ind_deltas_df = pd.concat([ind_deltas_df, curr_diff])
    del curr_diff

# %%
nonDelta_df = state_data.copy()

non_delta_cols = [
    "year",
    "state_fips",
    "inventory",
    "sale_4_slaughter_head",
    "population",
    "feed_expense",
    "number_of_farm_operation",
    "crp_wetland_acr",
    "rangeland_acre",
    "state_area_acre",
    "rangeland_fraction",
    "herb_avg",
    "herb_std",
    "herb_area_acr",
    "irr_hay_area",
    "total_area_irrHayRelated",
    "irr_hay_as_perc",
    "total_inventory",
    "inventory_share",
    "log_ratio_of_shares_Y2PrevY",
    "maxNDVI_DoY_stateMean",
    "maxNDVI_DoY_stateMedian",
    "maxNDVI_DoY_stateStd",
    "maxNDVI_DoY_stateMin",
    "maxNDVI_DoY_stateMax",
]
nonDelta_df = nonDelta_df[non_delta_cols]
nonDelta_df.head(3)

# %%
ind_deltas_df.head(3)

# %%
delta_data = pd.merge(ind_deltas_df, nonDelta_df, on=["year", "state_fips"], how="left")
del nonDelta_df
delta_data.head(3)

# %%
## Check the Census data which is every 5 years.
delta_data[["sale_4_slaughter_head", "population"]].head(3)

# %%
delta_data.columns

# %% [markdown]
# ### Normalize delta dataframe

# %%
normalize_cols = [
    "state_unit_npp",
    "state_total_npp",
    "annual_statemean_total_precip",
    "annual_avg_tavg",
    "maxNDVI_DoY_stateMean",
    "maxNDVI_DoY_stateMedian",
    "maxNDVI_DoY_stateStd",
    "maxNDVI_DoY_stateMin",
    "maxNDVI_DoY_stateMax",
    "s1_sum_modis_ndvi",
    "s2_sum_modis_ndvi",
    "s3_sum_modis_ndvi",
    "s4_sum_modis_ndvi",
    "s1_mean_modis_ndvi",
    "s2_mean_modis_ndvi",
    "s3_mean_modis_ndvi",
    "s4_mean_modis_ndvi",
    "s1_max_modis_ndvi",
    "s2_max_modis_ndvi",
    "s3_max_modis_ndvi",
    "s4_max_modis_ndvi",
    "s1_statemean_avg_tavg",
    "s2_statemean_avg_tavg",
    "s3_statemean_avg_tavg",
    "s4_statemean_avg_tavg",
    "s1_statemean_total_precip",
    "s2_statemean_total_precip",
    "s3_statemean_total_precip",
    "s4_statemean_total_precip",
    "sale_4_slaughter_head",
    "population",
    "feed_expense",
    "number_of_farm_operation",
    "crp_wetland_acr",
    "rangeland_acre",
    "state_area_acre",
    "rangeland_fraction",
    "herb_avg",
    "herb_std",
    "herb_area_acr",
    "irr_hay_area",
    "total_area_irrHayRelated",
    "irr_hay_as_perc",
]


delta_data_normal = delta_data.copy()

delta_data_normal[normalize_cols] = (
    delta_data_normal[normalize_cols] - delta_data_normal[normalize_cols].mean()
) / delta_data_normal[normalize_cols].std(ddof=1)
delta_data_normal.head(3)

# %%
print(delta_data_normal.columns)

# %%
indp_vars = ["state_total_npp"] + ["rangeland_acre"]
y_var = "inventory_share"
#################################################################
curr_all = delta_data_normal[indp_vars + [y_var] + ["state_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = curr_all[y_var].astype(float)
#################################################################
model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
indp_vars = ["state_total_npp"] + ["rangeland_acre"]
y_var = "log_ratio_of_shares_Y2PrevY"
#################################################################
curr_all = delta_data_normal[indp_vars + [y_var] + ["state_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = curr_all[y_var].astype(float)
#################################################################
model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%

# %%

# %%

# %%

# %%

# %%
