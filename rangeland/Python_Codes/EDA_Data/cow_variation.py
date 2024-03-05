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

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

# %% [markdown]
# ### Directories

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

plots_dir = data_dir_base + "plots/"

# %%
abb_dict = pd.read_pickle(param_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
d = {
    "state": abb_dict["abb_2_full"].keys(),
    "full_state": abb_dict["abb_2_full"].values(),
}

# creating a Dataframe object
state_abbrFull_df = pd.DataFrame(d)
state_abbrFull_df = state_abbrFull_df[state_abbrFull_df.full_state.isin(SoI)]
state_abbrFull_df.reset_index(drop=True, inplace=True)
state_abbrFull_df.head(2)

# %% [markdown]
# #### List of county names and FIPs

# %%
county_id_name_fips = pd.read_csv(Min_data_base + "county_id_name_fips.csv")
county_id_name_fips.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

county_id_name_fips.sort_values(by=["state", "county"], inplace=True)
county_id_name_fips.rename(columns={"county": "county_fips"}, inplace=True)

county_id_name_fips = rc.correct_Mins_county_6digitFIPS(
    df=county_id_name_fips, col_="county_fips"
)
county_id_name_fips.reset_index(drop=True, inplace=True)
county_id_name_fips.head(2)


# %%

# %%
USDA_data = pd.read_pickle(reOrganized_dir + "USDA_data.sav")
print("-----------  reading USDA data  -----------")
cattle_inventory = USDA_data["cattle_inventory"]
#
# pick only 25 states we want
#
print("----------- subset to 25 states -----------")
cattle_inventory = cattle_inventory[cattle_inventory.state.isin(SoI)].copy()
cattle_inventory.sort_values(by=["year", "county_fips"], inplace=True)
print()

print(f"{cattle_inventory.data_item.unique() = }")
print(f"{cattle_inventory.commodity.unique() = }")
print()
print(f"{len(cattle_inventory.state.unique()) = }")

census_years = sorted(list(cattle_inventory.year.unique()))
print(f"{census_years = }")

# pick only useful columns
inv_col_ = "cattle_cow_beef_inventory"
cattle_inventory = cattle_inventory[["year", "county_fips", inv_col_]]

print(f"{len(cattle_inventory.county_fips.unique()) = }")
cattle_inventory.head(2)

# %% [markdown]
# ## Read ```NPP``` and ```SW```
#
# **Min has an extra "1" as leading digit in FIPS!!**

# %%
NPP = pd.read_csv(Min_data_base + "county_annual_MODIS_NPP.csv")
NPP.rename(columns={"NPP": "modis_npp"}, inplace=True)

NPP = rc.correct_Mins_county_6digitFIPS(df=NPP, col_="county")
NPP.rename(columns={"county": "county_fips"}, inplace=True)

print(f"{NPP.year.min() = }")
NPP.head(2)

# %%
filename = reOrganized_dir + "county_seasonal_temp_ppt_weighted.sav"
seasonal_weather = pd.read_pickle(filename)
print(f"{seasonal_weather.keys() = }")
seasonal_weather = seasonal_weather["seasonal"]
seasonal_weather.head(2)

# %%
seasonal_var_cols = seasonal_weather.columns[2:10]
for a_col in seasonal_var_cols:
    seasonal_weather[a_col] = seasonal_weather[a_col].astype(float)

# %%
# pick only census years
NPP = NPP[NPP.year.isin(census_years)]
NPP.reset_index(drop=True, inplace=True)
NPP.head(2)

# pick only census years
seasonal_weather = seasonal_weather[seasonal_weather.year.isin(census_years)]
seasonal_weather.reset_index(drop=True, inplace=True)
seasonal_weather.head(2)

# %%
print(f"{len(NPP.county_fips.unique()) = }")
print(f"{len(seasonal_weather.county_fips.unique()) = }")

print(f"{NPP.shape = }")
print(f"{len(NPP.county_fips.unique()) = }")
NPP = NPP[NPP.county_fips.isin(list(county_id_name_fips.county_fips.unique()))].copy()
print(f"{NPP.shape = }")
print(f"{len(NPP.county_fips.unique()) = }")
NPP.head(2)

# %%
print(f"{seasonal_weather.shape = }")
print(f"{len(seasonal_weather.county_fips.unique()) = }")
LL = list(county_id_name_fips.county_fips.unique())
seasonal_weather = seasonal_weather[seasonal_weather.county_fips.isin(LL)].copy()
print(f"{len(seasonal_weather.county_fips.unique()) = }")
print(f"{seasonal_weather.shape = }")

# %%
county_id_name_fips.head(2)

# %%
print(f"{(NPP.year.unique()) = }")
print(f"{len(NPP.county_fips.unique()) = }")
print(f"{len(cattle_inventory.county_fips.unique()) = }")

# %%
for a_year in NPP.year.unique():
    df = NPP[NPP.year == a_year]
    print(f"{len(df.county_fips.unique()) = }")

NPP.head(2)

# %% [markdown]
# ### Rangeland area

# %%
# Rangeland area and Total area:
county_RA_and_TA_fraction = pd.read_csv(
    reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv"
)
county_RA_and_TA_fraction.rename(columns={"fips_id": "county_fips"}, inplace=True)

county_RA_and_TA_fraction = rc.correct_Mins_county_6digitFIPS(
    df=county_RA_and_TA_fraction, col_="county_fips"
)
L = len(county_RA_and_TA_fraction.county_fips.unique())
print("number of counties in county_RA_and_TA_fraction are {}.".format(L))
print(county_RA_and_TA_fraction.shape)
county_RA_and_TA_fraction.head(2)

# %%
county_annual_NPP_Ra = pd.merge(
    NPP, county_RA_and_TA_fraction, on=["county_fips"], how="left"
)
county_annual_NPP_Ra.head(2)

# %%
county_annual_NPP_Ra = rc.covert_unitNPP_2_total(
    NPP_df=county_annual_NPP_Ra,
    npp_unit_col_="modis_npp",
    acr_area_col_="rangeland_acre",
    npp_area_col_="county_rangeland_npp",
)
### Security check to not make mistake later:
county_annual_NPP_Ra.drop(columns=["modis_npp"], inplace=True)
county_annual_NPP_Ra.head(2)

# %%
county_annual_SW_Ra = pd.merge(
    seasonal_weather, county_RA_and_TA_fraction, on=["county_fips"], how="left"
)
county_annual_SW_Ra.head(2)

# %%
print(f"{sorted(cattle_inventory.year.unique())     = }")
print(f"{sorted(county_annual_NPP_Ra.year.unique()) = }")
print(f"{sorted(county_annual_SW_Ra.year.unique()) = }")

# %%
common_years = (
    set(cattle_inventory.year.unique())
    .intersection(set(county_annual_NPP_Ra.year.unique()))
    .intersection(set(county_annual_SW_Ra.year.unique()))
)
common_years

# %%
cattle_inventory = cattle_inventory[cattle_inventory.year.isin(list(common_years))]
county_annual_SW_Ra = county_annual_SW_Ra[county_annual_SW_Ra.year.isin(common_years)]

# %%
print(len(cattle_inventory.county_fips.unique()))
print(len(county_annual_NPP_Ra.county_fips.unique()))
print(len(county_annual_SW_Ra.county_fips.unique()))

# %% [markdown]
# ## NPP has a lot of missing counties
#
#  - Min says he had a threshld about rangeland/pasture.
#  - subset the ```NPP``` and ```Cattle``` to the intersection of counties present.
#  - It seems there are different number of counties in each year in cattle inventory. Find intersection of those as well.

# %%
all_cattle_counties = set(cattle_inventory.county_fips.unique())
print(f"{len(all_cattle_counties) = }")

for a_year in sorted(cattle_inventory.year.unique()):
    curr_cow = cattle_inventory[cattle_inventory.year == a_year].copy()
    curr_cow_counties = set(curr_cow.county_fips.unique())
    all_cattle_counties = all_cattle_counties.intersection(curr_cow_counties)
    print(a_year)
    print(f"{len(all_cattle_counties) = }")
    print("====================================================================")

# %%
all_county_annual_NPP_Ra = set(county_annual_NPP_Ra.county_fips.unique())
print(f"{len(all_county_annual_NPP_Ra) = }")

for a_year in sorted(county_annual_NPP_Ra.year.unique()):
    curr = county_annual_NPP_Ra[county_annual_NPP_Ra.year == a_year].copy()
    curr_counties = set(curr.county_fips.unique())
    all_county_annual_NPP_Ra = all_county_annual_NPP_Ra.intersection(curr_counties)
    print(a_year)
    print(f"{len(all_county_annual_NPP_Ra) = }")
    print("====================================================================")

# %%
all_county_annual_SW_Ra = set(county_annual_SW_Ra.county_fips.unique())
print(f"{len(all_county_annual_SW_Ra) = }")

for a_year in sorted(county_annual_SW_Ra.year.unique()):
    curr = county_annual_SW_Ra[county_annual_SW_Ra.year == a_year].copy()
    curr_counties = set(curr.county_fips.unique())
    all_county_annual_SW_Ra = all_county_annual_SW_Ra.intersection(curr_counties)
    print(a_year)
    print(f"{len(all_county_annual_SW_Ra) = }")
    print("====================================================================")

# %%
# choose only the counties that are present in all years:
cattle_inventory = cattle_inventory[
    cattle_inventory.county_fips.isin(list(all_cattle_counties))
]

# %%
SW_counties = set(county_annual_SW_Ra.county_fips.unique())
NPP_counties = set(county_annual_NPP_Ra.county_fips.unique())
cow_counties = set(cattle_inventory.county_fips.unique())

county_intersection = NPP_counties.intersection(cow_counties)
county_intersection = county_intersection.intersection(SW_counties)

# %%
county_annual_SW_Ra = county_annual_SW_Ra[
    county_annual_SW_Ra.county_fips.isin(list(county_intersection))
]
county_annual_NPP_Ra = county_annual_NPP_Ra[
    county_annual_NPP_Ra.county_fips.isin(list(county_intersection))
]
cattle_inventory = cattle_inventory[
    cattle_inventory.county_fips.isin(list(county_intersection))
]

print(f"{county_annual_SW_Ra.shape = }")
print(f"{county_annual_NPP_Ra.shape = }")
print(f"{cattle_inventory.shape     = }")
print()
print(f"{len(county_annual_SW_Ra.county_fips.unique())  = }")
print(f"{len(county_annual_NPP_Ra.county_fips.unique()) = }")
print(f"{len(cattle_inventory.county_fips.unique())     = }")
print()
print(f"{sorted(county_annual_SW_Ra.year.unique())  = }")
print(f"{sorted(county_annual_NPP_Ra.year.unique()) = }")
print(f"{sorted(cattle_inventory.year.unique())     = }")

# %%
county_annual_NPP_Ra_cattleInv = pd.merge(
    county_annual_NPP_Ra, cattle_inventory, on=["county_fips", "year"], how="left"
)

print(f"{cattle_inventory.shape = }")
print(f"{county_annual_NPP_Ra.shape = }")
print(f"{county_annual_NPP_Ra_cattleInv.shape = }")
county_annual_NPP_Ra_cattleInv.head(2)

# %%
county_annual_SW_Ra_cattleInv = pd.merge(
    county_annual_SW_Ra, cattle_inventory, on=["county_fips", "year"], how="left"
)

print(f"{cattle_inventory.shape = }")
print(f"{county_annual_SW_Ra.shape = }")
print(f"{county_annual_SW_Ra_cattleInv.shape = }")
county_annual_SW_Ra_cattleInv.head(2)

# %%
county_annual_NPP_Ra_cattleInv.sort_values(by=["year", "county_fips"], inplace=True)
county_annual_NPP_Ra_cattleInv.reset_index(drop=True, inplace=True)
county_annual_NPP_Ra_cattleInv.head(2)

# %%
county_annual_SW_Ra_cattleInv.sort_values(by=["year", "county_fips"], inplace=True)
county_annual_SW_Ra_cattleInv.reset_index(drop=True, inplace=True)
county_annual_SW_Ra_cattleInv.head(2)

# %% [markdown]
# ## County-level Change

# %%
perc_change_col_ = "cow_beef_inv_change%"

# %%
columns_ = ["county_fips", "window", "cow_beef_inv_change", perc_change_col_]
num_rows = len(cattle_inventory.county_fips.unique()) * (
    len(cattle_inventory.year.unique()) - 1
)

inv_change_cnty_yr2yr = pd.DataFrame(index=range(num_rows), columns=columns_)

print(f"{inv_change_cnty_yr2yr.shape = }")

inv_change_cnty_yr2yr["county_fips"] = inv_change_cnty_yr2yr["county_fips"].astype(str)
inv_change_cnty_yr2yr["window"] = inv_change_cnty_yr2yr["window"].astype(str)
inv_change_cnty_yr2yr[perc_change_col_] = inv_change_cnty_yr2yr[
    perc_change_col_
].astype(float)
inv_change_cnty_yr2yr[perc_change_col_] = inv_change_cnty_yr2yr[
    perc_change_col_
].astype(float)

inv_change_cnty_yr2yr.head(2)

# %%
pointer = 0

for a_cnty in cattle_inventory.county_fips.unique():
    curr_df = county_annual_NPP_Ra_cattleInv[
        county_annual_NPP_Ra_cattleInv.county_fips == a_cnty
    ]
    inv_change = (
        curr_df.iloc[1:]["cattle_cow_beef_inventory"].values
        - curr_df.iloc[:-1]["cattle_cow_beef_inventory"].values
    )

    inv_change_perc = (
        inv_change * 100 / curr_df.iloc[:-1]["cattle_cow_beef_inventory"].values
    ).round(2)

    L = [x + "-" for x in list(curr_df.iloc[:-1]["year"].astype(str))]
    window_ = [i + j for i, j in zip(L, list(curr_df.iloc[1:]["year"].astype(str)))]

    inv_change_cnty_yr2yr.loc[
        pointer : pointer + len(inv_change) - 1, "county_fips"
    ] = a_cnty
    inv_change_cnty_yr2yr.loc[
        pointer : pointer + len(inv_change) - 1, "window"
    ] = window_
    inv_change_cnty_yr2yr.loc[
        pointer : pointer + len(inv_change) - 1, "cow_beef_inv_change"
    ] = inv_change
    inv_change_cnty_yr2yr.loc[
        pointer : pointer + len(inv_change) - 1, perc_change_col_
    ] = inv_change_perc

    pointer += len(inv_change)
    del (a_cnty, window_, inv_change, inv_change_perc)

inv_change_cnty_yr2yr = pd.merge(
    inv_change_cnty_yr2yr,
    county_id_name_fips[["county_fips", "state", "county_name"]],
    on=["county_fips"],
    how="left",
)

# %%
inv_change_cnty_yr2yr.head(4)

# %% [markdown]
# ### Change in all counties and all years

# %%

# %%
print(("\033[1m" + "In all counties over all years:" + "\033[0m"))
print()

s_idx = inv_change_cnty_yr2yr[perc_change_col_].idxmin()
s_ = inv_change_cnty_yr2yr.loc[s_idx, perc_change_col_]
st_cnty = (
    inv_change_cnty_yr2yr.loc[s_idx, "county_name"]
    + ", "
    + inv_change_cnty_yr2yr.loc[s_idx, "state"]
)
print("Minimum change is [{}%]: {}.".format(s_, st_cnty))
del (s_, s_idx, st_cnty)

s_idx = inv_change_cnty_yr2yr[perc_change_col_].idxmax()
s_ = inv_change_cnty_yr2yr.loc[s_idx, perc_change_col_]
st_cnty = (
    inv_change_cnty_yr2yr.loc[s_idx, "county_name"]
    + ", "
    + inv_change_cnty_yr2yr.loc[s_idx, "state"]
)
print("Maximum change is [{}%]: {}.".format(s_, st_cnty))
del (s_, s_idx, st_cnty)

s_ = inv_change_cnty_yr2yr[perc_change_col_].mean().round(2)
print("Average change is [{}%].".format(s_))
del s_

s_ = (np.float64(inv_change_cnty_yr2yr[perc_change_col_].std())).round(2)
print("SD.  of change is [{}%].".format(s_))
del s_

# %%


# %%
print(("\033[1m" + "In all counties over all years (absolute values):" + "\033[0m"))
print()

s_idx = np.abs(inv_change_cnty_yr2yr[perc_change_col_]).idxmin()
s_ = inv_change_cnty_yr2yr.loc[s_idx, perc_change_col_]
st_cnty = (
    inv_change_cnty_yr2yr.loc[s_idx, "county_name"]
    + ", "
    + inv_change_cnty_yr2yr.loc[s_idx, "state"]
)
print("Minimum change is [{}%]: {}.".format(s_, st_cnty))
del (s_, s_idx, st_cnty)

s_idx = np.abs(inv_change_cnty_yr2yr[perc_change_col_]).idxmax()
s_ = inv_change_cnty_yr2yr.loc[s_idx, perc_change_col_]
st_cnty = (
    inv_change_cnty_yr2yr.loc[s_idx, "county_name"]
    + ", "
    + inv_change_cnty_yr2yr.loc[s_idx, "state"]
)
print("Maximum change is [{}%]: {}.".format(s_, st_cnty))
del (s_, s_idx, st_cnty)

s_ = np.abs(inv_change_cnty_yr2yr[perc_change_col_]).mean().round(2)
print("Average change is [{}%].".format(s_))
del s_

# %%
for a_window in inv_change_cnty_yr2yr.window.unique():
    df_ = inv_change_cnty_yr2yr[inv_change_cnty_yr2yr.window == a_window]
    print(("\033[1m" + "In all counties in " + a_window + "\033[0m"))
    print()

    s_idx = df_[perc_change_col_].idxmin()
    s_ = df_.loc[s_idx, perc_change_col_]
    st_cnty = df_.loc[s_idx, "county_name"] + ", " + df_.loc[s_idx, "state"]
    print("Minimum change is [{}%]: {}.".format(s_, st_cnty))
    del (s_, s_idx, st_cnty)

    s_idx = df_[perc_change_col_].idxmax()
    s_ = df_.loc[s_idx, perc_change_col_]
    st_cnty = df_.loc[s_idx, "county_name"] + ", " + df_.loc[s_idx, "state"]
    print("Maximum change is [{}%]: {}.".format(s_, st_cnty))
    del (s_, s_idx, st_cnty)

    s_ = df_[perc_change_col_].mean().round(2)
    print("Average change is [{}%]".format(s_))
    del s_

    s_ = np.float64(df_[perc_change_col_].std()).round(2)
    print("SD.  of change is [{}]".format(s_))
    del s_
    print("=================================")


# %%
print(("\033[1m" + "absolute values:" + "\033[0m"))
print()

for a_window in inv_change_cnty_yr2yr.window.unique():
    df_ = inv_change_cnty_yr2yr[inv_change_cnty_yr2yr.window == a_window]
    print(("\033[1m" + "In all counties in " + a_window + "\033[0m"))
    print()

    s_idx = np.abs(df_[perc_change_col_]).idxmin()
    s_ = df_.loc[s_idx, perc_change_col_]
    st_cnty = df_.loc[s_idx, "county_name"] + ", " + df_.loc[s_idx, "state"]
    print("Minimum change is [{}%]: {}.".format(s_, st_cnty))
    del (s_, s_idx, st_cnty)

    s_idx = np.abs(df_[perc_change_col_]).idxmax()
    s_ = df_.loc[s_idx, perc_change_col_]
    st_cnty = df_.loc[s_idx, "county_name"] + ", " + df_.loc[s_idx, "state"]
    print("Maximum change is [{}%]: {}.".format(s_, st_cnty))
    del (s_, s_idx, st_cnty)

    s_ = np.abs(df_[perc_change_col_]).mean().round(2)
    print("Average change is [{}%]".format(s_))
    del s_
    print("=================================")

# %% [markdown]
# # State-Level Changes

# %%
county_annual_NPP_Ra_cattleInv.head(2)

# %%
county_id_name_fips.head(2)

# %%
county_annual_NPP_Ra_cattleInv = pd.merge(
    county_annual_NPP_Ra_cattleInv,
    county_id_name_fips[["county_fips", "state"]],
    on=["county_fips"],
    how="left",
)
county_annual_NPP_Ra_cattleInv.head(2)

# %%
state_annual_cattleInv = county_annual_NPP_Ra_cattleInv[
    ["year", "cattle_cow_beef_inventory", "state"]
].copy()

state_annual_cattleInv = (
    state_annual_cattleInv.groupby(["year", "state"])["cattle_cow_beef_inventory"]
    .sum()
    .reset_index()
)

state_annual_cattleInv = pd.merge(
    state_annual_cattleInv, state_abbrFull_df, on=["state"], how="left"
)
state_annual_cattleInv.head(2)

# %%
columns_ = ["state", "window", "cow_beef_inv_change", perc_change_col_]
num_rows = len(state_annual_cattleInv.state.unique()) * (
    len(state_annual_cattleInv.year.unique()) - 1
)

inv_change_state_yr2yr = pd.DataFrame(index=range(num_rows), columns=columns_)

print(f"{inv_change_state_yr2yr.shape = }")

inv_change_state_yr2yr["state"] = inv_change_state_yr2yr["state"].astype(str)
inv_change_state_yr2yr["window"] = inv_change_state_yr2yr["window"].astype(str)
inv_change_state_yr2yr["cow_beef_inv_change"] = inv_change_state_yr2yr[
    "cow_beef_inv_change"
].astype(float)
inv_change_state_yr2yr[perc_change_col_] = inv_change_state_yr2yr[
    perc_change_col_
].astype(float)

inv_change_state_yr2yr.head(2)

# %%
pointer = 0

for a_state in state_annual_cattleInv.state.unique():
    curr_df = state_annual_cattleInv[state_annual_cattleInv.state == a_state]
    inv_change = (
        curr_df.iloc[1:]["cattle_cow_beef_inventory"].values
        - curr_df.iloc[:-1]["cattle_cow_beef_inventory"].values
    )

    inv_change_perc = (
        inv_change * 100 / curr_df.iloc[:-1]["cattle_cow_beef_inventory"].values
    ).round(2)

    L = [x + "-" for x in list(curr_df.iloc[:-1]["year"].astype(str))]
    window_ = [i + j for i, j in zip(L, list(curr_df.iloc[1:]["year"].astype(str)))]

    inv_change_state_yr2yr.loc[
        pointer : pointer + len(inv_change) - 1, "state"
    ] = a_state
    inv_change_state_yr2yr.loc[
        pointer : pointer + len(inv_change) - 1, "window"
    ] = window_
    inv_change_state_yr2yr.loc[
        pointer : pointer + len(inv_change) - 1, perc_change_col_
    ] = inv_change
    inv_change_state_yr2yr.loc[
        pointer : pointer + len(inv_change) - 1, perc_change_col_
    ] = inv_change_perc

    pointer += len(inv_change)
    del (a_state, window_, inv_change, inv_change_perc)

inv_change_state_yr2yr = pd.merge(
    inv_change_state_yr2yr, state_abbrFull_df, on=["state"], how="left"
)

inv_change_state_yr2yr.head(4)

# %%
print(("\033[1m" + "Among all states and over all years:" + "\033[0m"))
print()


s_idx = inv_change_state_yr2yr[perc_change_col_].idxmin()
s_ = inv_change_state_yr2yr.loc[s_idx, perc_change_col_]
st_cnty = inv_change_state_yr2yr.loc[s_idx, "full_state"]
print("Minimum change is [{}%]: {}.".format(s_, st_cnty))
del (s_, s_idx, st_cnty)

s_idx = inv_change_state_yr2yr[perc_change_col_].idxmax()
s_ = inv_change_state_yr2yr.loc[s_idx, perc_change_col_]
st_cnty = inv_change_state_yr2yr.loc[s_idx, "full_state"]
print("Maximum change is [{}%]: {}.".format(s_, st_cnty))
del (s_, s_idx, st_cnty)

s_ = inv_change_state_yr2yr[perc_change_col_].mean().round(2)
print("Average change is [{}%].".format(s_))
del s_

s_ = np.float64(inv_change_state_yr2yr[perc_change_col_].std()).round(2)
print("SD.  of change is [{}%].".format(s_))
del s_

# %%
print(("\033[1m" + "Among all states and over all years (Absolue Values):" + "\033[0m"))
print()


s_idx = np.abs(inv_change_state_yr2yr[perc_change_col_]).idxmin()
s_ = inv_change_state_yr2yr.loc[s_idx, perc_change_col_]
st_cnty = inv_change_state_yr2yr.loc[s_idx, "full_state"]
print("Minimum change is [{}%]: {}.".format(s_, st_cnty))
del (s_, s_idx, st_cnty)

s_idx = np.abs(inv_change_state_yr2yr[perc_change_col_]).idxmax()
s_ = inv_change_state_yr2yr.loc[s_idx, perc_change_col_]
st_cnty = inv_change_state_yr2yr.loc[s_idx, "full_state"]
print("Maximum change is [{}%]: {}.".format(s_, st_cnty))
del (s_, s_idx, st_cnty)

s_ = np.abs(inv_change_state_yr2yr[perc_change_col_]).mean().round(2)
print("Average change is [{}%].".format(s_))
del s_

# %%
for a_window in inv_change_state_yr2yr.window.unique():
    df_ = inv_change_state_yr2yr[inv_change_state_yr2yr.window == a_window]
    print(("\033[1m" + "Among all states in " + a_window + "\033[0m"))
    print()

    s_idx = df_[perc_change_col_].idxmin()
    s_ = df_.loc[s_idx, perc_change_col_]
    st_cnty = df_.loc[s_idx, "full_state"]
    print("Minimum change is [{}%]: {}.".format(s_, st_cnty))
    del (s_, s_idx, st_cnty)

    s_idx = df_[perc_change_col_].idxmax()
    s_ = df_.loc[s_idx, perc_change_col_]
    st_cnty = df_.loc[s_idx, "full_state"]
    print("Maximum change is [{}%]: {}.".format(s_, st_cnty))
    del (s_, s_idx, st_cnty)

    s_ = df_[perc_change_col_].mean().round(2)
    print("Average change is [{}%]".format(s_))
    del s_

    s_ = np.float64(df_[perc_change_col_].std()).round(2)
    print("SD.  of change is [{}]".format(s_))
    del s_
    print("=================================")

# %%
print(("\033[1m" + "Absolue Values" + "\033[0m"))
print()

for a_window in inv_change_state_yr2yr.window.unique():
    df_ = inv_change_state_yr2yr[inv_change_state_yr2yr.window == a_window]
    print(("\033[1m" + "Among all states in " + a_window + "\033[0m"))
    print()

    s_idx = np.abs(df_[perc_change_col_]).idxmin()
    s_ = df_.loc[s_idx, perc_change_col_]
    st_cnty = df_.loc[s_idx, "full_state"]
    print("Minimum change is [{}%]: {}.".format(s_, st_cnty))
    del (s_, s_idx, st_cnty)

    s_idx = np.abs(df_[perc_change_col_]).idxmax()
    s_ = df_.loc[s_idx, perc_change_col_]
    st_cnty = df_.loc[s_idx, "full_state"]
    print("Maximum change is [{}%]: {}.".format(s_, st_cnty))
    del (s_, s_idx, st_cnty)

    s_ = np.abs(df_[perc_change_col_]).mean().round(2)
    print("Average change is [{}%]".format(s_))
    del s_
    print("=================================")

# %% [markdown]
# ### Shannon data is annual and state-level

# %%
shannon_annual = pd.read_csv(reOrganized_dir + "Shannon_Beef_Cows_fromCATINV.csv")
shannon_annual = shannon_annual[shannon_annual.state.isin(SoI_abb)]
shannon_annual.head(2)

# %%
# shannon_annual.columns.values[1:].astype(int)
wanted_years = np.arange(1979, 2022)
print(wanted_years[:3])
print(wanted_years[-3:])
cols_ = ["state"] + list(wanted_years.astype(str))
shannon_annual = shannon_annual[cols_]
shannon_annual.head(2)

# %%

# %%
window_ = [x.astype(str) + "-" + (x + 1).astype(str) for x in wanted_years[:-1]]
print(window_[:5])
print(window_[-5:])

# %%
columns_ = ["state", "window", "cow_beef_inv_change", perc_change_col_]
num_rows = len(shannon_annual.state.unique()) * (len(window_))

inv_change_state_yr2yr = pd.DataFrame(index=range(num_rows), columns=columns_)

print(f"{inv_change_state_yr2yr.shape = }")

inv_change_state_yr2yr["state"] = inv_change_state_yr2yr["state"].astype(str)
inv_change_state_yr2yr["window"] = inv_change_state_yr2yr["window"].astype(str)
inv_change_state_yr2yr["cow_beef_inv_change"] = inv_change_state_yr2yr[
    "cow_beef_inv_change"
].astype(float)
inv_change_state_yr2yr[perc_change_col_] = inv_change_state_yr2yr[
    perc_change_col_
].astype(float)

inv_change_state_yr2yr.head(2)

# %%
pointer = 0

for a_state in shannon_annual.state.unique():
    curr_df = shannon_annual[shannon_annual.state == a_state]
    inv_change = (
        curr_df[list(np.arange(1980, 2022).astype(str))].values
        - curr_df[list(np.arange(1979, 2021).astype(str))].values
    )
    inv_change = inv_change[0]

    inv_change_perc = (
        inv_change * 100 / curr_df[list(np.arange(1979, 2021).astype(str))].values
    ).round(2)
    inv_change_perc = inv_change_perc[0]

    inv_change_state_yr2yr.loc[
        pointer : pointer + len(inv_change) - 1, "state"
    ] = a_state
    inv_change_state_yr2yr.loc[
        pointer : pointer + len(inv_change) - 1, "window"
    ] = window_
    inv_change_state_yr2yr.loc[
        pointer : pointer + len(inv_change) - 1, "cow_beef_inv_change"
    ] = inv_change
    inv_change_state_yr2yr.loc[
        pointer : pointer + len(inv_change) - 1, perc_change_col_
    ] = inv_change_perc

    pointer += len(inv_change)
    del (a_state, inv_change, inv_change_perc)

inv_change_state_yr2yr = pd.merge(
    inv_change_state_yr2yr, state_abbrFull_df, on=["state"], how="left"
)

inv_change_state_yr2yr.head(4)

# %%
for a_state in inv_change_state_yr2yr.full_state.unique():
    df_ = inv_change_state_yr2yr[inv_change_state_yr2yr.full_state == a_state]
    print(("\033[1m" + a_state + "(1979-2021)" + "\033[0m"))
    print()
    s_idx = df_[perc_change_col_].idxmin()
    s_ = df_.loc[s_idx, perc_change_col_]
    st_cnty = df_.loc[s_idx, "full_state"]
    print("Minimum change is [{}%].".format(s_))
    del (s_, s_idx, st_cnty)

    s_idx = df_[perc_change_col_].idxmax()
    s_ = df_.loc[s_idx, perc_change_col_]
    st_cnty = df_.loc[s_idx, "full_state"]
    print("Maximum change is [{}%].".format(s_))
    del (s_, s_idx, st_cnty)

    s_ = df_[perc_change_col_].mean().round(2)
    print("Average change is [{}%]".format(s_))
    del s_

    s_ = np.float64(df_[perc_change_col_].std()).round(2)
    print("SD.  of change is [{}]".format(s_))
    del s_
    print("=================================")

# %%
print(("\033[1m" + "Absolue Values" + "\033[0m"))
print()

for a_state in inv_change_state_yr2yr.full_state.unique():
    df_ = inv_change_state_yr2yr[inv_change_state_yr2yr.full_state == a_state]
    print(("\033[1m" + a_state + "(1979-2021)" + "\033[0m"))
    print()
    s_ = np.abs(df_[perc_change_col_]).min()
    s_idx = np.abs(df_[perc_change_col_]).idxmin()
    st_cnty = df_.loc[s_idx, "full_state"]
    print("Minimum change is [{}%].".format(s_))
    del (s_, s_idx, st_cnty)

    s_ = np.abs(df_[perc_change_col_]).max()
    s_idx = np.abs(df_[perc_change_col_]).idxmax()
    st_cnty = df_.loc[s_idx, "full_state"]
    print("Maximum change is [{}%].".format(s_))
    del (s_, s_idx, st_cnty)

    s_ = np.abs(df_[perc_change_col_]).mean().round(2)
    print("Average change is [{}%]".format(s_))
    del s_
    print("=================================")

# %%
print(("\033[1m" + "Absolue Values" + "\033[0m"))
print()

for a_state in inv_change_state_yr2yr.full_state.unique():
    df_ = inv_change_state_yr2yr[inv_change_state_yr2yr.full_state == a_state]
    print(("\033[1m" + a_state + "(1979-2021)" + "\033[0m"))
    print()
    s_ = np.abs(df_[perc_change_col_]).min()
    s_idx = np.abs(df_[perc_change_col_]).idxmin()
    st_cnty = df_.loc[s_idx, "full_state"]
    print("Minimum change is [{}%].".format(s_))
    del (s_, s_idx, st_cnty)

    s_ = np.abs(df_[perc_change_col_]).max()
    s_idx = np.abs(df_[perc_change_col_]).idxmax()
    st_cnty = df_.loc[s_idx, "full_state"]
    print("Maximum change is [{}%].".format(s_))
    del (s_, s_idx, st_cnty)

    s_ = np.abs(df_[perc_change_col_]).mean().round(2)
    print("Average change is [{}%]".format(s_))
    del s_
    print("=================================")

# %%
# sharey='col', # sharex=True, sharey=True,
import itertools

num_sub_plots = len(inv_change_state_yr2yr.state.unique())
fig, axs = plt.subplots(
    num_sub_plots,
    1,
    figsize=(10, 3 * num_sub_plots),
    sharex=True,
    gridspec_kw={"hspace": 0.15, "wspace": 0.05},
)

# state_, full_state_ = "ID", "Idaho"
for count, state_ in enumerate(inv_change_state_yr2yr.state.unique()):
    axs[count].grid(axis="y", which="both")
    B = inv_change_state_yr2yr[inv_change_state_yr2yr.state == state_]
    B.set_index("window", inplace=True)
    B.sort_index(inplace=True)
    axs[count].plot(
        B.index, B[perc_change_col_].values, c="dodgerblue", linewidth=2, label=state_
    )

    # add every other ticks
    odd_i = itertools.islice(B.index, 0, None, 2)
    odd_i = list(itertools.chain(odd_i))
    # axs[ii].set_xticks(odd_i, odd_i, rotation ='vertical');
    axs[count].set_ylabel("% change")
    axs[count].legend(loc="best")

plt.tight_layout()
plt.xticks(odd_i, odd_i, rotation="vertical")

fig_name = plots_dir + "shannon_annual_inventory_change.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")
plt.close("all")

# %%
shannon_annual.head(2)

# %%
# sharey='col', # sharex=True, sharey=True,
import itertools

num_sub_plots = len(inv_change_state_yr2yr.state.unique())
fig, axs = plt.subplots(
    num_sub_plots,
    1,
    figsize=(10, 3 * num_sub_plots),
    sharex=True,
    gridspec_kw={"hspace": 0.15, "wspace": 0.05},
)

# state_, full_state_ = "ID", "Idaho"
for count, state_ in enumerate(shannon_annual.state.unique()):
    axs[count].grid(axis="y", which="both")
    B = shannon_annual[shannon_annual.state == state_]
    axs[count].plot(
        B.columns[1:], B.values[0][1:], c="dodgerblue", linewidth=2, label=state_
    )

    # add every other ticks
    odd_i = itertools.islice(B.columns[1:], 0, None, 2)
    odd_i = list(itertools.chain(odd_i))
    # axs[ii].set_xticks(odd_i, odd_i, rotation ='vertical');
    axs[count].set_ylabel("inventory")
    axs[count].legend(loc="best")

plt.tight_layout()
plt.xticks(odd_i, odd_i, rotation="vertical")

fig_name = plots_dir + "shannon_annual_inventory.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")
plt.close("all")

# %%
inv_change_state_yr2yr.head(2)

# %%
inv_change_state_yr2yr.window.unique()

# %%

# %%

# %%

# %%
