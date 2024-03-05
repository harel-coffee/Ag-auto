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
# I want to have a direct stab at forming the datatables so they are usable for training the model.
# Let's just do it.

# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

import matplotlib
import matplotlib.pyplot as plt

sys.path.append("/Users/hn/Documents/00_GitHub/Rangeland/Python_Codes/")
import rangeland_core as rc

# import geopandas
# A = geopandas.read_file("/Users/hn/Desktop/amin/shapefile.shp")
# A.head(2)

# %% [markdown]
# According to Google Map the coordinate ```30.59375N 88.40625W``` is in Alabama. According to Bhupi's file it
# is in Alabama by state but in Mississipi by ```fips```. That place is on the border of the two states. I will go by fips.

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
# Min_data_dir_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"
seasonal_dir = reOrganized_dir + "My_seasonal_variables_Trash/02_merged_mean_over_county/"

# %%
# file_name = "countyMean_seasonalVars_wFips.csv"
# countyMean_seasonalVars = pd.read_csv(seasonal_dir + file_name)
# print (f"{len(countyMean_seasonalVars.state.unique())=}")

# # round numbers
#  countyMean_seasonalVars = countyMean_seasonalVars.round(decimals=2)
# countyMean_seasonalVars.head(2)

# %%
# file_name = "countyMean_seasonalVars_wFips.sav"
# countyMean_seasonalVars = pickle.load(open(seasonal_dir + file_name, "rb"))
# del(file_name)
# countyMean_seasonalVars = countyMean_seasonalVars["countyMean_seasonalVars_wFips"]
# countyMean_seasonalVars.head(2)

# %%
# Min_FIPS = pd.read_csv("/Users/hn/Documents/01_research_data/Sid/Analog/parameters/Min_counties.csv")
# Min_FIPS = Min_FIPS[["state", "county", "fips"]]
# Min_FIPS.drop_duplicates(inplace=True)
# Min_FIPS.reset_index(drop=True, inplace=True)
# Min_FIPS.head(2)

# %%
# Min_FIPS[Min_FIPS.state == "AL"].sort_values(by=['county'])

# %%
Bhupi = pd.read_csv(param_dir + "Bhupi_25states_clean.csv")
Bhupi["SC"] = Bhupi.state + "-" + Bhupi.county
Bhupi.head(2)

# %%

# %%
# cntyMean_seasonVars_wide = pickle.load(open(seasonal_dir + "wide_seasonal_vars_cntyMean_wFips.sav", "rb"))

# pandas >= 2.0.0
cntyMean_seasonVars_wide = pd.read_pickle(seasonal_dir + "wide_seasonal_vars_cntyMean_wFips.sav")
cntyMean_seasonVars_wide = cntyMean_seasonVars_wide["wide_seasonal_vars_cntyMean_wFips"]

cntyMean_seasonVars_wide.sort_values(by=["state", "county", "year"], inplace=True)
cntyMean_seasonVars_wide.reset_index(drop=True, inplace=True)
cntyMean_seasonVars_wide = cntyMean_seasonVars_wide.round(decimals=2)
cntyMean_seasonVars_wide.head(2)

# %%
len(cntyMean_seasonVars_wide.county_fips.unique())

# %%
USDA_data = pd.read_pickle(reOrganized_dir + "USDA_data.sav")
print(USDA_data.keys())
AgLand = USDA_data["AgLand"]
feed_expense = USDA_data["feed_expense"]
CRPwetLand_area = USDA_data["wetLand_area"]
cattle_inventory = USDA_data["cattle_inventory"]
# FarmOperation = USDA_data["FarmOperation"] # not needed. create by NASS guy.

# %%
abb_dict = pd.read_pickle(param_dir + "county_fips.sav")
SoI = abb_dict['SoI']
SoI_abb = []
for x in SoI:
    SoI_abb = SoI_abb + [abb_dict["full_2_abb"][x]]

# %%
AgLand = AgLand[AgLand.state.isin(SoI)].copy()
feed_expense = feed_expense[feed_expense.state.isin(SoI)].copy()
CRPwetLand_area = CRPwetLand_area[CRPwetLand_area.state.isin(SoI)].copy()
cattle_inventory = cattle_inventory[cattle_inventory.state.isin(SoI)].copy()

# %%

# %%
# feed_expense.rename(
#     columns={"value": "feed_expense", "cv_(%)": "feed_expense_cv_(%)"}, inplace=True
# )

# CRPwetLand_area.rename(
#     columns={"value": "CRP_wetLand_acr", "cv_(%)": "CRP_wetLand_acr_cv_(%)"},
#     inplace=True,
# )

# cattle_inventory.rename(
#     columns={"value": "cattle_cow_inventory", "cv_(%)": "cattle_cow_inventory_cv_(%)"},
#     inplace=True,
# )

# %%
print (len(AgLand.state.unique()))
print (len(AgLand.county_fips.unique()))

# %%
CRPwetLand_area.head(2)

# %%
print(f"{AgLand.shape = }")
AgLand = rc.clean_census(df=AgLand, col_="value")
print(f"{AgLand.shape = }")
print()

print(f"{feed_expense.shape = }")
feed_expense = rc.clean_census(df=feed_expense, col_="feed_expense")
print(f"{feed_expense.shape = }")
print()

print(f"{CRPwetLand_area.shape = }")
CRPwetLand_area = rc.clean_census(df=CRPwetLand_area, col_="CRP_wetLand_acr")
print(f"{CRPwetLand_area.shape = }")
print()

print(f"{cattle_inventory.shape = }")
cattle_inventory = rc.clean_census(df=cattle_inventory, col_="cattle_cow_inventory")
print(f"{cattle_inventory.shape = }")

# %%
print (len(AgLand.state.unique()))
print (len(AgLand.county_fips.unique()))

# %% [markdown]
# ### Compute irrigated percentages.
#
# **Not all counties have both irrigated area and farm operation area, apparently.**
# So, we cannot have percentages for all counties in there.

# %%
AgLand.rename(columns={"value": "area"}, inplace=True)

AgLand_FarmOper = AgLand[AgLand.data_item == "FARM OPERATIONS - ACRES OPERATED"].copy()
AgLand_FarmOper.reset_index(drop=True, inplace=True)

AgLand_irrigated = AgLand[AgLand.data_item == "AG LAND, IRRIGATED - ACRES"].copy()
AgLand_irrigated.reset_index(drop=True, inplace=True)
AgLand_irrigated.head(2)

print(f"{len(AgLand_irrigated.county_fips.unique()) = }")
print(f"{len(AgLand_FarmOper.county_fips.unique()) = }")

AgLand_irrigated.head(2)

# %%
AgLand_irrigated_Fips_notInAgLand_FarmOper = [
    x
    for x in AgLand_irrigated.county_fips.unique()
    if not (x in AgLand_FarmOper.county_fips.unique())
]

AgLand_irrigated_Fips_notInAgLand_FarmOper

# %%
AgLand[AgLand.county_fips == "32021"]

# %%
AgLand_FarmOper_Fips_notInAgLand_irrigated = [
    x
    for x in AgLand_FarmOper.county_fips.unique()
    if not (x in AgLand_irrigated.county_fips.unique())
]

print(AgLand_FarmOper_Fips_notInAgLand_irrigated)

# %%
AgLand[AgLand.county_fips == "13239"]

# %%
# I do not know why I have "irrigated.columns" below.
# GitHub said CRPwetLand. So, I fixed it above.
if "area" in AgLand_irrigated.columns:
    AgLand_irrigated.rename(
        columns={"area": "irrigated_area", "cv_(%)": "irrigated_area_cv_(%)"},
        inplace=True,
    )

print(f"{AgLand_irrigated.data_item.unique() = }")
AgLand_irrigated.head(2)

# %%
AgLand_FarmOper.head(2)

# %%
irr_areas_perc_df = pd.merge(
    AgLand_irrigated[["state", "county", "year", "county_fips", "irrigated_area"]],
    AgLand_FarmOper[["state", "county", "year", "county_fips", "area"]],
    on=["state", "county", "year", "county_fips"],
    how="left",
)
irr_areas_perc_df["irr_as_perc"] = (irr_areas_perc_df.irrigated_area / irr_areas_perc_df.area) * 100
irr_areas_perc_df.iloc[:, 1:] = irr_areas_perc_df.iloc[:, 1:].round(2)
irr_areas_perc_df.head(5)

# %%
print(f"{len(AgLand_irrigated.county_fips.unique()) = }")
print(f"{len(AgLand_FarmOper.county_fips.unique()) = }")
print(f"{len(irr_areas_perc_df.county_fips.unique()) = }")

# %%
irr_areas_perc_df[irr_areas_perc_df.county_fips == "32021"]

# %%
print(irr_areas_perc_df.shape)
irr_areas_perc_df.dropna(subset=["irr_as_perc"], inplace=True)
print(irr_areas_perc_df.shape)

# %%
cattle_inventory.head(2)

# %%
feed_expense[(feed_expense.state == "Alabama") & (feed_expense.county == "Baldwin")]

# %%
feed_expense[(feed_expense.state == "Alabama") & (feed_expense.county == "Washington")]

# %%
print(f"{len(feed_expense.state.unique())=}")
feed_expense.data_item.unique()

# %%
feed_expense.head(2)

# %%
feed_expense = feed_expense[feed_expense.state.isin(SoI)].copy()

print(feed_expense.shape)
print(len(feed_expense.state.unique()))
print(len(feed_expense.county.unique()))
print(len(feed_expense.year.unique()))

feed_expense.head(2)

# %%
print(cattle_inventory.shape)
cattle_inventory = cattle_inventory[cattle_inventory.state.isin(SoI)].copy()

print(cattle_inventory.shape)
print(len(cattle_inventory.state.unique()))
print(len(cattle_inventory.county.unique()))
print(len(cattle_inventory.year.unique()))

# %% [markdown]
# ### List all counties and years we want so we can fill the damn gaps

# %%
season_counties = cntyMean_seasonVars_wide.county_fips.unique()
print(f"{len(season_counties) = }")


# %%
### Subset seasonal vars to every 5 years that is in the USDA NASS

USDA_years = list(feed_expense.year.unique())
seasonal_5yearLapse = cntyMean_seasonVars_wide[
    cntyMean_seasonVars_wide.year.isin(USDA_years)
].copy()
seasonal_5yearLapse.reset_index(drop=True, inplace=True)
seasonal_5yearLapse.head(2)

# %%
irr_areas_perc_df.head(2)

# %% [markdown]
# ### Fill the gaps
#
# - First plan:  find counties for which there is less than 4 instances (2017, 2012, 2007, 2002, 1997).
# - Second plan: Mike said if a county has a missing value forget about it.

# %%
a_cnty = irr_areas_perc_df.county_fips.unique()[0]

A = irr_areas_perc_df[irr_areas_perc_df.county_fips == a_cnty]
aa = A.year.unique()
missin_yrs = [x for x in USDA_years if x not in aa]

aa

# %%
print(f"{len(USDA_years) = }")
len(aa) < (len(USDA_years))

# %%
#
# if min_yrs_needed is len(USDA_years) then we will have
# all the data for all years must be fully present
# if min_yrs_needed is (len(USDA_years) - 1) then we allow one year
# of missing data and we need to fill it with interpolation.
#
min_yrs_needed = len(USDA_years)

# %%
irr_areas_perc_cnty_toss = {}

for a_cnty in irr_areas_perc_df.county_fips.unique():
    A = irr_areas_perc_df[irr_areas_perc_df.county_fips == a_cnty]
    aa = A.year.unique()
    missin_yrs = [x for x in USDA_years if x not in aa]
    if len(aa) < (min_yrs_needed):
        irr_areas_perc_cnty_toss[a_cnty] = missin_yrs

print(len(irr_areas_perc_cnty_toss))
# irr_areas_perc_cnty_toss

# %%
feed_expense_cnty_toss = {}

for a_cnty in feed_expense.county_fips.unique():
    A = feed_expense[feed_expense.county_fips == a_cnty]
    aa = A.year.unique()
    missin_yrs = [x for x in USDA_years if x not in aa]
    if len(aa) < (min_yrs_needed):
        feed_expense_cnty_toss[a_cnty] = missin_yrs

print(len(feed_expense_cnty_toss))
# feed_expense_cnty_toss

# %%
CRPwetLand_area_cnty_toss = {}

for a_cnty in CRPwetLand_area.county_fips.unique():
    A = CRPwetLand_area[CRPwetLand_area.county_fips == a_cnty]
    aa = A.year.unique()
    missin_yrs = [x for x in USDA_years if x not in aa]
    if len(aa) < (min_yrs_needed):
        CRPwetLand_area_cnty_toss[a_cnty] = missin_yrs

print(f"{len(CRPwetLand_area_cnty_toss) = }")
# CRPwetLand_area_cnty_toss

# %%
cattle_inventory_cnty_toss = {}

for a_cnty in cattle_inventory.county_fips.unique():
    A = cattle_inventory[cattle_inventory.county_fips == a_cnty]
    aa = A.year.unique()
    missin_yrs = [x for x in USDA_years if x not in aa]
    if len(aa) < (min_yrs_needed):
        cattle_inventory_cnty_toss[a_cnty] = missin_yrs

print(f"{len(cattle_inventory_cnty_toss) = }")
# cattle_inventory_cnty_toss

# %%

# %%
cntyMean_seasonVars_wide.head(2)

# %%
seasonal_5yearLapse.head(2)

# %% [markdown]
# ### Toss all the counties for which there is not enough data

# %%
CRPwetLand_area.head(2)

# %%
# check if there are missing values
# for a_year in USDA_years:
#     for season_counties


CRPwetLand_area.data_item.unique()

# %% [markdown]
# # Do the analysis with and without CRPwetLand area
# since there are too many missing values.
#
# and ignore all the counties for which we do not have full cattle inventory data.

# %%
feed_expense_noCRPwetLand = feed_expense.copy()
cattle_inventory_noCRPwetLand = cattle_inventory.copy()
irr_areas_perc_df_noCRPwetLand = irr_areas_perc_df.copy()
seasonal_5yearLapse_noCRPwetLand = seasonal_5yearLapse.copy()

# %%
# irr_areas_perc_df

# %%
# set(CRPwetLand_area_cnty_toss.keys())
# set(cattle_inventory_cnty_toss.keys())
# set(irr_areas_perc_cnty_toss.keys())
# set(feed_expense_cnty_toss.keys())

# %%
toss_counties = (
    set(feed_expense_cnty_toss.keys())
    .union(set(irr_areas_perc_cnty_toss.keys()))
    .union(set(cattle_inventory_cnty_toss.keys()))
    .union(set(CRPwetLand_area_cnty_toss.keys()))
)

# %%
irr_areas_perc_df = irr_areas_perc_df[~irr_areas_perc_df.county_fips.isin(toss_counties)]
CRPwetLand_area = CRPwetLand_area[~CRPwetLand_area.county_fips.isin(toss_counties)]
feed_expense = feed_expense[~feed_expense.county_fips.isin(toss_counties)]
cattle_inventory = cattle_inventory[~cattle_inventory.county_fips.isin(toss_counties)]
seasonal_5yearLapse = seasonal_5yearLapse[~seasonal_5yearLapse.county_fips.isin(toss_counties)]

# %% [markdown]
# ### now find all counties that they share.

# %%
common_counties = (
    set(irr_areas_perc_df.county_fips.unique())
    .intersection(set(CRPwetLand_area.county_fips.unique()))
    .intersection(set(feed_expense.county_fips.unique()))
    .intersection(set(cattle_inventory.county_fips.unique()))
    .intersection(set(seasonal_5yearLapse.county_fips.unique()))
)

len(common_counties)

# %%
irr_areas_perc_df = irr_areas_perc_df[irr_areas_perc_df.county_fips.isin(common_counties)]
CRPwetLand_area = CRPwetLand_area[CRPwetLand_area.county_fips.isin(common_counties)]
feed_expense = feed_expense[feed_expense.county_fips.isin(common_counties)]
cattle_inventory = cattle_inventory[cattle_inventory.county_fips.isin(common_counties)]
seasonal_5yearLapse = seasonal_5yearLapse[seasonal_5yearLapse.county_fips.isin(common_counties)]

# %%
print(f"{len(irr_areas_perc_df.county_fips.unique()) = }")
print(f"{len(CRPwetLand_area.county_fips.unique()) = }")
print(f"{len(feed_expense.county_fips.unique()) = }")
print(f"{len(cattle_inventory.county_fips.unique()) = }")
print(f"{len(seasonal_5yearLapse.county_fips.unique()) = }")

# %% [markdown]
# #### Exclude CRPwetLand

# %%
toss_counties_excludeCRPwetLand = (
    set(feed_expense_cnty_toss.keys())
    .union(set(irr_areas_perc_cnty_toss.keys()))
    .union(set(cattle_inventory_cnty_toss.keys()))
)

# %%
irr_areas_perc_df_noCRPwetLand = irr_areas_perc_df_noCRPwetLand[
    ~irr_areas_perc_df_noCRPwetLand.county_fips.isin(toss_counties_excludeCRPwetLand)
]

feed_expense_noCRPwetLand = feed_expense_noCRPwetLand[
    ~feed_expense_noCRPwetLand.county_fips.isin(toss_counties_excludeCRPwetLand)
]

cattle_inventory_noCRPwetLand = cattle_inventory_noCRPwetLand[
    ~cattle_inventory_noCRPwetLand.county_fips.isin(toss_counties_excludeCRPwetLand)
]

seasonal_5yearLapse_noCRPwetLand = seasonal_5yearLapse_noCRPwetLand[
    ~seasonal_5yearLapse_noCRPwetLand.county_fips.isin(toss_counties_excludeCRPwetLand)
]

# %%
common_counties_noCRPwetLand = (
    set(irr_areas_perc_df_noCRPwetLand.county_fips.unique())
    .intersection(set(feed_expense_noCRPwetLand.county_fips.unique()))
    .intersection(set(cattle_inventory_noCRPwetLand.county_fips.unique()))
    .intersection(set(seasonal_5yearLapse_noCRPwetLand.county_fips.unique()))
)

len(common_counties_noCRPwetLand)

# %%
irr_areas_perc_df_noCRPwetLand = irr_areas_perc_df_noCRPwetLand[
    irr_areas_perc_df_noCRPwetLand.county_fips.isin(common_counties_noCRPwetLand)
]

feed_expense_noCRPwetLand = feed_expense_noCRPwetLand[
    feed_expense_noCRPwetLand.county_fips.isin(common_counties_noCRPwetLand)
]

cattle_inventory_noCRPwetLand = cattle_inventory_noCRPwetLand[
    cattle_inventory_noCRPwetLand.county_fips.isin(common_counties_noCRPwetLand)
]


seasonal_5yearLapse_noCRPwetLand = seasonal_5yearLapse_noCRPwetLand[
    seasonal_5yearLapse_noCRPwetLand.county_fips.isin(common_counties_noCRPwetLand)
]

# %%
print(f"{len(irr_areas_perc_df_noCRPwetLand.county_fips.unique()) = }")
print(f"{len(feed_expense_noCRPwetLand.county_fips.unique()) = }")
print(f"{len(cattle_inventory_noCRPwetLand.county_fips.unique()) = }")
print(f"{len(seasonal_5yearLapse_noCRPwetLand.county_fips.unique()) = }")

# %% [markdown]
# ## Fill the gaps

# %%
print(f"{min_yrs_needed = }")
cattle_inventory_noCRPwetLand.head(2)

# %%
A = cattle_inventory_noCRPwetLand[["state", "year", "cattle_cow_inventory"]]
    .groupby(["year", "state"]).sum().reset_index()

A.year = pd.to_datetime(A.year, format="%Y")
A.head(2)

# %%
A.set_index("year", inplace=True)
A.sort_values("cattle_cow_inventory", inplace=True)
A.head(2)

# %%
A[A.state.isin(A.state.unique()[:10])].groupby("state")["cattle_cow_inventory"].plot(legend=True);

# %%
A[A.state.isin(A.state.unique()[10:20])].groupby("state")["cattle_cow_inventory"].plot(legend=True);

# %%
# sharey='col', # sharex=True, sharey=True,
# fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, gridspec_kw={'hspace': 0.35, 'wspace': .05});
A[A.state.isin(A.state.unique()[20:])].groupby("state")["cattle_cow_inventory"].plot(legend=True);

# %%
A[A.state == "Texas"]

# %%
Beef_Cows_fromCATINV = pd.read_csv(reOrganized_dir + "Shannon_Beef_Cows_fromCATINV.csv")
Beef_Cows_fromCATINV.head(2)

# %%
abb_dict = pd.read_pickle(param_dir + "county_fips.sav")
state_25_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
Beef_Cows_fromCATINV = Beef_Cows_fromCATINV[Beef_Cows_fromCATINV.state.isin(state_25_abb)]
Beef_Cows_fromCATINV.reset_index(drop=True, inplace=True)
Beef_Cows_fromCATINV.head(2)

# %%
shannon_years = [str(x) for x in np.arange(1997, 2018)]
cols = ["state"] + shannon_years
Beef_Cows_fromCATINV = Beef_Cows_fromCATINV[cols]
Beef_Cows_fromCATINV.head(2)

# %%
fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=False, # sharey='col', # sharex=True, sharey=True,
    gridspec_kw={"hspace": 0.35, "wspace": 0.05},
)
axs[0].grid(axis="y", which="both")
axs[1].grid(axis="y", which="both")
axs[2].grid(axis="y", which="both")

##########################################################################################

state_ = "TX"
axs[0].plot(pd.to_datetime(shannon_years, format="%Y"),
        Beef_Cows_fromCATINV.loc[Beef_Cows_fromCATINV.state == state_, shannon_years].values[0],
        c="dodgerblue", linewidth=2, label=state_ + " Shannon")

state_ = "Texas"
B = A[A.state == state_].copy()
B.sort_index(inplace=True)
axs[0].plot(B.index, B.cattle_cow_inventory.values, c="red", linewidth=2, label=state_)
axs[0].legend(loc="best")
##########################################################################################
state_ = "AL"
axs[1].plot(pd.to_datetime(shannon_years, format="%Y"),
            Beef_Cows_fromCATINV.loc[Beef_Cows_fromCATINV.state == state_, shannon_years].values[0],
            c="dodgerblue", linewidth=2, label=state_ + " Shannon")

state_ = "Alabama"
B = A[A.state == state_].copy()
B.sort_index(inplace=True)
axs[1].plot(B.index, B.cattle_cow_inventory.values, c="red", linewidth=2, label=state_)
axs[1].legend(loc="best")
##########################################################################################
state_ = "OK"
axs[2].plot(pd.to_datetime(shannon_years, format="%Y"),
            Beef_Cows_fromCATINV.loc[Beef_Cows_fromCATINV.state == state_, shannon_years].values[0],
            c="dodgerblue", linewidth=2, label=state_ + " Shannon")

state_ = "Oklahoma"
B = A[A.state == state_].copy()
B.sort_index(inplace=True)
axs[2].plot(B.index, B.cattle_cow_inventory.values, c="red", linewidth=2, label=state_)

axs[2].legend(loc="best");

# %%
B

# %% [markdown]
# ## Merge different variables

# %%
seasonal_5yearLapse.head(2)

# %%
feed_expense.head(2)

# %%
need_cols = ["year", "county_fips", "feed_expense"]
season_Feed = pd.merge(
    seasonal_5yearLapse,
    feed_expense[need_cols].drop_duplicates(),
    on=["year", "county_fips"],
    how="left",
)

del need_cols

print(f"{len(seasonal_5yearLapse.county_fips.unique()) = }")
print(f"{len(feed_expense.county_fips.unique()) = }")
print(f"{seasonal_5yearLapse.shape = }")
print(f"{feed_expense.shape = }")
print(f"{season_Feed.shape = }")
season_Feed.head(2)

# %%
feed_expense_FIPS_yr = (
    feed_expense["county_fips"].astype(str) + "_" + feed_expense["year"].astype(str)
)

seasonal_5yearLapse_FIPS_yr = (
    seasonal_5yearLapse["county_fips"].astype(str) + "_" + seasonal_5yearLapse["year"].astype(str)
)

A = [x for x in list(seasonal_5yearLapse_FIPS_yr) if not (x in list(feed_expense_FIPS_yr))]
print(f"{len(A) = }")
A[:10]

# %% [markdown]
# ## Merge different variables

# %% [markdown]
# ### CRP
# ```CRPwetLand``` file:
#
# Acres enrolled in Conservation Reserve, CRPwetLands Reserve, Farmable CRPwetLands, or Conservation Reserve Enhancement Programs, 1997-2017:\\
#
# https://quickstats.nass.usda.gov/#3A734A89-9829-3674-AFC6-C4764DF7B728

# %%
CRPwetLand_area.head(2)

# %%
if "value" in CRPwetLand_area.columns:
    CRPwetLand_area.rename(
        columns={"value": "CRP_wetLand_acr", "cv_(%)": "CRP_wetLand_acr_cv_(%)"},
        inplace=True,
    )

print(f"{CRPwetLand_area.data_item.unique() = }")
CRPwetLand_area.head(2)

# %%
CRPwetLand_area = CRPwetLand_area[CRPwetLand_area.state.isin(SoI)]
print(f"{CRPwetLand_area.shape = }")
print(f"{len(CRPwetLand_area.state.unique()) = }")
print(f"{len(CRPwetLand_area.county_fips.unique()) = }")
print(f"{len(CRPwetLand_area.year.unique()) = }")

# %%
need_cols = ["year", "county_fips", "CRP_wetLand_acr"]
season_Feed_CRP = pd.merge(
    season_Feed,
    CRPwetLand_area[need_cols].drop_duplicates(),
    on=["year", "county_fips"],
    how="left",
)
del need_cols

# %%
print(f"{season_Feed.shape = }")
print(f"{CRPwetLand_area.shape = }")
print(f"{season_Feed_CRP.shape = }")

# %% [markdown]
# ### Irrigated Acre
#
# ```AgLand```. Irrigated acres and total land in farms by county, 1997-2017.
#
# https://quickstats.nass.usda.gov/#B2688D70-61AC-3E14-AA15-11882355E95E
#
# Ask for **Kirti**'s input.

# %%
# print("AgLand.data_item are {}".format(list(AgLand.data_item.unique())))
# print()
# AgLand.head(2)

# CRPwetLand_area_25state = CRPwetLand_area[CRPwetLand_area.state.isin(SoI)]
# print(AgLand.shape)
# print(len(AgLand.state.unique()))
# print(len(AgLand.county_fips.unique()))
# print(len(AgLand.year.unique()))

# AgLand.head(2)

# %%
irr_areas_perc_df.head(2)

# %%
season_Feed_CRP.head(2)

# %%
irr_areas_perc_df.head(2)

# %%
need_cols = ["state", "county", "year", "county_fips", "irr_as_perc"]
season_Feed_CRP_irr = pd.merge(
    season_Feed_CRP,
    irr_areas_perc_df[need_cols].drop_duplicates(),
    on=["year", "county_fips", "state", "county"],
    how="left",
)
del need_cols

season_Feed_CRP_irr.head(2)

# %% [markdown]
# ## County Population
#
# This is population of people. I do not know how this is relevant. Each farm would send beef outside of the county, no?

# %%
human_population = pd.read_pickle(reOrganized_dir + "human_population.sav")
human_population.keys()

# %%
pop_wide = human_population["human_population"]
pop_wide.head(2)

# %%
print (f"{pop_wide.year.unique()            = }")
print (f"{season_Feed_CRP_irr.year.unique() = }")

# %%
print("46113" in list(pop_wide.county_fips))
print("51515" in list(pop_wide.county_fips))
print("46102" in list(pop_wide.county_fips))
print ()
print("46113" in list(season_Feed_CRP_irr.county_fips))
print("51515" in list(season_Feed_CRP_irr.county_fips))
print("46102" in list(season_Feed_CRP_irr.county_fips))

# %%

# %%
season_Feed_CRP_irr.head(2)

# %%
pop_wide.head(2)

# %% [markdown]
# # 1990-1999
# Population file did not have state name in it. So, it is not filtered by **SoI**. So, number of FIPs is high.

# %%
# AgLand_25state = AgLand[AgLand.state.isin(SoI)]
print(pop_wide.shape)
print(len(pop_wide.county_fips.unique()))
# print(len(pop_wide.county.unique()))
# print(len(pop_wide.year.unique()))

# %%
county_fips_digit_count = 5
short_fips = []
long_fips = []
for a_fip in pop_wide.county_fips.unique():
    if len(a_fip) < county_fips_digit_count:
        short_fips += [a_fip]
        
    if len(a_fip) > county_fips_digit_count:
        long_fips += [a_fip]
        
print (short_fips)
print (long_fips)

# %%

# %%
print(season_Feed_CRP_irr.shape)
print(len(season_Feed_CRP_irr.county_fips.unique()))

# %%
season_Feed_CRP_irr_pop = pd.merge(season_Feed_CRP_irr, pop_wide, on=["year", "county_fips"], how="left")
season_Feed_CRP_irr_pop.head(2)

# %%
print(f"{season_Feed_CRP_irr.shape = }")
print(f"{season_Feed_CRP_irr_pop.shape = }")
season_Feed_CRP_irr_pop.head(5)

# %% [markdown]
# ### Beef Cow Inventory (heads)
#
#  - Total Beef Cow inventory: https://quickstats.nass.usda.gov/#ADD6AB04-62EF-3DBF-9F83-5C23977C6DC7
#  - Inventory of Beef Cows: https://quickstats.nass.usda.gov/#B2688D70-61AC-3E14-AA15-11882355E95E

# %%
# totalBeefCowInv = pd.read_csv(USDA_data_dir + "totalBeefCowInv.csv")
# totalBeefCowInv.head(2)

# totalBeefCowInv.rename(
#     columns={"value": "total_beefCowInv", "cv_(%)": "total_beefCowInv_(%)"},
#     inplace=True,
# )
# totalBeefCowInv.head(2)

# totalBeefCowInv = totalBeefCowInv[totalBeefCowInv.state.isin(SoI)].copy()

# print(totalBeefCowInv.shape)
# print(len(totalBeefCowInv.state.unique()))
# print(len(totalBeefCowInv.county.unique()))
# print(len(totalBeefCowInv.year.unique()))

# totalBeefCowInv.head(2)

# %%
print(len(cattle_inventory.state.unique()))
print(len(cattle_inventory.county_fips.unique()))
print(len(cattle_inventory.year.unique()))

cattle_inventory.head(2)

# %%

# %%
need_cols = ["year", "county_fips", "cattle_cow_inventory"]
season_Feed_CRP_irr_pop_beef = pd.merge(season_Feed_CRP_irr_pop,
                                        cattle_inventory[need_cols].drop_duplicates(),
                                        on=["year", "county_fips"],
                                        how="left")

# %%
season_Feed_CRP_irr_pop_beef.head(2)

# %%

# %%
