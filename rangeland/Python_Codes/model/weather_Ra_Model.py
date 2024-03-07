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
# ## Nov 7.
#
# - ```SW```: Seasonal Weather: temp. and precip.
#
# On Nov. 6 Mike wanted to model cattle inventory using only ```NPP```/```SW``` and rangeland area for one year.
#
# **Min's data are inconsistent:** Let us subset the counties that are in common between ```NPP``` and ```SW```, and cattle inventory.
#
# #### Seasons in Tonsor are
# - S1: Jan - Mar
# - S2: Apr - Jul
# - S3: Aug - Sep
# - S4: Oct - Dec

# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

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
seasonal_dir = reOrganized_dir + "seasonal_variables/02_merged_mean_over_county/"

# %%

# %%
Bhupi = pd.read_csv(param_dir + "Bhupi_25states_clean.csv")
Bhupi["SC"] = Bhupi.state + "-" + Bhupi.county

print(f"{len(Bhupi.state.unique()) = }")
print(f"{len(Bhupi.county_fips.unique()) = }")
Bhupi.head(2)

# %%

# %%
abb_dict = pd.read_pickle(param_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = []
for x in SoI:
    SoI_abb = SoI_abb + [abb_dict["full_2_abb"][x]]

# %%
USDA_data = pd.read_pickle(reOrganized_dir + "USDA_data.sav")

cattle_inventory = USDA_data["cattle_inventory"]

# pick only 25 states we want
cattle_inventory = cattle_inventory[cattle_inventory.state.isin(SoI)].copy()

print(f"{cattle_inventory.data_item.unique() = }")
print(f"{cattle_inventory.commodity.unique() = }")
print(f"{cattle_inventory.year.unique() = }")

census_years = list(cattle_inventory.year.unique())
# pick only useful columns
cattle_inventory = cattle_inventory[["year", "county_fips", "cattle_cow_inventory"]]

print(f"{len(cattle_inventory.county_fips.unique()) = }")
cattle_inventory.head(2)

# %%
print(cattle_inventory.shape)
cattle_inventory = rc.clean_census(df=cattle_inventory, col_="cattle_cow_inventory")
print(cattle_inventory.shape)

# %% [markdown]
# ### Min has an extra "1" as leading digit in FIPS!!

# %%
filename = reOrganized_dir + "county_seasonal_temp_ppt_weighted.sav"
seasonal_weather = pd.read_pickle(filename)
seasonal_weather.keys()

# %%
seasonal_weather = seasonal_weather["seasonal"]
seasonal_weather.head(2)

# %%
census_years

# %%
needed_cols = seasonal_weather.columns[2:10]
for a_col in needed_cols:
    seasonal_weather[a_col] = seasonal_weather[a_col].astype(float)

# %%

# %%
# pick only census years
seasonal_weather = seasonal_weather[seasonal_weather.year.isin(census_years)]
seasonal_weather.reset_index(drop=True, inplace=True)
seasonal_weather.head(2)

# %%
county_id_name_fips = pd.read_csv(Min_data_base + "county_id_name_fips.csv")
county_id_name_fips = county_id_name_fips[
    county_id_name_fips.STATE.isin(SoI_abb)
].copy()

county_id_name_fips.sort_values(by=["STATE", "county"], inplace=True)

county_id_name_fips = rc.correct_Mins_FIPS(df=county_id_name_fips, col_="county")

county_id_name_fips.rename(columns={"county": "county_fips"}, inplace=True)
county_id_name_fips.reset_index(drop=True, inplace=True)
county_id_name_fips.head(2)

# %%
print(f"{len(seasonal_weather.county_fips.unique()) = }")

# %%

# %%
print(seasonal_weather.shape)
LL = list(county_id_name_fips.county_fips.unique())
seasonal_weather = seasonal_weather[seasonal_weather.county_fips.isin(LL)].copy()
print(seasonal_weather.shape)
seasonal_weather.head(2)

# %%
print(f"{(seasonal_weather.year.unique()) = }")
print(f"{len(seasonal_weather.county_fips.unique()) = }")
print(f"{len(cattle_inventory.county_fips.unique()) = }")

# %%
for a_year in seasonal_weather.year.unique():
    df = seasonal_weather[seasonal_weather.year == a_year]
    print(f"{len(df.county_fips.unique()) = }")

# %%
# Rangeland area and Total area:
county_RA_and_TA_fraction = pd.read_csv(
    reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv"
)
print(county_RA_and_TA_fraction.shape)
county_RA_and_TA_fraction.head(5)

# %%
county_RA_and_TA_fraction.rename(columns={"fips_id": "county_fips"}, inplace=True)
county_RA_and_TA_fraction = rc.correct_Mins_FIPS(
    df=county_RA_and_TA_fraction, col_="county_fips"
)
county_RA_and_TA_fraction.head(2)

# %%
county_annual_SW_Ra = pd.merge(
    seasonal_weather, county_RA_and_TA_fraction, on=["county_fips"], how="left"
)
county_annual_SW_Ra.head(2)

# %%
print(f"{sorted(cattle_inventory.year.unique())     = }")
print(f"{sorted(county_annual_SW_Ra.year.unique()) = }")

# %%
cattle_inventory = cattle_inventory[
    cattle_inventory.year.isin(list(county_annual_SW_Ra.year.unique()))
]
sorted(cattle_inventory.year.unique())

# %%
print(len(cattle_inventory.county_fips.unique()))
print(len(county_annual_SW_Ra.county_fips.unique()))

# %%
cattle_inventory_cnty_missing_from_SW = [
    x
    for x in cattle_inventory.county_fips.unique()
    if not (x in county_annual_SW_Ra.county_fips.unique())
]
len(cattle_inventory_cnty_missing_from_SW)

# %%
SW_cnty_missing_from_cattle = [
    x
    for x in county_annual_SW_Ra.county_fips.unique()
    if not (x in cattle_inventory.county_fips.unique())
]
len(SW_cnty_missing_from_cattle)

# %%
print("01001" in list(county_annual_SW_Ra.county_fips.unique()))
print("01001" in list(cattle_inventory.county_fips.unique()))

# %% [markdown]
# ## SW has a lot of missing counties
#
#  - Min says he had a threshld about rangeland/pasture.
#  - subset the NPP and Cattle to the intersection of counties present.
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
cow_counties = set(cattle_inventory.county_fips.unique())
county_intersection = SW_counties.intersection(cow_counties)

# %%
county_annual_SW_Ra = county_annual_SW_Ra[
    county_annual_SW_Ra.county_fips.isin(list(county_intersection))
]
cattle_inventory = cattle_inventory[
    cattle_inventory.county_fips.isin(list(county_intersection))
]

print(f"{county_annual_SW_Ra.shape = }")
print(f"{cattle_inventory.shape     = }")
print()
print(f"{len(county_annual_SW_Ra.county_fips.unique()) = }")
print(f"{len(cattle_inventory.county_fips.unique())     = }")
print()
print(f"{sorted(county_annual_SW_Ra.year.unique()) = }")
print(f"{sorted(cattle_inventory.year.unique())     = }")

# %%
county_annual_SW_Ra_cattleInv = pd.merge(
    county_annual_SW_Ra, cattle_inventory, on=["county_fips", "year"], how="left"
)

print(f"{cattle_inventory.shape = }")
print(f"{county_annual_SW_Ra.shape = }")
print(f"{county_annual_SW_Ra_cattleInv.shape = }")
county_annual_SW_Ra_cattleInv.head(2)

# %%
county_annual_SW_Ra_cattleInv.sort_values(by=["year", "county_fips"], inplace=True)
county_annual_SW_Ra_cattleInv.reset_index(drop=True, inplace=True)
county_annual_SW_Ra_cattleInv.head(2)

# %% [markdown]
# ## Least Squares based on 2017 ```weather```

# %%
SW_Ra_cattleInv_2017 = county_annual_SW_Ra_cattleInv[
    county_annual_SW_Ra_cattleInv.year == 2017
].copy()

# %%

# %%
needed_cols = SW_Ra_cattleInv_2017.columns[2:11]
expl_var_2017 = SW_Ra_cattleInv_2017[needed_cols].values
y_2017 = SW_Ra_cattleInv_2017[["cattle_cow_inventory"]].values.reshape(-1)
print(f"{y_2017.shape = }")
y_2017

# %%
expl_var_interc_2017 = np.hstack(
    [expl_var_2017, np.ones(len(expl_var_2017)).reshape(-1, 1)]
)
print(expl_var_interc_2017.shape)
expl_var_interc_2017

# %%
expl_var_interc_2017[0]

# %%
solution_2017, RSS_2017, rank_2017, singular_vals_2017 = np.linalg.lstsq(
    expl_var_interc_2017, y_2017
)

# %%
county_annual_SW_Ra_cattleInv[county_annual_SW_Ra_cattleInv.year == 2017].head(2)

# %%
solution_2017

# %%
# y_hat_2017 = expl_var_interc_2017 @ solution_2017
# res = y_2017 - y_hat_2017
# RSS = np.dot(res, res)
# RSS

RSS_2017

# %%

# %% [markdown]
# ### Apply 2017 model to 2012 data

# %%
SW_Ra_cattleInv_2012 = county_annual_SW_Ra_cattleInv[
    county_annual_SW_Ra_cattleInv.year == 2012
].copy()

y_2012 = SW_Ra_cattleInv_2012[["cattle_cow_inventory"]].values.reshape(-1)

expl_var_2012 = SW_Ra_cattleInv_2012[needed_cols].values
expl_var_interc_2012 = np.hstack(
    [expl_var_2012, np.ones(len(expl_var_2012)).reshape(-1, 1)]
)
expl_var_interc_2012

# %%
SW_Ra_cattleInv_2012.head(2)

# %%
y_hat_2012_Model2017 = expl_var_interc_2012 @ solution_2017

res_2012_Model2017 = y_2012 - y_hat_2012_Model2017
RSS_2012_Model2017 = np.dot(res_2012_Model2017, res_2012_Model2017)
RSS_2012_Model2017 / len(expl_var_2012)

# %%
print(f"{SW_Ra_cattleInv_2012.cattle_cow_inventory.min()=}")
print(f"{SW_Ra_cattleInv_2012.cattle_cow_inventory.max()=}")

# %%

# %%

# %%

# %%
