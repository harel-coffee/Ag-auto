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
# Hence, do not merge inventory with ```NPP``` and ```SW```, otherwise, we miss a lot in
# the same data table!

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

from sklearn import preprocessing
import statistics
import statsmodels.api as sm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
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
# for bold print
start_b = "\033[1m"
end_b = "\033[0;0m"
print ("This is " + start_b + "a_bold_text" + end_b + "!")

# %%
SoI = [
    "Alabama",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Florida",
    "Georgia",
    "Idaho",
    "Illinois",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Mexico",
    "North Dakota",
    "Oklahoma",
    "Oregon",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Virginia",
    "Washington",
    "Wyoming",
]

abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI_abb = []
for x in SoI:
    SoI_abb = SoI_abb + [abb_dict["full_2_abb"][x]]

# %% [markdown]
# ## County Fips

# %%
county_fips = pd.read_pickle(reOrganized_dir + "county_fips.sav")
county_fips = county_fips["county_fips"]

print (f"{len(county_fips.state.unique()) = }")
county_fips = county_fips[county_fips.state.isin(SoI_abb)].copy()
county_fips.drop_duplicates(inplace=True)
county_fips.reset_index(drop=True, inplace=True)
county_fips = county_fips[["county_fips", "county_name", "state", "EW"]]
print (f"{len(county_fips.state.unique()) = }")

county_fips.head(2)

# %%
cnty_interest_list = list(county_fips.county_fips.unique())
len(cnty_interest_list)

# %% [markdown]
# ## Inventory

# %%
USDA_data = pd.read_pickle(reOrganized_dir + "USDA_data.sav")
inventory = USDA_data["cattle_inventory"]

inventory.rename(columns={"cattle_cow_beef_inventory": "inventory"}, inplace=True)

# pick only the counties we want
# cattle_inventory = cattle_inventory[cattle_inventory.county_fips.isin(cnty_interest_list)].copy()

print(f"{inventory.data_item.unique() = }")
print(f"{inventory.commodity.unique() = }")
print()
print(f"{len(inventory.state.unique())= }")

inventory.head(2)

# %%
census_years = sorted(list(inventory.year.unique()))
print(f"{census_years = }")

# pick only useful columns
inv_col_ = "inventory"
inventory = inventory[["year", "county_fips", inv_col_]]

print(f"{len(inventory.county_fips.unique()) = }")
inventory.head(2)

# %% [markdown]
# ### See how many counties and how many data points are incomplete in inventory

# %%
all_cattle_counties = list(inventory.county_fips.unique())
# print(f"{len(all_cattle_counties) = }")
incomplete_counties = {}
for a_cnty_fip in all_cattle_counties:
    curr_cow = inventory[inventory.county_fips == a_cnty_fip].copy()
    missing_yr = [x for x in census_years if not(x in list(curr_cow.year))]
    if (len(missing_yr)>0):
        incomplete_counties[a_cnty_fip] = missing_yr
        
lic = len(incomplete_counties)
la = len(all_cattle_counties)
print ("There are {} incomlete counties out of {} for census years!!!".format(lic, la))

{key:value for key,value in list(incomplete_counties.items())[0:3]}

# %% [markdown]
# ## NPP exist only after 2001! 
# So let us use subset of cattle inventory from census

# %%
inventory = inventory[inventory.year >= 2001]
inventory.reset_index(drop=True, inplace=True)

census_years = sorted(list(inventory.year.unique()))
inventory.head(2)

# %%
all_cattle_counties = list(inventory.county_fips.unique())
# print(f"{len(all_cattle_counties) = }")
incomplete_counties = {}
for a_cnty_fip in all_cattle_counties:
    curr_cow = inventory[inventory.county_fips == a_cnty_fip].copy()
    missing_yr = [x for x in census_years if not(x in list(curr_cow.year))]
    if (len(missing_yr) > 0):
        incomplete_counties[a_cnty_fip] = missing_yr
        
lic = len(incomplete_counties)
la = len(all_cattle_counties)
print ("There are {} incomlete counties out of {} for census years!!!".format(lic, la))

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
#  - rangeland acre, 
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
####################

# This cell is replaced by the one below after filtering counties based on
# their rangeland area

####################
# RA = pd.read_csv(reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv")
# RA.rename(columns={"fips_id": "county_fips"}, inplace=True)
# RA = rc.correct_Mins_county_FIPS(df=RA, col_ = "county_fips")
# print (f"{len(RA.county_fips.unique()) = }")
# RA = RA[RA.county_fips.isin(cnty_interest_list)]
# print (f"{len(RA.county_fips.unique()) = }")
# RA.reset_index(drop=True, inplace=True)
# RA.head(2)

# %%
RA = pd.read_pickle(param_dir + "filtered_counties_RAsizePallavi.sav")
RA = RA["filtered_counties"]

print (f"{len(RA.county_fips.unique()) = }")
print (f"{len(RA.state.unique()) = }")
RA.head(2)

# %%
print (f"{list(RA.state.unique()) = }")
print ()
miss_cnty = [x for x in SoI_abb if x not in RA.state.unique()]
print (miss_cnty)

# %%
print (len(county_fips.state.unique()))
county_fips.head(2)

# %% [markdown]
# ### Filter Inventory according to rangeland area Pallavi Filter

# %%
print ("number of counties are [{}]".format(len(inventory.county_fips.unique())))
print ("number of instances are [{}]".format(inventory.shape[0]))
print ()
print (start_b + "After filtering the counties based on rangeland area (Pallavi) condition" + end_b)
L = len(inventory[inventory.county_fips.isin(RA.county_fips.unique())].county_fips.unique())
print ("number of counties are [{}]".format(L))
L = inventory[inventory.county_fips.isin(RA.county_fips.unique())].shape[0]
print ("number of instances are [{}]".format(L))

# %%
good_size_counties = RA.county_fips.unique()
inventory_RA_filtered = inventory[inventory.county_fips.isin(good_size_counties)]
# inventory_RA_filtered = pd.merge(inventory_RA_filtered, 
#                                  RA[["county_fips", "rangeland_acre", "rangeland_fraction", "state"]], 
#                                  on=["county_fips"], how="left")

inventory_RA_filtered = pd.merge(inventory_RA_filtered,
                                 county_fips,
                                 on=["county_fips"], how="left")

inventory_RA_filtered.reset_index(drop=True, inplace=True)
inventory_RA_filtered.head(2)

# %%
print (len(inventory.county_fips.unique()))
print (len(inventory_RA_filtered.county_fips.unique()))

# %%
print (len(inventory_RA_filtered.state.unique()))
inventory_RA_filtered.head(2)

# %%
cty_yr_GPP_NPP_prod = pd.read_csv(reOrganized_dir + "county_annual_GPP_NPP_productivity.csv")

cty_yr_GPP_NPP_prod.rename(columns={"county" : "county_fips",
                                    "MODIS_NPP" : "unit_npp"}, inplace=True)
cty_yr_GPP_NPP_prod = rc.correct_Mins_county_6digitFIPS(df=cty_yr_GPP_NPP_prod, col_ = "county_fips")

print (f"{len(cty_yr_GPP_NPP_prod.county_fips.unique()) = }")
cty_yr_GPP_NPP_prod = cty_yr_GPP_NPP_prod[cty_yr_GPP_NPP_prod.county_fips.isin(cnty_interest_list)]
print (f"{len(cty_yr_GPP_NPP_prod.county_fips.unique()) = }")
cty_yr_GPP_NPP_prod.head(3)

# %%
cty_yr_GPP_NPP_prod = pd.merge(cty_yr_GPP_NPP_prod, 
                               RA[["county_fips", "rangeland_acre"]], 
                               on=["county_fips"], how="left")

cty_yr_GPP_NPP_prod = rc.covert_unitNPP_2_total(NPP_df=cty_yr_GPP_NPP_prod, 
                                                npp_unit_col_ = "unit_npp", 
                                                acr_area_col_ = "rangeland_acre", 
                                                npp_area_col_ = "county_total_npp")

cty_yr_GPP_NPP_prod.head(2)

# %%
cty_yr_npp = cty_yr_GPP_NPP_prod[["year", "county_fips", "county_total_npp"]]
cty_yr_npp.dropna(how="any", inplace=True)
cty_yr_npp.reset_index(drop=True, inplace=True)
cty_yr_npp.head(2)

# %% [markdown]
# ### Weather Variables

# %%
filename = reOrganized_dir + "county_seasonal_temp_ppt_weighted.sav"
SW = pd.read_pickle(filename)
print(f"{SW.keys() = }")
SW = SW["seasonal"]
# SW = SW[SW.year >= 2001]
SW = pd.merge(SW, county_fips,  on=["county_fips"], how="left")
SW.head(2)

# %%
seasonal_precip_vars = ["S1_countyMean_total_precip", "S2_countyMean_total_precip",
                        "S3_countyMean_total_precip", "S4_countyMean_total_precip",]

seasonal_temp_vars = ["S1_countyMean_avg_Tavg", "S2_countyMean_avg_Tavg",
                      "S3_countyMean_avg_Tavg", "S4_countyMean_avg_Tavg"]

SW_vars = seasonal_precip_vars + seasonal_temp_vars
for a_col in SW_vars:
    SW[a_col] = SW[a_col].astype(float)

# %%
SW["yr_countyMean_total_precip"] = SW[seasonal_precip_vars].sum(axis=1)

# SW["yr_countyMean_avg_Tavg"]     = SW[seasonal_temp_vars].sum(axis=1)
# SW["yr_countyMean_avg_Tavg"]     = SW["yr_countyMean_avg_Tavg"]/4

SW = SW.round(3)
SW.head(2)

# %% [markdown]
# ### WARNING
# Here, above, I did average of temperatures over 4 season. That is close enought to average of months (below).
#
# Too many changes occur in demands. Cannot keep up with doing everything immaculately and manicure everything.
#
# if you want to have everything manicured fucking do it. I ran the code already!

# %%
# SW_annual = pd.read_csv(reOrganized_dir + "county_gridmet_mean_indices.csv")
# SW_annual.head(2)

# SW_annual = SW_annual[["year", 'county', 'PPT', 'TAVG_AVG']]
# SW_annual.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
# SW_annual.rename(columns={"county": "county_fips",
#                           "tavg_avg" : "Tavg_avg"}, inplace=True)
# SW_annual = rc.correct_Mins_county_6digitFIPS(df=SW_annual, col_="county_fips")
# SW_annual.head(2)

# A = SW_annual.groupby(["year", "county_fips"])["ppt"].sum().reset_index()
# A.rename(columns={"ppt": "total_precip"}, inplace=True)
# A.head(2)

# A = SW_annual.groupby(["year", "county_fips"])["Tavg_avg"].sum().reset_index()
# A.rename(columns={"Tavg_avg": "avg_Tavg_avg"}, inplace=True)
# A["avg_Tavg_avg"] = A["avg_Tavg_avg"]/12
# A.head(2)

# %%
cty_yr_npp.head(2)

# %%
SW.head(2)

# %%
cty_yr_npp = cty_yr_npp[cty_yr_npp.county_fips.isin(inventory_RA_filtered.county_fips.unique())]
SW = SW[SW.county_fips.isin(inventory_RA_filtered.county_fips.unique())]

cty_yr_npp.reset_index(drop=True, inplace=True)
SW.reset_index(drop=True, inplace=True)

# %%
print (len(SW.county_fips.unique()))
print (len(cty_yr_npp.county_fips.unique()))

# %%
NPP_SW = pd.merge(cty_yr_npp, SW, on=["year", "county_fips"], how="outer")
NPP_SW.head(2)

# %%
filename = reOrganized_dir + "county_annual_avg_Tavg.sav"
county_annual_avg_Tavg = pd.read_pickle(filename)
county_annual_avg_Tavg = county_annual_avg_Tavg["annual_temp"]

cc = inventory_RA_filtered.county_fips.unique()
county_annual_avg_Tavg = county_annual_avg_Tavg[county_annual_avg_Tavg.county_fips.isin(cc)]

# county_annual_avg_Tavg = county_annual_avg_Tavg[county_annual_avg_Tavg.year >= 2001]

county_annual_avg_Tavg.reset_index(drop=True, inplace=True)
county_annual_avg_Tavg.head(2)

# %%
NPP_SW = pd.merge(NPP_SW, county_annual_avg_Tavg, on=["year", "county_fips"], how="outer")
NPP_SW.head(2)

# %% [markdown]
# ### Heat index
#
#
# - Normal: Number of days $\text{THI} < 75$
# - Alert: Number of days $\text{THI} \ge 75$ and $\text{THI}<79$
# - Danger: Number of days $\text{THI} \ge 79$ and $\text{THI}<84$
# - Emergency: Number of days $\text{THI} \ge 84$
#

# %%
cnty_grid_mean_idx = pd.read_csv(Min_data_base + "county_gridmet_mean_indices.csv")
cnty_grid_mean_idx.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
cnty_grid_mean_idx.rename(columns={"county": "county_fips"}, inplace=True)
cnty_grid_mean_idx = rc.correct_Mins_county_6digitFIPS(df=cnty_grid_mean_idx, col_="county_fips")
cnty_grid_mean_idx.head(2)

# %%
cnty_grid_mean_idx = cnty_grid_mean_idx[["year", "month", "county_fips",
                                         "normal", "alert", "danger", "emergency"]]

for a_col in ["normal", "alert", "danger", "emergency"]:
    cnty_grid_mean_idx[a_col] = cnty_grid_mean_idx[a_col].astype(int)
    
cnty_grid_mean_idx.head(2)

# %%
cnty_grid_mean_idx["dangerEncy"] = cnty_grid_mean_idx["danger"] + cnty_grid_mean_idx["emergency"]
cnty_grid_mean_idx.head(3)

# %%
dangerEncy_yr = cnty_grid_mean_idx.groupby(["year", "county_fips"])["dangerEncy"].sum()
dangerEncy_yr = pd.DataFrame(dangerEncy_yr).reset_index()
dangerEncy_yr.head(2)

# %%
NPP_SW_heat = pd.merge(NPP_SW, dangerEncy_yr, on=["year", "county_fips"], how="outer")
NPP_SW_heat.head(2)

# %% [markdown]
# # Model Snapshot

# %%
inventory_2017 = inventory_RA_filtered[inventory_RA_filtered.year==2017].copy()
inventory_2017 = inventory_2017[["year", "county_fips", "inventory"]]
inventory_2017.head(2)

# %%
NPP_SW_heat.head(2)

# %%

# %%

# %%

# %% [markdown]
# ### Herb Ratio

# %%

# %%

# %%
herb = pd.read_pickle(data_dir_base + "Supriya/Nov30_HerbRatio/county_herb_ratio.sav")
herb = herb["county_herb_ratio"]
herb.head(2)
print (herb.shape)
herb = herb[herb.county_fips.isin(cnty_interest_list)]
print (herb.shape)

herb.dropna(how="any", inplace=True)
print (herb.shape)

herb.reset_index(drop=True, inplace=True)
herb.head(3)

# %%
RA_herb = pd.merge(RA, herb, on=["county_fips"], how="left")
# RA_herb.dropna(how="any", inplace=True)
RA_herb.reset_index(drop=True, inplace=True)
RA_herb.head(2)

# %%
# RA_herb_outer = pd.merge(RA, herb, on=["county_fips"], how="outer")
# RA_herb_outer.reset_index(drop=True, inplace=True)
# RA_herb_outer.head(5)

# print (RA_herb_outer.shape)
# print (RA_herb.shape)

# outer_not_in_left = RA_herb_outer[~RA_herb_outer.county_fips.isin(RA_herb.county_fips)]
# outer_not_in_left.shape
# outer_not_in_left

# %%
inventory_RA_herb = pd.merge(inventory, RA_herb, on=["county_fips"], how="left")
inventory_RA_herb.head(2)

# %%

# %%

# %%

# %%
inventory_RA_herb.head(2)

# %%
inventory_RA_herb_NPP = pd.merge(inventory_RA_herb, cty_yr_npp, 
                                 on=["county_fips", "year"], how="left")

inventory_RA_herb_NPP.head(2)

# %%
