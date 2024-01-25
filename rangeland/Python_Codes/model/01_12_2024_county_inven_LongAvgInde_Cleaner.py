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
SoI = ["Alabama", "Arizona", "Arkansas", "California", "Colorado",
       "Florida", "Georgia", "Idaho", "Illinois", "Iowa", "Kansas",
       "Kentucky", "Louisiana", "Mississippi",
       "Missouri", "Montana", "Nebraska",
       "Nevada", "New Mexico", "North Dakota",
       "Oklahoma", "Oregon", "South Dakota",
       "Tennessee", "Texas", "Utah",
       "Virginia", "Washington", "Wyoming"]

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
# cnty_interest_list = list(county_fips.county_fips.unique())
# len(cnty_interest_list)

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
inventory[inventory.county_fips == "04003"]

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
    if (len(missing_yr) > 0):
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
print("04001" in inventory.county_fips.unique())
print("04003" in inventory.county_fips.unique())

# %%

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
RA = pd.read_pickle(param_dir + "filtered_counties_RAsizePallavi.sav")
RA = RA["filtered_counties_29States"]
print (f"{len(RA.county_fips.unique()) = }")
print (f"{len(RA.state.unique()) = }")
RA.head(2)

# %%
print ("04001" in list(RA.county_fips))
print ("04003" in list(RA.county_fips))

# %%
print (f"{list(RA.state.unique()) = }")
print ()
miss_cnty = [x for x in SoI_abb if x not in RA.state.unique()]
print (miss_cnty)

# %%
print (len(county_fips.state.unique()))
county_fips.head(2)

# %% [markdown]
# ### Filter Inventory according to Pallavi Filter ( rangeland area criteria)

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
inventory_Pallavi_filtered = inventory[inventory.county_fips.isin(good_size_counties)]
# inventory_Pallavi_filtered = pd.merge(inventory_Pallavi_filtered, 
#                                       RA[["county_fips", "rangeland_acre", "rangeland_fraction", "state"]], 
#                                       on=["county_fips"], how="left")

inventory_Pallavi_filtered = pd.merge(inventory_Pallavi_filtered,
                                      county_fips,
                                      on=["county_fips"], how="left")

inventory_Pallavi_filtered.reset_index(drop=True, inplace=True)
inventory_Pallavi_filtered.head(2)

# %%
#################################################################################
#########
######### We can do this waaay up there to use its
######### county_fips to filter all other variables.
#########
#################################################################################
inventory_2017 = inventory_Pallavi_filtered[inventory_Pallavi_filtered.year==2017].copy()
inventory_2017 = inventory_2017[["year", "county_fips", "inventory"]]

inventory_2017 = pd.merge(inventory_2017, county_fips, on=["county_fips"], how="left")
inventory_2017.head(2)

# %%
print (len(inventory.county_fips.unique()))
print (len(inventory_Pallavi_filtered.county_fips.unique()))
print (len(inventory_2017.county_fips.unique()))

# %%
print (len(inventory_Pallavi_filtered.state.unique()))
inventory_Pallavi_filtered.head(2)

# %%
#### To use to filter all other variables:

inv_2017_Pallavi_cnty_list = list(inventory_2017.county_fips.unique())
len(inv_2017_Pallavi_cnty_list)

# %%
cty_yr_npp = pd.read_csv(reOrganized_dir + "county_annual_GPP_NPP_productivity.csv")

cty_yr_npp.rename(columns={"county" : "county_fips",
                           "MODIS_NPP" : "unit_npp"}, inplace=True)

cty_yr_npp = rc.correct_Mins_county_6digitFIPS(df=cty_yr_npp, col_ = "county_fips")

###################################
######
###### Keep the counties that haved passed Pallavi's test
######
print (cty_yr_npp.shape)
cty_yr_npp = cty_yr_npp[cty_yr_npp.county_fips.isin(inv_2017_Pallavi_cnty_list)]
cty_yr_npp.reset_index(drop=True, inplace=True)
print (cty_yr_npp.shape)
print (f"{len(cty_yr_npp.county_fips.unique()) = }")
cty_yr_npp.head(3)

# %%
cty_yr_npp = pd.merge(cty_yr_npp, RA[["county_fips", "rangeland_acre", "state"]], 
                      on=["county_fips"], how="left")
cty_yr_npp.head(2)

# %%
cty_yr_npp = rc.covert_unitNPP_2_total(NPP_df=cty_yr_npp, 
                                       npp_unit_col_ = "unit_npp", 
                                       acr_area_col_ = "rangeland_acre", 
                                       npp_area_col_ = "county_total_npp")

# %%
cty_yr_npp = cty_yr_npp[["year", "county_fips", "county_total_npp"]]
cty_yr_npp.dropna(how="any", inplace=True)
cty_yr_npp.reset_index(drop=True, inplace=True)
cty_yr_npp.head(2)

# %% [markdown]
# ### Weather Variables

# %%
filename = reOrganized_dir + "county_seasonal_temp_ppt_weighted.sav"
SW = pd.read_pickle(filename)
SW = SW["seasonal"]
SW = pd.merge(SW, county_fips,  on=["county_fips"], how="left")

print (f"{len(SW.county_fips.unique())=}")
SW = SW[SW.county_fips.isin(inv_2017_Pallavi_cnty_list)]
SW.reset_index(drop=True, inplace=True)
print (f"{len(SW.county_fips.unique())=}")
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
# SW["yr_countyMean_avg_Tavg"]   = SW[seasonal_temp_vars].sum(axis=1)
# SW["yr_countyMean_avg_Tavg"]   = SW["yr_countyMean_avg_Tavg"]/4

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
print (len(SW.county_fips.unique()))
print (len(cty_yr_npp.county_fips.unique()))

# %%
NPP_SW = pd.merge(cty_yr_npp, SW, on=["year", "county_fips"], how="outer")
print(len(NPP_SW.county_fips.unique()))
print (NPP_SW.year.min())
print (NPP_SW.year.max())
NPP_SW.head(2)

# %%
filename = reOrganized_dir + "county_annual_avg_Tavg.sav"
cnty_yr_avg_Tavg = pd.read_pickle(filename)
cnty_yr_avg_Tavg = cnty_yr_avg_Tavg["annual_temp"]

cnty_yr_avg_Tavg = cnty_yr_avg_Tavg[cnty_yr_avg_Tavg.county_fips.isin(inv_2017_Pallavi_cnty_list)]

# county_annual_avg_Tavg = county_annual_avg_Tavg[county_annual_avg_Tavg.year >= 2001]

cnty_yr_avg_Tavg.reset_index(drop=True, inplace=True)
cnty_yr_avg_Tavg.head(2)

# %%
NPP_SW = pd.merge(NPP_SW, cnty_yr_avg_Tavg, on=["year", "county_fips"], how="outer")
print(len(NPP_SW.county_fips.unique()))
print (NPP_SW.year.min())
print (NPP_SW.year.max())
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

cnty_grid_mean_idx = cnty_grid_mean_idx[["year", "month", "county_fips",
                                         "normal", "alert", "danger", "emergency"]]

for a_col in ["normal", "alert", "danger", "emergency"]:
    cnty_grid_mean_idx[a_col] = cnty_grid_mean_idx[a_col].astype(int)

cnty_grid_mean_idx = cnty_grid_mean_idx[cnty_grid_mean_idx.county_fips.isin(inv_2017_Pallavi_cnty_list)]
cnty_grid_mean_idx.reset_index(drop=True, inplace=True)
cnty_grid_mean_idx.head(2)

# %%
# Arizona is safe?
cnty_grid_mean_idx[(cnty_grid_mean_idx.county_fips == "04005") & (cnty_grid_mean_idx.year == 2017)]

# %%
cnty_grid_mean_idx["dangerEncy"] = cnty_grid_mean_idx["danger"] + cnty_grid_mean_idx["emergency"]
cnty_grid_mean_idx.head(3)

# %%
dangerEncy_yr = cnty_grid_mean_idx.groupby(["year", "county_fips"])["dangerEncy"].sum()
dangerEncy_yr = pd.DataFrame(dangerEncy_yr).reset_index()
dangerEncy_yr.head(2)

# %%
NPP_SW_heat = pd.merge(NPP_SW, dangerEncy_yr, on=["year", "county_fips"], how="outer")

print(len(NPP_SW_heat.county_fips.unique()))
print (NPP_SW_heat.year.min())
print (NPP_SW_heat.year.max())
NPP_SW_heat.head(2)

# %% [markdown]
# # Model Snapshot 
# **for 2017 Inventory and long run avg of independent variables**

# %%
L = len(inventory_2017.state.unique())
print ("There are " + start_b + "<<< " + str(L) + " >>>" + end_b + " states left in 2017 after all filters!")

# %%
# print (len(inventory_Pallavi_filtered.state.unique()))
# mc = [x for x in inventory_Pallavi_filtered.state.unique() if not (x in inventory_2017.state.unique())]
# inventory_Pallavi_filtered[inventory_Pallavi_filtered.state.isin(mc)]

# %% [markdown]
# ## Compute long run averages
#
# Since ```NPP``` exist after 2001, we filter ```SW``` from 2001 as well. Otherwise, there is no other reason.

# %%
print (NPP_SW_heat.year.min())
print (NPP_SW_heat.year.max())
NPP_SW_heat.head(2)

# %%
################################################################################
######
######   Final Filter check so we have data of the counties in 2017 inventory
######   that is filtered by Pallavi test.
######
print (NPP_SW_heat.shape)
NPP_SW_heat = NPP_SW_heat[NPP_SW_heat.county_fips.isin(inv_2017_Pallavi_cnty_list)]
print (NPP_SW_heat.shape)

# %%
NPP_SW_heat = NPP_SW_heat[NPP_SW_heat.year >= 2001]
NPP_SW_heat = NPP_SW_heat[NPP_SW_heat.year <= 2017]
NPP_SW_heat.drop(labels=['county_name', 'state', 'EW'], axis=1, inplace=True)
NPP_SW_heat.head(2)

# %%
NPP_SW_heat_avg = NPP_SW_heat.groupby("county_fips").mean()
NPP_SW_heat_avg.reset_index(drop=False, inplace=True)
NPP_SW_heat_avg = NPP_SW_heat_avg.round(3)

NPP_SW_heat_avg.drop(labels=['year'], axis=1, inplace=True)
NPP_SW_heat_avg.head(3)

# %%
NPP_SW_heat.loc[NPP_SW_heat.county_fips == "04005", "S4_countyMean_total_precip"].mean().round(3)

# %%
print (len(NPP_SW_heat_avg.index))
print (len(inventory_2017.county_fips.unique()))
print (len(NPP_SW_heat_avg.index.unique()) == len(NPP_SW_heat_avg.index))

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
print (NPP_SW_heat_avg.shape)
print (inventory_2017.shape)

# %% [markdown]
# # CAREFUL
#
# Let us merge average data and inventory data. Keep in mind that the year will be 2017 but the data are averaged over the years (except for the inventory). 
#
# Merge them so that counties are in the same order. My sick mind!

# %%
inventory_2017.head(2)

# %%
inventory_2017 = inventory_2017[["county_fips", "inventory", "EW"]]

# %%
inv_2017_NPP_SW_heat_avg = pd.merge(inventory_2017, NPP_SW_heat_avg, on=["county_fips"], how="left")
inv_2017_NPP_SW_heat_avg.head(2)

# %%
inv_2017_NPP_SW_heat_avg.columns

# %% [markdown]
# ### Normalize

# %%
HS_var = ['dangerEncy']
npp_var = ['county_total_npp']
AW_vars = ['yr_countyMean_total_precip', 'annual_avg_Tavg']

SW_vars = ['S1_countyMean_total_precip', 'S2_countyMean_total_precip', 
           'S3_countyMean_total_precip', 'S4_countyMean_total_precip', 
           'S1_countyMean_avg_Tavg', 'S2_countyMean_avg_Tavg', 
           'S3_countyMean_avg_Tavg', 'S4_countyMean_avg_Tavg']

all_indp_vars = list(set(HS_var + AW_vars + SW_vars + npp_var))
all_indp_vars = sorted(all_indp_vars)
all_indp_vars

# %%
inv_2017_NPP_SW_heat_avg_normal = inv_2017_NPP_SW_heat_avg.copy()
inv_2017_NPP_SW_heat_avg_normal[all_indp_vars] = (inv_2017_NPP_SW_heat_avg_normal[all_indp_vars] - 
                                                  inv_2017_NPP_SW_heat_avg_normal[all_indp_vars].mean()) / \
                                                  inv_2017_NPP_SW_heat_avg_normal[all_indp_vars].std(ddof=1)
inv_2017_NPP_SW_heat_avg_normal.head(2)

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
Y = inv_2017_NPP_SW_heat_avg_normal[y_var].astype(float)
npp_inv_model = sm.OLS(Y, X_npp)
npp_inv_model_result = npp_inv_model.fit()
npp_inv_model_result.summary()

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

# %% [markdown]
# ### Inventory vs normal ```SW``` averaged over 2001-2017

# %%
indp_vars = SW_vars
y_var = "inventory"

#################################################################
X_SW = inv_2017_NPP_SW_heat_avg_normal[indp_vars]
X_SW = sm.add_constant(X_SW)
Y = np.log(inv_2017_NPP_SW_heat_avg_normal[y_var].astype(float))
SW_inv_model = sm.OLS(Y, X_SW)
SW_inv_model_result = SW_inv_model.fit()
SW_inv_model_result.summary()

# %%
del(X_SW, SW_inv_model, SW_inv_model_result)

# %% [markdown]
# ### Inventory vs normal ```AW``` averaged over 2001-2017

# %%
del(X_npp, npp_inv_model, npp_inv_model_result)

# %%
indp_vars = AW_vars
y_var = "inventory"

#################################################################
X_AW = inv_2017_NPP_SW_heat_avg_normal[indp_vars]
X_AW = sm.add_constant(X_AW)
Y = np.log(inv_2017_NPP_SW_heat_avg_normal[y_var].astype(float))
AW_inv_model = sm.OLS(Y, X_AW)
AW_inv_model_result = AW_inv_model.fit()
AW_inv_model_result.summary()

# %%
del(X_AW, AW_inv_model, AW_inv_model_result)

# %% [markdown]
# ### Inventory vs normal ```NPP``` AND ```dangerEncy``` averaged over 2001-2017

# %%
indp_vars = npp_var + HS_var
y_var = "inventory"

#################################################################
X_npp_dangerEncy = inv_2017_NPP_SW_heat_avg_normal[indp_vars]
X_npp_dangerEncy = sm.add_constant(X_npp_dangerEncy)
Y = np.log(inv_2017_NPP_SW_heat_avg_normal[y_var].astype(float))
npp_dangerEncy_inv_model = sm.OLS(Y, X_npp_dangerEncy)
npp_dangerEncy_inv_model_result = npp_dangerEncy_inv_model.fit()
npp_dangerEncy_inv_model_result.summary()

# %%
del(X_npp_dangerEncy, npp_dangerEncy_inv_model, npp_dangerEncy_inv_model_result)

# %% [markdown]
# ### Inventory vs normal ```AW``` AND ```dangerEncy``` averaged over 2001-2017

# %%
indp_vars = AW_vars + HS_var
y_var = "inventory"

#################################################################
X_AW_dangerEncy = inv_2017_NPP_SW_heat_avg_normal[indp_vars]
X_AW_dangerEncy = sm.add_constant(X_AW_dangerEncy)
Y = np.log(inv_2017_NPP_SW_heat_avg_normal[y_var].astype(float))
AW_dangerEncy_inv_model = sm.OLS(Y, X_AW_dangerEncy)
AW_dangerEncy_inv_model_result = AW_dangerEncy_inv_model.fit()
AW_dangerEncy_inv_model_result.summary()

# %%
del(X_AW_dangerEncy, AW_dangerEncy_inv_model, AW_dangerEncy_inv_model_result)

# %% [markdown]
# ### Inventory vs normal ```SW``` AND ```dangerEncy``` averaged over 2001-2017

# %%
indp_vars = SW_vars + HS_var
y_var = "inventory"

#################################################################
X_SW_dangerEncy = inv_2017_NPP_SW_heat_avg_normal[indp_vars]
X_SW_dangerEncy = sm.add_constant(X_SW_dangerEncy)
Y = np.log(inv_2017_NPP_SW_heat_avg_normal[y_var].astype(float))
SW_dangerEncy_inv_model = sm.OLS(Y, X_SW_dangerEncy)
SW_dangerEncy_inv_model_result = SW_dangerEncy_inv_model.fit()
SW_dangerEncy_inv_model_result.summary()

# %%
del(X_SW_dangerEncy, SW_dangerEncy_inv_model, SW_dangerEncy_inv_model_result)

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
RA = RA[["county_fips", "rangeland_fraction", "rangeland_acre"]]
RA.head(2)

# %%
herb = pd.read_pickle(data_dir_base + "Supriya/Nov30_HerbRatio/county_herb_ratio.sav")
herb = herb["county_herb_ratio"]
herb.dropna(how="any", inplace=True)

## Compute total herb area.
herb = rc.compute_herbRatio_totalArea(herb)
herb.reset_index(drop=True, inplace=True)
herb = herb.round(3)

herb = herb[["county_fips", "herb_avg", "herb_area_acr"]]
herb.head(2)

# %%
irr_hay = pd.read_pickle(reOrganized_dir + "irr_hay.sav")
irr_hay = irr_hay["irr_hay_perc"]

irr_hay.rename(columns={"value_irr": "irr_hay_area"}, inplace=True)

irr_hay = irr_hay[["county_fips", "irr_hay_area", "irr_hay_as_perc"]]

irr_hay.head(2)
# irr_hay = irr_hay[["county_fips", "irr_hay_as_perc"]]

# %%
feed_expense = USDA_data["feed_expense"]
feed_expense = feed_expense[["year", "county_fips", "feed_expense"]]

human_population = pd.read_pickle(reOrganized_dir + "human_population.sav")
human_population = human_population["human_population"]

slaughter_Q1 = pd.read_pickle(reOrganized_dir + "slaughter_Q1.sav")
slaughter_Q1 = slaughter_Q1["slaughter_Q1"]
slaughter_Q1.rename(columns={"cattle_on_feed_sale_4_slaughter": "slaughter"}, inplace=True)
slaughter_Q1 = slaughter_Q1[["year", "county_fips", "slaughter"]]
print ("max slaughter sale is [{}]".format(slaughter_Q1.slaughter.max()))

# %%
irr_hay.head(2)

# %%
controls = pd.merge(human_population, slaughter_Q1, on=["county_fips", "year"], how="outer")
controls = pd.merge(controls, feed_expense, on=["county_fips", "year"], how="outer")
controls = pd.merge(controls, irr_hay, on=["county_fips"], how="outer")
controls = pd.merge(controls, herb, on=["county_fips"], how="outer")

controls.head(2)

# %% [markdown]
# ## RA is already satisfies Pallavi condition

# %%
RA.head(2)

# %%
print (f"{controls.shape = }")
print (f"{len(controls.county_fips.unique()) = }")
print ()
controls = controls[controls.county_fips.isin(RA.county_fips)]

controls = pd.merge(controls, RA, on=["county_fips"], how="outer")

###########
########### NPP stuff are after 2001
###########
controls = controls[controls.year>=2001]
controls = controls[controls.year<=2017]
controls.reset_index(drop=True, inplace=True)

print (f"{controls.shape = }")
print (f"{len(controls.county_fips.unique()) = }")
controls.head(2)

# %%
inv_2017_NPP_SW_heat_avg_normal.head(2)

# %%
inventory_2017.head(2)

# %%
NPP_SW_heat_avg.head(2)

# %% [markdown]
# ## One more layer of filter according to 2017 inventory

# %%
print (f"{controls.shape = }")
print (f"{len(controls.county_fips.unique()) = }")
print ()

controls = controls[controls.county_fips.isin(inv_2017_NPP_SW_heat_avg_normal.county_fips.unique())]
controls.reset_index(drop=True, inplace=True)

print (f"{controls.shape = }")
print (f"{len(controls.county_fips.unique()) = }")

# %%
control_counties_allAvailable = controls.dropna(how="any", inplace=False).county_fips.unique()
counties_wNoControl = [x for x in NPP_SW_heat_avg.county_fips.unique() if not (x in control_counties_allAvailable)]
len(counties_wNoControl)

# %%
print (controls[~(controls.population.isna())].year.unique())
print (controls.dropna(how="any", inplace=False).shape)
print (len(controls.dropna(how="any", inplace=False).county_fips.unique()))

a = controls.dropna(how="any", inplace=False).county_fips.unique()
main_cnties = inv_2017_NPP_SW_heat_avg_normal.county_fips.unique()
len([x for x in main_cnties if x in a])

# %%
controls.head(2)

# %%

# %% [markdown]
# # Take Averages then Normalize

# %%
variable_controls = ["population", "slaughter", "feed_expense"]

constant_controls = ["herb_avg", "herb_area_acr",
                     "rangeland_fraction", "rangeland_acre", 
                     "irr_hay_area", "irr_hay_as_perc"]

constant_control_df = controls[["county_fips"] + constant_controls].drop_duplicates()
constant_control_df.reset_index(drop=True, inplace=True)
constant_control_df.head(2)

# %%
controls_avg = controls[["county_fips"] + variable_controls].groupby("county_fips").mean()
controls_avg.reset_index(drop=False, inplace=True)
controls_avg = controls_avg.round(3)
controls_avg.head(2)

# %%
controls_avg = pd.merge(controls_avg, constant_control_df, on=["county_fips"], how="left")
print (controls_avg.shape)
controls_avg.head(2)

# %%
controls_avg["irr_hay_as_perc_categ"] = controls_avg["irr_hay_as_perc"]

controls_avg.loc[(controls_avg.irr_hay_as_perc <= 6), "irr_hay_as_perc_categ"] = 0

controls_avg.loc[(controls_avg.irr_hay_as_perc > 6) & \
                 (controls_avg.irr_hay_as_perc <= 96), "irr_hay_as_perc_categ"] = 1

controls_avg.loc[(controls_avg.irr_hay_as_perc > 96), "irr_hay_as_perc_categ"] = 2

controls_avg.head(2)

# %%

# %%
normalize_cols = ["population", "slaughter", "feed_expense", 
                  "herb_avg", "herb_area_acr", 
                  "rangeland_fraction", "rangeland_acre", 
                  "irr_hay_area", "irr_hay_as_perc"]

controls_avg_normal = controls_avg.copy()

controls_avg_normal[normalize_cols] = (controls_avg_normal[normalize_cols] - \
                                       controls_avg_normal[normalize_cols].mean()) / \
                                       controls_avg_normal[normalize_cols].std(ddof=1)
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
print (f"{len(A)=}, {len(B)=}")

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
del(A)
L = inv_2017_NPP_SW_heat_avg_normal.county_fips.unique()
A = controls_avg_normal[controls_avg_normal.county_fips.isin(L)]
A.shape

unique_counties = A.county_fips.unique()
repeated_counties = []

for cnty in unique_counties:
    if len(A[A.county_fips == cnty])>1:
        repeated_counties = repeated_counties + [cnty]
        
repeated_counties


# %%
controls_avg_normal[controls_avg_normal.county_fips == "04009"]

# %%
print (inv_2017_NPP_SW_heat_avg_normal.shape)
all_df = pd.merge(inv_2017_NPP_SW_heat_avg_normal, controls_avg_normal, on=["county_fips"], how="outer")
print (all_df.shape)
all_df.head(2)

# %%
indp_vars = ["county_total_npp", "rangeland_acre"]
y_var = "inventory"

#################################################################
curr_all = all_df.copy()
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]]
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del(indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["county_total_npp", "rangeland_acre", "herb_area_acr"]
y_var = "inventory"

#################################################################
curr_all = all_df.copy()
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]]
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%

# %%
indp_vars = ["county_total_npp", "rangeland_acre", "herb_area_acr", "irr_hay_area"]
y_var = "inventory"

#################################################################
curr_all = all_df.copy()
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]]
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del(indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["county_total_npp", "rangeland_acre", "irr_hay_area"]
y_var = "inventory"

#################################################################
curr_all = all_df.copy()
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]]
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del(indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["county_total_npp", "population"]
y_var = "inventory"

#################################################################
curr_all = all_df.copy()
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]]
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del(indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["county_total_npp", "rangeland_acre", "population"]
y_var = "inventory"

#################################################################
curr_all = all_df.copy()
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]]
print (len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print (len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del(indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["county_total_npp", "rangeland_acre", "herb_area_acr", "population"]
y_var = "inventory"

#################################################################
curr_all = all_df.copy()
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]]
print (len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print (len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()


# %%

# %%
del(indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = ["county_total_npp", "rangeland_acre", "irr_hay_area", "population"]
y_var = "inventory"
#################################################################
curr_all = all_df.copy()
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]]
print (len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print (len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del(indp_vars, X, Y, model_, model_result, curr_all)

# %% [markdown]
# # SW and Controls

# %%
SW_vars

# %%
all_df.head(2)

# %%
indp_vars = SW_vars + ["rangeland_acre"]
y_var = "inventory"
#################################################################
curr_all = all_df.copy()
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]]
print (len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print (len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del(indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = SW_vars + ["rangeland_acre", "herb_area_acr"]
y_var = "inventory"

#################################################################
curr_all = all_df.copy()
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]]
print (len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print (len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del(indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = SW_vars + ["rangeland_acre", "herb_area_acr", "irr_hay_area"]
y_var = "inventory"

#################################################################
curr_all = all_df.copy()
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]]
print (len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print (len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del(indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = SW_vars + ["rangeland_acre", "herb_area_acr", "irr_hay_area", "population"]
y_var = "inventory"

#################################################################
curr_all = all_df.copy()
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]]
print (len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print (len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del(indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = SW_vars + ["rangeland_acre", "population"]
y_var = "inventory"
#################################################################
curr_all = all_df.copy()
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]]
print (len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print (len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del(indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = SW_vars + ["rangeland_acre", "herb_area_acr", "population"]
y_var = "inventory"
#################################################################
curr_all = all_df.copy()
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]]
print (len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print (len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del(indp_vars, X, Y, model_, model_result, curr_all)

# %%
all_df.head(2)

# %%

# %%

# %%

# %%

# %%
import scipy

# %%
x = np.array([0, 0, 100, 0, 0]) / 100
y = np.array([0, 30, 40, 30, 0]) / 100
z = np.array([30, 0, 40, 0, 30]) / 100

# %%
scipy.special.kl_div(x, y)

# %%
import sklearn
from sklearn import metrics
print (sklearn.metrics.mutual_info_score(x, y))
print (sklearn.metrics.mutual_info_score(x, z))

# %%
print (scipy.stats.entropy(x, y))
print (scipy.stats.entropy(x, z))

# %%

# %%
