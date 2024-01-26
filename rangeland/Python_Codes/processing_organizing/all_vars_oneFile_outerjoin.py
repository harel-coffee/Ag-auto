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
# **Jan 26, 2024**
#
# The Jan 24, 2024 models (long run avg of NPP for 2017 inventory modeling) was a lesson to put all variables in one file and just take outer join of everything.
#
# **Forgotten lesson** Keep everything: ***all states, not just 25***

# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

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

# %% [markdown]
# ## Inventory

# %%
USDA_data = pd.read_pickle(reOrganized_dir + "USDA_data.sav")
inventory = USDA_data["cattle_inventory"]
inventory.rename(columns={"cattle_cow_beef_inventory": "inventory"}, inplace=True)

print(f"{inventory.data_item.unique() = }")
print(f"{inventory.commodity.unique() = }")
print()
print(f"{len(inventory.state.unique())= }")

inventory = inventory[["year", "county_fips", "inventory"]]
inventory.head(2)

# %% [markdown]
# ### RA
#
# We need RA to convert unit NPP to total NPP.

# %%
RA = pd.read_csv(reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv")
RA.rename(columns={"fips_id": "county_fips"}, inplace=True)
RA = rc.correct_Mins_county_6digitFIPS(df=RA, col_ = "county_fips")
print (f"{len(RA.county_fips.unique()) = }")
RA.reset_index(drop=True, inplace=True)
RA.head(2)

# %%
RA_Pallavi = pd.read_pickle(param_dir + "filtered_counties_RAsizePallavi.sav")
RA_Pallavi = RA_Pallavi["filtered_counties_29States"]
print (f"{len(RA_Pallavi.county_fips.unique()) = }")
print (f"{len(RA_Pallavi.state.unique()) = }")


Pallavi_counties = list(RA_Pallavi.county_fips.unique())
RA_Pallavi.head(2)

# %%
cty_yr_npp = pd.read_csv(reOrganized_dir + "county_annual_GPP_NPP_productivity.csv")

cty_yr_npp.rename(columns={"county" : "county_fips",
                           "MODIS_NPP" : "unit_npp"}, inplace=True)

cty_yr_npp = rc.correct_Mins_county_6digitFIPS(df=cty_yr_npp, col_ = "county_fips")

cty_yr_npp = cty_yr_npp[["year", "county_fips", "unit_npp"]]

# Some counties do not have unit NPPs
cty_yr_npp.dropna(subset=["unit_npp"], inplace=True)
cty_yr_npp.reset_index(drop=True, inplace=True)

cty_yr_npp.head(2)

# %%
cty_yr_npp = pd.merge(cty_yr_npp, RA[["county_fips", "rangeland_acre"]], 
                      on=["county_fips"], how="left")

cty_yr_npp = rc.covert_unitNPP_2_total(NPP_df = cty_yr_npp, 
                                       npp_unit_col_ = "unit_npp", 
                                       acr_area_col_ = "rangeland_acre", 
                                       npp_area_col_ = "county_total_npp")

cty_yr_npp.head(2)

# %% [markdown]
# ### Weather

# %%
filename = reOrganized_dir + "county_annual_avg_Tavg.sav"
cnty_yr_avg_Tavg = pd.read_pickle(filename)
cnty_yr_avg_Tavg = cnty_yr_avg_Tavg["annual_temp"]

cnty_yr_avg_Tavg.reset_index(drop=True, inplace=True)
cnty_yr_avg_Tavg.head(2)

# %%
filename = reOrganized_dir + "county_seasonal_temp_ppt_weighted.sav"
SW = pd.read_pickle(filename)
SW = SW["seasonal"]
SW = pd.merge(SW, county_fips,  on=["county_fips"], how="left")

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
    
    
SW["yr_countyMean_total_precip"] = SW[seasonal_precip_vars].sum(axis=1)
# SW["yr_countyMean_avg_Tavg"]   = SW[seasonal_temp_vars].sum(axis=1)
# SW["yr_countyMean_avg_Tavg"]   = SW["yr_countyMean_avg_Tavg"]/4
SW = pd.merge(SW, cnty_yr_avg_Tavg, on=["county_fips", "year"], how="outer")
del(cnty_yr_avg_Tavg)
SW = SW.round(3)
SW.head(2)

# %%
cnty_grid_mean_idx = pd.read_csv(Min_data_base + "county_gridmet_mean_indices.csv")
cnty_grid_mean_idx.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
cnty_grid_mean_idx.rename(columns={"county": "county_fips"}, inplace=True)
cnty_grid_mean_idx = rc.correct_Mins_county_6digitFIPS(df=cnty_grid_mean_idx, col_="county_fips")

cnty_grid_mean_idx = cnty_grid_mean_idx[["year", "month", "county_fips",
                                         "normal", "alert", "danger", "emergency"]]

for a_col in ["normal", "alert", "danger", "emergency"]:
    cnty_grid_mean_idx[a_col] = cnty_grid_mean_idx[a_col].astype(int)
cnty_grid_mean_idx.reset_index(drop=True, inplace=True)
cnty_grid_mean_idx.head(2)

# %% [markdown]
# # Others (Controls)
#
#  - herb ratio
#  - irrigated hay
#  - feed expense
#  - population
#  - slaughter
#  
#  ### Herb

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

# %% [markdown]
# ### irrigated hay
#
# **Need to find 2012 and fill in some of those (D)/missing stuff**

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
AgLand = USDA_data["AgLand"]
wetLand_area = USDA_data["wetLand_area"]
FarmOperation = USDA_data["FarmOperation"]

# %%
# %who

# %%

# %%
import pickle
from datetime import datetime

filename = reOrganized_dir + "all_data_forOuterJoin.sav"

export_ = {"AgLand": AgLand,
           "FarmOperation": FarmOperation,
           "Pallavi_counties" : Pallavi_counties,
           "RA" : RA,
           "RA_Pallavi" :RA_Pallavi,
           "SW" : SW,
           "SoI" : SoI,
           "SoI_abb" : SoI_abb,
           "abb_dict" : abb_dict,
           "heat" : cnty_grid_mean_idx,
           "county_fips" : county_fips,
           "npp" : cty_yr_npp,
           "feed_expense": feed_expense,
           "herb" : herb, 
           "irr_hay" : irr_hay,
           "human_population" : human_population,
           "slaughter_Q1" : slaughter_Q1,
           "wetLand_area": wetLand_area,
           "cattle_inventory" : inventory,
           "source_code" : "all_vars_oneFile_outerjoin",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%

# %%

# %%
