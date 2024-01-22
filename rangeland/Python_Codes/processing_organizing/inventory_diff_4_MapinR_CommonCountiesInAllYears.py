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
# This is ```update of inventory_diff_4_MapinR.ipynb```
#
# The update is that here we keep all the counties that are present in all years. 
#
# This way, percentage changes might add up to zero.

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
SoI = ["Alabama", "Arizona", "Arkansas", "California", 
       "Colorado", "Florida", "Georgia", 
       "Idaho", "Illinois", "Iowa", 
       "Kansas", "Kentucky", "Louisiana", 
       "Mississippi", "Missouri", "Montana", 
       "Nebraska", "Nevada", "New Mexico", "North Dakota", 
       "Oklahoma", "Oregon", "South Dakota", 
       "Tennessee", "Texas", "Utah","Virginia", "Washington",
       "Wyoming"]

abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI_abb = []
for x in SoI:
    SoI_abb = SoI_abb + [abb_dict["full_2_abb"][x]]

# %%
len(SoI)

# %% [markdown]
# # Pallavi Condition

# %%
RA = pd.read_pickle(param_dir + "filtered_counties_RAsizePallavi.sav")
RA = RA["filtered_counties_29States"]
print (f"{len(RA.county_fips.unique()) = }")
print (f"{len(RA.state.unique()) = }")
RA.head(2)

# %%
USDA_data = pd.read_pickle(reOrganized_dir + "USDA_data.sav")
print ("-----------  reading USDA data  -----------")
cattle_inventory = USDA_data["cattle_inventory"]
cattle_inventory = cattle_inventory[["year", "cattle_cow_beef_inventory", "county_fips"]]
cattle_inventory.head(2)

# %%
county_fips = pd.read_pickle(reOrganized_dir + "county_fips.sav")
county_fips = county_fips["county_fips"]
county_fips = county_fips[["county_fips", "county_name", "state"]]

print (len(county_fips.state.unique()))
county_fips = county_fips[county_fips.state.isin(SoI_abb)]
county_fips.reset_index(drop=True, inplace=True)
print (len(county_fips.state.unique()))
county_fips.head(2)

# %%
cattle_inventory = pd.merge(cattle_inventory, county_fips, on = ["county_fips"], how = "left")
print (cattle_inventory.shape)
cattle_inventory.dropna(how='any', inplace=True)
print (cattle_inventory.shape)
len(cattle_inventory.state.unique())

cattle_inventory.head(2)

# %%
common_counties = set(cattle_inventory.county_fips.unique())

for a_year in cattle_inventory.year.unique():
    df = cattle_inventory[cattle_inventory.year == a_year].copy()
    curr_counties = set(df.county_fips.unique())
    common_counties = common_counties.intersection(curr_counties)

# %%
print (len(common_counties))
print (len(cattle_inventory.county_fips.unique()))

# %%
print (cattle_inventory.loc[cattle_inventory.year==1997, "cattle_cow_beef_inventory"].sum())
print (cattle_inventory.loc[cattle_inventory.year==2017, "cattle_cow_beef_inventory"].sum())

# %%
invent_commonCnty = cattle_inventory[cattle_inventory.county_fips.isin(list(common_counties))].copy()

# %%
print (f"{cattle_inventory.shape  = }")
print (f"{invent_commonCnty.shape = }")

# %%
invent_commonCnty_Pallavi = invent_commonCnty[invent_commonCnty.county_fips.isin(RA.county_fips.unique())]

# %%
print (len(invent_commonCnty.county_fips.unique()))
print (len(invent_commonCnty_Pallavi.county_fips.unique()))

# %%
invent_commonCnty.head(2)

# %%
invent_commonCnty_1997 = invent_commonCnty[invent_commonCnty.year == 1997].copy()
invent_commonCnty_2017 = invent_commonCnty[invent_commonCnty.year == 2017].copy()

invent_commonCnty_1997.reset_index(drop=True, inplace=True)
invent_commonCnty_2017.reset_index(drop=True, inplace=True)

# %%
invent_commonCnty_2017_total = invent_commonCnty_2017["cattle_cow_beef_inventory"].sum()
invent_commonCnty_1997_total = invent_commonCnty_1997["cattle_cow_beef_inventory"].sum()

# %%
inv_col = "cattle_cow_beef_inventory"
invent_commonCnty_2017.rename(columns={inv_col: "inven_2017"}, inplace=True)
invent_commonCnty_1997.rename(columns={inv_col: "inven_1997"}, inplace=True)

invent_commonCnty_2017.head(2)

# %%
invent_commonCnty_1997_2017 = pd.merge(invent_commonCnty_1997[["county_fips", "inven_1997"]], 
                                       invent_commonCnty_2017[["county_fips", "inven_2017"]], 
                                       on = ["county_fips"], how = "outer")

invent_commonCnty_1997_2017.head(2)

# %%
print (invent_commonCnty_1997_2017.shape)
print (invent_commonCnty_1997_2017.dropna(how='any', inplace=False).shape)

# %%
invent_commonCnty_1997_2017["inven_1997_share"] = invent_commonCnty_1997_2017["inven_1997"]/\
                                                                    invent_commonCnty_1997_total

invent_commonCnty_1997_2017["inven_2017_share"] = invent_commonCnty_1997_2017["inven_2017"]/\
                                                                    invent_commonCnty_2017_total

invent_commonCnty_1997_2017["share_change_1997_2017"] = invent_commonCnty_1997_2017["inven_2017_share"] - \
                                                        invent_commonCnty_1997_2017["inven_1997_share"]
invent_commonCnty_1997_2017.head(2)

# %%
invent_commonCnty_1997_2017['share_change_1997_2017'].sum()

# %% [markdown]
# # Ignore taking common counties.

# %%
inventory_1997 = cattle_inventory[cattle_inventory.year == 1997].copy()
inventory_2017 = cattle_inventory[cattle_inventory.year == 2017].copy()

inventory_1997.reset_index(drop=True, inplace=True)
inventory_2017.reset_index(drop=True, inplace=True)

inventory_2017_total = inventory_2017["cattle_cow_beef_inventory"].sum()
inventory_1997_total = inventory_1997["cattle_cow_beef_inventory"].sum()

# %%
inv_col = "cattle_cow_beef_inventory"
inventory_2017.rename(columns={inv_col: "inven_2017"}, inplace=True)
inventory_1997.rename(columns={inv_col: "inven_1997"}, inplace=True)

inventory_2017.head(2)

# %%
inventory_1997.head(2)

# %%
inventory_1997_2017 = pd.merge(inventory_1997[["county_fips", "inven_1997"]], 
                               inventory_2017[["county_fips", "inven_2017"]], 
                               on = ["county_fips"], how = "outer")

inventory_1997_2017.head(2)

# %%
inventory_1997_2017["inven_1997_share"] = inventory_1997_2017["inven_1997"]/\
                                                                    inventory_1997_total

inventory_1997_2017["inven_2017_share"] = inventory_1997_2017["inven_2017"]/\
                                                                    inventory_2017_total

inventory_1997_2017["share_change_1997_2017"] = inventory_1997_2017["inven_2017_share"] - \
                                                        invent_commonCnty_1997_2017["inven_1997_share"]
inventory_1997_2017.head(2)

# %%
inventory_1997_2017["share_change_1997_2017"].sum()

# %%
inventory_1997_2017.shape

# %%
inventory_1997_2017.dropna(how='any', inplace=False).shape

# %%

# %% [markdown]
# # For Kirti's panel

# %%
census_years = sorted(cattle_inventory.year.unique())[::-1]
all_df_ = county_fips.copy()
inv_col = "cattle_cow_beef_inventory"
while census_years:
    small_yr = census_years.pop()
    for yr in census_years:
        cattle_inventory_smYr = cattle_inventory[cattle_inventory.year == small_yr].copy()
        cattle_inventory_smYr.reset_index(drop=True, inplace=True)
        n_smallYr = "inven_" + str(small_yr)
        cattle_inventory_smYr.rename(columns={inv_col: n_smallYr}, inplace=True)

        cattle_inventory_Yr = cattle_inventory[cattle_inventory.year == yr].copy()
        cattle_inventory_Yr.reset_index(drop=True, inplace=True)
        n_Yr = "inven_" + str(yr)
        cattle_inventory_Yr.rename(columns={inv_col: n_Yr}, inplace=True)
        
        # pick common counties
        common_cnty = set(cattle_inventory_smYr["county_fips"]).intersection(\
                                                    set(cattle_inventory_Yr["county_fips"]))
        common_cnty = list(common_cnty)
        cattle_inventory_smYr = cattle_inventory_smYr[cattle_inventory_smYr.county_fips.isin(common_cnty)]
        cattle_inventory_Yr = cattle_inventory_Yr[cattle_inventory_Yr.county_fips.isin(common_cnty)]
        
        invent_smYr_Yr = pd.merge(cattle_inventory_smYr[["county_fips", n_smallYr]], 
                                  cattle_inventory_Yr[["county_fips", n_Yr]], 
                                  on = ["county_fips"], how = "outer")
        
        # invent_smYr_Yr.dropna(how='any', inplace=True)
        invent_smYr_Yr.reset_index(drop=True, inplace=True)
        invent_smYr_Yr.dropna(how="any",inplace=True) # sanity check
        
        change_colN = "inv_change" + str(small_yr) + "to" + str(yr)
        invent_smYr_Yr[change_colN] = invent_smYr_Yr[n_Yr] - invent_smYr_Yr[n_smallYr]
        
        change_colN_asPerc = change_colN + "_asPerc"
        invent_smYr_Yr[change_colN_asPerc] = 100 * invent_smYr_Yr[change_colN] / invent_smYr_Yr[n_smallYr]
        invent_smYr_Yr[change_colN_asPerc] = invent_smYr_Yr[change_colN_asPerc].round(2)

        needed_cols = ["county_fips", change_colN, change_colN_asPerc]
        all_df_ = pd.merge(all_df_, invent_smYr_Yr[needed_cols], on=["county_fips"], how="outer")
        
# add individual years inventory.
census_years = sorted(cattle_inventory.year.unique())[::-1]
for yr in census_years:
    cattle_inventory_Yr = cattle_inventory[cattle_inventory.year == yr].copy()
    cattle_inventory_Yr.reset_index(drop=True, inplace=True)
    n_Yr = "inven_" + str(yr)
    cattle_inventory_Yr.rename(columns={inv_col: n_Yr}, inplace=True)
    all_df_ = pd.merge(all_df_, cattle_inventory_Yr[["county_fips", n_Yr]], on=["county_fips"], how="outer")

# %%
out_name = data_dir_base + "data_4_plot/" + "inventory_AbsChange_4panel_commonCounties.csv"
all_df_.to_csv(out_name, index=False)

# %% [markdown]
# # All for panel in terms of national shares

# %%
census_years = sorted(cattle_inventory.year.unique())[::-1]
all_df_ = county_fips.copy()
inv_col = "cattle_cow_beef_inventory"
while census_years:
    small_yr = census_years.pop()
    for yr in census_years:
        inventory_smYr = cattle_inventory[cattle_inventory.year == small_yr].copy()
        inventory_smYr.reset_index(drop=True, inplace=True)
        n_smallYr = "inven_" + str(small_yr)
        inventory_smYr.rename(columns={inv_col: n_smallYr}, inplace=True)
    
        inventory_Yr = cattle_inventory[cattle_inventory.year == yr].copy()
        inventory_Yr.reset_index(drop=True, inplace=True)
        n_Yr = "inven_" + str(yr)
        inventory_Yr.rename(columns={inv_col: n_Yr}, inplace=True)
        
        # pick common counties
        common_cnty = set(inventory_Yr.county_fips.unique()).intersection(\
                                    set(inventory_smYr.county_fips.unique()))

        common_cnty = list(common_cnty)
        
        inventory_Yr = inventory_Yr[inventory_Yr.county_fips.isin(common_cnty)]
        inventory_smYr = inventory_smYr[inventory_smYr.county_fips.isin(common_cnty)]
        
        # Compute each county's share:
        inven_smYr_total = inventory_smYr[n_smallYr].sum()
        inven_Yr_total = inventory_Yr[n_Yr].sum()

        n_smallYr_shareCol = "inv_" + str(small_yr) + "_asPercShare"
        inventory_smYr[n_smallYr_shareCol] = 100 *(inventory_smYr[n_smallYr] / inven_smYr_total)        
        
        n_Yr_shareCol = "inv_" + str(yr) + "_asPercShare"
        inventory_Yr[n_Yr_shareCol] = 100 * (inventory_Yr[n_Yr] / inven_Yr_total)

        invent_asPerc = pd.merge(
            inventory_smYr[[n_smallYr_shareCol, "county_fips"]],
            inventory_Yr[[n_Yr_shareCol, "county_fips"]],
            on=["county_fips"],
            how="outer",
        )

        # invent_asPerc.dropna(how="any", inplace=True)

        diff_colName = "change_" + str(small_yr) + "_" + str(yr) + "_asPercShare"
        invent_asPerc[diff_colName] = (
            invent_asPerc[n_Yr_shareCol] - invent_asPerc[n_smallYr_shareCol]
        )

        needed_cols = ["county_fips", diff_colName]
        all_df_ = pd.merge(
            all_df_, invent_asPerc[needed_cols], on=["county_fips"], how="outer"
        )

census_years = sorted(cattle_inventory.year.unique())[::-1]
for yr in census_years:
    inventory_smYr = cattle_inventory[cattle_inventory.year == yr].copy()
    inventory_smYr.reset_index(drop=True, inplace=True)
    smallYr_inv_NewcolName = "inven_" + str(yr)
    inventory_smYr.rename(
        columns={inv_col: smallYr_inv_NewcolName}, inplace=True
    )
    inven_smYr_total = inventory_smYr[smallYr_inv_NewcolName].sum()

    smallYr_inv_NewcolName_shareCol = "inv_" + str(yr) + "_asPercShare"
    inventory_smYr[smallYr_inv_NewcolName_shareCol] = 100 * (
        inventory_smYr[smallYr_inv_NewcolName] / inven_smYr_total
    )
    inventory_smYr = inventory_smYr[["county_fips", smallYr_inv_NewcolName_shareCol]]
    
    all_df_ = pd.merge(all_df_, inventory_smYr, on=["county_fips"], how="outer")

all_df_.head(2)

# %%

# %%
out_name = data_dir_base + "data_4_plot/" + "inventory_ShareChange_4panel_commonCounties.csv"
all_df_.to_csv(out_name, index=False)

# %%

# %%

# %%
