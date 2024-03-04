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
# ### State level:
#
# - [state_slaughter](https://quickstats.nass.usda.gov/#79E47847-EA4F-33E4-8665-5DBEC5AB1947)
#
# - [state_feed_cost](https://quickstats.nass.usda.gov/#333604E3-E7AA-3207-A5CA-E7D093D656C5)
#
# - [state_wetLand_area](https://quickstats.nass.usda.gov/#D00341ED-2A09-3F5E-85BC-4F36827B0EDF)
#
# - [state_AgLand](https://quickstats.nass.usda.gov/#6C3CEC1E-7829-336B-B3BD-7486E5A2C92F)
#
# - [state_FarmOperation](https://quickstats.nass.usda.gov/#212EA12D-A220-3650-A70E-C30A2317B1D7)
#
#
#
#
# ### County level:
#
# - [Feed expense by county, 1997-2017:](https://quickstats.nass.usda.gov/#EF899E9D-F162-3655-89D9-5C423132E97F)
#  
# - [Acres enrolled in Conservation Reserve, Wetlands Reserve, Farmable Wetlands, or Conservation Reserve Enhancement Programs, 1997-2017](https://quickstats.nass.usda.gov/#3A734A89-9829-3674-AFC6-C4764DF7B728)
#
# - [Number of farm operations 1997-2017](https://quickstats.nass.usda.gov/#7310AC8E-D9CF-3BD9-8DC7-A4EF053FC56E) 
#
# - [Irrigated acres and total land in farms by county, 1997-2017](https://quickstats.nass.usda.gov/#B2688D70-61AC-3E14-AA15-11882355E95E)
#    
# __________________________________________________________________
#
#  - [Total Beef Cow inventory](https://quickstats.nass.usda.gov/#ADD6AB04-62EF-3DBF-9F83-5C23977C6DC7)
#  - [Inventory of Beef Cows](https://quickstats.nass.usda.gov/#B8E64DC0-E63E-31EA-8058-3E86C0DCB74E)

# %%
import shutup
shutup.please()

import pandas as pd
import numpy as np
import os

from datetime import datetime, date

import os, os.path, pickle, sys

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
param_dir = data_dir_base + "parameters/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %% [markdown]
# # First
# do annual_stateLevel_inventoryDeltas. This was a different notebook called ```annual_state_level_inventory_deltas.ipynb```

# %%
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI = abb_dict['SoI']
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

county_id_name_fips = abb_dict["county_fips"]
county_id_name_fips.head(2)

# %%
state_abb_state_fips = county_id_name_fips[["state", "state_fips", "EW"]]
state_abb_state_fips.drop_duplicates(inplace=True)
state_abb_state_fips.reset_index(drop=True, inplace=True)
print (state_abb_state_fips.shape)
state_abb_state_fips.head(2)

# %% [markdown]
# ### Read inventory
#
# ```Beef_Cows_fromCATINV.csv``` and ```Shannon_Beef_Cows_fromCATINV.csv``` are the same.

# %%
shannon_annual = pd.read_csv(reOrganized_dir + "Shannon_Beef_Cows_fromCATINV.csv")
shannon_annual = shannon_annual[shannon_annual.state.isin(list(state_abb_state_fips.state.unique()))]
print (shannon_annual.state.unique())
shannon_annual.reset_index(drop=True, inplace=True)
shannon_annual.head(2)

# %% [markdown]
# ## Compute deltas

# %%
# form deltas: inventort(t+1) - inventory (t)
inv_deltas = shannon_annual[list(shannon_annual.columns)[2:]].values - \
               shannon_annual[list(shannon_annual.columns)[1:-1]].values

delta_columns = [(str(x) + "_" + str(x-1)) for x in np.arange(1921, 2022)]
# form deltas dataframe
inventory_annual_deltas = pd.DataFrame(data = inv_deltas, columns=delta_columns)
inventory_annual_deltas["state"] = shannon_annual["state"]

# re-order columns
inventory_annual_deltas = inventory_annual_deltas[["state"]+delta_columns]
inventory_annual_deltas.head(2)

# %% [markdown]
# ## Compute Ratios

# %%
# form deltas: inventort(t+1) - inventory (t)
inv_ratios = shannon_annual[list(shannon_annual.columns)[2:]].values / \
               shannon_annual[list(shannon_annual.columns)[1:-1]].values

delta_columns = [(str(x) + "_" + str(x-1)) for x in np.arange(1921, 2022)]
# form ratios dataframe
inventory_annual_ratios = pd.DataFrame(data = inv_ratios, columns=delta_columns)
inventory_annual_ratios["state"] = shannon_annual["state"]

# re-order columns
inventory_annual_ratios = inventory_annual_ratios[["state"]+delta_columns]
inventory_annual_ratios.head(2)

# %%
shannon_annual.head(2)

# %% [markdown]
# ### convert to tall format

# %%
years = list(inventory_annual_deltas.columns[1:])
num_years = len(years)

inventory_deltas_tall=pd.DataFrame(data=None, index=range(num_years*len(inventory_annual_deltas.state.unique())), 
                                    columns=["state", "year", "inventory_delta"], 
                                    dtype=None, copy=False)

idx_ = 0 
for a_state in inventory_annual_deltas.state.unique():
    curr = inventory_annual_deltas[inventory_annual_deltas.state == a_state]
    inventory_deltas_tall.loc[idx_: idx_ + num_years - 1 , "inventory_delta"] = curr[years].values[0]
    inventory_deltas_tall.loc[idx_: idx_ + num_years - 1 , "state"] = a_state
    inventory_deltas_tall.loc[idx_: idx_ + num_years - 1 , "year"] = years
    idx_ = idx_ + num_years
    
inventory_deltas_tall.head(2)

# %%
years = list(inventory_annual_ratios.columns[1:])
num_years = len(years)

inventory_ratios_tall=pd.DataFrame(data=None, index=range(num_years*len(inventory_annual_ratios.state.unique())), 
                                     columns=["state", "year", "inventory_ratio"], 
                                     dtype=None, copy=False)

idx_ = 0 
for a_state in inventory_annual_ratios.state.unique():
    curr = inventory_annual_ratios[inventory_annual_ratios.state == a_state]
    inventory_ratios_tall.loc[idx_: idx_ + num_years - 1 , "inventory_ratio"] = curr[years].values[0]
    inventory_ratios_tall.loc[idx_: idx_ + num_years - 1 , "state"] = a_state
    inventory_ratios_tall.loc[idx_: idx_ + num_years - 1 , "year"] = years
    idx_ = idx_ + num_years
    
inventory_ratios_tall.head(2)

# %%
print (state_abb_state_fips.shape)
state_abb_state_fips.head(2)

# %%
inventory_annual_deltas = pd.merge(inventory_annual_deltas, state_abb_state_fips, on = ["state"], how = "left")
inventory_deltas_tall = pd.merge(inventory_deltas_tall, state_abb_state_fips, on = ["state"], how = "left")
inventory_annual_deltas.head(2)

# %%
inventory_annual_ratios = pd.merge(inventory_annual_ratios, state_abb_state_fips, on = ["state"], how = "left")
inventory_ratios_tall = pd.merge(inventory_ratios_tall, state_abb_state_fips, on = ["state"], how = "left")
inventory_annual_ratios.head(2)

# %%
# filename = reOrganized_dir + "Shannon_Beef_Cows_fromCATINV_deltas.sav"

# export_ = {"shannon_annual_inventory_deltas": inventory_annual_deltas, 
#            "shannon_annual_inventory_deltas_tall": inventory_deltas_tall,

#            "shannon_annual_inventory_ratios" : inventory_annual_ratios,
#            "shannon_annual_inventory_ratios_tall" : inventory_ratios_tall,
           
#            "source_code" : "annual_state_level_inventory_deltas",
#            "Author": "HN",
#            "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# pickle.dump(export_, open(filename, 'wb'))

# %%

# %% [markdown]
# # Second: USDA data

# %%
USDA_files = sorted([x for x in os.listdir(NASS_downloads_state) if x.endswith(".csv")])
USDA_files

# %%
AgLand = pd.read_csv(NASS_downloads_state + "state_AgLand.csv")
FarmOperation = pd.read_csv(NASS_downloads_state + "state_FarmOperation.csv")
feed_expense = pd.read_csv(NASS_downloads_state + "state_feed_expense.csv")
slaughter = pd.read_csv(NASS_downloads_state + "state_slaughter.csv")
wetLand_area = pd.read_csv(NASS_downloads_state + "state_wetLand_area.csv")

# %%
feed_expense.head(2)

# %%
feed_expense.Year.unique()

# %%
AgLand.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
wetLand_area.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
FarmOperation.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
feed_expense.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
slaughter.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True) 

sorted(list(feed_expense.columns))

# %%
print (f"{AgLand.shape = }")
print (f"{wetLand_area.shape = }")
print (f"{FarmOperation.shape = }")
print (f"{feed_expense.shape = }")
print (f"{slaughter.shape = }")

# %%
print ((feed_expense.columns == AgLand.columns).all())
print ((feed_expense.columns == wetLand_area.columns).all())
print ((feed_expense.columns == FarmOperation.columns).all())
print ((feed_expense.columns == slaughter.columns).all())

# %%
print (AgLand.zip_code.unique())
print (wetLand_area.zip_code.unique())
print (feed_expense.zip_code.unique())
print (FarmOperation.zip_code.unique())
print (slaughter.zip_code.unique())
print ()
print (AgLand.week_ending.unique())
print (wetLand_area.week_ending.unique())
print (feed_expense.week_ending.unique())
print (FarmOperation.week_ending.unique())
print (slaughter.week_ending.unique())

# %%
print (AgLand.watershed.unique())
print (wetLand_area.watershed.unique())
print (feed_expense.watershed.unique())
print (FarmOperation.watershed.unique())
print (slaughter.watershed.unique())
print ()
print (AgLand.domain_category.unique())
print (wetLand_area.domain_category.unique())
print (feed_expense.domain_category.unique())
print (FarmOperation.domain_category.unique())
print (slaughter.domain_category.unique())

# %%
FarmOperation.state_ansi.unique()

# %%
print (AgLand.domain.unique())
print (wetLand_area.domain.unique())
print (feed_expense.domain.unique())
print (FarmOperation.domain.unique())
print (slaughter.domain.unique())
print ()
print (AgLand.watershed_code.unique())
print (wetLand_area.watershed_code.unique())
print (feed_expense.watershed_code.unique())
print (FarmOperation.watershed_code.unique())
print (slaughter.watershed_code.unique())

# %%
print (AgLand.watershed.unique())
print (wetLand_area.watershed.unique())
print (feed_expense.watershed.unique())
print (FarmOperation.watershed.unique())
print (slaughter.watershed.unique())

# %%
print (AgLand.region.unique())
print (wetLand_area.region.unique())
print (feed_expense.region.unique())
print (FarmOperation.region.unique())
print (slaughter.region.unique())

# %%
print (AgLand.program.unique())
print (wetLand_area.program.unique())
print (feed_expense.program.unique())
print (FarmOperation.program.unique())
print (slaughter.program.unique())

# %%
print (AgLand.period.unique())
print (wetLand_area.period.unique())
print (feed_expense.period.unique())
print (FarmOperation.period.unique())
print (slaughter.period.unique())
print ()
print (AgLand.geo_level.unique())
print (wetLand_area.geo_level.unique())
print (feed_expense.geo_level.unique())
print (FarmOperation.geo_level.unique())
print (slaughter.geo_level.unique())

# %%
print (AgLand.data_item.unique())
print (wetLand_area.data_item.unique())
print (feed_expense.data_item.unique())
print (FarmOperation.data_item.unique())
print (slaughter.data_item.unique())

# %%
AgLand.columns

# %%
print (AgLand.county.unique())
print (wetLand_area.county.unique())
print (feed_expense.county.unique())
print (FarmOperation.county.unique())
print (slaughter.county.unique())

# %%
print (AgLand.county_ansi.unique())
print (wetLand_area.county_ansi.unique())
print (feed_expense.county_ansi.unique())
print (FarmOperation.county_ansi.unique())
print (slaughter.county_ansi.unique())

# %%
print (AgLand.ag_district_code.unique())
print (wetLand_area.ag_district_code.unique())
print (feed_expense.ag_district_code.unique())
print (FarmOperation.ag_district_code.unique())
print (slaughter.ag_district_code.unique())

print (AgLand.ag_district.unique())
print (wetLand_area.ag_district.unique())
print (feed_expense.ag_district.unique())
print (FarmOperation.ag_district.unique())
print (slaughter.ag_district.unique())

# %%
bad_cols  = ["watershed", "watershed_code", 
             "domain", "domain_category", "ag_district", "ag_district_code",
             "region", "period", "county", "county_ansi",
             "week_ending", "zip_code", "program", "geo_level"]

meta_cols = ["state", "state_ansi"]

# %%
FarmOperation["state_ansi"] = FarmOperation["state_ansi"].astype('int32')
FarmOperation["state_ansi"] = FarmOperation["state_ansi"].astype('str')
FarmOperation.state = FarmOperation.state.str.title()
FarmOperation[["state", "state_ansi"]].head(5)

# %%
for idx in FarmOperation.index:
    if len(FarmOperation.loc[idx, "state_ansi"]) == 1:
        FarmOperation.loc[idx, "state_ansi"] = "0" + FarmOperation.loc[idx, "state_ansi"]

# %%
FarmOperation[["state", "state_ansi"]].head(5)

# %%
meta_DF = FarmOperation[meta_cols].copy()
meta_DF.head(2)

# %% [markdown]
# # Alaska 
# has problem with ansi's

# %%
print (f"{meta_DF.shape = }")
print (f"{meta_DF.drop_duplicates().shape = }")

# %%
meta_DF.drop_duplicates(inplace=True)
meta_DF.head(2)

# %%
meta_DF.to_csv(reOrganized_dir + "state_USDA_NASS_Census_metadata.csv", index=False)

# %%
AgLand.head(2)

# %%
AgLand.drop(bad_cols, axis="columns", inplace=True)
wetLand_area.drop(bad_cols, axis="columns", inplace=True)
feed_expense.drop(bad_cols, axis="columns", inplace=True)
FarmOperation.drop(bad_cols, axis="columns", inplace=True)
slaughter.drop(bad_cols, axis="columns", inplace=True)

# %%
AgLand.head(2)

# %%
feed_expense[(feed_expense.state=="CALIFORNIA")]

# %%
AgLand["state_ansi"] = AgLand["state_ansi"].astype('int32')
AgLand["state_ansi"] = AgLand["state_ansi"].astype('str')
AgLand.state = AgLand.state.str.title()

for idx in AgLand.index:
    if len(AgLand.loc[idx, "state_ansi"]) == 1:
        AgLand.loc[idx, "state_ansi"] = "0" + AgLand.loc[idx, "state_ansi"]

AgLand[["state", "state_ansi"]].head(5)

# %%
wetLand_area["state_ansi"] = wetLand_area["state_ansi"].astype('int32')
wetLand_area["state_ansi"] = wetLand_area["state_ansi"].astype('str')
wetLand_area.state = wetLand_area.state.str.title()


for idx in wetLand_area.index:
    if len(wetLand_area.loc[idx, "state_ansi"]) == 1:
        wetLand_area.loc[idx, "state_ansi"] = "0" + wetLand_area.loc[idx, "state_ansi"]

wetLand_area[["state", "state_ansi"]].head(5)

# %%
feed_expense["state_ansi"] = feed_expense["state_ansi"].astype('int32')
feed_expense["state_ansi"] = feed_expense["state_ansi"].astype('str')

feed_expense.state = feed_expense.state.str.title()

for idx in feed_expense.index:
    if len(feed_expense.loc[idx, "state_ansi"]) == 1:
        feed_expense.loc[idx, "state_ansi"] = "0" + feed_expense.loc[idx, "state_ansi"]

feed_expense[["state", "state_ansi"]].head(5)

# %%
slaughter["state_ansi"] = slaughter["state_ansi"].astype('int32')
slaughter["state_ansi"] = slaughter["state_ansi"].astype('str')
slaughter.state = slaughter.state.str.title()

for idx in slaughter.index:
    if len(slaughter.loc[idx, "state_ansi"]) == 1:
        slaughter.loc[idx, "state_ansi"] = "0" + slaughter.loc[idx, "state_ansi"]

slaughter[["state", "state_ansi"]].head(5)

# %%
slaughter.head(2)

# %%
feed_expense[(feed_expense.state=="Alabama")]

# %%
print (slaughter.data_item.unique())
slaughter.head(2)

# %%
feed_expense.rename(columns={"value": "feed_expense", "cv_(%)": "feed_expense_cv_(%)"}, inplace=True)

wetLand_area.rename(columns={"value": "CRP_wetLand_acr", "cv_(%)": "CRP_wetLand_acr_cv_(%)"}, inplace=True)

slaughter.rename(columns={"value": "sale_4_slaughter_head", 
                          "cv_(%)": "sale_4_slaughter_head_cv_(%)"},
                        inplace=True)

FarmOperation.rename(columns={"state_ansi": "state_fips",
                              "value" : "number_of_farm_operation",
                              "cv_(%)" : "number_of_farm_operation_cv_(%)"}, inplace=True)


slaughter.head(2)

# %%
feed_expense.head(2)

# %%
wetLand_area.head(2)

# %%
import sys
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/Rangeland/Python_Codes/")
import rangeland_core as rc

# %%
wetLand_area.head(2)

# %%
print (wetLand_area.shape)
wetLand_area = rc.clean_census(df=wetLand_area, col_="CRP_wetLand_acr")
print (wetLand_area.shape)

# %%
print (slaughter.shape)
slaughter = rc.clean_census(df=slaughter, col_="sale_4_slaughter_head")
print (slaughter.shape)

# %%
print (feed_expense.shape)
feed_expense = rc.clean_census(df=feed_expense, col_="feed_expense")
print (feed_expense.shape)

# %%
print (FarmOperation.shape)
FarmOperation = rc.clean_census(df=FarmOperation, col_="number_of_farm_operation")
print (FarmOperation.shape)

# %% [markdown]
# ### Beef inventory

# %%
shannon_annual = pd.merge(shannon_annual, 
                          state_abb_state_fips[["state", "state_fips"]], 
                          on=["state"], how="left")

shannon_annual.head(2)

# %%
AgLand.rename(columns={"state_ansi": "state_fips", 
                       "value": "AgLand"}, inplace=True)

AgLand = AgLand[["year", "state_fips", "data_item", "AgLand", "cv_(%)"]]
AgLand.head(2)

# %%
wetLand_area.rename(columns={"state_ansi": "state_fips"}, inplace=True)
wetLand_area = wetLand_area[["year", "state_fips", "data_item", "crp_wetland_acr", "crp_wetland_acr_cv_(%)"]]
wetLand_area.head(2)

# %%
print (feed_expense.data_item.unique())
feed_expense.rename(columns={"state_ansi": "state_fips"}, inplace=True)
feed_expense = feed_expense[["year", "state_fips", "data_item", "feed_expense", "feed_expense_cv_(%)"]]
feed_expense.head(2)

# %%
print (FarmOperation.data_item.unique())
FarmOperation = FarmOperation[["year", "state_fips", "data_item", "number_of_farm_operation", \
                               "number_of_farm_operation_cv_(%)"]]
FarmOperation.head(2)

# %%
print (slaughter.commodity.unique())
print (slaughter.data_item.unique())
slaughter.rename(columns={"state_ansi": "state_fips"}, inplace=True)
slaughter = slaughter[["year", "state_fips", "data_item", "sale_4_slaughter_head", \
                       "sale_4_slaughter_head_cv_(%)"]]

slaughter.head(2)

# %%
shannon_annual.head(2)

# %%
Shannon_Beef_Cows_fromCATINV_tall = pd.read_pickle(reOrganized_dir + "Shannon_Beef_Cows_fromCATINV_tall.sav")
print (Shannon_Beef_Cows_fromCATINV_tall.keys())
Shannon_Beef_Cows_fromCATINV_tall = Shannon_Beef_Cows_fromCATINV_tall["CATINV_annual_tall"]
Shannon_Beef_Cows_fromCATINV_tall.head(2)

# %%
Shannon_Beef_Cows_fromCATINV_tall.head(2)

# %%
filename = reOrganized_dir + "state_USDA_ShannonCattle.sav"

export_ = {"AgLand": AgLand, 
           "wetLand_area": wetLand_area, 
           "feed_expense": feed_expense, 
           "FarmOperation": FarmOperation,
           "slaughter" : slaughter,
           "beef_fromCATINV_csv" : shannon_annual,
           "Shannon_beef_fromCATINV_deltas" : inventory_annual_deltas,
           "shannon_annual_inventory_deltas_tall": inventory_deltas_tall,
           
           "Shannon_beef_fromCATINV_ratios" : inventory_annual_ratios,
           "shannon_annual_inventory_ratios_tall" : inventory_ratios_tall,
           
           "Shannon_Beef_Cows_fromCATINV_tall" : Shannon_Beef_Cows_fromCATINV_tall,
           "source_code" : "00_state_clean_USDA_data_addFIPS",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%

# %%

# %%
