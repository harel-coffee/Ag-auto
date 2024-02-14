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
# - Feed expense by county, 1997-2017:
#
#    https://quickstats.nass.usda.gov/#EF899E9D-F162-3655-89D9-5C423132E97F
# __________________________________________________________________
#  
# - Acres enrolled in Conservation Reserve, Wetlands Reserve, Farmable Wetlands, or Conservation Reserve Enhancement Programs, 1997-2017:
#
#    https://quickstats.nass.usda.gov/#3A734A89-9829-3674-AFC6-C4764DF7B728
# __________________________________________________________________
#
# - Number of farm operations by county, 1997-2017:
#
#    https://quickstats.nass.usda.gov/#7310AC8E-D9CF-3BD9-8DC7-A4EF053FC56E
# __________________________________________________________________
#
# - Irrigated acres and total land in farms by county, 1997-2017:
#
#    https://quickstats.nass.usda.gov/#B2688D70-61AC-3E14-AA15-11882355E95E
#    
# __________________________________________________________________
#
#  - Total Beef Cow inventory: https://quickstats.nass.usda.gov/#ADD6AB04-62EF-3DBF-9F83-5C23977C6DC7
#  - Inventory of Beef Cows:  https://quickstats.nass.usda.gov/#B8E64DC0-E63E-31EA-8058-3E86C0DCB74E

# %%
import pandas as pd
import numpy as np
import os

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
USDA_data_dir = data_dir_base + "/NASS_downloads/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
USDA_files = [x for x in os.listdir(USDA_data_dir) if x.endswith(".csv")]
USDA_files

# %%
AgLand = pd.read_csv(USDA_data_dir + "AgLand.csv")
wetLand_area = pd.read_csv(USDA_data_dir + "wetLand_area.csv")
FarmOperation = pd.read_csv(USDA_data_dir + "FarmOperation.csv")
feed_expense = pd.read_csv(USDA_data_dir + "feed_expense.csv")
# totalBeefCowInv = pd.read_csv(USDA_data_dir + "totalBeefCowInv.csv")

## This is the bad cattle inventory. Where they suggested on Nov. 13 meeting that
## this might include dairy. At this point (Nov. 16), I am waiting for their response
## on the new queries I fetched. I am going w/ Q4; look at myBeef_vs_Shannon.ipynb

# totalBeefCowInv = pd.read_csv(USDA_data_dir + "totalBeefCowInv.csv")
# cattle_inventory = pd.read_csv(USDA_data_dir + "cattle_inventory.csv")

# Q4:
cattle_inventory = pd.read_csv(USDA_data_dir + "/cow_inventory_Qs/"+ "Q4_Updated_AZ_WA.csv")
cattle_inventory.head(2)

# %%
# totalBeefCowInv.head(3)

# %%
cattle_inventory.head(3)

# %%
# totalBeefCowInv.iloc[1]

# %%
cattle_inventory.iloc[1]

# %%
feed_expense.head(2)

# %%
feed_expense.Year.unique()

# %%
AgLand.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
wetLand_area.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
FarmOperation.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
feed_expense.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)

# totalBeefCowInv.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
cattle_inventory.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True) 

sorted(list(feed_expense.columns))

# %%
print (f"{AgLand.shape = }")
print (f"{wetLand_area.shape = }")
print (f"{FarmOperation.shape = }")
print (f"{feed_expense.shape = }")
print (f"{cattle_inventory.shape = }")

# %%
print ((feed_expense.columns == AgLand.columns).all())
print ((feed_expense.columns == wetLand_area.columns).all())
print ((feed_expense.columns == FarmOperation.columns).all())
print ((feed_expense.columns == cattle_inventory.columns).all())

# %%

# %%
print (AgLand.zip_code.unique())
print (wetLand_area.zip_code.unique())
print (feed_expense.zip_code.unique())
print (FarmOperation.zip_code.unique())
print (cattle_inventory.zip_code.unique())
print ()
print (AgLand.week_ending.unique())
print (wetLand_area.week_ending.unique())
print (feed_expense.week_ending.unique())
print (FarmOperation.week_ending.unique())
print (cattle_inventory.week_ending.unique())

# %%
print (AgLand.watershed.unique())
print (wetLand_area.watershed.unique())
print (feed_expense.watershed.unique())
print (FarmOperation.watershed.unique())
print (cattle_inventory.watershed.unique())
print ()
print (AgLand.domain_category.unique())
print (wetLand_area.domain_category.unique())
print (feed_expense.domain_category.unique())
print (FarmOperation.domain_category.unique())
print (cattle_inventory.domain_category.unique())

# %%
FarmOperation.state_ansi.unique()

# %%
print (AgLand.domain.unique())
print (wetLand_area.domain.unique())
print (feed_expense.domain.unique())
print (FarmOperation.domain.unique())
print (cattle_inventory.domain.unique())
print ()
print (AgLand.watershed_code.unique())
print (wetLand_area.watershed_code.unique())
print (feed_expense.watershed_code.unique())
print (FarmOperation.watershed_code.unique())
print (cattle_inventory.watershed_code.unique())

# %%
print (sorted(AgLand.ag_district_code.unique()))
print (sorted(wetLand_area.ag_district_code.unique()))
print (sorted(feed_expense.ag_district_code.unique()))
print (sorted(FarmOperation.ag_district_code.unique()))
print (sorted(cattle_inventory.ag_district_code.unique()))

# %%
print (AgLand.watershed.unique())
print (wetLand_area.watershed.unique())
print (feed_expense.watershed.unique())
print (FarmOperation.watershed.unique())
print (cattle_inventory.watershed.unique())

# %%
print (AgLand.region.unique())
print (wetLand_area.region.unique())
print (feed_expense.region.unique())
print (FarmOperation.region.unique())
print (cattle_inventory.region.unique())

# %%
print (AgLand.program.unique())
print (wetLand_area.program.unique())
print (feed_expense.program.unique())
print (FarmOperation.program.unique())
print (cattle_inventory.program.unique())

# %%
print (AgLand.period.unique())
print (wetLand_area.period.unique())
print (feed_expense.period.unique())
print (FarmOperation.period.unique())
print (cattle_inventory.period.unique())
print ()
print (AgLand.geo_level.unique())
print (wetLand_area.geo_level.unique())
print (feed_expense.geo_level.unique())
print (FarmOperation.geo_level.unique())
print (cattle_inventory.geo_level.unique())

# %%
print (AgLand.data_item.unique())
print (wetLand_area.data_item.unique())
print (feed_expense.data_item.unique())
print (FarmOperation.data_item.unique())
print (cattle_inventory.data_item.unique())

# %%
bad_cols  = ["watershed", "watershed_code", 
             "domain", "domain_category", 
             "region", "period",
             "week_ending", "zip_code", "program", "geo_level"]

meta_cols = ["state", "county", "county_ansi", "state_ansi", "ag_district_code"]

# %%
FarmOperation['county_ansi'].fillna(666, inplace=True)

FarmOperation["state_ansi"] = FarmOperation["state_ansi"].astype('int32')
FarmOperation["county_ansi"] = FarmOperation["county_ansi"].astype('int32')

FarmOperation["state_ansi"] = FarmOperation["state_ansi"].astype('str')
FarmOperation["county_ansi"] = FarmOperation["county_ansi"].astype('str')

FarmOperation.state = FarmOperation.state.str.title()
FarmOperation.county = FarmOperation.county.str.title()

# %%
FarmOperation[["state", "county", "state_ansi", "county_ansi"]].head(5)

# %%
for idx in FarmOperation.index:
    if len(FarmOperation.loc[idx, "state_ansi"]) == 1:
        FarmOperation.loc[idx, "state_ansi"] = "0" + FarmOperation.loc[idx, "state_ansi"]
        
    if len(FarmOperation.loc[idx, "county_ansi"]) == 1:
        FarmOperation.loc[idx, "county_ansi"] = "00" + FarmOperation.loc[idx, "county_ansi"]
    elif len(FarmOperation.loc[idx, "county_ansi"]) == 2:
        FarmOperation.loc[idx, "county_ansi"] = "0" + FarmOperation.loc[idx, "county_ansi"]

# %%
FarmOperation[["state", "county", "state_ansi", "county_ansi"]].head(5)

# %%
FarmOperation["county_fips"] = FarmOperation["state_ansi"] + FarmOperation["county_ansi"] 

# %%
meta_DF = FarmOperation[meta_cols].copy()
meta_DF.head(2)

# %% [markdown]
# # Alaska 
# has problem with ansi's

# %%
# meta_DF[meta_DF['county_ansi'].isnull()]

# values = {"county_ansi": 666}
# meta_DF.fillna(value=values, inplace=True)

# %%
# meta_DF["county_ansi"] = meta_DF["county_ansi"].astype(int)
meta_DF[meta_DF['county_ansi'].isnull()]

# %%
# meta_DF["county_ansi"] = meta_DF["county_ansi"].astype(int)

# %%
print (f"{meta_DF.shape = }")
print (f"{meta_DF.drop_duplicates().shape = }")

# %%
meta_DF.drop_duplicates(inplace=True)
meta_DF.head(2)

# %%
meta_DF.to_csv(reOrganized_dir + "USDA_NASS_Census_metadata.csv", index=False)

# %%
AgLand.head(2)

# %%
AgLand.drop(bad_cols, axis="columns", inplace=True)
wetLand_area.drop(bad_cols, axis="columns", inplace=True)
feed_expense.drop(bad_cols, axis="columns", inplace=True)
FarmOperation.drop(bad_cols, axis="columns", inplace=True)
cattle_inventory.drop(bad_cols, axis="columns", inplace=True)

# %%
AgLand.head(2)

# %%
feed_expense[(feed_expense.state=="CALIFORNIA") & (feed_expense.county=="SAN FRANCISCO")]

# %%
AgLand['county_ansi'].fillna(666, inplace=True)

AgLand["state_ansi"] = AgLand["state_ansi"].astype('int32')
AgLand["county_ansi"] = AgLand["county_ansi"].astype('int32')

AgLand["state_ansi"] = AgLand["state_ansi"].astype('str')
AgLand["county_ansi"] = AgLand["county_ansi"].astype('str')

AgLand.state = AgLand.state.str.title()
AgLand.county = AgLand.county.str.title()

for idx in AgLand.index:
    if len(AgLand.loc[idx, "state_ansi"]) == 1:
        AgLand.loc[idx, "state_ansi"] = "0" + AgLand.loc[idx, "state_ansi"]
        
    if len(AgLand.loc[idx, "county_ansi"]) == 1:
        AgLand.loc[idx, "county_ansi"] = "00" + AgLand.loc[idx, "county_ansi"]
    elif len(AgLand.loc[idx, "county_ansi"]) == 2:
        AgLand.loc[idx, "county_ansi"] = "0" + AgLand.loc[idx, "county_ansi"]

AgLand["county_fips"] = AgLand["state_ansi"] + AgLand["county_ansi"]
AgLand[["state", "county", "state_ansi", "county_ansi", "county_fips"]].head(5)

# %%
wetLand_area['county_ansi'].fillna(666, inplace=True)

wetLand_area["state_ansi"] = wetLand_area["state_ansi"].astype('int32')
wetLand_area["county_ansi"] = wetLand_area["county_ansi"].astype('int32')

wetLand_area["state_ansi"] = wetLand_area["state_ansi"].astype('str')
wetLand_area["county_ansi"] = wetLand_area["county_ansi"].astype('str')

wetLand_area.state = wetLand_area.state.str.title()
wetLand_area.county = wetLand_area.county.str.title()

for idx in wetLand_area.index:
    if len(wetLand_area.loc[idx, "state_ansi"]) == 1:
        wetLand_area.loc[idx, "state_ansi"] = "0" + wetLand_area.loc[idx, "state_ansi"]
        
    if len(wetLand_area.loc[idx, "county_ansi"]) == 1:
        wetLand_area.loc[idx, "county_ansi"] = "00" + wetLand_area.loc[idx, "county_ansi"]
    elif len(wetLand_area.loc[idx, "county_ansi"]) == 2:
        wetLand_area.loc[idx, "county_ansi"] = "0" + wetLand_area.loc[idx, "county_ansi"]

wetLand_area["county_fips"] = wetLand_area["state_ansi"] + wetLand_area["county_ansi"]
wetLand_area[["state", "county", "state_ansi", "county_ansi", "county_fips"]].head(5)

# %%
feed_expense['county_ansi'].fillna(666, inplace=True)

feed_expense["state_ansi"] = feed_expense["state_ansi"].astype('int32')
feed_expense["county_ansi"] = feed_expense["county_ansi"].astype('int32')

feed_expense["state_ansi"] = feed_expense["state_ansi"].astype('str')
feed_expense["county_ansi"] = feed_expense["county_ansi"].astype('str')

feed_expense.state = feed_expense.state.str.title()
feed_expense.county = feed_expense.county.str.title()

for idx in feed_expense.index:
    if len(feed_expense.loc[idx, "state_ansi"]) == 1:
        feed_expense.loc[idx, "state_ansi"] = "0" + feed_expense.loc[idx, "state_ansi"]
        
    if len(feed_expense.loc[idx, "county_ansi"]) == 1:
        feed_expense.loc[idx, "county_ansi"] = "00" + feed_expense.loc[idx, "county_ansi"]
    elif len(feed_expense.loc[idx, "county_ansi"]) == 2:
        feed_expense.loc[idx, "county_ansi"] = "0" + feed_expense.loc[idx, "county_ansi"]
        
        
feed_expense["county_fips"] = feed_expense["state_ansi"] + feed_expense["county_ansi"]
feed_expense[["state", "county", "state_ansi", "county_ansi", "county_fips"]].head(5)

# %%
# totalBeefCowInv.head(2)
cattle_inventory.head(2)

# %%
# totalBeefCowInv['county_ansi'].fillna(666, inplace=True)

# totalBeefCowInv["state_ansi"] = totalBeefCowInv["state_ansi"].astype('int32')
# totalBeefCowInv["county_ansi"] = totalBeefCowInv["county_ansi"].astype('int32')

# totalBeefCowInv["state_ansi"] = totalBeefCowInv["state_ansi"].astype('str')
# totalBeefCowInv["county_ansi"] = totalBeefCowInv["county_ansi"].astype('str')

# totalBeefCowInv.state = totalBeefCowInv.state.str.title()
# totalBeefCowInv.county = totalBeefCowInv.county.str.title()

# for idx in totalBeefCowInv.index:
#     if len(totalBeefCowInv.loc[idx, "state_ansi"]) == 1:
#         totalBeefCowInv.loc[idx, "state_ansi"] = "0" + totalBeefCowInv.loc[idx, "state_ansi"]
        
#     if len(totalBeefCowInv.loc[idx, "county_ansi"]) == 1:
#         totalBeefCowInv.loc[idx, "county_ansi"] = "00" + totalBeefCowInv.loc[idx, "county_ansi"]
#     elif len(totalBeefCowInv.loc[idx, "county_ansi"]) == 2:
#         totalBeefCowInv.loc[idx, "county_ansi"] = "0" + totalBeefCowInv.loc[idx, "county_ansi"]
        
        
# totalBeefCowInv["county_fips"] = totalBeefCowInv["state_ansi"] + totalBeefCowInv["county_ansi"]
# totalBeefCowInv[["state", "county", "state_ansi", "county_ansi", "county_fips"]].head(5)

cattle_inventory['county_ansi'].fillna(666, inplace=True)

cattle_inventory["state_ansi"] = cattle_inventory["state_ansi"].astype('int32')
cattle_inventory["county_ansi"] = cattle_inventory["county_ansi"].astype('int32')

cattle_inventory["state_ansi"] = cattle_inventory["state_ansi"].astype('str')
cattle_inventory["county_ansi"] = cattle_inventory["county_ansi"].astype('str')

cattle_inventory.state = cattle_inventory.state.str.title()
cattle_inventory.county = cattle_inventory.county.str.title()

for idx in cattle_inventory.index:
    if len(cattle_inventory.loc[idx, "state_ansi"]) == 1:
        cattle_inventory.loc[idx, "state_ansi"] = "0" + cattle_inventory.loc[idx, "state_ansi"]
        
    if len(cattle_inventory.loc[idx, "county_ansi"]) == 1:
        cattle_inventory.loc[idx, "county_ansi"] = "00" + cattle_inventory.loc[idx, "county_ansi"]
    elif len(cattle_inventory.loc[idx, "county_ansi"]) == 2:
        cattle_inventory.loc[idx, "county_ansi"] = "0" + cattle_inventory.loc[idx, "county_ansi"]
        
        
cattle_inventory["county_fips"] = cattle_inventory["state_ansi"] + cattle_inventory["county_ansi"]
cattle_inventory[["state", "county", "state_ansi", "county_ansi", "county_fips"]].head(5)

# %%
# totalBeefCowInv.head(2)
cattle_inventory.head(2)

# %%
feed_expense[(feed_expense.state=="Alabama") & (feed_expense.county=="Washington")]

# %%
# AgLand.drop(["county_ansi", "state_ansi", "ag_district_code"], axis="columns", inplace=True)
# wetLand_area.drop(["county_ansi", "state_ansi", "ag_district_code"], axis="columns", inplace=True)
# feed_expense.drop(["county_ansi", "state_ansi", "ag_district_code"], axis="columns", inplace=True)
# FarmOperation.drop(["county_ansi", "state_ansi", "ag_district_code"], axis="columns", inplace=True)

# %%
feed_expense.rename(columns={"value": "feed_expense", "cv_(%)": "feed_expense_cv_(%)"}, inplace=True)

wetLand_area.rename(columns={"value": "CRP_wetLand_acr", "cv_(%)": "CRP_wetLand_acr_cv_(%)"}, inplace=True)

cattle_inventory.rename(columns={"value": "cattle_cow_beef_inventory", 
                                 "cv_(%)": "cattle_cow_beef_inventory_cv_(%)"},
                        inplace=True)

cattle_inventory.head(2)

# %%
feed_expense.head(2)

# %%
wetLand_area.head(2)

# %%
import sys
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/Rangeland/Python_Codes/")
import rangeland_core as rc

# %%
print (wetLand_area.shape)
wetLand_area = rc.clean_census(df=wetLand_area, col_="CRP_wetLand_acr")
print (wetLand_area.shape)

# %%
print (cattle_inventory.shape)
cattle_inventory = rc.clean_census(df=cattle_inventory, col_="cattle_cow_beef_inventory")
print (cattle_inventory.shape)

# %%
print (feed_expense.shape)
feed_expense = rc.clean_census(df=feed_expense, col_="feed_expense")
print (feed_expense.shape)

# %%
# AgLand.to_csv(reOrganized_dir + "USDA_AgLand_cleaned_01.csv", index=False)
# wetLand_area.to_csv(reOrganized_dir  + "USDA_wetLand_area_cleaned_01.csv",  index=False)
# feed_expense.to_csv(reOrganized_dir  + "USDA_feed_expense_cleaned_01.csv",  index=False)
# FarmOperation.to_csv(reOrganized_dir + "USDA_FarmOperation_cleaned_01.csv", index=False)


import pickle
from datetime import datetime

filename = reOrganized_dir + "county_USDA_data.sav"

export_ = {"AgLand": AgLand, 
           "wetLand_area": wetLand_area, 
           "feed_expense": feed_expense, 
           "FarmOperation": FarmOperation,
           "cattle_inventory":cattle_inventory,
           "source_code" : "00_clean_USDA_data_addFIPS",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
feed_expense.county_fips[1]

# %%
len(cattle_inventory.county_fips.unique())

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# # Tonsor
# - CRP
# - Feed Expense
# - Population/county (**missing. need to contact Census Bureau**)
# - Percentage of irrigated acres
# - FarmOperation not needed. NASS guy had created this.

# %%
