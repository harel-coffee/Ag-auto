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
# - [Hay Prices. March 6, 2024.](https://quickstats.nass.usda.gov/#4CF5F365-9FA1-3248-A7E2-DDBAE1E247B2)
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
Mike_dir = data_dir_base + "Mike/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_fips = abb_dict["state_fips"]
state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
print(len(state_fips_SoI))
state_fips_SoI.head(2)

# %%

# %%
filename = reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
all_data_dict = pd.read_pickle(filename)
print (all_data_dict["Date"])
list(all_data_dict.keys())

# %%
all_df = all_data_dict["all_df_outerjoined"]
dummy_ = [x for x in list(all_df.columns) if "dummy" in x]
all_df.drop(columns=dummy_, inplace=True)
all_df.head(2)

# %%
all_df = rc.convert_lb_2_kg(df=all_df, 
                            matt_total_npp_col="total_matt_npp", 
                            new_col_name="metric_total_matt_npp")

# %%
all_df = rc.convert_lbperAcr_2_kg_in_sqM(df=all_df, 
                                         matt_unit_npp_col = "unit_matt_npp", 
                                         new_col_name = "metric_unit_matt_npp")

# %%

# %% [markdown]
# # Inventory
#
# Wrote it up using Shannon's CSV.

# %%
inventory = all_df[["year", "state_fips", "inventory"]].copy()
inventory = pd.merge(inventory, state_fips_SoI, how="left", on="state_fips")

inventory = inventory[inventory.state_fips.isin(list(state_fips_SoI.state_fips))]

inventory.dropna(subset=["inventory"], inplace=True)
inventory.reset_index(drop=True, inplace=True)

inventory.head(2)

# %%
print (inventory.inventory.mean())
print (inventory.inventory.std())

# %%
inventory.describe()

# %%
print (inventory.year.min())
print (inventory.year.max())
print (len(inventory.state_fips.unique()))

# %%
for a_state in inventory.state_full.unique():
    df = inventory[inventory.state_full == a_state].copy()
    print ("{}: {}, {}, {}".format(a_state, df.year.min(), df.year.max(), len(df.year)))

# %%
print (len(inventory))
print (29 * (2022-1919+1))

# %% [markdown]
# # NDVIs

# %%
# statefips_monthly_AVHRR_NDVI statefips_monthly_GIMMS_NDVI statefips_monthly_MODIS_NDVI
A = pd.read_csv(Min_data_dir_base + "statefips_monthly_GIMMS_NDVI.csv")

A.rename({"statefips90m": "state_fips",}, axis=1, inplace=True)
A['state_fips'] = A['state_fips'].astype("str").str.slice(start=1, stop=3)
A = A[A.state_fips.isin(list(state_fips_SoI.state_fips))].copy()

A = pd.merge(A, state_fips_SoI, on="state_fips", how="left")

print (A.year.min())
print (A.year.max())
A.head(2)

# %%
[x for x in all_df.columns if "max_ndvi_in_year" in x]

# %%
ndvi_col = "max_ndvi_in_year_avhrr"

ndvi = all_df[["year", "state_fips", ndvi_col ]].copy()
ndvi = pd.merge(ndvi, state_fips_SoI, how="left", on="state_fips")

ndvi = ndvi[ndvi.state_fips.isin(list(state_fips_SoI.state_fips))]

ndvi.dropna(subset=[ndvi_col], inplace=True)
ndvi.reset_index(drop=True, inplace=True)

print (f"{len(ndvi.state.unique()) = }")
print (f"{ndvi.year.min() = }")
print (f"{ndvi.year.max() = }")
print ()
print (f"{round(ndvi[ndvi_col].mean(), 3) = }")
print (f"{round(ndvi[ndvi_col].std(), 3) = }")

print (f"{ndvi.shape = }")
ndvi.head(2)

# %%
min_year, max_year = ndvi.year.min(), ndvi.year.max()

for a_state in ndvi.state_full.unique():
    df = ndvi[ndvi.state_full == a_state].copy()
    if (df.year.min() != min_year) or (df.year.max() != max_year):
        print ("{}: {}, {}, {}".format(a_state, df.year.min(), df.year.max(), len(df.year)))

# %%

# %%

# %%

# %% [markdown]
# # NPPs
#
# ### Matt NPP

# %%
A = pd.read_csv(Min_data_dir_base + "county_annual_productivity.csv")
print (A.year.max())
A.head(2)

# %%
# A = pd.read_csv(Min_data_dir_base + "/statefips_annual_productivity.csv")
# A.rename({"statefips90m": "state_fips",}, axis=1, inplace=True)
# A['state_fips'] = A['state_fips'].astype("str").str.slice(start=1, stop=3)
# A = A[A.state_fips.isin(list(state_fips_SoI.state_fips))].copy()

# A = pd.merge(A, state_fips_SoI, how="left", on="state_fips")
# print (f"{A.shape = }")
# A.head(2)
# for a_state in A.state_full.unique():
#     df = A[A.state_full == a_state].copy()
#     print ("{}: {}, {}, {}".format(a_state, df.year.min(), df.year.max(), len(df.year)))

# %%
[x for x in list(all_df.columns) if "npp" in x]

# %%
npp_col = "metric_total_matt_npp"
matt_npp = all_df[["year", "state_fips", npp_col ]].copy()
matt_npp = pd.merge(matt_npp, state_fips_SoI, how="left", on="state_fips")

matt_npp = matt_npp[matt_npp.state_fips.isin(list(state_fips_SoI.state_fips))]

matt_npp.dropna(subset=[npp_col], inplace=True)
matt_npp.reset_index(drop=True, inplace=True)

print (f"{len(matt_npp.state.unique()) = }")
print (f"{matt_npp.shape = }")
matt_npp.head(2)

# %%
print (matt_npp[npp_col].mean())
print (matt_npp[npp_col].std())

# %%
npp_col

# %%
for a_state in matt_npp.state_full.unique():
    df = matt_npp[matt_npp.state_full == a_state].copy()
    print ("{}: {}, {}, {}".format(a_state, df.year.min(), df.year.max(), len(df.year)))

# %%

# %% [markdown]
# ### MODIS-NPP

# %%
d_dir = "/Users/hn/Documents/01_research_data/RangeLand/"
MODIS_NPP = pd.read_csv(d_dir + "Data_large_notUsedYet/Min_data/statefips_annual_MODIS_NPP.csv")
MODIS_NPP.rename({"statefips90m": "state_fips",}, axis=1, inplace=True)
MODIS_NPP['state_fips'] = MODIS_NPP['state_fips'].astype("str").str.slice(start=1, stop=3)
MODIS_NPP = MODIS_NPP[MODIS_NPP.state_fips.isin(list(state_fips_SoI.state_fips))].copy()

MODIS_NPP = pd.merge(MODIS_NPP, state_fips_SoI, how="left", on="state_fips")
print (f"{MODIS_NPP.shape = }")
print (len(MODIS_NPP.state_full.unique()))
print ()
print (f'{round(MODIS_NPP["NPP"].mean(), 3) =}')
print (f'{round(MODIS_NPP["NPP"].std(), 3)  =}')
print (len(MODIS_NPP.state_fips))
print ()
MODIS_NPP.head(2)
for a_state in MODIS_NPP.state_full.unique():
    df = MODIS_NPP[MODIS_NPP.state_full == a_state].copy()
    print ("{}: {}, {}, {}".format(a_state, df.year.min(), df.year.max(), len(df.year)))

# %%

# %%

# %% [markdown]
# # RA
# Rangeland area is important to understand the filtering process: Kentucky and such.

# %%
state_RA_area = pd.read_pickle(reOrganized_dir + "state_RA_area.sav")
state_RA_area = state_RA_area["state_RA_area"]
state_RA_area.rename({"state_fip": "state_fips",}, axis=1, inplace=True)
state_RA_area = state_RA_area[state_RA_area.state_fips.isin(list(state_fips_SoI.state_fips))].copy()
state_RA_area.reset_index(drop=True, inplace=True)

state_RA_area = pd.merge(state_RA_area, state_fips_SoI, on="state_fips", how="left")
state_RA_area.head(2)

# %%
print (state_RA_area["rangeland_acre"].mean())
print (state_RA_area["rangeland_acre"].std())
len(state_RA_area.state_fips)

# %%
chosen_sts = ['Arizona', 'Nevada', 'Utah', 'Washington', "Kentucky"]
five_states_RA = state_RA_area[state_RA_area.state_full.isin(chosen_sts)].copy()
five_states_RA = five_states_RA.sort_values(by="state_full")
five_states_RA.reset_index(inplace=True, drop=True)
five_states_RA["rangeland_acre"] = five_states_RA["rangeland_acre"].astype(int)
five_states_RA

# %%
print (state_RA_area.shape)
list(five_states_RA["rangeland_acre"].values)

# %%
state_RA_area["rangeland_acre"] = state_RA_area["rangeland_acre"].astype(int)
state_RA_area.to_csv(data_dir_base + "for_paper/" + "States_on_Map.csv", index=False)

# %%
state_RA_area.head(2)

# %%

# %%

# %%

# %% [markdown]
# # State-level Hay Price
#
# There is a link on the first cell that gives the hay price on Monthly and Marketing-Year levels.
# It is possible that hay prices came from Mike. Lets see.
#
# Upon checking codes, seems I have used the priced from the link in cell 1 here and the file that Mike had sent
# is not used. Why? I cannot recall.
#
# The file that Mike had sent was called "Feed Grains Yearbook Tables-All Years.xlsx" which I cannot find in the emails.

# %%

# %%

# %%
hay_price_inModels = all_df[["year", "state_fips", "hay_price_at_1982"]].copy()
hay_price_inModels.dropna(how="any", inplace=True)
hay_price_inModels.reset_index(drop=True, inplace=True)
hay_price_inModels.head(5)

# %%
hay_price_inModels = hay_price_inModels[hay_price_inModels.state_fips.isin(state_fips_SoI.state_fips)]
hay_price_inModels = pd.merge(hay_price_inModels, state_fips_SoI, on="state_fips", how="left")
hay_price_inModels.head(2)

# %%
print (f"{hay_price_inModels.year.min() = }")
print (f"{hay_price_inModels.year.max() = }")

# %%
for a_state in hay_price_inModels.state_full.unique():
    df = hay_price_inModels[hay_price_inModels.state_full == a_state].copy()
    print ("{}: {}, {}, {}".format(a_state, df.year.min(), df.year.max(), len(df.year)))

# %%
unit_matt_npp = all_df[["year", "unit_matt_npp", "state_fips"]].copy()
unit_matt_npp.dropna(how="any", inplace=True)
unit_matt_npp.year.max()

# %%

# %%

# %%
# filename = reOrganized_dir + "old/" + "beef_hay_cost_fromMikeLinkandFile.sav"
# Mike_beef_hay_cost = pd.read_pickle(filename)
# Mike_beef_hay_cost.keys()
# Mike_beef_hay_cost = Mike_beef_hay_cost["beef_hay_cost_MikeLinkandFile"].copy()
# Mike_beef_hay_cost.head(2)

# %% [markdown]
# # Beef Price

# %%
beef_price = pd.read_csv(Mike_dir + "Census_BeefPriceMikeMarch62024Email.csv")
beef_price = beef_price[beef_price.Year < 2024].copy()
beef_price.reset_index(drop=True, inplace=True)

print(beef_price.shape)
beef_price.head(2)

# %%

# The following file is the same as "Census_BeefPriceMikeMarch62024Email.csv"
# beef_price_down = pd.read_csv("/Users/hn/Downloads/DAAB6D8D-5E25-3608-9D89-BB8096AC01A1.csv")
beef_price = pd.read_csv(Mike_dir + "Census_BeefPriceMikeMarch62024Email.csv")
# beef_price_down = pd.read_csv("/Users/hn/Downloads/DAAB6D8D-5E25-3608-9D89-BB8096AC01A1.csv")
# beef_price_down = beef_price_down[beef_price_down.Year < 2024].copy()
# beef_price_down.reset_index(drop=True, inplace=True)

# print (beef_price_down.shape)
# beef_price_down.equals(beef_price)

# %% [markdown]
# # Regional Data

# %%
filename = reOrganized_dir + "monthly_NDVI_beef_slaughter.sav"
monthly_NDVI_beef_slaughter = pd.read_pickle(filename)
monthly_NDVI_beef_slaughter.keys()

# %%
monthly_NDVI_slaughter = monthly_NDVI_beef_slaughter["monthly_NDVI_beef_slaughter"].copy()
weekly_beef_slaughter_wide = monthly_NDVI_beef_slaughter["weekly_beef_slaughter_wide"].copy()

# %%
monthly_NDVI_slaughter.head(2)

# %%

# %%

# %%

# %%
