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
print (len(state_fips_SoI))
state_fips_SoI.head(2)

# %%

# %%
beef_price = pd.read_csv(Mike_dir + "Census_BeefPriceMikeMarch62024Email.csv")
beef_price = beef_price[beef_price.Year < 2024].copy()
beef_price.reset_index(drop=True, inplace=True)

print (beef_price.shape)
beef_price.head(2)

# %%
<<<<<<< Updated upstream
# The following file is the same as "Census_BeefPriceMikeMarch62024Email.csv"
# beef_price_down = pd.read_csv("/Users/hn/Downloads/DAAB6D8D-5E25-3608-9D89-BB8096AC01A1.csv")
beef_price = pd.read_csv(Mike_dir + "Census_BeefPriceMikeMarch62024Email.csv")

=======
# beef_price_down = pd.read_csv("/Users/hn/Downloads/DAAB6D8D-5E25-3608-9D89-BB8096AC01A1.csv")
>>>>>>> Stashed changes
# beef_price_down = beef_price_down[beef_price_down.Year < 2024].copy()
# beef_price_down.reset_index(drop=True, inplace=True)

# print (beef_price_down.shape)
# beef_price_down.equals(beef_price)

# %%
prod = pd.read_csv(Min_data_dir_base + "statefips_annual_productivity.csv")
prod = pd.read_csv(Min_data_dir_base + "statefips_annual_productivity.csv")

# %%
f_name = "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
A = pd.read_pickle(reOrganized_dir + f_name)
list(A.keys())

# %%
all_df_outerjoined = A["all_df_outerjoined"]
all_df_outerjoined.head(2)

# %%
list(all_df_outerjoined.columns)

# %%
[x for x in list(all_df_outerjoined.columns) if "state" in x]

# %%
all_df_outerjoined=all_df_outerjoined[["state_fips", "unit_matt_npp"]]
all_df_outerjoined.head(2)

# %%
all_df_outerjoined = pd.merge(all_df_outerjoined, state_fips_SoI, on="state_fips", how="left")
all_df_outerjoined.dropna(how="any", inplace=True)
all_df_outerjoined.reset_index(drop=True, inplace=True)
all_df_outerjoined.head(2)

# %%
all_df_outerjoined[all_df_outerjoined.state=="KY"]

# %%
dir_ = "/Users/hn/Documents/01_research_data/RangeLand/Data_large_notUsedYet/Min_data/"
A = pd.read_csv(dir_ + "county_annual_MODIS_NPP.csv")
A.head(2)

# %%
A = rc.correct_Mins_county_6digitFIPS(df=A, col_="county")
A.head(2)

# %%
A["state_fips"] = A.county.str.slice(start=0, stop=2)
A.head(4)

# %%
state_fips_SoI[state_fips_SoI.state=="KY"]

# %%
A[A.state_fips == "21"]

# %%
beef_price_down.head(2)

# %%
