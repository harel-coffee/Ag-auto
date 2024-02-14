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

# %%
import shutup
shutup.please()

import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
NASS_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %%
slaughter_dir = NASS_dir + "slaughter_Qs/"

# %%
slaughter_Q1 = pd.read_csv(slaughter_dir + "slaughter_Q1.csv")
slaughter_Q1.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
slaughter_Q1.head(2)

# %%
print (slaughter_Q1["data_item"].unique())
print (slaughter_Q1["domain_category"].unique())

# %%
sorted(slaughter_Q1.columns)

# %%
print (slaughter_Q1.zip_code.unique())
print (slaughter_Q1.week_ending.unique())
print ()
print (slaughter_Q1.watershed.unique())
print (slaughter_Q1.domain_category.unique())
print ()
print (slaughter_Q1.domain.unique())
print (slaughter_Q1.watershed_code.unique())
print ()
print (slaughter_Q1.watershed.unique())
print (slaughter_Q1.region.unique())
print ()
print (slaughter_Q1.program.unique())
print (slaughter_Q1.period.unique())
print ()
print (slaughter_Q1.geo_level.unique())
print (slaughter_Q1.data_item.unique())

# %%
bad_cols  = ["watershed", "watershed_code", 
             "domain", "domain_category", 
             "region", "period",
             "week_ending", "zip_code", "program", "geo_level"]

slaughter_Q1.drop(bad_cols, axis="columns", inplace=True)
slaughter_Q1.head(2)

# %%
slaughter_Q1['county_ansi'].fillna(666, inplace=True)

slaughter_Q1["state_ansi"] = slaughter_Q1["state_ansi"].astype('int32')
slaughter_Q1["county_ansi"] = slaughter_Q1["county_ansi"].astype('int32')

slaughter_Q1["state_ansi"] = slaughter_Q1["state_ansi"].astype('str')
slaughter_Q1["county_ansi"] = slaughter_Q1["county_ansi"].astype('str')

slaughter_Q1.state = slaughter_Q1.state.str.title()
slaughter_Q1.county = slaughter_Q1.county.str.title()
slaughter_Q1.ag_district = slaughter_Q1.ag_district.str.title()
slaughter_Q1.head(2)

# %%
for idx in slaughter_Q1.index:
    if len(slaughter_Q1.loc[idx, "state_ansi"]) == 1:
        slaughter_Q1.loc[idx, "state_ansi"] = "0" + slaughter_Q1.loc[idx, "state_ansi"]
        
    if len(slaughter_Q1.loc[idx, "county_ansi"]) == 1:
        slaughter_Q1.loc[idx, "county_ansi"] = "00" + slaughter_Q1.loc[idx, "county_ansi"]
    elif len(slaughter_Q1.loc[idx, "county_ansi"]) == 2:
        slaughter_Q1.loc[idx, "county_ansi"] = "0" + slaughter_Q1.loc[idx, "county_ansi"]
        
slaughter_Q1.head(2)

# %%
slaughter_Q1["county_fips"] = slaughter_Q1["state_ansi"] + slaughter_Q1["county_ansi"]
slaughter_Q1.head(2)

# %%
slaughter_Q1.rename(columns={"value": "cattle_on_feed_sale_4_slaughter", 
                             "cv_(%)": "cattle_on_feed_sale_4_slaughter_cv_(%)"}, inplace=True)

# %%
print (slaughter_Q1.shape)
slaughter_Q1 = rc.clean_census(df=slaughter_Q1, col_="cattle_on_feed_sale_4_slaughter")
print (slaughter_Q1.shape)

# %%
slaughter_Q1.head(2)

# %%
print (slaughter_Q1.commodity.unique())
print (slaughter_Q1.data_item.unique())

# %%
slaughter_Q1 = slaughter_Q1[["year", "data_item", 
                             "cattle_on_feed_sale_4_slaughter", 
                             "cattle_on_feed_sale_4_slaughter_cv_(%)",
                             "county_fips"]]
slaughter_Q1.head(2)

# %%
slaughter_Q1.reset_index(drop=True, inplace=True)

# %%
filename = reOrganized_dir + "slaughter_Q1.sav"

export_ = {"slaughter_Q1": slaughter_Q1, 
           "source_code" : "clean_slaughter",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
