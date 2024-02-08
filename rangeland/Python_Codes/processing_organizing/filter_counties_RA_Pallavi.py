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
# In this notebook we filter counties that have enough rangeland area and
#
#
#
# -------------------------------------------------------
# **Pallavi's Notes:**
#
# The final list of the counties was shortlisted after considering the criterion
#
#
# - (Area > 50,000 acres) or (Area <= 50,000 and coverage% >= 10%), where
#
# Area = Area of county covered by rangelands (in acres)
#
# and
#
#
# Coverage % = Area of county covered by Rangelands Total area of the county * 100
#
# • The total number of counties considered: 896+48 = 944.
# • This accounts for 40% of the total counties which have at least 1 pixel of their area
# being covered by rangeland.
#
# -------------------------------------------------------
#
# I do not know what she is saying in terms of rectangles. I will use the CSV file provided to me by Min.
#

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
print("This is " + start_b + "a_bold_text" + end_b + "!")

# %%
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI = abb_dict['SoI']
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
RA = pd.read_csv(reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv")
RA.rename(columns={"fips_id": "county_fips"}, inplace=True)
RA = rc.correct_Mins_county_6digitFIPS(df=RA, col_="county_fips")
print(f"{len(RA.county_fips.unique()) = }")
RA.reset_index(drop=True, inplace=True)
RA.head(2)

# %%
county_fips = pd.read_pickle(reOrganized_dir + "county_fips.sav")

county_fips = county_fips["county_fips"]

print(f"{len(county_fips.state.unique()) = }")
county_fips.drop_duplicates(inplace=True)
county_fips.reset_index(drop=True, inplace=True)
print(f"{len(county_fips.state.unique()) = }")

county_fips.head(2)

# %%
print (county_fips[county_fips.state=="SD"].state_fip.unique())
print (county_fips[county_fips.state=="NJ"].state_fip.unique())

# %%
print (county_fips[county_fips.state_fip=="46"].state.unique())
print (county_fips[county_fips.state_fip=="34"].state.unique())

# %%
RA = pd.merge(RA, county_fips[["county_fips", "state"]], on=["county_fips"], how="left")

# %%
print (len(RA.state.unique()))
print (len(county_fips.state.unique()))

# %%
RA.head(2)

# %%
RA[RA.state.isna()]

# %%
RA.dropna(how="any", inplace=True)

county_fips = county_fips[county_fips.state.isin(SoI_abb)].copy()
RA = RA[RA.state.isin(SoI_abb)].copy()

print (len(RA.state.unique()))
print (len(county_fips.state.unique()))

# %%

# %%
large_counties = RA[RA.rangeland_acre >= 50000].copy()
small_counties = RA[RA.rangeland_acre < 50000].copy()
small_counties_largeRA = small_counties[small_counties.rangeland_fraction >= 0.1].copy()

# %%
print (len(large_counties.state.unique()))
print (len(small_counties_largeRA.state.unique()))

# %%
filtered_counties = pd.concat(
    [large_counties, small_counties_largeRA], ignore_index=True
)
filtered_counties.reset_index(drop=True, inplace=True)
filtered_counties.head(2)

# %%
len(filtered_counties.state.unique())

# %%
print(len(filtered_counties.county_fips))
print(len(filtered_counties.county_fips.unique()))

# %%
print(len(RA.county_fips))
print(len(RA.county_fips.unique()))

# %%
filtered_counties_29States = filtered_counties[
    filtered_counties.county_fips.isin(list(county_fips.county_fips.unique()))
].copy()

# %%
print(len(filtered_counties_29States.county_fips))
print(len(filtered_counties_29States.county_fips.unique()))

# %%
filtered_counties

# %%

# %%
import pickle
from datetime import datetime

filename = param_dir + "filtered_counties_RAsizePallavi.sav"

export_ = {
    "filtered_counties": filtered_counties,
    "filtered_counties_29States": filtered_counties_29States,
    "SoI": SoI,
    "source_code": "filter_counties_RA_Pallavi",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Desciption": "Some counties have small portion of RA etc.",
}

pickle.dump(export_, open(filename, "wb"))

# %%
filtered_counties_29States.head(2)

# %%
A = filtered_counties_29States.groupby("state")["county_fips"].count()
A
