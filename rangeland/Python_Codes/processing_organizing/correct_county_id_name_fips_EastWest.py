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

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/Rangeland/Python_Codes/")
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
county_id_name_fips = pd.read_csv(Min_data_base + "county_id_name_fips.csv")
county_id_name_fips.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

county_id_name_fips.sort_values(by=["state", "county"], inplace=True)

county_id_name_fips = rc.correct_Mins_county_6digitFIPS(
    df=county_id_name_fips, col_="county"
)
county_id_name_fips.rename(columns={"county": "county_fips"}, inplace=True)

county_id_name_fips["state_fips"] = county_id_name_fips.county_fips.str.slice(0, 2)

county_id_name_fips.reset_index(drop=True, inplace=True)
print(len(county_id_name_fips.state.unique()))
county_id_name_fips.head(2)

# %%
county_id_name_fips[county_id_name_fips.state=="SD"].state_fips.unique()

# %%
county_id_name_fips[county_id_name_fips.state_fips=="34"].state.unique()

# %%
# Shannon county changed to Oglala Lakota county in 2014
# and is missing from Min's data!
county_id_name_fips[county_id_name_fips.county_fips == "46102"]

# %%
county_id_name_fips.tail(3)

# %%
# county_id_name_fips = county_id_name_fips.append(df2, ignore_index = True)
county_id_name_fips.loc[len(county_id_name_fips.index)] = ["46102",
                                                           "Oglala Lakota County",
                                                           "46102",
                                                           "SD",
                                                           "46"]
county_id_name_fips.tail(3)

# %%
county_id_name_fips[county_id_name_fips.state=="SD"].state_fips.unique()

# %%
county_id_name_fips[county_id_name_fips.state_fips=="34"].state.unique()

# %%
county_id_name_fips["EW"] = "E"
county_id_name_fips.head(2)

# %%
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI = abb_dict['SoI']
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
abb_dict.keys()

# %%
# list(abb_dict["full_2_abb"].keys())[0:4]

n = 4
{key: value for key, value in list(abb_dict["full_2_abb"].items())[0:n]}

# %%
West_of_Mississippi = [
    "Alaska",
    "Washington",
    "Oregon",
    "California",
    "Idaho",
    "Nevada",
    "Utah",
    "Arizona",
    "Montana",
    "Wyoming",
    "Colorado",
    "New Mexico",
    "Texas",
    "North Dakota",
    "South Dakota",
    "Nebraska",
    "Kansas",
    "Oklahoma",
    "Hawaii",
]
len(West_of_Mississippi)

# %%
len([x for x in West_of_Mississippi if x in abb_dict["full_2_abb"].keys()])

# %%
West_of_Mississippi_abb = [
    value
    for key, value in list(abb_dict["full_2_abb"].items())
    if key in West_of_Mississippi
]
West_of_Mississippi_abb[:3]

# %%
county_id_name_fips.head(2)

# %%
county_id_name_fips.loc[county_id_name_fips.state.isin(West_of_Mississippi_abb), "EW"] = "W"

# %%
county_id_name_fips[county_id_name_fips.state.isin(West_of_Mississippi_abb)].EW.unique()

# %%
county_id_name_fips.loc[~(county_id_name_fips.state.isin(West_of_Mississippi_abb)), "EW"].unique()

# %%
county_id_name_fips[county_id_name_fips.state=="SD"].state_fips.unique()

# %%
county_id_name_fips[county_id_name_fips.state=="NJ"].state_fips.unique()

# %%
county_id_name_fips[county_id_name_fips.state_fips=="34"].state.unique()

# %%
county_id_name_fips[county_id_name_fips.state_fips=="46"].state.unique()

# %%
import pickle
from datetime import datetime

filename = reOrganized_dir + "county_fips.sav"

export_ = {
    "county_fips": county_id_name_fips,
    "source_code": "correct_county_id_name_fips_EastWest.ipynb",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %%
A = pd.read_pickle(reOrganized_dir + "county_fips.sav")
A = A['county_fips']
# A.rename(columns={"state_fip": "state_fips"}, inplace=True)
A

# %%

# %%
