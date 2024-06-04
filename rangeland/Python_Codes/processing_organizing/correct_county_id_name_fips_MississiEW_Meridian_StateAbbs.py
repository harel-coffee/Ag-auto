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
# This notebook only corrected county fips of Min that had 1 or 2 mistakes in it. and saved the file as
# ```county_fips.sav```.
#
# Now, I have merged 2 other notebooks here as well: ```state_abbreviations``` and ```filter_counties_RA_Pallavi```.
#
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

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/Rangeland/Python_Codes/")
import rangeland_core as rc

datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %%
county_fips = pd.read_csv(Min_data_base + "county_id_name_fips.csv")
county_fips.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

county_fips.sort_values(by=["state", "county"], inplace=True)

county_fips = rc.correct_Mins_county_6digitFIPS(df=county_fips, col_="county")
county_fips.rename(columns={"county": "county_fips"}, inplace=True)

county_fips["state_fips"] = county_fips.county_fips.str.slice(0, 2)

county_fips.reset_index(drop=True, inplace=True)
print(len(county_fips.state.unique()))
county_fips.head(2)

# %%
county_fips[county_fips.state=="SD"].state_fips.unique()

# %%
county_fips[county_fips.state_fips=="34"].state.unique()

# %%
# Shannon county changed to Oglala Lakota county in 2014
# and is missing from Min's data!
county_fips[county_fips.county_fips == "46102"]

# %%
county_fips.tail(3)

# %%
# county_fips = county_fips.append(df2, ignore_index = True)
county_fips.loc[len(county_fips.index)] = ["46102",
                                           "Oglala Lakota County",
                                           "46102",
                                           "SD",
                                           "46"]
county_fips.tail(3)

# %%
county_fips[county_fips.state=="SD"].state_fips.unique()

# %%
county_fips[county_fips.state_fips=="34"].state.unique()

# %%

# %%
# abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
# SoI = abb_dict['SoI']
# SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
state_to_abbrev = {"Alabama": "AL", 
                   "Alaska": "AK", 
                   "Arizona": "AZ",
                   "Arkansas": "AR", 
                   "California": "CA", 
                   "Colorado": "CO",
                   "Connecticut": "CT", 
                   "Delaware": "DE", 
                   "Florida": "FL",
                   "Georgia": "GA", 
                   "Hawaii": "HI", 
                   "Idaho": "ID",
                   "Illinois": "IL", 
                   "Indiana": "IN",
                   "Iowa": "IA",
                   "Kansas": "KS", 
                   "Kentucky": "KY",
                   "Louisiana": "LA",
                   "Maine": "ME",
                   "Maryland": "MD",
                   "Massachusetts": "MA",
                   "Michigan": "MI",
                   "Minnesota": "MN",
                   "Mississippi": "MS",
                   "Missouri": "MO",
                   "Montana": "MT",
                   "Nebraska": "NE",
                   "Nevada": "NV",
                   "New Hampshire": "NH",
                   "New Jersey": "NJ",
                   "New Mexico": "NM",
                   "New York": "NY",
                   "North Carolina": "NC",
                   "North Dakota": "ND",
                   "Ohio": "OH",
                   "Oklahoma": "OK",
                   "Oregon": "OR",
                   "Pennsylvania": "PA",
                   "Rhode Island": "RI",
                   "South Carolina": "SC",
                   "South Dakota": "SD",
                   "Tennessee": "TN",
                   "Texas": "TX",
                   "Utah": "UT",
                   "Vermont": "VT",
                   "Virginia": "VA",
                   "Washington": "WA",
                   "West Virginia": "WV",
                   "Wisconsin": "WI",
                   "Wyoming": "WY",
                   "District of Columbia": "DC",
                   "American Samoa": "AS",
                   "Guam": "GU",
                   "Northern Mariana Islands": "MP",
                   "Puerto Rico": "PR",
                   "United States Minor Outlying Islands": "UM",    
                   "U.S. Virgin Islands": "VI"}

states_abb_list = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                    'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                    'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
                    'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                    'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

abb_2_full_dict = {'AK': 'Alaska',
                   'AL': 'Alabama',
                   'AR': 'Arkansas',    
                   'AZ': 'Arizona',
                   'CA': 'California',
                   'CO': 'Colorado',
                   'CT': 'Connecticut',
                   'DC': 'District of Columbia',
                   'DE': 'Delaware',
                   'FL': 'Florida',
                   'GA': 'Georgia',
                   'HI': 'Hawaii',
                   'IA': 'Iowa',
                   'ID': 'Idaho',
                   'IL': 'Illinois',
                   'IN': 'Indiana',
                   'KS': 'Kansas',
                   'KY': 'Kentucky',
                   'LA': 'Louisiana',
                   'MA': 'Massachusetts',
                   'MD': 'Maryland',
                   'ME': 'Maine',
                   'MI': 'Michigan',
                   'MN': 'Minnesota',
                   'MO': 'Missouri',
                   'MS': 'Mississippi',
                   'MT': 'Montana',
                   'NC': 'North Carolina',
                   'ND': 'North Dakota',
                   'NE': 'Nebraska',
                   'NH': 'New Hampshire',
                   'NJ': 'New Jersey',
                   'NM': 'New Mexico',
                   'NV': 'Nevada',
                   'NY': 'New York',
                   'OH': 'Ohio',
                   'OK': 'Oklahoma',
                   'OR': 'Oregon',
                   'PA': 'Pennsylvania',
                   'RI': 'Rhode Island',
                   'SC': 'South Carolina',
                   'SD': 'South Dakota',
                   'TN': 'Tennessee',
                   'TX': 'Texas',
                   'UT': 'Utah',
                   'VA': 'Virginia',
                   'VT': 'Vermont',
                   'WA': 'Washington',
                   'WI': 'Wisconsin',
                   'WV': 'West Virginia',
                   'WY': 'Wyoming'}


print (len(state_to_abbrev))
print (len(states_abb_list))
print (len(abb_2_full_dict))

# %%
# list(abb_dict["full_2_abb"].keys())[0:4]

n = 4
{key: value for key, value in list(state_to_abbrev.items())[0:n]}

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
West_of_meridian = [
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
len(West_of_meridian)

# %%
len([x for x in West_of_meridian if x in state_to_abbrev.keys()])

# %%
West_of_meridian_abb = [
    value
    for key, value in list(state_to_abbrev.items())
    if key in West_of_Mississippi
]
West_of_meridian_abb[:3]

# %%
West_of_Mississippi_abb = [
    value
    for key, value in list(state_to_abbrev.items())
    if key in West_of_Mississippi
]
West_of_Mississippi_abb[:3]

# %%
county_fips["EW_Mississippi"] = "E"
county_fips.loc[county_fips.state.isin(West_of_Mississippi_abb), "EW_Mississippi"] = "W"

print (county_fips[county_fips.state.isin(West_of_Mississippi_abb)].EW_Mississippi.unique())
print (county_fips.loc[~(county_fips.state.isin(West_of_Mississippi_abb)), "EW_Mississippi"].unique())

county_fips.head(2)

# %%
county_fips["EW_meridian"] = "E"
county_fips.loc[county_fips.state.isin(West_of_meridian_abb), "EW_meridian"] = "W"

print (county_fips[county_fips.state.isin(West_of_meridian_abb)].EW_Mississippi.unique())
print (county_fips.loc[~(county_fips.state.isin(West_of_meridian_abb)), "EW_meridian"].unique())

county_fips.head(2)

# %%
(county_fips.EW_Mississippi == county_fips.EW_meridian).sum() == len(county_fips)

# %%

# %%
county_fips[county_fips.state=="SD"].state_fips.unique()

# %%
county_fips[county_fips.state=="NJ"].state_fips.unique()

# %%
county_fips[county_fips.state_fips=="34"].state.unique()

# %%
county_fips[county_fips.state_fips=="46"].state.unique()

# %%
len(county_fips.state_fips.unique())

# %% [markdown]
# ###--- End of correct_county_id_name_fips_EastWest and----
# beginning of state_abbreviations

# %%
SoI = [
    "Alabama",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Florida",
    "Georgia",
    "Idaho",
    "Illinois",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Mexico",
    "North Dakota",
    "Oklahoma",
    "Oregon",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Virginia",
    "Washington",
    "Wyoming",
]

# %%
SoI_abb = [state_to_abbrev[x] for x in SoI]
SoI_abb[:4]

# %%
county_fips.head(2)

# %%
county_fips[county_fips.state=="NJ"]

# %%
print (county_fips[county_fips.state=="SD"].state_fips.unique())
print (county_fips[county_fips.state=="NJ"].state_fips.unique())

# %%
print (county_fips[county_fips.state_fips=="34"].state.unique())
print (county_fips[county_fips.state_fips=="46"].state.unique())

# %%
state_fips = county_fips.copy()
state_fips = state_fips[["state", "state_fips"]]
print (state_fips.shape)
state_fips.drop_duplicates(inplace=True)
state_fips.reset_index(drop=True, inplace=True)
print (state_fips.shape)

state_fips.head(2)

# %%
# data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
# param_dir = data_dir_base + "parameters/"

# filename = param_dir + "state_abbreviations.sav"

# export_ = {"full_2_abb" : state_to_abbrev,
#            "states_abb_list" : states_abb_list,
#            "abb_2_full" : abb_2_full_dict,
#            "SoI" : SoI,
#            "county_fips" : county_fips, 
#            "state_fips" : state_fips,
#            "source_code": "state_abbreviations",
#            "Author": "HN",
#            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#           }

# pickle.dump(export_, open(filename, "wb"))

# %% [markdown]
# # filter_counties_RA_Pallavi.ipynb:
#
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
RA = pd.read_csv(reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv")
RA.rename(columns={"fips_id": "county_fips"}, inplace=True)
RA = rc.correct_Mins_county_6digitFIPS(df=RA, col_="county_fips")
print(f"{len(RA.county_fips.unique()) = }")
RA.reset_index(drop=True, inplace=True)
RA.head(2)

# %%
print (county_fips[county_fips.state=="SD"].state_fips.unique())
print (county_fips[county_fips.state=="NJ"].state_fips.unique())

# %%
print (county_fips[county_fips.state_fips=="46"].state.unique())
print (county_fips[county_fips.state_fips=="34"].state.unique())

# %%
RA = pd.merge(RA, county_fips[["county_fips", "state"]], on=["county_fips"], how="left")

# %%
print (len(RA.state.unique()))
print (len(county_fips.state.unique()))

# %%
RA[RA.state.isna()]

# %%
RA.dropna(how="any", inplace=True)

# county_fips_SoI = county_fips[county_fips.state.isin(SoI_abb)].copy()
# RA = RA[RA.state.isin(SoI_abb)].copy()

print (len(RA.state.unique()))
print (len(county_fips.state.unique()))

# %%
large_counties = RA[RA.rangeland_acre >= 50000].copy()
small_counties = RA[RA.rangeland_acre < 50000].copy()
small_counties_largeRA = small_counties[small_counties.rangeland_fraction >= 0.1].copy()

# %%
print (len(large_counties.state.unique()))
print (len(small_counties_largeRA.state.unique()))

# %%
filtered_counties = pd.concat([large_counties, small_counties_largeRA], ignore_index=True)
filtered_counties.reset_index(drop=True, inplace=True)
filtered_counties.head(2)

# %%
print (len(filtered_counties.state.unique()))
print(len(filtered_counties.county_fips))
print(len(filtered_counties.county_fips.unique()))
print ()
print(len(RA.county_fips))
print(len(RA.county_fips.unique()))

# %%
filtered_counties.head(2)

# %%
filtered_counties_29States = filtered_counties[filtered_counties.state.isin(SoI_abb)].copy()

# %%
print(len(filtered_counties_29States.county_fips))
print(len(filtered_counties_29States.county_fips.unique()))

# %%
print (len(filtered_counties.state.unique()))
print (len(filtered_counties_29States.state.unique()))

# %%
filtered_counties["Pallavi"] = "True"
filtered_counties_29States["Pallavi"] = "True"

# %%
len(county_fips.state.unique())

# %%
filtered_counties.head(2)

# %%
county_fips.head(2)

# %%
county_fips = pd.merge(county_fips, filtered_counties[["county_fips", "Pallavi"]], 
                       on=["county_fips"], how="left")
county_fips.head(2)

# %%
NA_filling_values = {"Pallavi": "False"}
county_fips.fillna(value = NA_filling_values, inplace=True)
county_fips.head(2)

# %%

# %%
county_fips.head(2)

# %%
# filename = param_dir + "filtered_counties_RAsizePallavi.sav"

# export_ = {
#     "filtered_counties": filtered_counties,
#     "filtered_counties_29States": filtered_counties_29States,
#     "SoI": SoI,
#     "source_code": "filter_counties_RA_Pallavi",
#     "Author": "HN",
#     "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     "Desciption": "Some counties have small portion of RA etc.",
# }

# pickle.dump(export_, open(filename, "wb"))

# %%
list(abb_2_full_dict.keys()) == sorted(states_abb_list)

# %%
county_fips.head(2)

# %%
filtered_counties_29States.head(2)

# %%
state_fips = pd.merge(state_fips, county_fips[["state_fips", "EW_meridian"]].drop_duplicates(), 
                      how="left", on="state_fips")
state_fips.head(2)

# %%
abb_full_df = pd.DataFrame.from_dict(abb_2_full_dict, orient='index').reset_index(drop=False)
abb_full_df.rename(columns={"index": "state", 0: "state_full"}, inplace=True)

abb_full_df.head(2)

# %%
state_fips = pd.merge(state_fips, abb_full_df, on="state", how="outer")
state_fips.head(2)

# %%
county_fips = pd.merge(county_fips, abb_full_df, on="state", how="outer")
county_fips.head(2)

# %%

# %%

# %%
filename = reOrganized_dir + "county_fips.sav"

export_ = {
    "county_fips": county_fips,
    "full_2_abb" : state_to_abbrev,
    "abb_2_full_dict" : abb_2_full_dict,
    "abb_full_df" : abb_full_df,
    'filtered_counties_29States':filtered_counties_29States,
    "SoI" : SoI,
    "state_fips" : state_fips,
    "source_code": "correct_county_id_name_fips_MississiEW_Meridian_StateAbbs.ipynb",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %%
# export_ = {"full_2_abb" : state_to_abbrev,
#            "abb_2_full" : abb_2_full_dict,
#            "SoI" : SoI,
#            "county_fips" : county_fips, 
#            "state_fips" : state_fips,
#            "source_code": "state_abbreviations",
#            "Author": "HN",
#            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#           }

