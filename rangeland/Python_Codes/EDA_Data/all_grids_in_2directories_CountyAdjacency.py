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

# %% [raw]
# dir1 = "/data/adam/data/metdata/VIC_ext_maca_v2_binary_westUSA/bcc-csm1-1/historical/"
# dir2 = "/data/project/agaid/rajagopalan_agroecosystems/commondata/meteorologicaldata/" + \
#        "gridded/gridMET/gridmet/historical/"
#
# dir1_files = os.listdir(dir1)
# dir2_files = os.listdir(dir2)
# all_files = dir2_files + dir1_files
#
# files_DF = pd.DataFrame()
# files_DF["grids"] = all_files
# files_DF.drop_duplicates(inplace=True)
#
# export_dir = "/data/project/agaid/h.noorazar/"
# output_name = "all_grids_in_2directories.csv"
# files_DF.head.to_csv(export_dir + output_name, index = False)

# %%

# %%
import pandas as pd
import numpy as np

# %%
research_dir = "/Users/hn/Documents/01_research_data/"
rangeland_dir = research_dir + "/RangeLand/Data/"

# %%
FIPS = pd.read_csv(
    "/Users/hn/Documents/01_research_data/Sid/Analog/parameters/Min_counties.csv"
)
FIPS = FIPS[["state", "county", "fips"]]
FIPS.drop_duplicates(inplace=True)
FIPS.reset_index(drop=True, inplace=True)
FIPS["county_name"] = FIPS["county"] + ", " + FIPS["state"]
FIPS.head(2)

# %%
Supriya_FIPS = pd.read_csv(
    rangeland_dir + "Supriya_FIPS.csv", encoding="unicode_escape"
)
Supriya_FIPS = Supriya_FIPS[
    ["STATEFP", "COUNTYFP", "GEOID", "NAME", "STUSPS", "NAME_1"]
]
Supriya_FIPS.rename(columns=lambda x: x.lower(), inplace=True)
Supriya_FIPS.rename(
    columns={
        "name": "county",
        "geoid": "fips",
        "stusps": "state",
        "name_1": "state_full_name",
    },
    inplace=True,
)
Supriya_FIPS["county_name"] = Supriya_FIPS["county"] + ", " + Supriya_FIPS["state"]


Supriya_FIPS

# %%
f = rangeland_dir + "Supriya_FIPS_clean.csv"
Supriya_FIPS.to_csv(f, index=False)

# %%
print(f"{len(sorted(list(Supriya_FIPS.state.unique()))) = }")
# sorted(list(Supriya_FIPS.state.unique()))

# %%
# sorted(list(Supriya_FIPS.state_full_name.unique()))

# %%
Supriya_FIPS[Supriya_FIPS.state == "VA"].sort_values(by=["county"])

# %% [markdown]
# ## Min's Fips is missing Alaska and Hawaii
#
# and some other places. Add those you can

# %%
FIPS.head(2)

# %%
missing_fips = Supriya_FIPS[~Supriya_FIPS.fips.isin(list(FIPS.fips))].copy()
FIPS = pd.concat(
    [FIPS, missing_fips[["state", "county", "fips", "county_name"]]], ignore_index=True
)
FIPS.reset_index(drop=True, inplace=True)
FIPS.head(2)

# %%
f = rangeland_dir + "Supriya_Min_FIPS.csv"
FIPS.to_csv(f, index=False)

# %%
all_grids_in_2directories = pd.read_csv(research_dir + "all_grids_in_2directories.csv")
f = "Supriya_all_grids_in_2directories_state_county.csv"
all_grids_in_2directories_state_county = pd.read_csv(research_dir + f)

# %%
all_grids_in_2directories.tail(10)

# %%
all_grids_in_2directories_state_county.head(10)

# %%
all_grids_in_2directories_state_county["grid"] = all_grids_in_2directories["grids"]
all_grids_in_2directories_state_county.tail(10)

# %%
all_grids_in_2directories_state_county = all_grids_in_2directories_state_county[
    ["grid", "County", "State"]
]
all_grids_in_2directories_state_county

# %%
all_grids_in_2directories_state_county.rename(columns=lambda x: x.lower(), inplace=True)

# %%

# %%
f = research_dir + "all_grids_in_2directories_state_county.csv"
all_grids_in_2directories_state_county.to_csv(f, index=False)

# %% [markdown]
# # 25 states of Rangeland

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"

abb_dict = pd.read_pickle(param_dir + "county_fips.sav")
states_25 = abb_dict["SoI"]  # it was 25 before. then we added 4 more.

# %%
len(SoI)

# %%

# %%

# %%
A = all_grids_in_2directories_state_county.copy()
grids_25states = A[A.state.isin(states_25)]

# %%
f = rangeland_dir + "grids_25states.csv"
grids_25states.to_csv(f, index=False)

# %% [markdown]
# # Neighboring/Adjacent Counties.
# ```.txt``` file is downloaded from [CENSUS](https://www.census.gov/geographies/reference-files/2010/geo/county-adjacency.html#:~:text=The%20county%20adjacency%20file%20lists,and%20the%20U.S.%20Virgin%20Islands)
#
# The [actual text is here](https://www2.census.gov/geo/docs/reference/county_adjacency.txt)
#
# This link also is the same as [CENSUS above](https://www.census.gov/geographies/reference-files/2010/geo/county-adjacency.html)

# %%
county_adjacency_file = "www2.census.gov_geo_docs_reference_county_adjacency.txt"

# county_adjacency = pd.read_csv(rangeland_dir + county_adjacency_file, header=0, sep=" ", on_bad_lines='skip')
# county_adjacency

county_adjacency = pd.read_csv(
    rangeland_dir + county_adjacency_file,
    header=0,
    sep="\t",
    on_bad_lines="skip",
    encoding="unicode_escape",
)

county_adjacency.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

f = rangeland_dir + "www2.census.gov_geo_docs_reference_county_adjacency.csv"
county_adjacency.to_csv(f, index=False)

county_adjacency

# %%
print(f"{len(county_adjacency.county_geoid.unique()) = }")
print(f"{len(county_adjacency.neighbor_geoid.unique()) = }")

# %%

# %%
county_adjacency[county_adjacency.county_name == "Todd County, MN"]

# %%
import math

for a_idx in county_adjacency.index:
    """
    There are some instances in which county_deoid is present but county_name is missing!!!!!
    Handle them!
    """
    if (not math.isnan(county_adjacency.loc[a_idx, "county_geoid"])) and (
        type((county_adjacency.loc[a_idx, "county_name"])) != str
    ):
        curr_county = list(
            FIPS.loc[
                FIPS.fips == county_adjacency.loc[a_idx, "county_geoid"], "county_name"
            ]
        )[0]
        county_adjacency.loc[a_idx, "county_name"] = curr_county

    if not math.isnan(county_adjacency.loc[a_idx, "county_geoid"]):
        curr_county = county_adjacency.loc[a_idx, "county_name"]
        curr_geoid = county_adjacency.loc[a_idx, "county_geoid"]

    if math.isnan(county_adjacency.loc[a_idx, "county_geoid"]):
        county_adjacency.loc[a_idx, "county_name"] = curr_county
        county_adjacency.loc[a_idx, "county_geoid"] = curr_geoid

county_adjacency.county_geoid = county_adjacency.county_geoid.astype(int)
county_adjacency

# %% [markdown]
# # A Mistake
# ```Todd County, MN    27111.0``` is mistake
# and
# ```Todd County, MN    27153.0``` is correct:

# %%
FIPS[FIPS.county == "Todd County"]

# %%
FIPS[FIPS.fips == 27111]

# %%
FIPS[FIPS.county_name == "Todd County, MN"]

# %%
county_adjacency[county_adjacency.neighbor_name == "Todd County, MN"]

# %%
county_adjacency[county_adjacency.county_name == "Todd County, MN"]

# %% [markdown]
# # Morrison is neighbor of Todd but not Otter Tail

# %%
county_adjacency[county_adjacency.county_name == "Todd County, MN"]

# %%
county_adjacency.loc[county_adjacency.county_geoid == 27111, "county_name"] = list(
    FIPS[FIPS.fips == 27111].county_name
)[0]

# %%
county_adjacency[county_adjacency.county_name == "Todd County, MN"]

# %%
county_adjacency[county_adjacency.county_geoid == 27111]

# %%
county_adjacency[county_adjacency.county_geoid == 27111.0]

# %%
county_adjacency[county_adjacency.county_name == "Otter Tail County, MN"]

# %%
county_adjacency[county_adjacency.neighbor_name == "Otter Tail County, MN"]

# %%
county_adjacency[county_adjacency.county_geoid == 27111]

# %%
FIPS[FIPS.fips == 27111]

# %%

# %%
county_adjacency.head(20)

# %%
county_adjacency.tail(20)

# %%
f = rangeland_dir + "county_adjacency_cleaned.csv"
county_adjacency.to_csv(f, index=False)

# %% [markdown]
# # Problematic Names
#
# Some county names are misspeled in different places. For example, ```San Sebastißn Municipio, PR```.
# So, we use the ```IDs``` and use unique names.
#
# **Wade Hampton Census Area, AK (2270)** has changed to **Kusilvak Census Area (2158)**!!!

# %%

df = {
    "state": ["AK"],
    "county": ["Wade Hampton Census Area"],
    "fips": [2270],
    "county_name": ["Wade Hampton Census Area, AK"],
}

df = pd.DataFrame(df)
FIPS = pd.concat([FIPS, df], ignore_index=True)

# %%
df = {
    "state": ["VA"],
    "county": ["Bedford city"],
    "fips": [51515],
    "county_name": ["Bedford city, VA"],
}

df = pd.DataFrame(df)
FIPS = pd.concat([FIPS, df], ignore_index=True)

# %%
county_adjacency.head(2)

# %%
geo_IDs = list(county_adjacency.county_geoid.unique())
neighbor_geoids = list(county_adjacency.neighbor_geoid.unique())

# %%
print(f"{len(geo_IDs) = }")
print(f"{len(neighbor_geoids) = }")

# %%
sorted(geo_IDs) == sorted(neighbor_geoids)

# %%

# %%
county_ID_dict = {}
for an_ID in geo_IDs:
    if not (an_ID in county_ID_dict.keys()):
        county_ID_dict[an_ID] = county_adjacency[
            county_adjacency.county_geoid == an_ID
        ].county_name.unique()[0]

# %%
neighbor_ID_dict = {}
for an_ID in neighbor_geoids:
    if not (an_ID in neighbor_ID_dict.keys()):
        neighbor_ID_dict[an_ID] = county_adjacency[
            county_adjacency.county_geoid == an_ID
        ].county_name.unique()[0]

# %%
sorted(list(county_ID_dict.keys())) == sorted(list(neighbor_ID_dict.keys()))

# %%
# print (f"{len(list(county_ID_dict.keys())) = }")
# print (f"{len(county_adjacency.county_name.unique()) = }")

# for a_county in county_adjacency.county_name.unique():
#     curr_slice = county_adjacency[county_adjacency.county_name==a_county]
#     if len(curr_slice.county_geoid.unique())>1:
#         print (a_county)
#         break;

# %%
a_key = list(county_ID_dict.keys())[0]
county_ID_dict[a_key] in list(county_adjacency.county_name.unique())

# %% [markdown]
# ### Make the names unique:

# %%

# %%
for an_idx in county_adjacency.index:
    ### Min's FIPS do not include Alaska and DC. Do that one differently!!!!
    a = list(
        FIPS[FIPS.fips == county_adjacency.loc[an_idx, "county_geoid"]]["county_name"]
    )[0]
    county_adjacency.loc[an_idx, "county_name"] = a
    if a != county_adjacency.loc[an_idx, "county_name"]:
        print(f"{an_idx = }, {a = }, {county_adjacency.loc[an_idx, 'county_name'] = }")

    b = list(
        FIPS[FIPS.fips == county_adjacency.loc[an_idx, "neighbor_geoid"]]["county_name"]
    )[0]
    county_adjacency.loc[an_idx, "neighbor_name"] = b

    if b != county_adjacency.loc[an_idx, "neighbor_name"]:
        print()
        print(f"{an_idx = }, {b = }, {county_adjacency.loc[an_idx, 'county_name'] = }")

# %%
county_adjacency

# %%
f = rangeland_dir + "county_adjacency_cleaned_corrected.csv"
county_adjacency.to_csv(f, index=False)

# %%

# %%

# %%

# %%
column_names = ["county"] + list(county_adjacency.county_name.unique())

adjacency_binary_matrix = pd.DataFrame(columns=column_names)
adjacency_binary_matrix["county"] = list(county_adjacency.county_name.unique())

adjacency_binary_matrix.fillna(0, inplace=True)
adjacency_binary_matrix.head(4)

for a_county in list(county_adjacency.county_name.unique()):
    curr_slice = county_adjacency[county_adjacency.county_name == a_county]
    curr_neighbors = list(curr_slice.neighbor_name)
    adjacency_binary_matrix.loc[
        adjacency_binary_matrix.county == a_county, curr_neighbors
    ] = 1

# %%
adjacency_binary_matrix

# %% [markdown]
# # Save all binary adjacency matrix

# %%
import os, sys, pickle
from datetime import datetime

# %%
filename = rangeland_dir + "county_adjacency_binary_matrix.sav"

export_ = {
    "adjacency_binary_matrix": adjacency_binary_matrix,
    "source_code": "all_grids_in_2directories_CountyAdjacency",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %%
county_states = county_adjacency.county_name.str.split(pat=", ", expand=True)[1]
countyNeighbors_states = county_adjacency.neighbor_name.str.split(
    pat=", ", expand=True
)[1]
county_adjacency["county_states"] = county_states
county_adjacency["countyNeighbors_states"] = countyNeighbors_states
county_adjacency.head(2)

# %% [markdown]
# # Subset 25 states and create binary adjacency matrix

# %%
state_to_abbrev = {
    "Alabama": "AL",
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
    "U.S. Virgin Islands": "VI",
}

states_abb_list = [
    "AK",
    "AL",
    "AR",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DC",
    "DE",
    "FL",
    "GA",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VA",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
]

abb_2_full_dict = {
    "AK": "Alaska",
    "AL": "Alabama",
    "AR": "Arkansas",
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DC": "District of Columbia",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VA": "Virginia",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
    "WY": "Wyoming",
}

# %%
state_25_abb = [state_to_abbrev[x] for x in states_25]

# %%
#### We can look at counties in a given state when they have neighbors in another state
#### We do that in this cell. However, I think we need to strict our attention to the
#### targeted states. i.e. the neighbors should be also inside those states!
"""
county_adjacency_25states = county_adjacency[county_adjacency.county_states.isin(state_25_abb)].copy()
county_adjacency_25states.reset_index(drop=True, inplace=True)

column_names = ["county"] + list(county_adjacency_25states.neighbor_name.unique())

adjacency_binary_matrix_25states = pd.DataFrame(columns=column_names)
adjacency_binary_matrix_25states["county"] = list(county_adjacency_25states.county_name.unique())

adjacency_binary_matrix_25states.fillna(0, inplace=True)
adjacency_binary_matrix_25states.head(4)
print (f"{adjacency_binary_matrix_25states.shape = }")

for a_county in list(county_adjacency_25states.county_name.unique()):
    curr_slice = county_adjacency_25states[county_adjacency_25states.county_name==a_county]
    curr_neighbors = list(curr_slice.neighbor_name)
    adjacency_binary_matrix_25states.loc[adjacency_binary_matrix_25states.county==a_county, curr_neighbors] = 1
    
adjacency_binary_matrix_25states.shape
"""

# %%
county_adjacency_25states = county_adjacency[
    county_adjacency.county_states.isin(state_25_abb)
].copy()

print(county_adjacency_25states.shape)
county_adjacency_25states = county_adjacency_25states[
    county_adjacency_25states.countyNeighbors_states.isin(state_25_abb)
].copy()

print(county_adjacency_25states.shape)

county_adjacency_25states.reset_index(drop=True, inplace=True)

column_names = ["county"] + list(county_adjacency_25states.neighbor_name.unique())

adjacency_binary_matrix_25states = pd.DataFrame(columns=column_names)
adjacency_binary_matrix_25states["county"] = list(
    county_adjacency_25states.county_name.unique()
)

adjacency_binary_matrix_25states.fillna(0, inplace=True)
adjacency_binary_matrix_25states.head(4)
print(f"{adjacency_binary_matrix_25states.shape = }")

for a_county in list(county_adjacency_25states.county_name.unique()):
    curr_slice = county_adjacency_25states[
        county_adjacency_25states.county_name == a_county
    ]
    curr_neighbors = list(curr_slice.neighbor_name)
    adjacency_binary_matrix_25states.loc[
        adjacency_binary_matrix_25states.county == a_county, curr_neighbors
    ] = 1

adjacency_binary_matrix_25states.shape

# %%
filename = rangeland_dir + "adjacency_binary_matrix_strict25states.sav"

export_ = {
    "adjacency_binary_matrix_strict25states": adjacency_binary_matrix_25states,
    "source_code": "all_grids_in_2directories_CountyAdjacency",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %%


# %%

# %%
county_adjacency_25states.shape

# %%

# %%
len(list(county_adjacency_25states.neighbor_name.unique()))

# %%
adjacency_binary_matrix.shape

# %%
adjacency_binary_matrix_25states.shape

# %%

# %%
adjacency_binary_matrix.loc[adjacency_binary_matrix.county == a_county, curr_neighbors]

# %%
curr_neighbors[-1] in list(adjacency_binary_matrix.columns)

# %%
"Camuy Municipio, PR" in list(adjacency_binary_matrix.columns)

# %%
list(adjacency_binary_matrix.columns)

# %%
