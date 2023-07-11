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
import pandas as pd
import numpy as np

# %%
research_dir = "/Users/hn/Documents/01_research_data/"
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
all_grids_in_2directories_state_county = all_grids_in_2directories_state_county[["grid", "County", "State"]]
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
states_25 = ["Alabama", "Arkansas", "California", 
             "Colorado", "Florida", "Georgia", "Idaho", 
             "Illinois", "Iowa", "Kansas", "Kentucky", 
             "Louisiana", "Missouri", "Mississippi", "Montana", 
             "Nebraska", "New Mexico", "North Dakota", 
             "Oklahoma", "Oregon", "South Dakota", "Tennessee", 
             "Texas", "Virginia", "Wyoming"]

# %%
A = all_grids_in_2directories_state_county.copy()
grids_25states = A[A.state.isin(states_25)]

# %%
rangeland_dir = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
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

# county_adjacency = pd.read_csv(rangeland_dir + county_adjacency_file, header=0, sep="	", on_bad_lines='skip')
# county_adjacency

county_adjacency = pd.read_csv(rangeland_dir + county_adjacency_file, 
                               header = 0, sep = "\t", on_bad_lines = 'skip',
                               encoding = 'unicode_escape')
county_adjacency

# %%
# with open(rangeland_dir + county_adjacency_file) as f:
#     contents = f.readlines()

# f = open(rangeland_dir + county_adjacency_file,'r')

# %%
county_adjacency.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
county_adjacency

# %%
f = rangeland_dir + "www2.census.gov_geo_docs_reference_county_adjacency.csv"
county_adjacency.to_csv(f, index=False)

# %%
county_adjacency.head(20)

# %%
import math
for a_idx in county_adjacency.index:
    if (not math.isnan(county_adjacency.loc[a_idx, "county_geoid"])):
        curr_county = county_adjacency.loc[a_idx, "county_name"]
        curr_geoid = county_adjacency.loc[a_idx, "county_geoid"]
        
    if (math.isnan(county_adjacency.loc[a_idx, "county_geoid"])):
        county_adjacency.loc[a_idx, "county_name"] = curr_county
        county_adjacency.loc[a_idx, "county_geoid"] = curr_geoid

# %%
county_adjacency.head(20)

# %%
county_adjacency.tail(20)

# %%
county_adjacency.county_geoid = county_adjacency.county_geoid.astype(int)

# %%
f = rangeland_dir + "county_adjacency_cleaned.csv"
county_adjacency.to_csv(f, index=False)

# %%

# %%
column_names = ["county"] + list(county_adjacency.county_name.unique())

adjacency_binary_matrix = pd.DataFrame(columns=column_names)
adjacency_binary_matrix["county"] = list(county_adjacency.county_name.unique())

# %%

# %%

# %%
