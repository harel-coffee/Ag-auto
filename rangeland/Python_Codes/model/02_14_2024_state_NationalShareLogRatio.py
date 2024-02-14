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
# <font color='red'>red</font>, <span style='color:green'>green</span>, $\color{blue}{\text{blue}}$

# %%
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

# %%
filename = reOrganized_dir + "state_data_OuterJoined.sav"
state_data_OuterJoined = pd.read_pickle(filename)
state_data_OuterJoined.keys()

# %%
state_data_OuterJoined["NOTE"]

# %%
state_data_OuterJoined = state_data_OuterJoined["all_df"]
print (f"{len(state_data_OuterJoined.columns) = }")
state_data_OuterJoined.head(2)

# %%
print (state_data_OuterJoined.dropna(subset=['state_unit_NPP'])['year'].min())
print (state_data_OuterJoined.dropna(subset=['state_unit_NPP'])['year'].max())

# %% [markdown]
# ### Subset: 2001 and 2020
#
# Since ```NPP``` exist only between 2001 and 2020.

# %%
state_data_OuterJoined = state_data_OuterJoined[(state_data_OuterJoined.year >= 2001) & \
                                                (state_data_OuterJoined.year <= 2020)].copy()

state_data_OuterJoined.reset_index(drop=True, inplace=True)
state_data_OuterJoined.head(2)

# %% [markdown]
# **Subset to states of interest**

# %%
state_fips_df = abb_dict['county_fips']
state_fips_df = state_fips_df[["state", "state_fips"]].copy()

state_fips_df.drop_duplicates(inplace=True)
state_fips_df = state_fips_df[state_fips_df.state.isin(SoI_abb)]
state_fips_df.reset_index(drop=True, inplace=True)

print (f"{len(state_fips_df) = }")
state_fips_df.head(2)

# %%
state_data_OuterJoined = state_data_OuterJoined[state_data_OuterJoined.state_fips.isin(state_fips_df.state_fips)]
state_data_OuterJoined.reset_index(drop=True, inplace=True)

print (f"{len(state_data_OuterJoined.state_fips.unique()) = }")
state_data_OuterJoined.head(2)

# %% [markdown]
# ## Compute national share of each state

# %%

# %%
