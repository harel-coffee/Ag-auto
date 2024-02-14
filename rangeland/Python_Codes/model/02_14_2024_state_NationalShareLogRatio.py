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
state_data = pd.read_pickle(filename)
state_data.keys()

# %%
state_data["NOTE"]

# %%
state_data = state_data["all_df"]
print (f"{len(state_data.columns) = }")
state_data.head(2)

# %%
print (state_data.dropna(subset=['state_unit_NPP'])['year'].min())
print (state_data.dropna(subset=['state_unit_NPP'])['year'].max())

# %% [markdown]
# ### Subset: 2001 and 2020
#
# Since ```NPP``` exist only between 2001 and 2020.

# %%
state_data = state_data[(state_data.year >= 2001) & (state_data.year <= 2020)].copy()

state_data.reset_index(drop=True, inplace=True)
state_data.head(2)

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
state_data = state_data[state_data.state_fips.isin(state_fips_df.state_fips)]
state_data.reset_index(drop=True, inplace=True)

print (f"{len(state_data.state_fips.unique()) = }")
state_data.head(2)

# %% [markdown]
# ## Compute national share of each state

# %%
total_inv = state_data[['year', 'inventory']].copy()
total_inv = total_inv.groupby(["year"]).sum().reset_index()
total_inv.rename(columns={"inventory": "total_inventory"}, inplace=True)
total_inv.head(2)

# %%
state_data = pd.merge(state_data, total_inv, on=["year"], how="left")
state_data.head(2)

# %%
state_data["inventory_share"] = (state_data["inventory"] / state_data["total_inventory"])*100
state_data.head(2)

# %%
### Sort values so we can be sure the ratios are correct
### we need a for-loop or sth. we cannot just do all of the df
### since the row at which state changes will be messed up.
state_data.sort_values(by=["state_fips", "year"], inplace=True)

# %%
cc = ["year", "state_fips", "log_ratio_of_shares_Y2PrevY"]
log_ratios_df = pd.DataFrame(columns=cc)

for a_state in state_data.state_fips.unique():
    curr_df = state_data[state_data.state_fips == a_state].copy()
    curr_ratios = (curr_df.inventory_share[1:].values / curr_df.inventory_share[:-1].values).astype(float)
    curr_ratios_log = np.log(curr_ratios)
    
    curr_ratio_df = pd.DataFrame(columns=cc)
    curr_ratio_df["year"] = curr_df.year[1:]
    curr_ratio_df["log_ratio_of_shares_Y2PrevY"] = curr_ratios_log
    curr_ratio_df["state_fips"] = a_state
    log_ratios_df = pd.concat([log_ratios_df, curr_ratio_df])
    del(curr_ratio_df)

# %%
log_ratios_df.head(2)

# %%
state_data = pd.merge(state_data, log_ratios_df, on=["state_fips", "year"], how="left")
state_data.head(2)

# %%
a = pd.read_pickle("/Users/hn/Downloads/state_data_OuterJoined.sav")
a['Date']

# %%

# %%

# %%

# %%
