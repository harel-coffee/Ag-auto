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
# ### May 20, 2024
# We want to merge high frequency slaughter with NDVI
#
# Slaughter data are weekly and on regional scale.

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
Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"

reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %% [markdown]
# ## Shannon Slaughter data
#
# We just need sheet B (beef cows) from Weekly Regional Cow Slaughter. Check this with others.
#
# Regions are
#
# | Region | States 
# | :- | :- 
# | 1 |  CT, ME, NH,VT, MA & RI 
# | 2 | NY & NJ
# | 3 | DE-MD, PA, WV & VA
# | 4 |  AL, FL, GA, KY, MS, NC, SC & TN
# | 5 | IL, IN, MI, MN ,OH & WI
# | 6 | AR, LA, NM, OK & TX
# | 7 | IA, KS, MO & NE
# | 8 | CO, MT, ND, SD, UT & WY
# | 9 | AZ, CA, HI & NV
# | 10 | AK, ID, OR & WA

# %%
# The following file is created in convertShannonData.ipynb

out_name = reOrganized_dir + "shannon_slaughter_data.sav"
slaughter = pd.read_pickle(out_name)


beef_slaughter_tall = slaughter["beef_slaughter_tall"]
beef_slaughter = slaughter["beef_slaughter"]

# There are some rows with NA in them
# slaughter['week'] = slaughter['week'].astype(int)
beef_slaughter_tall.head(4)

# %%

# %%

# %%

# %% [markdown]
# # Inventory
#
# Read inventory and see if slaughter numbers are making sense

# %%
all_data = pd.read_pickle(reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav")

# %%
all_df_outerjoined = all_data["all_df_outerjoined"].copy()
all_df_outerjoined = all_df_outerjoined[["year", "inventory", "state_fips"]]
all_df_outerjoined.dropna(subset="inventory", inplace=True)
all_df_outerjoined.reset_index(drop=True, inplace=True)

print (len(all_df_outerjoined.state_fips.unique()))
all_df_outerjoined.head(2)

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

state_fips = abb_dict["state_fips"]
state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
print(len(state_fips_SoI))
state_fips_SoI.head(2)

# %%
all_df_outerjoined = pd.merge(all_df_outerjoined, state_fips_SoI, on=["state_fips"], how="left")

all_df_outerjoined.dropna(subset="state", inplace=True)
all_df_outerjoined.reset_index(drop=True, inplace=True)

print (len(all_df_outerjoined.state_fips.unique()))
all_df_outerjoined.head(2)

# %%

# %%

# %% [markdown]
# ## Regional NDVI
#
# Is ```Regional``` different from ```subsection```, ```ecozone```, and ```econregion```.
#

# %%
