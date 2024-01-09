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
import os

# %%

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_dir = data_dir_base + "census/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_dir_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %%
FIPS = pd.read_csv("/Users/hn/Documents/01_research_data/Sid/Analog/parameters/Min_counties.csv")
FIPS = FIPS[["state", "county", "fips"]]
FIPS.drop_duplicates(inplace=True)
FIPS.reset_index(drop=True, inplace=True)
FIPS.head(2)

# %%
SoI = ["Alabama", "Arkansas", 
       "California", "Colorado", 
       "Florida", "Georgia",
       "Idaho", "Illinois", 
       "Iowa", "Kansas", 
       "Kentucky", "Louisiana", 
       "Mississippi", "Missouri", "Montana", 
       "Nebraska", "New Mexico", "North Dakota", 
       "Oklahoma", "Oregon", 
       "South Dakota", "Tennessee", 
       "Texas", "Virginia", "Wyoming"]

# %%
Bhupi = pd.read_csv(param_dir + "Bhupi_gridMET_counties_state.csv")
Bhupi["SC"] = Bhupi.state + "-" + Bhupi.county
Bhupi_25Satates = Bhupi[Bhupi.state.isin(SoI)]
Bhupi_25Satates = Bhupi_25Satates[["state", "county", "AFFGEOID"]].copy()
Bhupi_25Satates.drop_duplicates(inplace=True)
Bhupi_25Satates.reset_index(drop=True, inplace=True)
Bhupi_25Satates.head(2)

# %%
len(Bhupi_25Satates.state.unique())

# %%
Bhupi_25Satates_SC_df = Bhupi_25Satates.copy()
Bhupi_25Satates_SC_df["SC"] = Bhupi_25Satates.state + "-" + Bhupi_25Satates.county
Bhupi_25Satates_SC_df.head(2)

# %%
multiple_IDs = pd.DataFrame()
for ii in Bhupi_25Satates_SC_df.SC.unique():
    A = Bhupi_25Satates_SC_df[Bhupi_25Satates_SC_df.SC==ii].copy()
    if len(A)>1:
        multiple_IDs = pd.concat([multiple_IDs, A])

# %%
print (sorted(list(multiple_IDs.state.unique())))
print (multiple_IDs.shape)
multiple_IDs

# %%
print (Bhupi[Bhupi.SC == "Alabama-Jackson"].shape)
print (Bhupi[Bhupi.SC == "Idaho-Madison"].shape)
print (Bhupi[Bhupi.SC == "Missouri-St. Louis"].shape)
print (Bhupi[Bhupi.SC == "Nebraska-Holt"].shape)
print (Bhupi[Bhupi.SC == "Tennessee-Scott"].shape)
print (Bhupi[Bhupi.SC == "Virginia-Roanoke"].shape)
print (Bhupi[Bhupi.SC == "Virginia-Richmond"].shape)
print (Bhupi[Bhupi.SC == "Virginia-Fairfax"].shape)

# %%
print (len(Bhupi[Bhupi.SC == "Alabama-Jackson"]) + \
       len(Bhupi[Bhupi.SC == "Idaho-Madison"]) + \
       len(Bhupi[Bhupi.SC == "Missouri-St. Louis"])  + \
       len(Bhupi[Bhupi.SC == "Nebraska-Holt"]) + \
       len(Bhupi[Bhupi.SC == "Tennessee-Scott"]) + \
       len(Bhupi[Bhupi.SC == "Virginia-Roanoke"]) + \
       len(Bhupi[Bhupi.SC == "Virginia-Richmond"]) + \
       len(Bhupi[Bhupi.SC == "Virginia-Fairfax"])
      )

# %%
Bhupi[Bhupi.SC == "Alabama-Jackson"].AFFGEOID.unique()

# %%
Bhupi[(Bhupi.SC == "Alabama-Jackson") & (Bhupi.AFFGEOID== "0500000US28059")]

# %%

# %%
Bhupi[Bhupi.SC == "Idaho-Madison"].AFFGEOID.unique()

# %%
Bhupi[(Bhupi.SC == "Idaho-Madison") & (Bhupi.AFFGEOID== "0500000US30057")]

# %%

# %%
Bhupi[Bhupi.SC == "Missouri-St. Louis"].AFFGEOID.unique()

# %%
Bhupi[(Bhupi.SC == "Missouri-St. Louis") & (Bhupi.AFFGEOID == "0500000US29510")]

# %%
Bhupi[(Bhupi.SC == "Missouri-St. Louis") & (Bhupi.AFFGEOID == "0500000US29189")]

# %% [markdown]
# ### It seems mistakes are too few. Just drop them

# %%
Bhupi_25Satates = Bhupi[Bhupi.state.isin(SoI)]
Bhupi_25Satates.head(3)

# %%
len(Bhupi_25Satates.state.unique())

# %%
Bhupi_25Satates_clean = pd.DataFrame()

for a_SC in Bhupi_25Satates.SC.unique():
    curr_df = Bhupi_25Satates[Bhupi_25Satates.SC == a_SC].copy()
    AFFGEOID_max = curr_df.groupby('AFFGEOID').size().idxmax()
    correct_curr_df = curr_df[curr_df.AFFGEOID == AFFGEOID_max]
    Bhupi_25Satates_clean = pd.concat([Bhupi_25Satates_clean, correct_curr_df])

# %%
len(Bhupi_25Satates) - len(Bhupi_25Satates_clean)

# %%

# %%
Bhupi_25Satates_clean.head(2)

# %%
Bhupi_25Satates_clean = Bhupi_25Satates_clean[["county", "state", "AFFGEOID", "grid"]]
Bhupi_25Satates_clean.head(2)

# %%
Bhupi_25Satates_clean["county_fips"] = Bhupi_25Satates_clean.AFFGEOID.str.split(pat="0500000US", expand=True)[1]
Bhupi_25Satates_clean.head(2)

# %%
Bhupi_25Satates_clean = Bhupi_25Satates_clean[["grid", "state", "county", "county_fips", "AFFGEOID"]]
Bhupi_25Satates_clean.head(2)

# %%
out_name = param_dir + "Bhupi_25states_clean.csv"
Bhupi_25Satates_clean.to_csv(out_name, index = False)

# %%
len(Bhupi_25Satates_clean.state.unique())

# %%
param_dir

# %%
sorted(list((Bhupi_25Satates_clean.state + "-" + Bhupi_25Satates_clean.county).unique()))

# %%
for a_fip in Bhupi_25Satates_clean.county_fips.unique():
    df = Bhupi_25Satates_clean[Bhupi_25Satates_clean.county_fips==a_fip]
    if len(df)==1:
        print (a_fip)

# %%
Bhupi_25Satates_clean[Bhupi_25Satates_clean.county_fips=="51685"]

# %%
Bhupi_25Satates_clean.state.unique()

# %%
sorted(Bhupi_25Satates_clean.county_fips.unique())

# %%

# %%

# %%
states_abbr = {"Alabama": "AL",
               "Idaho" : "ID",
               "Missouri": "MO",
               "Nebraska": "NE",
               "Tennessee": "TN",
               "Virginia": "VA"
              }
multiple_IDs['state'] = multiple_IDs['state'].str.strip().replace(states_abbr)


FIPS_SoI_missing = FIPS[FIPS.state.isin(list(multiple_IDs.state.unique()))].copy()
# FIPS_SoI_missing = FIPS[FIPS.state.isin(list(multiple_IDs.county.unique()))].copy()
FIPS_SoI_missing.reset_index(drop=True, inplace=True)
FIPS_SoI_missing
