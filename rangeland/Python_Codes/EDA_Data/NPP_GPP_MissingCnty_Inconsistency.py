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
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# %% [markdown]
# ## Directories

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %%
NPP = pd.read_csv(Min_data_base + "county_annual_MODIS_NPP.csv")
NPP.rename(columns={"NPP": "modis_npp"}, inplace=True)

NPP = rc.correct_Mins_FIPS(df=NPP, col_="county")
NPP.rename(columns={"county": "county_fips"}, inplace=True)

print(f"{NPP.year.min() = }")
NPP.head(2)

# %%
GPP = pd.read_csv(Min_data_base + "county_annual_MODIS_GPP.csv")
GPP.rename(columns={"GPP": "modis_GPP"}, inplace=True)

GPP = rc.correct_Mins_FIPS(df=GPP, col_="county")
GPP.rename(columns={"county": "county_fips"}, inplace=True)

print(f"{GPP.year.min() = }")
GPP.head(2)

# %%
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
county_id_name_fips = pd.read_csv(Min_data_base + "county_id_name_fips.csv")
county_id_name_fips.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

county_id_name_fips.sort_values(by=["state", "county"], inplace=True)
county_id_name_fips.rename(columns={"county": "county_fips"}, inplace=True)

county_id_name_fips = rc.correct_Mins_FIPS(df=county_id_name_fips, col_="county_fips")
county_id_name_fips.reset_index(drop=True, inplace=True)
county_id_name_fips.head(2)

# %%
print(f"{len(list(county_id_name_fips.state.unique())) = }")
print(f"{len(list(county_id_name_fips.county_fips.unique())) = }")

# %%
NPP = pd.merge(NPP, county_id_name_fips, on=["county_fips"], how="left")
GPP = pd.merge(GPP, county_id_name_fips, on=["county_fips"], how="left")

# %%
NPP.head(2)

# %%
GPP.head(2)

# %%
print(GPP.state.isna().sum())
print(NPP.state.isna().sum())

# %%
county_id_name_fips = county_id_name_fips[
    county_id_name_fips.state.isin(SoI_abb)
].copy()
NPP = NPP[NPP.state.isin(SoI_abb)].copy()
GPP = GPP[GPP.state.isin(SoI_abb)].copy()

# %%
print(f"{len(NPP.county_fips.unique()) = }")
print(f"{len(GPP.county_fips.unique()) = }")

# %%
counties_inNPP_missinginGPP = [
    x for x in NPP.county_fips.unique() if not (x in GPP.county_fips.unique())
]

# %%
counties_inGPP_missinginNPP = [
    x for x in GPP.county_fips.unique() if not (x in NPP.county_fips.unique())
]

# %%
counties_inGPP_missinginNPP

# %%
county_id_name_fips[county_id_name_fips.county_fips.isin(counties_inGPP_missinginNPP)]

# %%
all_NPP_counties_len = len(NPP.county_fips.unique())
print(all_NPP_counties_len)
for a_year in NPP.year.unique():
    df = NPP[NPP.year == a_year]
    if len(NPP.county_fips.unique()) != all_NPP_counties_len:
        print(a_year)

# %%
all_GPP_counties_len = len(GPP.county_fips.unique())
print(all_GPP_counties_len)
for a_year in GPP.year.unique():
    df = GPP[GPP.year == a_year]
    if len(GPP.county_fips.unique()) != all_GPP_counties_len:
        print(a_year)

# %%
NPP_GPP = pd.merge(
    NPP[["year", "county_fips", "modis_npp"]],
    GPP[["year", "county_fips", "modis_GPP"]],
    on=["year", "county_fips"],
    how="left",
)

NPP_GPP = pd.merge(
    NPP_GPP,
    county_id_name_fips[["county_fips", "state"]],
    on=["county_fips"],
    how="left",
)
print(len(NPP_GPP.county_fips.unique()))
NPP_GPP.head(2)

# %%

# %%
