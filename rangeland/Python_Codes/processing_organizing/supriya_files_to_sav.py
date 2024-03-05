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
# ```county_herb_ratio_colab.sav``` and ```state_herb_ratio.sav``` are created on CoLab.

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
# state_herb_ratio = pd.read_pickle(data_dir_base + "Supriya/Nov30_HerbRatio/" + \
#                                              "state_herb_ratio_colab.sav")
# state_herb_ratio = state_herb_ratio["state_herb_ratio"]
# state_herb_ratio.head(2)

# %%
abb_dict = pd.read_pickle(param_dir + "county_fips.sav")

SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

county_fips = abb_dict["county_fips"]
state_fips = abb_dict["state_fips"]

state_fips.rename(columns={"state_fip": "state_fips"}, inplace=True)
state_fips.head(2)

# %%
state_fips[state_fips.state_fips == "34"]

# %%
State_NDVIDOY_Herb = pd.read_csv(
    data_dir_base + "Supriya/" + "State_NDVIDOY_Herb_Feb82024.csv"
)

State_NDVIDOY_Herb.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

cols_ = [
    "statefp",
    "stusps",
    "herbcount",
    "herbmean",
    "herbstdev",
    "vi_doymean",
    "vi_doymedi",
    "vi_doystde",
    "vi_doymin",
    "vi_doymax",
]
State_NDVIDOY_Herb = State_NDVIDOY_Herb[cols_]


State_NDVIDOY_Herb.sort_values(by=["statefp"], inplace=True)
State_NDVIDOY_Herb.reset_index(drop=True, inplace=True)
State_NDVIDOY_Herb.head(2)

# %%
State_NDVIDOY_Herb.rename(
    columns={
        "statefp": "state_fips",
        "stusps": "state",
        "herbcount": "pixel_count",
        "herbmean": "herb_avg",
        "herbstdev": "herb_std",
        "vi_doymean": "maxNDVI_DoY_stateMean",
        "vi_doymedi": "maxNDVI_DoY_stateMedian",
        "vi_doystde": "maxNDVI_DoY_stateStd",
        "vi_doymin": "maxNDVI_DoY_stateMin",
        "vi_doymax": "maxNDVI_DoY_stateMax",
    },
    inplace=True,
)

State_NDVIDOY_Herb = rc.correct_state_int_fips_to_str(
    df=State_NDVIDOY_Herb, col_="state_fips"
)
State_NDVIDOY_Herb.head(2)

# %%

# %%

# %%
County_NDVIDOY_Herb = pd.read_csv(
    data_dir_base + "Supriya/" + "County_NDVIDOY_Herb_Feb82024.csv",
    encoding="unicode_escape",
)
County_NDVIDOY_Herb.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

cols_ = [
    "statefp",
    "countyfp",
    "geoid",
    "name",
    "stusps",
    "herb_avgme",
    "herb_sdstd",
    "pixelscoun",
    "vi_doymean",
    "vi_doymedi",
    "vi_doystde",
    "vi_doymin",
    "vi_doymax",
]

County_NDVIDOY_Herb.sort_values(by=["statefp"], inplace=True)
County_NDVIDOY_Herb.reset_index(drop=True, inplace=True)

County_NDVIDOY_Herb = County_NDVIDOY_Herb[cols_]

County_NDVIDOY_Herb.head(2)

# %%
County_NDVIDOY_Herb.rename(
    columns={
        "statefp": "state_fips",
        "countyfp": "county_fips_standAlone",
        "geoid": "county_fips",
        "name": "county_name",
        "stusps": "state",
        "herb_avgme": "herb_avg",
        "herb_sdstd": "herb_std",
        "pixelscoun": "pixel_count",
        "vi_doymean": "maxNDVI_DoY_countyMean",
        "vi_doymedi": "maxNDVI_DoY_countyMedian",
        "vi_doystde": "maxNDVI_DoY_countyStd",
        "vi_doymin": "maxNDVI_DoY_countyMin",
        "vi_doymax": "maxNDVI_DoY_countyMax",
    },
    inplace=True,
)

County_NDVIDOY_Herb.head(2)

# %%
County_NDVIDOY_Herb = rc.correct_state_int_fips_to_str(
    df=County_NDVIDOY_Herb, col_="state_fips"
)
County_NDVIDOY_Herb = rc.correct_4digitFips(df=County_NDVIDOY_Herb, col_="county_fips")

County_NDVIDOY_Herb = rc.correct_2digit_countyStandAloneFips(
    df=County_NDVIDOY_Herb, col_="county_fips_standAlone"
)
County_NDVIDOY_Herb.head(2)

# %%
County_NDVIDOY_Herb.drop(
    columns=["state_fips", "state", "county_fips_standAlone", "county_name"],
    inplace=True,
)

County_NDVIDOY_Herb.sort_values(by=["county_fips"], inplace=True)
County_NDVIDOY_Herb.reset_index(drop=True, inplace=True)
County_NDVIDOY_Herb.head(2)

# %%
from datetime import datetime
import pickle

filename = reOrganized_dir + "county_state_NDVIDOY_Herb.sav"

export_ = {
    "State_NDVIDOY_Herb": State_NDVIDOY_Herb,
    "County_NDVIDOY_Herb": County_NDVIDOY_Herb,
    "source_code": "supriya_files_to_sav",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %%

# %%

# %%
