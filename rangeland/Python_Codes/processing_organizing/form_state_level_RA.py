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

sys.path.append("/Users/hn/Documents/00_GitHub/Rangeland/Python_Codes/")
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
county_id_name_fips.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)

county_id_name_fips.sort_values(by=["state", "county"], inplace=True)

county_id_name_fips = rc.correct_Mins_FIPS(df=county_id_name_fips, col_="county")
county_id_name_fips.rename(columns={"county": "county_fips"}, inplace=True)

county_id_name_fips["state_fip"] = county_id_name_fips.county_fips.str.slice(0, 2)

county_id_name_fips.reset_index(drop=True, inplace=True)
print (len(county_id_name_fips.state.unique()))
county_id_name_fips.head(2)

# %%
# Rangeland area and Total area:
county_RA_and_TA_fraction = pd.read_csv(reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv")
county_RA_and_TA_fraction.rename(columns={"fips_id": "county_fips"}, inplace=True)

county_RA_and_TA_fraction = rc.correct_Mins_FIPS(df=county_RA_and_TA_fraction, col_="county_fips")
L = len(county_RA_and_TA_fraction.county_fips.unique())
print ("number of counties are {}.".format(L))
print (county_RA_and_TA_fraction.shape)
county_RA_and_TA_fraction.head(2)

# %%
county_RA_and_TA_fraction["state_fip"] = county_RA_and_TA_fraction.county_fips.str.slice(0, 2)
county_RA_and_TA_fraction.head(2)

# %%
state_RA = county_RA_and_TA_fraction[["state_fip", "rangeland_acre"]].groupby("state_fip").sum()
state_RA.reset_index(drop=False, inplace=True)
state_RA = state_RA[state_RA.state_fip.isin(county_id_name_fips.state_fip)]
state_RA.reset_index(drop=True, inplace=True)
state_RA.head(2)

# %%
state_area = county_RA_and_TA_fraction[["state_fip", "county_area_acre"]].groupby("state_fip").sum()
state_area.rename(columns={"county_area_acre": "state_area_acre"}, inplace=True)

state_area.reset_index(drop=False, inplace=True)
state_area = state_area[state_area.state_fip.isin(county_id_name_fips.state_fip)]
state_area.reset_index(drop=True, inplace=True)
state_area.head(2)

# %%
state_RA_area = pd.merge(state_RA, state_area, on = ["state_fip"], how = "left")
state_RA_area.head(2)

# %%
filename = reOrganized_dir + "state_RA_area.sav"

export_ = {"state_RA_area": state_RA_area, 
           "source_code" : "form_state_level_RA",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%

# %%
