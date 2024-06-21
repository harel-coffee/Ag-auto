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
# # Convert Satate-level NDVI to regional

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_fips = abb_dict["state_fips"]
# state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
# state_fips_SoI.reset_index(drop=True, inplace=True)
# print (len(state_fips_SoI))
print (state_fips.shape)
state_fips.head(2)

# %%
# county_fips = abb_dict["county_fips"]

# print(f"{len(county_fips.state.unique()) = }")
# # county_fips = county_fips[county_fips.state.isin(SoI_abb)].copy()
# county_fips.drop_duplicates(inplace=True)
# county_fips.reset_index(drop=True, inplace=True)
# print(f"{len(county_fips.state.unique()) = }")

# county_fips = county_fips[["county_fips", "county_name", "state_fips", "EW_meridian"]]
# print(f"{len(county_fips.state_fips.unique()) = }")

# county_fips.head(2)

# %% [markdown]
# ### Rangeland areas

# %%
state_RA_area = pd.read_pickle(reOrganized_dir + "state_RA_area.sav")
state_RA_area = state_RA_area["state_RA_area"]
print (f"{state_RA_area.shape = }")
state_RA_area.rename(columns={"state_fip": "state_fips",
                              "rangeland_acre" : "state_rangeland_acre"}, inplace=True)

state_RA_area = pd.merge(state_RA_area, state_fips, on="state_fips", how="left")
state_RA_area.head(2)

# %%

# %% [markdown]
# ### NDVIs

# %%
NDVI_files = os.listdir(Min_data_dir_base)
NDVI_files = [x for x in NDVI_files if "NDVI" in x]
NDVI_files = [x for x in NDVI_files if "monthly" in x]
NDVI_files = [x for x in NDVI_files if "state" in x]
NDVI_files

# %%
NDVIs = pd.DataFrame()
for ii in range(len(NDVI_files)):
    curr_NDVI = pd.read_csv(Min_data_dir_base + NDVI_files[ii])
    source = NDVI_files[ii].split("_")[2]

    curr_NDVI.rename(columns={"statefips90m": "state_fips",
                              "NDVI" : source + "_NDVI"}, inplace=True)

    curr_NDVI["state_fips"] = curr_NDVI["state_fips"].astype(str)
    curr_NDVI["state_fips"] = curr_NDVI["state_fips"].str.slice(start=1, stop=3)


    if len(NDVIs) == 0:
        NDVIs = curr_NDVI.copy()
    else:
        NDVIs = pd.merge(NDVIs, curr_NDVI, on=["state_fips", "year", "month"], how="outer")
        
NDVIs.head(3)

# %%
NDVIs = pd.merge(NDVIs, state_RA_area, on=["state_fips",], how="left")
NDVIs.head(3)

# %%
# regions_dict = {"region_1" : ['CT', 'ME', 'NH', 'VT', 'MA', 'RI'], 
#                 "region_2" : ['NY', 'NJ'],
#                 "region_3" : ['DE', 'MD', 'PA', 'WV', 'VA'],
#                 "region_4" : ['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN'],
#                 "region_5" : ['IL', 'IN', 'MI', 'MN', 'OH', 'WI'],
#                 "region_6" : ['AR', 'LA', 'NM', 'OK', 'TX'],
#                 "region_7" : ['IA', 'KS', 'MO', 'NE'],
#                 "region_8" : ['CO', 'MT', 'ND', 'SD', 'UT', 'WY'],
#                 "region_9" : ['AZ', 'CA', 'HI', 'NV'],
#                 "region_10": ['AK', 'ID', 'OR', 'WA']}

shannon_regions_dict_abbr = {"region_1_region_2" : ['CT', 'ME', 'NH', 'VT', 'MA', 'RI', 'NY', 'NJ'], 
                             "region_3" : ['DE', 'MD', 'PA', 'WV', 'VA'],
                             "region_4" : ['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN'],
                             "region_5" : ['IL', 'IN', 'MI', 'MN', 'OH', 'WI'],
                             "region_6" : ['AR', 'LA', 'NM', 'OK', 'TX'],
                             "region_7" : ['IA', 'KS', 'MO', 'NE'],
                             "region_8" : ['CO', 'MT', 'ND', 'SD', 'UT', 'WY'],
                             "region_9" : ['AZ', 'CA', 'HI', 'NV'],
                             "region_10": ['AK', 'ID', 'OR', 'WA']}

# %%
ts = []
for a_key in shannon_regions_dict_abbr.keys():
    ts += shannon_regions_dict_abbr[a_key]
len(ts)

# %%
# shannon_regions_dict_fips = {}
# for a_key in shannon_regions_dict_abbr.keys():
#     curr_list = shannon_regions_dict_abbr[a_key]
#     curr_list_fips = []    
#     for a_state in curr_list:
#         curr_list_fips += [state_fips.loc[state_fips.state == a_state, "state_fips"].item()]
    
#     shannon_regions_dict_fips[a_key] = curr_list_fips
    
# ts = []
# for a_key in shannon_regions_dict_fips.keys():
#     ts += shannon_regions_dict_fips[a_key]
# len(ts)

# %%
# state_fips.loc[state_fips.state == a_state, "state_fips"].squeeze()
# state_fips.loc[state_fips.state == a_state, "state_fips"].item()

# %%

# %%
print ("states missing from NDVIs are: ")
print ([x for x in ts if not (x in NDVIs.state.unique())])
print ()
print ("states missing from Regions are: ")
print ([x for x in NDVIs.state.unique() if not (x in ts)])

# %% [markdown]
# ## Mismatch states 
#
# NDVI data is missing Alaska and Hawaii and Regions do not include DC.
#

# %%
print (len(NDVIs.state.unique()))
NDVIs[NDVIs.state.isin(["AK", "HI"])]

# %% [markdown]
# #### Drop DC from NDVI

# %%
NDVIs = NDVIs[NDVIs.state!='DC'].copy()
state_RA_area = state_RA_area[state_RA_area.state!='DC'].copy()

# %%
NDVIs.head(2)

# %%
NDVIs["region"] = "NA"
for a_key in shannon_regions_dict_abbr.keys():
    NDVIs.loc[NDVIs["state"].isin(shannon_regions_dict_abbr[a_key]), 'region'] = a_key
NDVIs.head(2)

# %%
# We must do this on state_RA_area. Otherwise, there are repetitions in NDVIs!
state_RA_area["region"] = "NA"
for a_key in shannon_regions_dict_abbr.keys():
    state_RA_area.loc[state_RA_area["state"].isin(shannon_regions_dict_abbr[a_key]), 'region'] = a_key

state_RA_area.head(2)

# %%
region_acres = state_RA_area.groupby(["region"])["state_rangeland_acre"].sum().reset_index()
region_acres.rename(columns={"state_rangeland_acre": "region_rangeland_acre"}, inplace=True)
region_acres.head(3)

# %%
state_RA_area = pd.merge(state_RA_area, region_acres, on=["region"], how="left")
state_RA_area.head(2)

# %%
state_RA_area["state_RA_div_region_RA"] = state_RA_area["state_rangeland_acre"] / \
                                                state_RA_area["region_rangeland_acre"]
state_RA_area.head(2)

# %%
NDVIs.head(2)

# %%
NDVIs = pd.merge(NDVIs, state_RA_area[["state_fips", "state_RA_div_region_RA"]], 
                 on=["state_fips"], how="left")
NDVIs.head(2)

# %%
NDVI_cols = ["AVHRR_NDVI", "MODIS_NDVI", "GIMMS_NDVI"]
for a_col in NDVI_cols:
    NDVIs["weighted_" + a_col] = NDVIs[a_col]* NDVIs["state_RA_div_region_RA"]
    
NDVIs.head(2)

# %%
# missing values turn to zero after groupby operation
# Let us replace it with -666 and then turn them to NAs.
w_NDVI_cols = ["weighted_AVHRR_NDVI", "weighted_MODIS_NDVI", "weighted_GIMMS_NDVI"]
regional_NDVIs = NDVIs.copy()
regional_NDVIs.fillna(-66666, inplace=True)
regional_NDVIs = regional_NDVIs.groupby(["year", "month", "region"])[w_NDVI_cols].sum().reset_index()
regional_NDVIs.head(3)

# %%
regional_NDVIs.loc[regional_NDVIs["weighted_AVHRR_NDVI"] < -1, "weighted_AVHRR_NDVI"] = np.nan
regional_NDVIs.loc[regional_NDVIs["weighted_MODIS_NDVI"] < -1, "weighted_MODIS_NDVI"] = np.nan
regional_NDVIs.loc[regional_NDVIs["weighted_GIMMS_NDVI"] < -1, "weighted_GIMMS_NDVI"] = np.nan
regional_NDVIs.rename(columns=lambda x: x.replace('weighted_', ''), inplace=True)

regional_NDVIs.head(2)

# %%
# pd.DataFrame.from_dict(shannon_regions_dict, orient='columns')

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

# beef_slaughter_tall = slaughter["beef_slaughter_tall"]

# There are some rows with NA in them
# slaughter['week'] = slaughter['week'].astype(int)
# beef_slaughter_tall.head(4)

# %%
beef_slaught_complete_months_tall = slaughter["beef_slaught_complete_months_tall"]
# beef_slaught_complete_months = slaughter["beef_slaught_complete_months_tall"]

beef_slaught_complete_months_tall.head(2)

# %%
group_cols = ["year", "month", "region"]
monthly_beef_slaughter = beef_slaught_complete_months_tall.groupby(group_cols)["slaughter_count"].sum()\
                              .reset_index()
monthly_beef_slaughter.head(2)

# %%
regional_NDVIs.head(2)

# %%
monthly_NDVI_beef_slaughter_tall = pd.merge(regional_NDVIs, monthly_beef_slaughter, 
                                            on=["year", "month", "region"], how="outer")
monthly_NDVI_beef_slaughter_tall.head(2)

# %%
monthly_NDVI_beef_slaughter_tall.region.unique()

# %%
monthly_NDVI_beef_slaughter_tall.head(2)

# %%
filename = reOrganized_dir + "monthly_NDVI_beef_slaughter.sav"

export_ = {"monthly_NDVI_beef_slaughter": monthly_NDVI_beef_slaughter_tall,
           "source_code" : "merge_regional_monthly_NDVI_Slaughter",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           "Author": "HN"}

pickle.dump(export_, open(filename, 'wb'))


# %%
df = monthly_NDVI_beef_slaughter_tall[monthly_NDVI_beef_slaughter_tall["region"] == "region_8"].copy()
df = df[["year", "region", "slaughter_count"]].copy()
df.dropna(how="any", inplace=True)
print (df.shape)

df = monthly_NDVI_beef_slaughter_tall[monthly_NDVI_beef_slaughter_tall["region"] == "region_9"].copy()
df = df[["year", "region", "slaughter_count"]].copy()
df.dropna(how="any", inplace=True)

print (df.shape)

# %%
monthly_NDVI_beef_slaughter_tall.head(2)

# %%
