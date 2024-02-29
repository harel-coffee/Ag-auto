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

# %% [markdown]
# I want to see if there is any difference between my way of going from county to state-level NDVI and Min's state-level NDVI.

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

# %% [markdown]
# ## counties and state fips

# %%
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI = abb_dict['SoI']
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
county_fips = abb_dict["county_fips"]
county_fips = county_fips[county_fips.state.isin(SoI_abb)]
SoI_abb_fips = county_fips.state_fips.unique()
print (len(SoI_abb_fips))
SoI_abb_fips

# %%
# county_fips = pd.read_pickle(reOrganized_dir + "county_fips.sav")
# county_fips = county_fips["county_fips"]

# L=len(county_fips[county_fips.state == "SD"])
# print ("number of counties in SD is {}".format(L))
# county_fips = county_fips[county_fips.state.isin(SoI_abb)]
# county_fips.reset_index(drop=True, inplace=True)
# county_fips.head(2)

# %%
# Min_files = os.listdir(Min_data_base)
# Min_files = [x for x in Min_files if x.endswith(".csv")]
# Min_files = [x for x in Min_files if "NDVI" in x]
# Min_files = [x for x in Min_files if "county_monthly" in x]

Min_files = ["statefips_monthly_AVHRR_NDVI.csv",
             "statefips_monthly_GIMMS_NDVI.csv",
             "statefips_monthly_MODIS_NDVI.csv"]

# %%
file_name = Min_files[0]
print (file_name)
print ("------------------------------------------------")

ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"statefips90m": "state_fips"}, inplace=True)

ndvi = rc.correct_3digitStateFips_Min(df=ndvi, col_ = "state_fips")
ndvi = ndvi[ndvi.state_fips.isin(SoI_abb_fips)]
print (ndvi.head(2))
print ("------------------------------------------------")
##############################################################################
###### Three new vars (Max NDVI, Max NDVI DoY, and NDVI std)
######
max_ndvi_per_yr_idx_df = ndvi.groupby(["state_fips", "year"])["NDVI"].idxmax().reset_index()
max_ndvi_per_yr_idx_df.rename(columns={"NDVI": "max_ndvi_per_year_idx"}, inplace=True)
max_ndvi_per_yr_idx_df.head(2)

max_ndvi_per_yr = ndvi.loc[max_ndvi_per_yr_idx_df.max_ndvi_per_year_idx]

max_ndvi_per_yr.rename(columns={"month" : "max_ndvi_month",
                                "NDVI" : "max_ndvi_in_year" }, inplace=True)

max_ndvi_per_yr.head(2)

max_ndvi_per_yr = max_ndvi_per_yr[["state_fips", "year", "max_ndvi_month", "max_ndvi_in_year"]]
max_ndvi_per_yr.sort_values(by=["state_fips", "year"], inplace=True)
max_ndvi_per_yr.reset_index(drop=True, inplace=True)
max_ndvi_per_yr.head(2)

ndvi_variance = ndvi.groupby(["state_fips"])["NDVI"].std(ddof=1).reset_index()
ndvi_variance.rename(columns={"NDVI": "ndvi_std"}, inplace=True)
ndvi_variance.head(2)
# ndvi_variance.to_frame()
ndvi_three_vars = pd.merge(max_ndvi_per_yr, ndvi_variance, on=["state_fips"], how="outer")
print (ndvi_three_vars.head(2))
print ("------------------------------------------------")
##############################################################################

ndvi["season"] = "s5"
ndvi['season'] = np.where(ndvi['month'].isin([1, 2, 3]), 's1', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([4, 5, 6, 7]), 's2', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([8, 9]), 's3', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([10, 11, 12]), 's4', ndvi.season)

ndvi = ndvi[["year",  "state_fips", "season", "NDVI"]]
ndvi = ndvi.groupby(["year", "state_fips", "season"]).mean().reset_index()
ndvi.head(2)

ndvi = ndvi.pivot(index=['year', 'state_fips'], columns='season', values='NDVI')
ndvi.reset_index(drop=False, inplace=True)
ndvi.head(2)

if "MODIS" in file_name:
    new_name = "modis_ndvi"
elif "AVHRR" in file_name:
    new_name = "avhrr_ndvi"
elif "GIMMS" in file_name:
    new_name = "gimms_ndvi"

ndvi.rename(columns={"s1": "s1_mean_" + new_name,
                     "s2": "s2_mean_" + new_name,
                     "s3": "s3_mean_" + new_name,
                     "s4": "s4_mean_" + new_name}, inplace=True)

seasonal_ndvi_avhrr = ndvi.copy()
del(ndvi)
seasonal_ndvi_avhrr.head(3)

seasonal_ndvi_avhrr = pd.merge(seasonal_ndvi_avhrr, ndvi_three_vars, on=["state_fips", "year"], how="outer")
postfix = "_" + file_name.split("_")[2].lower()
seasonal_ndvi_avhrr.rename(columns={"max_ndvi_month" : "max_ndvi_month" + postfix,
                                    "max_ndvi_in_year" : "max_ndvi_in_year" + postfix,
                                    "ndvi_std" : "ndvi_std" + postfix}, inplace=True)

seasonal_ndvi_avhrr.head(2)

# %%
seasonal_ndvi_avhrr["max_ndvi_season_avhrr"] = 666
seasonal_ndvi_avhrr['max_ndvi_season_avhrr'] = np.where(
    seasonal_ndvi_avhrr['max_ndvi_month_avhrr'].isin([1, 2, 3]), 1, seasonal_ndvi_avhrr.max_ndvi_season_avhrr)

seasonal_ndvi_avhrr['max_ndvi_season_avhrr'] = np.where(
    seasonal_ndvi_avhrr['max_ndvi_month_avhrr'].isin([4, 5, 6, 7]), 2, seasonal_ndvi_avhrr.max_ndvi_season_avhrr)

seasonal_ndvi_avhrr['max_ndvi_season_avhrr'] = np.where(
    seasonal_ndvi_avhrr['max_ndvi_month_avhrr'].isin([8, 9]), 3, seasonal_ndvi_avhrr.max_ndvi_season_avhrr)

seasonal_ndvi_avhrr['max_ndvi_season_avhrr'] = np.where(
    seasonal_ndvi_avhrr['max_ndvi_month_avhrr'].isin([10, 11, 12]), 4, seasonal_ndvi_avhrr.max_ndvi_season_avhrr)

######################################################################
seasonal_ndvi_avhrr["max_ndvi_season_avhrr_str"] = seasonal_ndvi_avhrr["max_ndvi_season_avhrr"].astype(str)
seasonal_ndvi_avhrr["max_ndvi_season_avhrr_str"] = "s" + seasonal_ndvi_avhrr["max_ndvi_season_avhrr_str"]
######################################################################
L = list(seasonal_ndvi_avhrr.columns)
L.pop()
seasonal_ndvi_avhrr = seasonal_ndvi_avhrr.pivot(index=L, 
                                                columns='max_ndvi_season_avhrr_str', 
                                                values='max_ndvi_season_avhrr')

seasonal_ndvi_avhrr.reset_index(drop=False, inplace=True)
seasonal_ndvi_avhrr.columns = seasonal_ndvi_avhrr.columns.values

seasonal_ndvi_avhrr.rename(columns={"s1": "max_ndvi_season_avhrr_s1",
                                    "s2": "max_ndvi_season_avhrr_s2",
                                    "s3": "max_ndvi_season_avhrr_s3",
                                    "s4": "max_ndvi_season_avhrr_s4"}, inplace=True)

seasonal_ndvi_avhrr.head(2)
######################################################################

seasonal_ndvi_avhrr.fillna(value = {"max_ndvi_season_avhrr_s1": 0,
                                    "max_ndvi_season_avhrr_s2": 0,
                                    "max_ndvi_season_avhrr_s3": 0, 
                                    "max_ndvi_season_avhrr_s4": 0},
                          inplace=True)

c = "max_ndvi_season_avhrr_s1"
seasonal_ndvi_avhrr[c] = np.where(seasonal_ndvi_avhrr[c].isin([1, 2, 3, 4]), 1, seasonal_ndvi_avhrr[c])
seasonal_ndvi_avhrr[c] = seasonal_ndvi_avhrr[c].astype(int)

c = "max_ndvi_season_avhrr_s2"
seasonal_ndvi_avhrr[c] = np.where(seasonal_ndvi_avhrr[c].isin([1, 2, 3, 4]), 1, seasonal_ndvi_avhrr[c])
seasonal_ndvi_avhrr[c] = seasonal_ndvi_avhrr[c].astype(int)

c = "max_ndvi_season_avhrr_s3"
seasonal_ndvi_avhrr[c] = np.where(seasonal_ndvi_avhrr[c].isin([1, 2, 3, 4]), 1, seasonal_ndvi_avhrr[c])
seasonal_ndvi_avhrr[c] = seasonal_ndvi_avhrr[c].astype(int)

c = "max_ndvi_season_avhrr_s4"
seasonal_ndvi_avhrr[c] = np.where(seasonal_ndvi_avhrr[c].isin([1, 2, 3, 4]), 1, seasonal_ndvi_avhrr[c])
seasonal_ndvi_avhrr[c] = seasonal_ndvi_avhrr[c].astype(int)

seasonal_ndvi_avhrr.head(2)

# %%

# %%
file_name = Min_files[1]
print (file_name)
print ("------------------------------------------------")

ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"statefips90m": "state_fips"}, inplace=True)
ndvi = rc.correct_3digitStateFips_Min(df=ndvi, col_ = "state_fips")
ndvi = ndvi[ndvi.state_fips.isin(SoI_abb_fips)]

print (ndvi.head(2))
print ("------------------------------------------------")
##############################################################################
###### Three new vars (Max NDVI, Max NDVI DoY, and NDVI std)
######
max_ndvi_per_yr_idx_df = ndvi.groupby(["state_fips", "year"])["NDVI"].idxmax().reset_index()
max_ndvi_per_yr_idx_df.rename(columns={"NDVI": "max_ndvi_per_year_idx"}, inplace=True)
max_ndvi_per_yr_idx_df.head(2)

max_ndvi_per_yr = ndvi.loc[max_ndvi_per_yr_idx_df.max_ndvi_per_year_idx]

max_ndvi_per_yr.rename(columns={"month" : "max_ndvi_month",
                                "NDVI" : "max_ndvi_in_year" }, inplace=True)

max_ndvi_per_yr.head(2)

max_ndvi_per_yr = max_ndvi_per_yr[["state_fips", "year", "max_ndvi_month", "max_ndvi_in_year"]]
max_ndvi_per_yr.sort_values(by=["state_fips", "year"], inplace=True)
max_ndvi_per_yr.reset_index(drop=True, inplace=True)
max_ndvi_per_yr.head(2)

ndvi_variance = ndvi.groupby(["state_fips"])["NDVI"].std(ddof=1).reset_index()
ndvi_variance.rename(columns={"NDVI": "ndvi_std"}, inplace=True)
ndvi_variance.head(2)

# ndvi_variance.to_frame()

ndvi_three_vars = pd.merge(max_ndvi_per_yr, ndvi_variance, on=["state_fips"], how="outer")
print (ndvi_three_vars.head(2))
print ("------------------------------------------------")
##############################################################################

ndvi["season"] = "s5"
ndvi['season'] = np.where(ndvi['month'].isin([1, 2, 3]), 's1', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([4, 5, 6, 7]), 's2', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([8, 9]), 's3', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([10, 11, 12]), 's4', ndvi.season)

ndvi = ndvi[["year",  "state_fips", "season", "NDVI"]]
ndvi = ndvi.groupby(["year", "state_fips", "season"]).mean().reset_index()
ndvi.head(2)


ndvi = ndvi.pivot(index=['year', 'state_fips'], columns='season', values='NDVI')
ndvi.reset_index(drop=False, inplace=True)
ndvi.head(2)

if "MODIS" in file_name:
    new_name = "modis_ndvi"
elif "AVHRR" in file_name:
    new_name = "avhrr_ndvi"
elif "GIMMS" in file_name:
    new_name = "gimms_ndvi"

ndvi.rename(columns={"s1": "s1_mean_" + new_name,
                     "s2": "s2_mean_" + new_name,
                     "s3": "s3_mean_" + new_name,
                     "s4": "s4_mean_" + new_name}, inplace=True)

seasonal_ndvi_gimms = ndvi.copy()
del(ndvi)
seasonal_ndvi_gimms.head(3)

seasonal_ndvi_gimms = pd.merge(seasonal_ndvi_gimms, ndvi_three_vars, on=["state_fips", "year"], how="outer")
postfix = "_" + file_name.split("_")[2].lower()
seasonal_ndvi_gimms.rename(columns={"max_ndvi_month" : "max_ndvi_month" + postfix,
                                    "max_ndvi_in_year" : "max_ndvi_in_year" + postfix,
                                    "ndvi_std" : "ndvi_std" + postfix}, inplace=True)

seasonal_ndvi_gimms.head(2)

# %%
seasonal_ndvi_gimms["max_ndvi_season_gimms"] = 666

seasonal_ndvi_gimms['max_ndvi_season_gimms'] = np.where(
    seasonal_ndvi_gimms['max_ndvi_month_gimms'].isin([1, 2, 3]), 1, seasonal_ndvi_gimms.max_ndvi_season_gimms)

seasonal_ndvi_gimms['max_ndvi_season_gimms'] = np.where(
    seasonal_ndvi_gimms['max_ndvi_month_gimms'].isin([4, 5, 6, 7]), 2, seasonal_ndvi_gimms.max_ndvi_season_gimms)

seasonal_ndvi_gimms['max_ndvi_season_gimms'] = np.where(
    seasonal_ndvi_gimms['max_ndvi_month_gimms'].isin([8, 9]), 3, seasonal_ndvi_gimms.max_ndvi_season_gimms)

seasonal_ndvi_gimms['max_ndvi_season_gimms'] = np.where(
    seasonal_ndvi_gimms['max_ndvi_month_gimms'].isin([10, 11, 12]), 4, seasonal_ndvi_gimms.max_ndvi_season_gimms)
######################################################################
seasonal_ndvi_gimms["max_ndvi_season_gimms_str"] = seasonal_ndvi_gimms["max_ndvi_season_gimms"].astype(str)
seasonal_ndvi_gimms["max_ndvi_season_gimms_str"] = "s" + seasonal_ndvi_gimms["max_ndvi_season_gimms_str"]
######################################################################
L = list(seasonal_ndvi_gimms.columns)
L.pop()
seasonal_ndvi_gimms = seasonal_ndvi_gimms.pivot(index=L, 
                                                columns='max_ndvi_season_gimms_str', 
                                                values='max_ndvi_season_gimms')

seasonal_ndvi_gimms.reset_index(drop=False, inplace=True)
seasonal_ndvi_gimms.columns = seasonal_ndvi_gimms.columns.values

seasonal_ndvi_gimms.rename(columns={"s1": "max_ndvi_season_gimms_s1",
                                    "s2": "max_ndvi_season_gimms_s2",
                                    "s3": "max_ndvi_season_gimms_s3",
                                    "s4": "max_ndvi_season_gimms_s4"}, inplace=True)

seasonal_ndvi_gimms.head(2)
######################################################################
seasonal_ndvi_gimms.fillna(value = {"max_ndvi_season_gimms_s1": 0,
                                    "max_ndvi_season_gimms_s2": 0,
                                    "max_ndvi_season_gimms_s3": 0, 
                                    "max_ndvi_season_gimms_s4": 0},
                          inplace=True)

c = "max_ndvi_season_gimms_s1"
seasonal_ndvi_gimms[c] = np.where(seasonal_ndvi_gimms[c].isin([1, 2, 3, 4]), 1, seasonal_ndvi_gimms[c])
seasonal_ndvi_gimms[c] = seasonal_ndvi_gimms[c].astype(int)

c = "max_ndvi_season_gimms_s2"
seasonal_ndvi_gimms[c] = np.where(seasonal_ndvi_gimms[c].isin([1, 2, 3, 4]), 1, seasonal_ndvi_gimms[c])
seasonal_ndvi_gimms[c] = seasonal_ndvi_gimms[c].astype(int)

c = "max_ndvi_season_gimms_s3"
seasonal_ndvi_gimms[c] = np.where(seasonal_ndvi_gimms[c].isin([1, 2, 3, 4]), 1, seasonal_ndvi_gimms[c])
seasonal_ndvi_gimms[c] = seasonal_ndvi_gimms[c].astype(int)

c = "max_ndvi_season_gimms_s4"
seasonal_ndvi_gimms[c] = np.where(seasonal_ndvi_gimms[c].isin([1, 2, 3, 4]), 1, seasonal_ndvi_gimms[c])
seasonal_ndvi_gimms[c] = seasonal_ndvi_gimms[c].astype(int)

seasonal_ndvi_gimms.head(2)

# %%
file_name = Min_files[2]
print (file_name)
print ("------------------------------------------------")

ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"statefips90m": "state_fips"}, inplace=True)
ndvi = rc.correct_3digitStateFips_Min(df=ndvi, col_ = "state_fips")
ndvi = ndvi[ndvi.state_fips.isin(SoI_abb_fips)]
print (ndvi.head(2))
print ("------------------------------------------------")
##############################################################################
###### Three new vars (Max NDVI, Max NDVI DoY, and NDVI std)
######
max_ndvi_per_yr_idx_df = ndvi.groupby(["state_fips", "year"])["NDVI"].idxmax().reset_index()
max_ndvi_per_yr_idx_df.rename(columns={"NDVI": "max_ndvi_per_year_idx"}, inplace=True)
max_ndvi_per_yr_idx_df.head(2)

max_ndvi_per_yr = ndvi.loc[max_ndvi_per_yr_idx_df.max_ndvi_per_year_idx]

max_ndvi_per_yr.rename(columns={"month" : "max_ndvi_month",
                                "NDVI" : "max_ndvi_in_year" }, inplace=True)

max_ndvi_per_yr.head(2)

max_ndvi_per_yr = max_ndvi_per_yr[["state_fips", "year", "max_ndvi_month", "max_ndvi_in_year"]]
max_ndvi_per_yr.sort_values(by=["state_fips", "year"], inplace=True)
max_ndvi_per_yr.reset_index(drop=True, inplace=True)
max_ndvi_per_yr.head(2)

ndvi_variance = ndvi.groupby(["state_fips"])["NDVI"].std(ddof=1).reset_index()
ndvi_variance.rename(columns={"NDVI": "ndvi_std"}, inplace=True)
ndvi_variance.head(2)

# ndvi_variance.to_frame()

ndvi_three_vars = pd.merge(max_ndvi_per_yr, ndvi_variance, on=["state_fips"], how="outer")
print (ndvi_three_vars.head(2))
print ("------------------------------------------------")
##############################################################################

ndvi["season"] = "s5"
ndvi['season'] = np.where(ndvi['month'].isin([1, 2, 3]), 's1', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([4, 5, 6, 7]), 's2', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([8, 9]), 's3', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([10, 11, 12]), 's4', ndvi.season)

ndvi = ndvi[["year",  "state_fips", "season", "NDVI"]]
ndvi = ndvi.groupby(["year", "state_fips", "season"]).mean().reset_index()
ndvi.head(2)


ndvi = ndvi.pivot(index=['year', 'state_fips'], columns='season', values='NDVI')
ndvi.reset_index(drop=False, inplace=True)
ndvi.head(2)

if "MODIS" in file_name:
    new_name = "modis_ndvi"
elif "AVHRR" in file_name:
    new_name = "avhrr_ndvi"
elif "GIMMS" in file_name:
    new_name = "gimms_ndvi"

ndvi.rename(columns={"s1": "s1_mean_" + new_name,
                     "s2": "s2_mean_" + new_name,
                     "s3": "s3_mean_" + new_name,
                     "s4": "s4_mean_" + new_name}, inplace=True)

seasonal_ndvi_modis = ndvi.copy()
del(ndvi)
seasonal_ndvi_modis.head(3)

seasonal_ndvi_modis = pd.merge(seasonal_ndvi_modis, ndvi_three_vars, on=["state_fips", "year"], how="outer")
postfix = "_" + file_name.split("_")[2].lower()
seasonal_ndvi_modis.rename(columns={"max_ndvi_month" : "max_ndvi_month" + postfix,
                                    "max_ndvi_in_year" : "max_ndvi_in_year" + postfix,
                                    "ndvi_std" : "ndvi_std" + postfix}, inplace=True)

seasonal_ndvi_modis.head(2)

# %%
seasonal_ndvi_modis["max_ndvi_season_modis"] = 666

seasonal_ndvi_modis['max_ndvi_season_modis'] = np.where(
    seasonal_ndvi_modis['max_ndvi_month_modis'].isin([1, 2, 3]), 1, seasonal_ndvi_modis.max_ndvi_season_modis)

seasonal_ndvi_modis['max_ndvi_season_modis'] = np.where(
    seasonal_ndvi_modis['max_ndvi_month_modis'].isin([4, 5, 6, 7]), 2, seasonal_ndvi_modis.max_ndvi_season_modis)

seasonal_ndvi_modis['max_ndvi_season_modis'] = np.where(
    seasonal_ndvi_modis['max_ndvi_month_modis'].isin([8, 9]), 3, seasonal_ndvi_modis.max_ndvi_season_modis)

seasonal_ndvi_modis['max_ndvi_season_modis'] = np.where(
    seasonal_ndvi_modis['max_ndvi_month_modis'].isin([10, 11, 12]), 4, seasonal_ndvi_modis.max_ndvi_season_modis)
######################################################################
seasonal_ndvi_modis["max_ndvi_season_modis_str"] = seasonal_ndvi_modis["max_ndvi_season_modis"].astype(str)
seasonal_ndvi_modis["max_ndvi_season_modis_str"] = "s" + seasonal_ndvi_modis["max_ndvi_season_modis_str"]
######################################################################
L = list(seasonal_ndvi_modis.columns)
L.pop()
seasonal_ndvi_modis = seasonal_ndvi_modis.pivot(index=L, 
                                                columns='max_ndvi_season_modis_str', 
                                                values='max_ndvi_season_modis')

seasonal_ndvi_modis.reset_index(drop=False, inplace=True)
seasonal_ndvi_modis.columns = seasonal_ndvi_modis.columns.values

seasonal_ndvi_modis.rename(columns={"s1": "max_ndvi_season_modis_s1",
                                    "s2": "max_ndvi_season_modis_s2",
                                    "s3": "max_ndvi_season_modis_s3",
                                    "s4": "max_ndvi_season_modis_s4"}, inplace=True)

seasonal_ndvi_modis.head(2)
######################################################################
seasonal_ndvi_modis.fillna(value = {"max_ndvi_season_modis_s1": 0,
                                    "max_ndvi_season_modis_s2": 0,
                                    "max_ndvi_season_modis_s3": 0, 
                                    "max_ndvi_season_modis_s4": 0},
                          inplace=True)

c = "max_ndvi_season_modis_s1"
seasonal_ndvi_modis[c] = np.where(seasonal_ndvi_modis[c].isin([1, 2, 3, 4]), 1, seasonal_ndvi_modis[c])
seasonal_ndvi_modis[c] = seasonal_ndvi_modis[c].astype(int)

c = "max_ndvi_season_modis_s2"
seasonal_ndvi_modis[c] = np.where(seasonal_ndvi_modis[c].isin([1, 2, 3, 4]), 1, seasonal_ndvi_modis[c])
seasonal_ndvi_modis[c] = seasonal_ndvi_modis[c].astype(int)

c = "max_ndvi_season_modis_s3"
seasonal_ndvi_modis[c] = np.where(seasonal_ndvi_modis[c].isin([1, 2, 3, 4]), 1, seasonal_ndvi_modis[c])
seasonal_ndvi_modis[c] = seasonal_ndvi_modis[c].astype(int)

c = "max_ndvi_season_modis_s4"
seasonal_ndvi_modis[c] = np.where(seasonal_ndvi_modis[c].isin([1, 2, 3, 4]), 1, seasonal_ndvi_modis[c])
seasonal_ndvi_modis[c] = seasonal_ndvi_modis[c].astype(int)

seasonal_ndvi_modis.head(2)

# %%
print(f"{seasonal_ndvi_avhrr.shape = }")
print(f"{seasonal_ndvi_gimms.shape = }")
print(f"{seasonal_ndvi_modis.shape = }")

# %%

# %%
print(len(seasonal_ndvi_avhrr.state_fips.unique()))
print(len(seasonal_ndvi_gimms.state_fips.unique()))
print(len(seasonal_ndvi_modis.state_fips.unique()))

# %%
seasonal_ndvi_modis.head(2)

# %%
seasonal_ndvi_mean = pd.merge(seasonal_ndvi_avhrr, seasonal_ndvi_gimms, on=["state_fips", "year"], how="outer")
seasonal_ndvi_mean = pd.merge(seasonal_ndvi_mean, seasonal_ndvi_modis, on=["state_fips", "year"], how="outer")
seasonal_ndvi_mean.head(2)

# %%
print(seasonal_ndvi_mean.shape)
print(seasonal_ndvi_avhrr.shape)

# %%
del(seasonal_ndvi_avhrr, seasonal_ndvi_gimms, seasonal_ndvi_modis)

# %% [markdown]
# ###   Max start here


# %%
file_name = Min_files[0]
print (f"{file_name = }")

ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"statefips90m": "state_fips"}, inplace=True)
ndvi = rc.correct_3digitStateFips_Min(df=ndvi, col_ = "state_fips")
ndvi = ndvi[ndvi.state_fips.isin(SoI_abb_fips)]

ndvi = ndvi[["year", "month", "NDVI", "state_fips"]]

ndvi = ndvi[ndvi.state_fips.isin(county_fips.state_fips.unique())]
ndvi.reset_index(drop=True, inplace=True)

ndvi.head(2)

ndvi["season"] = "s5"

ndvi['season'] = np.where(ndvi['month'].isin([1, 2, 3]), 's1', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([4, 5, 6, 7]), 's2', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([8, 9]), 's3', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([10, 11, 12]), 's4', ndvi.season)

ndvi = ndvi[["year",  "state_fips", "season", "NDVI"]]
ndvi = ndvi.groupby(["year", "state_fips", "season"]).max().reset_index()
ndvi.head(2)

ndvi = ndvi.pivot(index=['year', 'state_fips'], columns='season', values='NDVI')
ndvi.reset_index(drop=False, inplace=True)
ndvi.head(2)

if "MODIS" in file_name:
    new_name = "modis_ndvi"
elif "AVHRR" in file_name:
    new_name = "avhrr_ndvi"
elif "GIMMS" in file_name:
    new_name = "gimms_ndvi"

ndvi.rename(columns={"s1": "s1_max_" + new_name,
                     "s2": "s2_max_" + new_name,
                     "s3": "s3_max_" + new_name,
                     "s4": "s4_max_" + new_name}, inplace=True)
seasonal_ndvi_avhrr = ndvi.copy()
del(ndvi)
seasonal_ndvi_avhrr.head(3)

# %%
file_name = Min_files[1]
print (f"{file_name = }")

ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"statefips90m": "state_fips"}, inplace=True)
ndvi = rc.correct_3digitStateFips_Min(df=ndvi, col_ = "state_fips")
ndvi = ndvi[ndvi.state_fips.isin(SoI_abb_fips)]
ndvi = ndvi[["year", "month", "NDVI", "state_fips"]]

ndvi = ndvi[ndvi.state_fips.isin(county_fips.state_fips.unique())]
ndvi.reset_index(drop=True, inplace=True)
########################################
ndvi["season"] = "s5"

ndvi['season'] = np.where(ndvi['month'].isin([1, 2, 3]), 's1', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([4, 5, 6, 7]), 's2', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([8, 9]), 's3', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([10, 11, 12]), 's4', ndvi.season)

ndvi = ndvi[["year",  "state_fips", "season", "NDVI"]]
ndvi = ndvi.groupby(["year", "state_fips", "season"]).max().reset_index()
ndvi.head(2)

ndvi = ndvi.pivot(index=['year', 'state_fips'], columns='season', values='NDVI')
ndvi.reset_index(drop=False, inplace=True)
ndvi.head(2)


if "MODIS" in file_name:
    new_name = "modis_ndvi"
elif "AVHRR" in file_name:
    new_name = "avhrr_ndvi"
elif "GIMMS" in file_name:
    new_name = "gimms_ndvi"

ndvi.rename(columns={"s1": "s1_max_" + new_name,
                     "s2": "s2_max_" + new_name,
                     "s3": "s3_max_" + new_name,
                     "s4": "s4_max_" + new_name}, inplace=True)


seasonal_ndvi_gimms = ndvi.copy()
del(ndvi)
seasonal_ndvi_gimms.head(3)

# %%

# %%
file_name = Min_files[2]
print (f"{file_name = }")
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"statefips90m": "state_fips"}, inplace=True)
ndvi = rc.correct_3digitStateFips_Min(df=ndvi, col_ = "state_fips")
ndvi = ndvi[ndvi.state_fips.isin(SoI_abb_fips)]

ndvi = ndvi[["year", "month", "NDVI", "state_fips"]]
ndvi = ndvi[ndvi.state_fips.isin(county_fips.state_fips.unique())]
ndvi.reset_index(drop=True, inplace=True)
ndvi.head(2)
##################################################
ndvi["season"] = "s5"

ndvi['season'] = np.where(ndvi['month'].isin([1, 2, 3]), 's1', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([4, 5, 6, 7]), 's2', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([8, 9]), 's3', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([10, 11, 12]), 's4', ndvi.season)

ndvi = ndvi[["year",  "state_fips", "season", "NDVI"]]
ndvi = ndvi.groupby(["year", "state_fips", "season"]).max().reset_index()
ndvi.head(2)

ndvi = ndvi.pivot(index=['year', 'state_fips'], columns='season', values='NDVI')
ndvi.reset_index(drop=False, inplace=True)
ndvi.head(2)


if "MODIS" in file_name:
    new_name = "modis_ndvi"
elif "AVHRR" in file_name:
    new_name = "avhrr_ndvi"
elif "GIMMS" in file_name:
    new_name = "gimms_ndvi"

ndvi.rename(columns={"s1": "s1_max_" + new_name,
                     "s2": "s2_max_" + new_name,
                     "s3": "s3_max_" + new_name,
                     "s4": "s4_max_" + new_name}, inplace=True)


seasonal_ndvi_modis = ndvi.copy()
del(ndvi)
seasonal_ndvi_modis.head(3)

# %%
print(f"{seasonal_ndvi_avhrr.shape = }")
print(f"{seasonal_ndvi_gimms.shape = }")
print(f"{seasonal_ndvi_modis.shape = }")

# %%
seasonal_ndvi_modis.head(2)

# %%
seasonal_ndvi_max = pd.merge(seasonal_ndvi_avhrr, seasonal_ndvi_gimms, on=["state_fips", "year"], how="outer")
seasonal_ndvi_max = pd.merge(seasonal_ndvi_max, seasonal_ndvi_modis, on=["state_fips", "year"], how="outer")
seasonal_ndvi_max.head(2)

# %%
print(seasonal_ndvi_max.shape)
print(seasonal_ndvi_avhrr.shape)


# %%
del(seasonal_ndvi_avhrr, seasonal_ndvi_gimms, seasonal_ndvi_modis)

# %% [markdown]
# ###   Sum start here

# %%
file_name = Min_files[0]
print (f"{file_name = }")

ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"statefips90m": "state_fips"}, inplace=True)
ndvi = rc.correct_3digitStateFips_Min(df=ndvi, col_ = "state_fips")
ndvi = ndvi[ndvi.state_fips.isin(SoI_abb_fips)]

ndvi = ndvi[["year", "month", "NDVI", "state_fips"]]
ndvi = ndvi[ndvi.state_fips.isin(county_fips.state_fips.unique())]
ndvi.reset_index(drop=True, inplace=True)

ndvi.head(2)

ndvi["season"] = "s5"

ndvi['season'] = np.where(ndvi['month'].isin([1, 2, 3]), 's1', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([4, 5, 6, 7]), 's2', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([8, 9]), 's3', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([10, 11, 12]), 's4', ndvi.season)

ndvi = ndvi[["year",  "state_fips", "season", "NDVI"]]
ndvi = ndvi.groupby(["year", "state_fips", "season"]).sum().reset_index()
ndvi.head(2)

ndvi = ndvi.pivot(index=['year', 'state_fips'], columns='season', values='NDVI')
ndvi.reset_index(drop=False, inplace=True)
ndvi.head(2)


if "MODIS" in file_name:
    new_name = "modis_ndvi"
elif "AVHRR" in file_name:
    new_name = "avhrr_ndvi"
elif "GIMMS" in file_name:
    new_name = "gimms_ndvi"

ndvi.rename(columns={"s1": "s1_sum_" + new_name,
                     "s2": "s2_sum_" + new_name,
                     "s3": "s3_sum_" + new_name,
                     "s4": "s4_sum_" + new_name}, inplace=True)


seasonal_ndvi_avhrr = ndvi.copy()
del(ndvi)
seasonal_ndvi_avhrr.head(3)

# %%

# %%
file_name = Min_files[1]
print (f"{file_name = }")

ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"statefips90m": "state_fips"}, inplace=True)
ndvi = rc.correct_3digitStateFips_Min(df=ndvi, col_ = "state_fips")
ndvi = ndvi[ndvi.state_fips.isin(SoI_abb_fips)]

ndvi = ndvi[["year", "month", "NDVI", "state_fips"]]
ndvi = ndvi[ndvi.state_fips.isin(county_fips.state_fips.unique())]
ndvi.reset_index(drop=True, inplace=True)
########################################
ndvi["season"] = "s5"

ndvi['season'] = np.where(ndvi['month'].isin([1, 2, 3]), 's1', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([4, 5, 6, 7]), 's2', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([8, 9]), 's3', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([10, 11, 12]), 's4', ndvi.season)

ndvi = ndvi[["year",  "state_fips", "season", "NDVI"]]
ndvi = ndvi.groupby(["year", "state_fips", "season"]).sum().reset_index()
ndvi.head(2)

ndvi = ndvi.pivot(index=['year', 'state_fips'], columns='season', values='NDVI')
ndvi.reset_index(drop=False, inplace=True)
ndvi.head(2)


if "MODIS" in file_name:
    new_name = "modis_ndvi"
elif "AVHRR" in file_name:
    new_name = "avhrr_ndvi"
elif "GIMMS" in file_name:
    new_name = "gimms_ndvi"

ndvi.rename(columns={"s1": "s1_sum_" + new_name,
                     "s2": "s2_sum_" + new_name,
                     "s3": "s3_sum_" + new_name,
                     "s4": "s4_sum_" + new_name}, inplace=True)


seasonal_ndvi_gimms = ndvi.copy()
del(ndvi)
seasonal_ndvi_gimms.head(3)

# %%

# %%
file_name = Min_files[2]
print (f"{file_name = }")
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"statefips90m": "state_fips"}, inplace=True)
ndvi = rc.correct_3digitStateFips_Min(df=ndvi, col_ = "state_fips")
ndvi = ndvi[ndvi.state_fips.isin(SoI_abb_fips)]

ndvi = ndvi[["year", "month", "NDVI", "state_fips"]]
ndvi = ndvi[ndvi.state_fips.isin(county_fips.state_fips.unique())]
ndvi.reset_index(drop=True, inplace=True)
ndvi.head(2)
##################################################
ndvi["season"] = "s5"

ndvi['season'] = np.where(ndvi['month'].isin([1, 2, 3]), 's1', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([4, 5, 6, 7]), 's2', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([8, 9]), 's3', ndvi.season)
ndvi['season'] = np.where(ndvi['month'].isin([10, 11, 12]), 's4', ndvi.season)

ndvi = ndvi[["year",  "state_fips", "season", "NDVI"]]
ndvi = ndvi.groupby(["year", "state_fips", "season"]).sum().reset_index()
ndvi.head(2)

ndvi = ndvi.pivot(index=['year', 'state_fips'], columns='season', values='NDVI')
ndvi.reset_index(drop=False, inplace=True)
ndvi.head(2)


if "MODIS" in file_name:
    new_name = "modis_ndvi"
elif "AVHRR" in file_name:
    new_name = "avhrr_ndvi"
elif "GIMMS" in file_name:
    new_name = "gimms_ndvi"

ndvi.rename(columns={"s1": "s1_sum_" + new_name,
                     "s2": "s2_sum_" + new_name,
                     "s3": "s3_sum_" + new_name,
                     "s4": "s4_sum_" + new_name}, inplace=True)


seasonal_ndvi_modis = ndvi.copy()
del(ndvi)
seasonal_ndvi_modis.head(3)

# %%
print(f"{seasonal_ndvi_avhrr.shape = }")
print(f"{seasonal_ndvi_gimms.shape = }")
print(f"{seasonal_ndvi_modis.shape = }")

# %%
seasonal_ndvi_sum = pd.merge(seasonal_ndvi_avhrr, seasonal_ndvi_gimms, on=["state_fips", "year"], how="outer")
seasonal_ndvi_sum = pd.merge(seasonal_ndvi_sum, seasonal_ndvi_modis, on=["state_fips", "year"], how="outer")
seasonal_ndvi_sum.head(2)

# %%
seasonal_ndvi_mean.head(2)

# %%
seasonal_ndvi_max.head(2)

# %%
print(seasonal_ndvi_sum.shape)
print(seasonal_ndvi_avhrr.shape)

# %%
seasonal_ndvi = pd.merge(seasonal_ndvi_sum, seasonal_ndvi_mean, on=["state_fips", "year"], how="outer")
seasonal_ndvi = pd.merge(seasonal_ndvi, seasonal_ndvi_max, on=["state_fips", "year"], how="outer")
seasonal_ndvi.head(2)

# %%
seasonal_ndvi.columns = seasonal_ndvi.columns.values
seasonal_ndvi.head(2)

# %%
import pickle
from datetime import datetime

filename = reOrganized_dir + "state_seasonal_ndvi_V2MinBased.sav"

export_ = {
    "seasonal_ndvi": seasonal_ndvi,
    "source_code": "state_monthly_NDVI_2_seasonal_V2MinBased",
    "Author": "HN",
    "Note": "The county level NDVI files have missing months in them and are not consistent.",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %%
seasonal_ndvi.columns

# %%
seasonal_ndvi.head(2)

# %%
modis_cols = [x for x in seasonal_ndvi.columns if "modis" in x ]
cos = ["year", "state_fips"] + modis_cols
seasonal_ndvi.loc[seasonal_ndvi.year==2001, cos]

# %%
