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
Min_files = os.listdir(Min_data_base)
Min_files = [x for x in Min_files if x.endswith(".csv")]
Min_files = [x for x in Min_files if "NDVI" in x]
Min_files = [x for x in Min_files if "county_monthly" in x]
Min_files

# %%
file_name = Min_files[0]
print (file_name)
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")

###### Three new vars (Max NDVI, Max NDVI DoY, and NDVI std)
max_ndvi_per_yr_idx_df = ndvi.groupby(["county_fips", "year"])["NDVI"].idxmax().reset_index()
max_ndvi_per_yr_idx_df.rename(columns={"NDVI": "max_ndvi_per_year_idx"}, inplace=True)
max_ndvi_per_yr_idx_df.head(2)

max_ndvi_per_yr = ndvi.loc[max_ndvi_per_yr_idx_df.max_ndvi_per_year_idx]

max_ndvi_per_yr.rename(columns={"month" : "max_ndvi_month",
                                "NDVI" : "max_ndvi_in_year" }, inplace=True)

max_ndvi_per_yr = max_ndvi_per_yr[["county_fips", "year", "max_ndvi_month", "max_ndvi_in_year"]]
max_ndvi_per_yr.sort_values(by=["county_fips", "year"], inplace=True)
max_ndvi_per_yr.reset_index(drop=True, inplace=True)
max_ndvi_per_yr.head(2)

ndvi_variance = ndvi.groupby(["county_fips"])["NDVI"].std(ddof=1).reset_index()
ndvi_variance.rename(columns={"NDVI": "ndvi_std"}, inplace=True)
ndvi_variance.head(2)

# ndvi_variance.to_frame()

ndvi_three_vars = pd.merge(max_ndvi_per_yr, ndvi_variance, on=["county_fips"], how="outer")
ndvi_three_vars.head(2)

##################################################################################
S1 = ndvi[ndvi.month.isin([1, 2, 3])]
S2 = ndvi[ndvi.month.isin([4, 5, 6, 7])]
S3 = ndvi[ndvi.month.isin([8, 9])]
S4 = ndvi[ndvi.month.isin([10, 11, 12])]

# drop "month"
S1 = S1[["year", "county_fips", "NDVI"]]
S2 = S2[["year", "county_fips", "NDVI"]]
S3 = S3[["year", "county_fips", "NDVI"]]
S4 = S4[["year", "county_fips", "NDVI"]]

S1 = S1.groupby(["year", "county_fips"]).mean().reset_index()
S2 = S2.groupby(["year", "county_fips"]).mean().reset_index()
S3 = S3.groupby(["year", "county_fips"]).mean().reset_index()
S4 = S4.groupby(["year", "county_fips"]).mean().reset_index()

if "MODIS" in file_name:
    new_name = "ndvi_modis"
elif "AVHRR" in file_name:
    new_name = "ndvi_avhrr"
elif "GIMMS" in file_name:
    new_name = "ndvi_gimms"

S1.rename(columns={"NDVI": "s1_mean_" + new_name}, inplace=True)
S2.rename(columns={"NDVI": "s2_mean_" + new_name}, inplace=True)
S3.rename(columns={"NDVI": "s3_mean_" + new_name}, inplace=True)
S4.rename(columns={"NDVI": "s4_mean_" + new_name}, inplace=True)

seasonal_ndvi_avhrr = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
seasonal_ndvi_avhrr = pd.merge(seasonal_ndvi_avhrr, S3, on=["county_fips", "year"], how="outer")
seasonal_ndvi_avhrr = pd.merge(seasonal_ndvi_avhrr, S4, on=["county_fips", "year"], how="outer")
seasonal_ndvi_avhrr = pd.merge(seasonal_ndvi_avhrr, ndvi_three_vars, on=["county_fips", "year"], how="outer")

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
del (S1, S2, S3, S4, ndvi, ndvi_three_vars)

file_name = Min_files[1]
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")
###### Three new vars (Max NDVI, Max NDVI DoY, and NDVI std)
max_ndvi_per_yr_idx_df = ndvi.groupby(["county_fips", "year"])["NDVI"].idxmax().reset_index()
max_ndvi_per_yr_idx_df.rename(columns={"NDVI": "max_ndvi_per_year_idx"}, inplace=True)
max_ndvi_per_yr_idx_df.head(2)

max_ndvi_per_yr = ndvi.loc[max_ndvi_per_yr_idx_df.max_ndvi_per_year_idx]

max_ndvi_per_yr.rename(columns={"month" : "max_ndvi_month",
                                "NDVI" : "max_ndvi_in_year" }, inplace=True)

max_ndvi_per_yr = max_ndvi_per_yr[["county_fips", "year", "max_ndvi_month", "max_ndvi_in_year"]]
max_ndvi_per_yr.sort_values(by=["county_fips", "year"], inplace=True)
max_ndvi_per_yr.reset_index(drop=True, inplace=True)
max_ndvi_per_yr.head(2)

ndvi_variance = ndvi.groupby(["county_fips"])["NDVI"].std(ddof=1).reset_index()
ndvi_variance.rename(columns={"NDVI": "ndvi_std"}, inplace=True)
ndvi_variance.head(2)

# ndvi_variance.to_frame()
ndvi_three_vars = pd.merge(max_ndvi_per_yr, ndvi_variance, on=["county_fips"], how="outer")
ndvi_three_vars.head(2)

##################################################################################
S1 = ndvi[ndvi.month.isin([1, 2, 3])]
S2 = ndvi[ndvi.month.isin([4, 5, 6, 7])]
S3 = ndvi[ndvi.month.isin([8, 9])]
S4 = ndvi[ndvi.month.isin([10, 11, 12])]

# drop "month"
S1 = S1[["year", "county_fips", "NDVI"]]
S2 = S2[["year", "county_fips", "NDVI"]]
S3 = S3[["year", "county_fips", "NDVI"]]
S4 = S4[["year", "county_fips", "NDVI"]]

S1 = S1.groupby(["year", "county_fips"]).mean().reset_index()
S2 = S2.groupby(["year", "county_fips"]).mean().reset_index()
S3 = S3.groupby(["year", "county_fips"]).mean().reset_index()
S4 = S4.groupby(["year", "county_fips"]).mean().reset_index()

if "MODIS" in file_name:
    new_name = "ndvi_modis"
elif "AVHRR" in file_name:
    new_name = "ndvi_avhrr"
elif "GIMMS" in file_name:
    new_name = "ndvi_gimms"

S1.rename(columns={"NDVI": "s1_mean_" + new_name}, inplace=True)
S2.rename(columns={"NDVI": "s2_mean_" + new_name}, inplace=True)
S3.rename(columns={"NDVI": "s3_mean_" + new_name}, inplace=True)
S4.rename(columns={"NDVI": "s4_mean_" + new_name}, inplace=True)

seasonal_ndvi_gimms = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
seasonal_ndvi_gimms = pd.merge(seasonal_ndvi_gimms, S3, on=["county_fips", "year"], how="outer")
seasonal_ndvi_gimms = pd.merge(seasonal_ndvi_gimms, S4, on=["county_fips", "year"], how="outer")
seasonal_ndvi_gimms = pd.merge(seasonal_ndvi_gimms, ndvi_three_vars, on=["county_fips", "year"], how="outer")

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
del (S1, S2, S3, S4, ndvi, ndvi_three_vars)

# %%
file_name = Min_files[2]
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")

###### Three new vars (Max NDVI, Max NDVI DoY, and NDVI std)
max_ndvi_per_yr_idx_df = ndvi.groupby(["county_fips", "year"])["NDVI"].idxmax().reset_index()
max_ndvi_per_yr_idx_df.rename(columns={"NDVI": "max_ndvi_per_year_idx"}, inplace=True)
max_ndvi_per_yr_idx_df.head(2)


max_ndvi_per_yr = ndvi.loc[max_ndvi_per_yr_idx_df.max_ndvi_per_year_idx]

max_ndvi_per_yr.rename(columns={"month" : "max_ndvi_month",
                                "NDVI" : "max_ndvi_in_year" }, inplace=True)

max_ndvi_per_yr = max_ndvi_per_yr[["county_fips", "year", "max_ndvi_month", "max_ndvi_in_year"]]
max_ndvi_per_yr.sort_values(by=["county_fips", "year"], inplace=True)
max_ndvi_per_yr.reset_index(drop=True, inplace=True)
max_ndvi_per_yr.head(2)

ndvi_variance = ndvi.groupby(["county_fips"])["NDVI"].std(ddof=1).reset_index()
ndvi_variance.rename(columns={"NDVI": "ndvi_std"}, inplace=True)
ndvi_variance.head(2)

# ndvi_variance.to_frame()

ndvi_three_vars = pd.merge(max_ndvi_per_yr, ndvi_variance, on=["county_fips"], how="outer")
ndvi_three_vars.head(2)

##################################################################################

S1 = ndvi[ndvi.month.isin([1, 2, 3])]
S2 = ndvi[ndvi.month.isin([4, 5, 6, 7])]
S3 = ndvi[ndvi.month.isin([8, 9])]
S4 = ndvi[ndvi.month.isin([10, 11, 12])]

# drop "month"
S1 = S1[["year", "county_fips", "NDVI"]]
S2 = S2[["year", "county_fips", "NDVI"]]
S3 = S3[["year", "county_fips", "NDVI"]]
S4 = S4[["year", "county_fips", "NDVI"]]

S1 = S1.groupby(["year", "county_fips"]).mean().reset_index()
S2 = S2.groupby(["year", "county_fips"]).mean().reset_index()
S3 = S3.groupby(["year", "county_fips"]).mean().reset_index()
S4 = S4.groupby(["year", "county_fips"]).mean().reset_index()

if "MODIS" in file_name:
    new_name = "ndvi_modis"
elif "AVHRR" in file_name:
    new_name = "ndvi_avhrr"
elif "GIMMS" in file_name:
    new_name = "ndvi_gimms"

S1.rename(columns={"NDVI": "s1_mean_" + new_name}, inplace=True)
S2.rename(columns={"NDVI": "s2_mean_" + new_name}, inplace=True)
S3.rename(columns={"NDVI": "s3_mean_" + new_name}, inplace=True)
S4.rename(columns={"NDVI": "s4_mean_" + new_name}, inplace=True)

seasonal_ndvi_modis = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
seasonal_ndvi_modis = pd.merge(seasonal_ndvi_modis, S3, on=["county_fips", "year"], how="outer")
seasonal_ndvi_modis = pd.merge(seasonal_ndvi_modis, S4, on=["county_fips", "year"], how="outer")
seasonal_ndvi_modis = pd.merge(seasonal_ndvi_modis, ndvi_three_vars, on=["county_fips", "year"], how="outer")

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
del (S1, S2, S3, S4, ndvi, ndvi_three_vars)

# %%
print(f"{seasonal_ndvi_avhrr.shape = }")
print(f"{seasonal_ndvi_gimms.shape = }")
print(f"{seasonal_ndvi_modis.shape = }")

# %%
print(len(seasonal_ndvi_avhrr.county_fips.unique()))
print(len(seasonal_ndvi_gimms.county_fips.unique()))
print(len(seasonal_ndvi_modis.county_fips.unique()))

# %%
ndvi_avhrr = pd.read_csv(Min_data_base + Min_files[0])
ndvi_gimms = pd.read_csv(Min_data_base + Min_files[1])
ndvi_modis = pd.read_csv(Min_data_base + Min_files[2])

# %%
# it seems they must have 381 rows per county
# and there are 22 years worth of data.
incomplete_counties_avhrr = []

for a_county in ndvi_avhrr.county.unique():
    df = ndvi_avhrr[ndvi_avhrr.county == a_county]
    if len(df) != 381:
        incomplete_counties_avhrr += [a_county]

len(incomplete_counties_avhrr)

# %%
# it seems they must have 381 rows per county
incomplete_counties_gimms = []

for a_county in ndvi_gimms.county.unique():
    df = ndvi_gimms[ndvi_gimms.county == a_county]
    if len(df) != 381:
        incomplete_counties_gimms += [a_county]

len(incomplete_counties_gimms)

# %%
# it seems they must have 381 rows per county
incomplete_counties_modis = []

for a_county in ndvi_modis.county.unique():
    df = ndvi_modis[ndvi_modis.county == a_county]
    if len(df) != 381:
        incomplete_counties_modis += [a_county]

len(incomplete_counties_modis)

# %%
ndvi_modis[ndvi_modis.county == incomplete_counties_modis[0]].year.unique()

# %%
# for a_year in range(2001, 2023):
#     for a_county in incomplete_counties_modis:
#         df = ndvi_modis[ndvi_modis.county == a_county]
#         df = df[df.year == a_year]


# %% [markdown]
# # Some Months are missing.

# %%
a_county = incomplete_counties_modis[0]
# for a_year in range(2001, 2023):
#     df = ndvi_modis[ndvi_modis.county == a_county]
#     df = df[df.year == a_year]
#     print (df.shape)
#     print (a_year)
#     print ()

df = ndvi_modis[ndvi_modis.county == a_county]
df = df[df.year == 2001]
df

# %%
print(f"{seasonal_ndvi_avhrr.shape = }")
print(f"{seasonal_ndvi_gimms.shape = }")
print(f"{seasonal_ndvi_modis.shape = }")

# %% [markdown]
# # Check if some counties have missing years

# %%
print(ndvi_avhrr.year.unique())
print()
print(ndvi_gimms.year.unique())

print()
print(ndvi_modis.year.unique())

# %%
print(sorted(ndvi_modis.county.unique()) == sorted(ndvi_avhrr.county.unique()))
print(sorted(ndvi_modis.county.unique()) == sorted(ndvi_gimms.county.unique()))

# %%
cnty_missing_years_modis = []
max_no_years = len(ndvi_modis.year.unique())

for a_county in ndvi_modis.county.unique():
    df = ndvi_modis[ndvi_modis.county == a_county]
    if len(df.year.unique()) != max_no_years:
        cnty_missing_years_modis += [a_county]

len(cnty_missing_years_modis)

# %%
cnty_missing_years_avhrr = []
max_no_years = len(ndvi_avhrr.year.unique())

for a_county in ndvi_avhrr.county.unique():
    df = ndvi_avhrr[ndvi_avhrr.county == a_county]
    if len(df.year.unique()) != max_no_years:
        cnty_missing_years_avhrr += [a_county]

len(cnty_missing_years_avhrr)

# %%
cnty_missing_years_gimms = []
max_no_years = len(ndvi_gimms.year.unique())

for a_county in ndvi_gimms.county.unique():
    df = ndvi_gimms[ndvi_gimms.county == a_county]
    if len(df.year.unique()) != max_no_years:
        cnty_missing_years_gimms += [a_county]

len(cnty_missing_years_gimms)

# %%

# %%
# for a_year in range(2001, 2023):
#     df = ndvi_modis[ndvi_modis.county == a_county]
#     df = df[df.year == a_year]
#     print (df.shape)
#     print (a_year)
#     print ()

# %%
seasonal_ndvi_gimms.head(2)

# %%
seasonal_ndvi_modis.head(2)

# %%
seasonal_ndvi_mean = pd.merge(seasonal_ndvi_avhrr, seasonal_ndvi_gimms, on=["county_fips", "year"], how="outer")
seasonal_ndvi_mean = pd.merge(seasonal_ndvi_mean, seasonal_ndvi_modis, on=["county_fips", "year"], how="outer")
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
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")

S1 = ndvi[ndvi.month.isin([1, 2, 3])]
S2 = ndvi[ndvi.month.isin([4, 5, 6, 7])]
S3 = ndvi[ndvi.month.isin([8, 9])]
S4 = ndvi[ndvi.month.isin([10, 11, 12])]

# drop "month"
S1 = S1[["year", "county_fips", "NDVI"]]
S2 = S2[["year", "county_fips", "NDVI"]]
S3 = S3[["year", "county_fips", "NDVI"]]
S4 = S4[["year", "county_fips", "NDVI"]]

S1 = S1.groupby(["year", "county_fips"]).max().reset_index()
S2 = S2.groupby(["year", "county_fips"]).max().reset_index()
S3 = S3.groupby(["year", "county_fips"]).max().reset_index()
S4 = S4.groupby(["year", "county_fips"]).max().reset_index()

if "MODIS" in file_name:
    new_name = "ndvi_modis"
elif "AVHRR" in file_name:
    new_name = "ndvi_avhrr"
elif "GIMMS" in file_name:
    new_name = "ndvi_gimms"

S1.rename(columns={"NDVI": "s1_max_" + new_name}, inplace=True)
S2.rename(columns={"NDVI": "s2_max_" + new_name}, inplace=True)
S3.rename(columns={"NDVI": "s3_max_" + new_name}, inplace=True)
S4.rename(columns={"NDVI": "s4_max_" + new_name}, inplace=True)

seasonal_ndvi_avhrr = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
seasonal_ndvi_avhrr = pd.merge(seasonal_ndvi_avhrr, S3, on=["county_fips", "year"], how="outer")
seasonal_ndvi_avhrr = pd.merge(seasonal_ndvi_avhrr, S4, on=["county_fips", "year"], how="outer")
seasonal_ndvi_avhrr.head(2)

# %%
del (S1, S2, S3, S4, ndvi)

# %%
file_name = Min_files[1]
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")

S1 = ndvi[ndvi.month.isin([1, 2, 3])]
S2 = ndvi[ndvi.month.isin([4, 5, 6, 7])]
S3 = ndvi[ndvi.month.isin([8, 9])]
S4 = ndvi[ndvi.month.isin([10, 11, 12])]

# drop "month"
S1 = S1[["year", "county_fips", "NDVI"]]
S2 = S2[["year", "county_fips", "NDVI"]]
S3 = S3[["year", "county_fips", "NDVI"]]
S4 = S4[["year", "county_fips", "NDVI"]]

S1 = S1.groupby(["year", "county_fips"]).max().reset_index()
S2 = S2.groupby(["year", "county_fips"]).max().reset_index()
S3 = S3.groupby(["year", "county_fips"]).max().reset_index()
S4 = S4.groupby(["year", "county_fips"]).max().reset_index()

if "MODIS" in file_name:
    new_name = "ndvi_modis"
elif "AVHRR" in file_name:
    new_name = "ndvi_avhrr"
elif "GIMMS" in file_name:
    new_name = "ndvi_gimms"

S1.rename(columns={"NDVI": "s1_max_" + new_name}, inplace=True)
S2.rename(columns={"NDVI": "s2_max_" + new_name}, inplace=True)
S3.rename(columns={"NDVI": "s3_max_" + new_name}, inplace=True)
S4.rename(columns={"NDVI": "s4_max_" + new_name}, inplace=True)

seasonal_ndvi_gimms = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
seasonal_ndvi_gimms = pd.merge(seasonal_ndvi_gimms, S3, on=["county_fips", "year"], how="outer")
seasonal_ndvi_gimms = pd.merge(seasonal_ndvi_gimms, S4, on=["county_fips", "year"], how="outer")

del (S1, S2, S3, S4, ndvi)
seasonal_ndvi_gimms.head(2)

# %%
file_name = Min_files[2]
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")

S1 = ndvi[ndvi.month.isin([1, 2, 3])]
S2 = ndvi[ndvi.month.isin([4, 5, 6, 7])]
S3 = ndvi[ndvi.month.isin([8, 9])]
S4 = ndvi[ndvi.month.isin([10, 11, 12])]

# drop "month"
S1 = S1[["year", "county_fips", "NDVI"]]
S2 = S2[["year", "county_fips", "NDVI"]]
S3 = S3[["year", "county_fips", "NDVI"]]
S4 = S4[["year", "county_fips", "NDVI"]]

S1 = S1.groupby(["year", "county_fips"]).max().reset_index()
S2 = S2.groupby(["year", "county_fips"]).max().reset_index()
S3 = S3.groupby(["year", "county_fips"]).max().reset_index()
S4 = S4.groupby(["year", "county_fips"]).max().reset_index()

if "MODIS" in file_name:
    new_name = "ndvi_modis"
elif "AVHRR" in file_name:
    new_name = "ndvi_avhrr"
elif "GIMMS" in file_name:
    new_name = "ndvi_gimms"

S1.rename(columns={"NDVI": "s1_max_" + new_name}, inplace=True)
S2.rename(columns={"NDVI": "s2_max_" + new_name}, inplace=True)
S3.rename(columns={"NDVI": "s3_max_" + new_name}, inplace=True)
S4.rename(columns={"NDVI": "s4_max_" + new_name}, inplace=True)

seasonal_ndvi_modis = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
seasonal_ndvi_modis = pd.merge(seasonal_ndvi_modis, S3, on=["county_fips", "year"], how="outer")
seasonal_ndvi_modis = pd.merge(seasonal_ndvi_modis, S4, on=["county_fips", "year"], how="outer")
seasonal_ndvi_modis.head(2)

# %%
print(f"{seasonal_ndvi_avhrr.shape = }")
print(f"{seasonal_ndvi_gimms.shape = }")
print(f"{seasonal_ndvi_modis.shape = }")

# %%
seasonal_ndvi_modis.head(2)

# %%
seasonal_ndvi_max = pd.merge(seasonal_ndvi_avhrr, seasonal_ndvi_gimms, on=["county_fips", "year"], how="outer")
seasonal_ndvi_max = pd.merge(seasonal_ndvi_max, seasonal_ndvi_modis, on=["county_fips", "year"], how="outer")
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
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")

S1 = ndvi[ndvi.month.isin([1, 2, 3])]
S2 = ndvi[ndvi.month.isin([4, 5, 6, 7])]
S3 = ndvi[ndvi.month.isin([8, 9])]
S4 = ndvi[ndvi.month.isin([10, 11, 12])]

# drop "month"
S1 = S1[["year", "county_fips", "NDVI"]]
S2 = S2[["year", "county_fips", "NDVI"]]
S3 = S3[["year", "county_fips", "NDVI"]]
S4 = S4[["year", "county_fips", "NDVI"]]

S1 = S1.groupby(["year", "county_fips"]).sum().reset_index()
S2 = S2.groupby(["year", "county_fips"]).sum().reset_index()
S3 = S3.groupby(["year", "county_fips"]).sum().reset_index()
S4 = S4.groupby(["year", "county_fips"]).sum().reset_index()

if "MODIS" in file_name:
    new_name = "ndvi_modis"
elif "AVHRR" in file_name:
    new_name = "ndvi_avhrr"
elif "GIMMS" in file_name:
    new_name = "ndvi_gimms"

S1.rename(columns={"NDVI": "s1_sum_" + new_name}, inplace=True)
S2.rename(columns={"NDVI": "s2_sum_" + new_name}, inplace=True)
S3.rename(columns={"NDVI": "s3_sum_" + new_name}, inplace=True)
S4.rename(columns={"NDVI": "s4_sum_" + new_name}, inplace=True)

seasonal_ndvi_avhrr = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
seasonal_ndvi_avhrr = pd.merge(seasonal_ndvi_avhrr, S3, on=["county_fips", "year"], how="outer")
seasonal_ndvi_avhrr = pd.merge(seasonal_ndvi_avhrr, S4, on=["county_fips", "year"], how="outer")
seasonal_ndvi_avhrr.head(2)

# %%
del (S1, S2, S3, S4, ndvi)

# %%
file_name = Min_files[1]
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")

S1 = ndvi[ndvi.month.isin([1, 2, 3])]
S2 = ndvi[ndvi.month.isin([4, 5, 6, 7])]
S3 = ndvi[ndvi.month.isin([8, 9])]
S4 = ndvi[ndvi.month.isin([10, 11, 12])]

# drop "month"
S1 = S1[["year", "county_fips", "NDVI"]]
S2 = S2[["year", "county_fips", "NDVI"]]
S3 = S3[["year", "county_fips", "NDVI"]]
S4 = S4[["year", "county_fips", "NDVI"]]

S1 = S1.groupby(["year", "county_fips"]).sum().reset_index()
S2 = S2.groupby(["year", "county_fips"]).sum().reset_index()
S3 = S3.groupby(["year", "county_fips"]).sum().reset_index()
S4 = S4.groupby(["year", "county_fips"]).sum().reset_index()

if "MODIS" in file_name:
    new_name = "ndvi_modis"
elif "AVHRR" in file_name:
    new_name = "ndvi_avhrr"
elif "GIMMS" in file_name:
    new_name = "ndvi_gimms"

S1.rename(columns={"NDVI": "s1_sum_" + new_name}, inplace=True)
S2.rename(columns={"NDVI": "s2_sum_" + new_name}, inplace=True)
S3.rename(columns={"NDVI": "s3_sum_" + new_name}, inplace=True)
S4.rename(columns={"NDVI": "s4_sum_" + new_name}, inplace=True)

seasonal_ndvi_gimms = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
seasonal_ndvi_gimms = pd.merge(seasonal_ndvi_gimms, S3, on=["county_fips", "year"], how="outer")
seasonal_ndvi_gimms = pd.merge(seasonal_ndvi_gimms, S4, on=["county_fips", "year"], how="outer")
seasonal_ndvi_gimms.head(2)

del (S1, S2, S3, S4, ndvi)

# %%
file_name = Min_files[2]
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")

S1 = ndvi[ndvi.month.isin([1, 2, 3])]
S2 = ndvi[ndvi.month.isin([4, 5, 6, 7])]
S3 = ndvi[ndvi.month.isin([8, 9])]
S4 = ndvi[ndvi.month.isin([10, 11, 12])]

# drop "month"
S1 = S1[["year", "county_fips", "NDVI"]]
S2 = S2[["year", "county_fips", "NDVI"]]
S3 = S3[["year", "county_fips", "NDVI"]]
S4 = S4[["year", "county_fips", "NDVI"]]

S1 = S1.groupby(["year", "county_fips"]).sum().reset_index()
S2 = S2.groupby(["year", "county_fips"]).sum().reset_index()
S3 = S3.groupby(["year", "county_fips"]).sum().reset_index()
S4 = S4.groupby(["year", "county_fips"]).sum().reset_index()

if "MODIS" in file_name:
    new_name = "ndvi_modis"
elif "AVHRR" in file_name:
    new_name = "ndvi_avhrr"
elif "GIMMS" in file_name:
    new_name = "ndvi_gimms"

S1.rename(columns={"NDVI": "s1_sum_" + new_name}, inplace=True)
S2.rename(columns={"NDVI": "s2_sum_" + new_name}, inplace=True)
S3.rename(columns={"NDVI": "s3_sum_" + new_name}, inplace=True)
S4.rename(columns={"NDVI": "s4_sum_" + new_name}, inplace=True)

seasonal_ndvi_modis = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
seasonal_ndvi_modis = pd.merge(seasonal_ndvi_modis, S3, on=["county_fips", "year"], how="outer")
seasonal_ndvi_modis = pd.merge(seasonal_ndvi_modis, S4, on=["county_fips", "year"], how="outer")
seasonal_ndvi_modis.head(2)

# %%
print(f"{seasonal_ndvi_avhrr.shape = }")
print(f"{seasonal_ndvi_gimms.shape = }")
print(f"{seasonal_ndvi_modis.shape = }")

# %%
ndvi_avhrr = pd.read_csv(Min_data_base + Min_files[0])
ndvi_gimms = pd.read_csv(Min_data_base + Min_files[1])
ndvi_modis = pd.read_csv(Min_data_base + Min_files[2])

# %%
print(f"{seasonal_ndvi_avhrr.shape = }")
print(f"{seasonal_ndvi_gimms.shape = }")
print(f"{seasonal_ndvi_modis.shape = }")


# %%
seasonal_ndvi_gimms.head(2)

# %%
seasonal_ndvi_modis.head(2)

# %%
seasonal_ndvi_sum = pd.merge(seasonal_ndvi_avhrr, seasonal_ndvi_gimms, on=["county_fips", "year"], how="outer")
seasonal_ndvi_sum = pd.merge(seasonal_ndvi_sum, seasonal_ndvi_modis, on=["county_fips", "year"], how="outer")
seasonal_ndvi_sum.head(2)

# %%
seasonal_ndvi_mean.head(2)

# %%
seasonal_ndvi_max.head(2)

# %%
print(seasonal_ndvi_sum.shape)
print(seasonal_ndvi_avhrr.shape)

# %%
seasonal_ndvi = pd.merge(seasonal_ndvi_mean, seasonal_ndvi_sum, on=["county_fips", "year"], how="outer")
seasonal_ndvi = pd.merge(seasonal_ndvi, seasonal_ndvi_max, on=["county_fips", "year"], how="outer")
seasonal_ndvi.head(2)

# %%
seasonal_ndvi.columns

# %%
import pickle
from datetime import datetime

filename = reOrganized_dir + "county_seasonal_ndvi.sav"

export_ = {
    "seasonal_ndvi": seasonal_ndvi,
    "source_code": "county_monthly_NDVI_2_seasonal",
    "Author": "HN",
    "Note": "The county level NDVI files have missing months in them and are not consistent.",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

pickle.dump(export_, open(filename, "wb"))

# %%
seasonal_ndvi.head(2)

# %%
