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
county_fips = pd.read_pickle(reOrganized_dir + "county_fips.sav")
county_fips = county_fips["county_fips"]


L=len(county_fips[county_fips.state == "SD"])
print ("number of counties in SD is {}".format(L))
county_fips = county_fips[county_fips.state.isin(SoI_abb)]
county_fips.reset_index(drop=True, inplace=True)
county_fips.head(2)

# %%
len(county_fips.state_fips.unique())

# %%

# %%
Min_files = os.listdir(Min_data_base)
Min_files = [x for x in Min_files if x.endswith(".csv")]
Min_files = [x for x in Min_files if "NDVI" in x]
Min_files = [x for x in Min_files if "county_monthly" in x]
Min_files

# %%
print (Min_files[0])
file_name = Min_files[0]
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")

ndvi.county_fips = ndvi.county_fips.astype(str)
ndvi["state_fips"] = ndvi.county_fips.str.slice(0, 2)
ndvi = ndvi[["year", "month", "NDVI", "state_fips"]]

ndvi = ndvi[ndvi.state_fips.isin(county_fips.state_fips.unique())]
ndvi.reset_index(drop=True, inplace=True)

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

# %%
file_name = Min_files[1]
print (f"{file_name = }")

ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")
ndvi.county_fips = ndvi.county_fips.astype(str)
ndvi["state_fips"] = ndvi.county_fips.str.slice(0, 2)
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

# %%
file_name = Min_files[2]
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")

ndvi.county_fips = ndvi.county_fips.astype(str)
ndvi["state_fips"] = ndvi.county_fips.str.slice(0, 2)
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

# %%
print(f"{seasonal_ndvi_avhrr.shape = }")
print(f"{seasonal_ndvi_gimms.shape = }")
print(f"{seasonal_ndvi_modis.shape = }")

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
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")

ndvi.county_fips = ndvi.county_fips.astype(str)
ndvi["state_fips"] = ndvi.county_fips.str.slice(0, 2)
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
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")
ndvi.county_fips = ndvi.county_fips.astype(str)
ndvi["state_fips"] = ndvi.county_fips.str.slice(0, 2)
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
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")

ndvi.county_fips = ndvi.county_fips.astype(str)
ndvi["state_fips"] = ndvi.county_fips.str.slice(0, 2)
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
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")

ndvi.county_fips = ndvi.county_fips.astype(str)
ndvi["state_fips"] = ndvi.county_fips.str.slice(0, 2)
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
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")
ndvi.county_fips = ndvi.county_fips.astype(str)
ndvi["state_fips"] = ndvi.county_fips.str.slice(0, 2)
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
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_="county_fips")

ndvi.county_fips = ndvi.county_fips.astype(str)
ndvi["state_fips"] = ndvi.county_fips.str.slice(0, 2)
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

filename = reOrganized_dir + "state_seasonal_ndvi.sav"

export_ = {
    "seasonal_ndvi": seasonal_ndvi,
    "source_code": "state_monthly_NDVI_2_seasonal",
    "Author": "HN",
    "Note": "The county level NDVI files have missing months in them and are not consistent.",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %%
