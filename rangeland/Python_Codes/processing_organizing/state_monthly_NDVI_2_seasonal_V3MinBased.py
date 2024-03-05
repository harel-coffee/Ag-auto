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
# This is a copy of ```state_monthly_NDVI_2_seasonal_V2MinBased```.
#
# I want to leave out sum and mean in each season as it seems they want max in each season. And write for-loops.
# ________________________________________________________________________________________________________________
#
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
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
list(abb_dict.keys())

# %%
SoI = abb_dict['SoI']
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
county_fips = abb_dict["county_fips"]
county_fips = county_fips[county_fips.state.isin(SoI_abb)]
SoI_abb_fips = county_fips.state_fips.unique()
print (len(SoI_abb_fips))
SoI_abb_fips

# %%
county_fips = county_fips[["state", "state_fips"]].copy()
county_fips.drop_duplicates(inplace=True)
county_fips.reset_index(drop=True, inplace=True)
county_fips

# %%
Min_files = ["statefips_monthly_AVHRR_NDVI.csv",
             "statefips_monthly_GIMMS_NDVI.csv",
             "statefips_monthly_MODIS_NDVI.csv"]

# %%
ndvi_all = pd.DataFrame()

for file_name in Min_files:
    satellite = file_name.split("_")[2].lower()
    print ("------------------------------------------------")
    ndvi = pd.read_csv(Min_data_base + file_name)
    ndvi.rename(columns={"statefips90m": "state_fips"}, inplace=True)

    ndvi = rc.correct_3digitStateFips_Min(df=ndvi, col_ = "state_fips")
    ndvi = ndvi[ndvi.state_fips.isin(SoI_abb_fips)]
    # print (ndvi.head(2))
    # print ("------------------------------------------------")
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
    # print (ndvi_three_vars.head(2))
    # print ("------------------------------------------------")
    ##############################################################################
    # find mean NDVI in each season.
    #     ndvi["season"] = "s5"
    #     ndvi['season'] = np.where(ndvi['month'].isin([1, 2, 3]), 's1', ndvi.season)
    #     ndvi['season'] = np.where(ndvi['month'].isin([4, 5, 6, 7]), 's2', ndvi.season)
    #     ndvi['season'] = np.where(ndvi['month'].isin([8, 9]), 's3', ndvi.season)
    #     ndvi['season'] = np.where(ndvi['month'].isin([10, 11, 12]), 's4', ndvi.season)

    #     ndvi = ndvi[["year",  "state_fips", "season", "NDVI"]]
    #     ndvi = ndvi.groupby(["year", "state_fips", "season"]).mean().reset_index()
    #     ndvi.head(2)
    #     ndvi = ndvi.pivot(index=['year', 'state_fips'], columns='season', values='NDVI')
    #     ndvi.reset_index(drop=False, inplace=True)
    #     new_name = satellite + "_ndvi"

    #     ndvi.rename(columns={"s1": "s1_mean_" + new_name,
    #                          "s2": "s2_mean_" + new_name,
    #                          "s3": "s3_mean_" + new_name,
    #                          "s4": "s4_mean_" + new_name}, inplace=True)
    #     ndvi = pd.merge(ndvi, ndvi_three_vars, on=["state_fips", "year"], how="outer")
    seasonal_ndvi = ndvi_three_vars.copy()
    postfix = "_" + satellite
    seasonal_ndvi.rename(columns={"max_ndvi_month" : "max_ndvi_month" + postfix,
                         "max_ndvi_in_year" : "max_ndvi_in_year" + postfix,
                         "ndvi_std" : "ndvi_std" + postfix}, inplace=True)

    c_ = "max_ndvi_currYrseason_" + satellite
    seasonal_ndvi[c_] = 666
    seasonal_ndvi[c_] = np.where(
        seasonal_ndvi['max_ndvi_month_' + satellite].isin([1, 2, 3]), 1, seasonal_ndvi[c_])
    
    seasonal_ndvi[c_] = np.where(
        seasonal_ndvi['max_ndvi_month_' + satellite].isin([4, 5, 6, 7]), 2, seasonal_ndvi[c_])
    
    seasonal_ndvi[c_] = np.where(
        seasonal_ndvi['max_ndvi_month_' + satellite].isin([8, 9]), 3, seasonal_ndvi[c_])
    
    seasonal_ndvi[c_] = np.where(
        seasonal_ndvi['max_ndvi_month_' + satellite].isin([10, 11, 12]), 4, seasonal_ndvi[c_])
    seasonal_ndvi.head(5)

    ###
    ### count the frequency of a season being important to a county
    ###
    season_count = seasonal_ndvi.groupby(["state_fips", c_])[c_].count()
    season_count = pd.DataFrame(season_count)
    season_count.rename(columns={c_: "max_ndvi_season_count"}, inplace=True)
    season_count = season_count.reset_index()
    season_count.head(10)

    maxIdx_df = season_count.groupby(["state_fips"])["max_ndvi_season_count"].idxmax().reset_index()
    maxIdx = maxIdx_df["max_ndvi_season_count"].values
    state_seasons = season_count.loc[maxIdx, ["state_fips", c_]]
    state_seasons.head(4)

    state_seasons[c_ + "_str"] = state_seasons[c_].astype(str)
    state_seasons[c_ + "_str"] = "s" + state_seasons[c_ + "_str"]
    state_seasons.head(2)


    L = list(state_seasons.columns)
    L.pop()

    state_seasons = state_seasons.pivot(index=L, columns = c_ + "_str", values = c_)
    state_seasons.head(2)

    state_seasons.reset_index(drop=False, inplace=True)
    state_seasons.columns = state_seasons.columns.values
    state_seasons.head(2)

    ### We may have only 2 seasons. So, S1 and S4 may not exist. 
    ### So, create them manually
    all_seasons_names = ["s1", "s2", "s3", "s4"]
    for a_seas in all_seasons_names:
        if not(a_seas in state_seasons.columns):
            state_seasons[a_seas] = np.nan


    state_seasons.rename(columns={"s1": "s1_bucket_" + satellite,
                                  "s2": "s2_bucket_" + satellite,
                                  "s3": "s3_bucket_" + satellite,
                                  "s4": "s4_bucket_" + satellite}, inplace=True)

    ######################################################################
    state_seasons.fillna(value = {"s1_bucket_" + satellite: 0, 
                                  "s2_bucket_" + satellite: 0,
                                  "s3_bucket_" + satellite: 0, 
                                  "s4_bucket_" + satellite: 0},
                          inplace=True)

    cs = [x + satellite for x in ["s1_bucket_", "s2_bucket_", "s3_bucket_", "s4_bucket_"]]
    for c in cs:
        state_seasons[c] = np.where(state_seasons[c].isin([1, 2, 3, 4]), 1, state_seasons[c])
        state_seasons[c] = state_seasons[c].astype(int)

    state_seasons.drop(labels=c_, axis=1, inplace=True)
    state_seasons.head(2)

    seasonal_ndvi = pd.merge(seasonal_ndvi, state_seasons, on=["state_fips"], how="outer")
    seasonal_ndvi.head(2)

    print(f"{file_name = }, {seasonal_ndvi.shape = }")
    if len(ndvi_all) == 0:
        ndvi_all = seasonal_ndvi.copy()
    else:
        ndvi_all = pd.merge(ndvi_all, seasonal_ndvi, on=["state_fips", "year"], how="outer")
    del(seasonal_ndvi)

# %%
print(len(ndvi_all.state_fips.unique()))

# %% [markdown]
# ###   Max start here


# %%
seasonal_ndvi_max = pd.DataFrame()
for file_name in Min_files:
    satellite = file_name.split("_")[2].lower()
    print (f"{file_name = }")

    ndvi = pd.read_csv(Min_data_base + file_name)
    ndvi.rename(columns={"statefips90m": "state_fips"}, inplace=True)
    ndvi = rc.correct_3digitStateFips_Min(df=ndvi, col_ = "state_fips")
    ndvi = ndvi[ndvi.state_fips.isin(SoI_abb_fips)]

    ndvi = ndvi[["year", "month", "NDVI", "state_fips"]]
    ndvi = ndvi[ndvi.state_fips.isin(county_fips.state_fips.unique())]
    ndvi.reset_index(drop=True, inplace=True)

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
    new_name = satellite + "_ndvi"
    ndvi.rename(columns={"s1": "s1_max_" + new_name,
                         "s2": "s2_max_" + new_name,
                         "s3": "s3_max_" + new_name,
                         "s4": "s4_max_" + new_name}, inplace=True)
    
    if len(seasonal_ndvi_max) == 0:
        seasonal_ndvi_max = ndvi.copy()
        del(ndvi)
    else:
        seasonal_ndvi_max = pd.merge(seasonal_ndvi_max, ndvi, 
                                     on=["state_fips", "year"], how="outer")
        del(ndvi)

seasonal_ndvi_max.head(3)

# %%
print(seasonal_ndvi_max.shape)


# %% [markdown]
# ###   Sum start here

# %%
seasonal_ndvi_sum = pd.DataFrame()
for file_name in Min_files:
    satellite = file_name.split("_")[2].lower()
    
    ndvi = pd.read_csv(Min_data_base + file_name)
    ndvi.rename(columns={"statefips90m": "state_fips"}, inplace=True)
    ndvi = rc.correct_3digitStateFips_Min(df=ndvi, col_ = "state_fips")
    ndvi = ndvi[ndvi.state_fips.isin(SoI_abb_fips)]

    ndvi = ndvi[["year", "month", "NDVI", "state_fips"]]
    ndvi = ndvi[ndvi.state_fips.isin(county_fips.state_fips.unique())]
    ndvi.reset_index(drop=True, inplace=True)

    ndvi["season"] = "s5"
    ndvi['season'] = np.where(ndvi['month'].isin([1, 2, 3]), 's1', ndvi.season)
    ndvi['season'] = np.where(ndvi['month'].isin([4, 5, 6, 7]), 's2', ndvi.season)
    ndvi['season'] = np.where(ndvi['month'].isin([8, 9]), 's3', ndvi.season)
    ndvi['season'] = np.where(ndvi['month'].isin([10, 11, 12]), 's4', ndvi.season)

    ndvi = ndvi[["year",  "state_fips", "season", "NDVI"]]
    ndvi = ndvi.groupby(["year", "state_fips", "season"]).sum().reset_index()

    ndvi = ndvi.pivot(index=['year', 'state_fips'], columns='season', values='NDVI')
    ndvi.reset_index(drop=False, inplace=True)
    
    new_name = satellite + "_ndvi"
    ndvi.rename(columns={"s1": "s1_sum_" + new_name,
                         "s2": "s2_sum_" + new_name,
                         "s3": "s3_sum_" + new_name,
                         "s4": "s4_sum_" + new_name}, inplace=True)
    
    print(f"{file_name = }, {ndvi.shape = }")
    if len(seasonal_ndvi_sum) == 0:
        seasonal_ndvi_sum = ndvi.copy()
        del(ndvi)
    else:
        seasonal_ndvi_sum = pd.merge(seasonal_ndvi_sum, ndvi, 
                                     on=["state_fips", "year"], how="outer")
        del(ndvi)
seasonal_ndvi_sum.head(3)

# %%
ndvi_all = pd.merge(ndvi_all, seasonal_ndvi_max, on=["state_fips", "year"], how="outer")
# ndvi_all = pd.merge(ndvi_all, seasonal_ndvi_sum, on=["state_fips", "year"], how="outer")
ndvi_all.head(2)

# %%
filename = reOrganized_dir + "state_seasonal_ndvi_V3MinBased.sav"

export_ = {
    "seasonal_ndvi": ndvi_all,
    "source_code": "state_monthly_NDVI_2_seasonal_V3MinBased",
    "Author": "HN",
    "Note": "The county level NDVI files have missing months in them and are not consistent.",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %%
ndvi_all.columns

# %%
ndvi_all.head(2)

# %%
modis_cols = [x for x in ndvi_all.columns if "modis" in x ]
cos = ["year", "state_fips"] + modis_cols
ndvi_all.loc[ndvi_all.year==2001, cos]

# %%
A = pd.read_pickle(reOrganized_dir + "old/state_seasonal_ndvi_V2MinBased.sav")
A = A["seasonal_ndvi"]
A.head(2)

# %%
list(A.columns)

# %%
ndvi_all.head(3)

# %%
A.sort_values(by=["state_fips", "year"], inplace=True)
ndvi_all.sort_values(by=["state_fips", "year"], inplace=True)

A.reset_index(drop=True, inplace=True)
ndvi_all.reset_index(drop=True, inplace=True)

A.fillna(666, inplace=True)
ndvi_all.fillna(666, inplace=True)

# %%

# %%

# %%
