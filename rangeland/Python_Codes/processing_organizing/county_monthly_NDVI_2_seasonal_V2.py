# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# This is a copy of ```county_monthly_NDVI_2_seasonal.ipynb```
#
# What we want here:
#
#   - Leave mean ```NDVI``` and sum ```NDVI``` out for now. It seems max is the one we want. To keep the file size 
#      small.
#   - And modify the county grouping (dummy variable) based on my bucket idea. The season at which NDVI max is attained most frequently, is the season of that county.

# +
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# -

from jupytext.config import global_jupytext_configuration_directories
list(global_jupytext_configuration_directories())

data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# for bold print
start_b = "\033[1m"
end_b = "\033[0;0m"
print("This is " + start_b + "a_bold_text" + end_b + "!")

Min_files = os.listdir(Min_data_base)
Min_files = [x for x in Min_files if x.endswith(".csv")]
Min_files = [x for x in Min_files if "NDVI" in x]
Min_files = [x for x in Min_files if "county_monthly" in x]
Min_files

# +
ndvi_all = pd.DataFrame()

for file_name in Min_files:
    satellite = file_name.split("_")[2].lower()
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
    # S1 = ndvi[ndvi.month.isin([1, 2, 3])]
    # S2 = ndvi[ndvi.month.isin([4, 5, 6, 7])]
    # S3 = ndvi[ndvi.month.isin([8, 9])]
    # S4 = ndvi[ndvi.month.isin([10, 11, 12])]

    # # drop "month"
    # S1 = S1[["year", "county_fips", "NDVI"]]
    # S2 = S2[["year", "county_fips", "NDVI"]]
    # S3 = S3[["year", "county_fips", "NDVI"]]
    # S4 = S4[["year", "county_fips", "NDVI"]]

    # S1 = S1.groupby(["year", "county_fips"]).mean().reset_index()
    # S2 = S2.groupby(["year", "county_fips"]).mean().reset_index()
    # S3 = S3.groupby(["year", "county_fips"]).mean().reset_index()
    # S4 = S4.groupby(["year", "county_fips"]).mean().reset_index()
    # new_name = "ndvi_" + satellite

    # S1.rename(columns={"NDVI": "s1_mean_" + new_name}, inplace=True)
    # S2.rename(columns={"NDVI": "s2_mean_" + new_name}, inplace=True)
    # S3.rename(columns={"NDVI": "s3_mean_" + new_name}, inplace=True)
    # S4.rename(columns={"NDVI": "s4_mean_" + new_name}, inplace=True)

    # seasonal_ndvi = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
    # seasonal_ndvi = pd.merge(seasonal_ndvi, S3, on=["county_fips", "year"], how="outer")
    # seasonal_ndvi = pd.merge(seasonal_ndvi, S4, on=["county_fips", "year"], how="outer")
    # seasonal_ndvi = pd.merge(seasonal_ndvi, ndvi_three_vars, on=["county_fips", "year"], how="outer")
    seasonal_ndvi = ndvi_three_vars.copy()

    postfix = "_" + satellite
    seasonal_ndvi.rename(columns={"max_ndvi_month" : "max_ndvi_month" + postfix,
                                  "max_ndvi_in_year" : "max_ndvi_in_year" + postfix,
                                  "ndvi_std" : "ndvi_std" + postfix}, inplace=True)

    seasonal_ndvi.head(2)

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
    season_count = seasonal_ndvi.groupby(["county_fips", c_])[c_].count()
    season_count = pd.DataFrame(season_count)
    season_count.rename(columns={c_: "max_ndvi_season_count"}, inplace=True)
    season_count = season_count.reset_index()
    season_count.head(10)

    maxIdx_df = season_count.groupby(["county_fips"])["max_ndvi_season_count"].idxmax().reset_index()
    maxIdx = maxIdx_df["max_ndvi_season_count"].values
    county_seasons = season_count.loc[maxIdx, ["county_fips", c_]]
    county_seasons.head(4)

    county_seasons[c_ + "_str"] = county_seasons[c_].astype(str)
    county_seasons[c_ + "_str"] = "s" + county_seasons[c_ + "_str"]
    county_seasons.head(2)

    L = list(county_seasons.columns)
    L.pop()

    county_seasons = county_seasons.pivot(index=L, columns = c_ + "_str", values = c_)
    county_seasons.head(2)

    county_seasons.reset_index(drop=False, inplace=True)
    county_seasons.columns = county_seasons.columns.values
    county_seasons.head(2)

    county_seasons.rename(columns={"s1": "s1_bucket_" + satellite,
                                   "s2": "s2_bucket_" + satellite,
                                   "s3": "s3_bucket_" + satellite,
                                   "s4": "s4_bucket_" + satellite}, inplace=True)

    ######################################################################
    county_seasons.fillna(value = {"s1_bucket_" + satellite: 0, 
                                   "s2_bucket_" + satellite: 0,
                                   "s3_bucket_" + satellite: 0, 
                                   "s4_bucket_" + satellite: 0},
                          inplace=True)

    cs = [x + satellite for x in ["s1_bucket_", "s2_bucket_", "s3_bucket_", "s4_bucket_"]]
    for c in cs:
        county_seasons[c] = np.where(county_seasons[c].isin([1, 2, 3, 4]), 1, county_seasons[c])
        county_seasons[c] = county_seasons[c].astype(int)

    county_seasons.drop(labels=c_, axis=1, inplace=True)
    county_seasons.head(2)

    seasonal_ndvi = pd.merge(seasonal_ndvi, county_seasons, on=["county_fips"], how="outer")
    seasonal_ndvi.head(2)
    
    print(f"{file_name = }, {seasonal_ndvi.shape = }")
    if len(ndvi_all) == 0:
        ndvi_all = seasonal_ndvi.copy()
    else:
        ndvi_all = pd.merge(ndvi_all, seasonal_ndvi, on=["county_fips", "year"], how="outer")
# -

print(len(ndvi_all.county_fips.unique()))
ndvi_all.shape

ndvi_avhrr = pd.read_csv(Min_data_base + Min_files[0])
ndvi_gimms = pd.read_csv(Min_data_base + Min_files[1])
ndvi_modis = pd.read_csv(Min_data_base + Min_files[2])

# +
# it seems they must have 381 rows per county
# and there are 22 years worth of data.
incomplete_counties_avhrr = []

for a_county in ndvi_avhrr.county.unique():
    df = ndvi_avhrr[ndvi_avhrr.county == a_county]
    if len(df) != 381:
        incomplete_counties_avhrr += [a_county]

len(incomplete_counties_avhrr)

# +
# it seems they must have 381 rows per county
incomplete_counties_gimms = []

for a_county in ndvi_gimms.county.unique():
    df = ndvi_gimms[ndvi_gimms.county == a_county]
    if len(df) != 381:
        incomplete_counties_gimms += [a_county]

len(incomplete_counties_gimms)

# +
# it seems they must have 381 rows per county
incomplete_counties_modis = []

for a_county in ndvi_modis.county.unique():
    df = ndvi_modis[ndvi_modis.county == a_county]
    if len(df) != 381:
        incomplete_counties_modis += [a_county]

len(incomplete_counties_modis)
# -

ndvi_modis[ndvi_modis.county == incomplete_counties_modis[0]].year.unique()

# +
# for a_year in range(2001, 2023):
#     for a_county in incomplete_counties_modis:
#         df = ndvi_modis[ndvi_modis.county == a_county]
#         df = df[df.year == a_year]
# -


# # Some Months are missing.

# +
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
# -

# # Check if some counties have missing years

# +
print(ndvi_avhrr.year.unique())
print()
print(ndvi_gimms.year.unique())

print()
print(ndvi_modis.year.unique())
# -

print(sorted(ndvi_modis.county.unique()) == sorted(ndvi_avhrr.county.unique()))
print(sorted(ndvi_modis.county.unique()) == sorted(ndvi_gimms.county.unique()))

# +
cnty_missing_years_modis = []
max_no_years = len(ndvi_modis.year.unique())

for a_county in ndvi_modis.county.unique():
    df = ndvi_modis[ndvi_modis.county == a_county]
    if len(df.year.unique()) != max_no_years:
        cnty_missing_years_modis += [a_county]

len(cnty_missing_years_modis)

# +
cnty_missing_years_avhrr = []
max_no_years = len(ndvi_avhrr.year.unique())

for a_county in ndvi_avhrr.county.unique():
    df = ndvi_avhrr[ndvi_avhrr.county == a_county]
    if len(df.year.unique()) != max_no_years:
        cnty_missing_years_avhrr += [a_county]

len(cnty_missing_years_avhrr)

# +
cnty_missing_years_gimms = []
max_no_years = len(ndvi_gimms.year.unique())

for a_county in ndvi_gimms.county.unique():
    df = ndvi_gimms[ndvi_gimms.county == a_county]
    if len(df.year.unique()) != max_no_years:
        cnty_missing_years_gimms += [a_county]

len(cnty_missing_years_gimms)
# -

# ###   Max start here


# +
seasonal_ndvi_max = pd.DataFrame()

for file_name in Min_files:
    satellite = file_name.split("_")[2].lower()

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

    new_name = "ndvi_" + satellite

    S1.rename(columns={"NDVI": "s1_max_" + new_name}, inplace=True)
    S2.rename(columns={"NDVI": "s2_max_" + new_name}, inplace=True)
    S3.rename(columns={"NDVI": "s3_max_" + new_name}, inplace=True)
    S4.rename(columns={"NDVI": "s4_max_" + new_name}, inplace=True)

    seasonal_ndvi = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
    seasonal_ndvi = pd.merge(seasonal_ndvi, S3, on=["county_fips", "year"], how="outer")
    seasonal_ndvi = pd.merge(seasonal_ndvi, S4, on=["county_fips", "year"], how="outer")
    seasonal_ndvi.head(2)
    print(f"{file_name = }, {seasonal_ndvi.shape = }")

    if len(seasonal_ndvi_max) == 0:
        seasonal_ndvi_max = seasonal_ndvi.copy()
        del(seasonal_ndvi)
    else:
        seasonal_ndvi_max = pd.merge(seasonal_ndvi_max, seasonal_ndvi, 
                                     on=["county_fips", "year"], how="outer")
        del(seasonal_ndvi)
# -

print (len(seasonal_ndvi_max.county_fips.unique()))
seasonal_ndvi_max.shape

# +
# A = pd.read_pickle(reOrganized_dir + "old/county_seasonal_ndvi.sav")
# A_seasonal_ndvi = A["seasonal_ndvi"]
# A_seasonal_ndvi.sort_values(by=["county_fips", "year"], inplace=True)
# A_seasonal_ndvi.reset_index(drop=True, inplace=True)
# A_seasonal_ndvi.fillna(666, inplace=True)

# A_seasonal_ndvi.head(2)
# -

# ###   Sum start here

# +
seasonal_ndvi_sum = pd.DataFrame()

for file_name in Min_files:
    satellite = file_name.split("_")[2].lower()
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

    new_name = "ndvi_" + satellite

    S1.rename(columns={"NDVI": "s1_sum_" + new_name}, inplace=True)
    S2.rename(columns={"NDVI": "s2_sum_" + new_name}, inplace=True)
    S3.rename(columns={"NDVI": "s3_sum_" + new_name}, inplace=True)
    S4.rename(columns={"NDVI": "s4_sum_" + new_name}, inplace=True)

    seasonal_ndvi = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
    seasonal_ndvi = pd.merge(seasonal_ndvi, S3, on=["county_fips", "year"], how="outer")
    seasonal_ndvi = pd.merge(seasonal_ndvi, S4, on=["county_fips", "year"], how="outer")
    del (S1, S2, S3, S4, ndvi)
    seasonal_ndvi.head(2)
    
    print(f"{file_name = }, {seasonal_ndvi.shape = }")
    if len(seasonal_ndvi_sum) == 0:
        seasonal_ndvi_sum = seasonal_ndvi.copy()
        del(seasonal_ndvi)
    else:
        seasonal_ndvi_sum = pd.merge(seasonal_ndvi_sum, seasonal_ndvi, 
                                     on=["county_fips", "year"], how="outer")
        del(seasonal_ndvi)
# -

seasonal_ndvi_sum.head(2)

ndvi_all = pd.merge(ndvi_all, seasonal_ndvi_max, on=["county_fips", "year"], how="outer")
# ndvi_all = pd.merge(ndvi_all, seasonal_ndvi_sum, on=["county_fips", "year"], how="outer")
ndvi_all.head(2)

ndvi_all.columns

# +
filename = reOrganized_dir + "county_seasonal_ndvi.sav"

export_ = {"seasonal_ndvi": ndvi_all,
           "source_code": "county_monthly_NDVI_2_seasonal_V2",
           "Author": "HN",
           "Note": "The county level NDVI files have missing months in them and are not consistent.",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

pickle.dump(export_, open(filename, "wb"))
# -

ndvi_all.head(2)

filename
