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

import matplotlib
import matplotlib.pyplot as plt
import calendar

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
seasonal_dir = reOrganized_dir + "seasonal_variables/02_merged_mean_over_county/"

# %%
abb_dict = pd.read_pickle(param_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
county_id_name_fips = pd.read_csv(Min_data_base + "county_id_name_fips.csv")

print(f"{len(county_id_name_fips.STATE.unique()) = }")
county_id_name_fips.head(2)

county_id_name_fips = county_id_name_fips[county_id_name_fips.STATE.isin(SoI_abb)]
print(f"{len(county_id_name_fips.STATE.unique()) = }")
print(f"{len(county_id_name_fips.county.unique()) = }")

county_id_name_fips.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
county_id_name_fips.rename(columns={"county": "county_fips"}, inplace=True)

county_id_name_fips.reset_index(drop=True, inplace=True)
county_id_name_fips.head(2)

# %%
county_id_name_fips.dropna(how="any", inplace=False).shape

# %%
county_fips = pd.read_pickle(reOrganized_dir + "county_fips.sav")
county_fips = county_fips["county_fips"]

print(f"{len(county_fips.state.unique()) = }")
county_fips = county_fips[county_fips.state.isin(SoI_abb)].copy()
county_fips.drop_duplicates(inplace=True)
county_fips.reset_index(drop=True, inplace=True)
county_fips = county_fips[["county_fips", "county_name", "state", "EW"]]
print(f"{len(county_fips.state.unique()) = }")
county_fips.head(2)

# %%
county_fips[county_fips.county_fips == "04001"]

# %%
county_gridmet_mean_indices = pd.read_csv(
    Min_data_base + "county_gridmet_mean_indices.csv"
)
county_gridmet_mean_indices.rename(
    columns=lambda x: x.lower().replace(" ", "_"), inplace=True
)
county_gridmet_mean_indices.rename(columns={"county": "county_fips"}, inplace=True)
county_gridmet_mean_indices.head(2)

# %%
county_gridmet_mean_indices[county_gridmet_mean_indices.county_fips == 104001]

# %%
sorted(county_gridmet_mean_indices.columns)

# %%
needed_cols = ["year", "month", "county_fips", "tavg_avg", "ppt"]
county_gridmet_mean_indices = county_gridmet_mean_indices[needed_cols]
county_gridmet_mean_indices.head(2)

# %%
print(f"{len(county_gridmet_mean_indices.county_fips.unique()) = }")

# %%
# %%time

missing = [
    x
    for x in county_gridmet_mean_indices.county_fips.unique()
    if not (x in county_id_name_fips.county_fips.unique())
]
print(len(missing))
sorted(missing)[:5]

# %%
# # %%time

# missing = [x for x in county_id_name_fips.county_fips.unique()\
#                                           if not(x in county_gridmet_mean_indices.county_fips.unique())]
# print (len(missing))
# sorted(missing)[:5]

# %% [markdown]
# ### Subset counties that are in the states of interest

# %%
print(f"{len(county_gridmet_mean_indices.county_fips.unique()) = }")

county_gridmet_mean_indices = county_gridmet_mean_indices[
    county_gridmet_mean_indices.county_fips.isin(list(county_id_name_fips.county_fips))
]
print(f"{len(county_gridmet_mean_indices.county_fips.unique()) = }")

# %%
2379 - 1648

# %% [markdown]
# ## Tonsor Seasons
#  - January–March
#  - April–July
#  - August – September
#  - October – December

# %%
tonsor_seasons = {
    "season_1": [1, 2, 3],
    "season_2": [4, 5, 6, 7],
    "season_3": [8, 9],
    "season_4": [10, 11, 12],
}

days_per_month = {
    "1": 31,
    "2": 28,
    "3": 31,
    "4": 30,
    "5": 31,
    "6": 30,
    "7": 31,
    "8": 31,
    "9": 30,
    "10": 31,
    "11": 30,
    "12": 31,
}

no_days_in_each_season = {
    "season_1": 90,
    "season_2": 122,
    "season_3": 61,
    "season_4": 92,
}

# %%
county_gridmet_mean_indices["sum_tavg"] = 666
county_gridmet_mean_indices.head(2)

# %%
# %%time
for a_year in county_gridmet_mean_indices.year.unique():
    leap_ = calendar.isleap(a_year)

    for a_month in county_gridmet_mean_indices.month.unique():
        curr_df = county_gridmet_mean_indices[
            (county_gridmet_mean_indices.year == a_year)
            & (county_gridmet_mean_indices.month == a_month)
        ]

        curr_locs = curr_df.index
        if leap_:
            if a_month == 2:
                county_gridmet_mean_indices.loc[
                    curr_locs, "sum_tavg"
                ] = county_gridmet_mean_indices.loc[curr_locs, "tavg_avg"] * (
                    days_per_month[str(a_month)] + 1
                )

        else:
            county_gridmet_mean_indices.loc[curr_locs, "sum_tavg"] = (
                county_gridmet_mean_indices.loc[curr_locs, "tavg_avg"]
                * days_per_month[str(a_month)]
            )

# %%
county_gridmet_mean_indices.head(5)

# %%
county_gridmet_mean_indices[county_gridmet_mean_indices.county_fips == 104001]

# %% [markdown]
# # <font color='red'>Warning</font>
# Takes about 5 min

# %%
# %%time

needed_cols = [
    "county_fips",
    "year",
    "s1_countymean_total_precip",
    "s2_countymean_total_precip",
    "s3_countymean_total_precip",
    "s4_countymean_total_precip",
    "s1_countymean_avg_tavg",
    "s2_countymean_avg_tavg",
    "s3_countymean_avg_tavg",
    "s4_countymean_avg_tavg",
]

nu_rows = len(county_gridmet_mean_indices.year.unique()) * len(
    county_gridmet_mean_indices.county_fips.unique()
)
seasonal = pd.DataFrame(columns=needed_cols, index=range(nu_rows))

wide_pointer = 0
for a_year in county_gridmet_mean_indices.year.unique():
    leap_ = calendar.isleap(a_year)

    for fip in county_gridmet_mean_indices.county_fips.unique():
        curr_df = county_gridmet_mean_indices[
            (county_gridmet_mean_indices.year == a_year)
            & (county_gridmet_mean_indices.county_fips == fip)
        ]

        curr_df_s1 = curr_df[curr_df.month.isin(tonsor_seasons["season_1"])]
        curr_df_s2 = curr_df[curr_df.month.isin(tonsor_seasons["season_2"])]
        curr_df_s3 = curr_df[curr_df.month.isin(tonsor_seasons["season_3"])]
        curr_df_s4 = curr_df[curr_df.month.isin(tonsor_seasons["season_4"])]

        seasonal.loc[wide_pointer, "county_fips"] = fip
        seasonal.loc[wide_pointer, "year"] = a_year

        seasonal.loc[wide_pointer, "s1_countymean_total_precip"] = curr_df_s1.ppt.sum()
        seasonal.loc[wide_pointer, "s2_countymean_total_precip"] = curr_df_s2.ppt.sum()
        seasonal.loc[wide_pointer, "s3_countymean_total_precip"] = curr_df_s3.ppt.sum()
        seasonal.loc[wide_pointer, "s4_countymean_total_precip"] = curr_df_s4.ppt.sum()

        if leap_:
            seasonal.loc[
                wide_pointer, "s1_countymean_avg_tavg"
            ] = curr_df_s1.sum_tavg.sum() / (no_days_in_each_season["season_1"] + 1)
        else:
            seasonal.loc[wide_pointer, "s1_countymean_avg_tavg"] = (
                curr_df_s1.sum_tavg.sum() / no_days_in_each_season["season_1"]
            )

        seasonal.loc[wide_pointer, "s2_countymean_avg_tavg"] = (
            curr_df_s2.sum_tavg.sum() / no_days_in_each_season["season_2"]
        )

        seasonal.loc[wide_pointer, "s3_countymean_avg_tavg"] = (
            curr_df_s3.sum_tavg.sum() / no_days_in_each_season["season_3"]
        )

        seasonal.loc[wide_pointer, "s4_countymean_avg_tavg"] = (
            curr_df_s4.sum_tavg.sum() / no_days_in_each_season["season_4"]
        )
        wide_pointer += 1

        del (curr_df, curr_df_s1, curr_df_s2, curr_df_s3, curr_df_s4)

seasonal.head(5)

# %%
seasonal[seasonal.county_fips == 104001]

# %%
for a_col in needed_cols[2:]:
    seasonal[a_col] = seasonal[a_col].astype(float)

seasonal = seasonal.round(decimals=2)
seasonal = rc.correct_Mins_county_6digitFIPS(df=seasonal, col_="county_fips")
seasonal.head(5)

# %%
seasonal[seasonal.county_fips == "04001"]

# %%
import pickle
from datetime import datetime

filename = reOrganized_dir + "county_seasonal_temp_ppt_weighted.sav"

export_ = {
    "seasonal": seasonal,
    "source_code": "county_monthly_SW_2_seasonal",
    "Author": "HN",
    "Min_file_used": "county_gridmet_mean_indices.csv",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %%
seasonal.head(2)

# %%
county_fips.head(2)

# %%
seasonal = pd.merge(seasonal, county_fips, on=["county_fips"], how="left")
seasonal.head(2)

# %%
len(seasonal.state.unique())

# %%
seasonal[seasonal.state.isna()]

# %%
county_gridmet_mean_indices.head(5)

# %% [markdown]
# # On Jan 12. HN, KR, MB
# had a meeting and we wanted to model one year (snapshot) as function of long run averages.
# Here I am doing annual temp!!!!!

# %%
# %%time

needed_cols = ["county_fips", "year", "annual_avg_tavg"]

nu_rows = len(county_gridmet_mean_indices.year.unique()) * len(
    county_gridmet_mean_indices.county_fips.unique()
)
annual_temp = pd.DataFrame(columns=needed_cols, index=range(nu_rows))

wide_pointer = 0
for a_year in county_gridmet_mean_indices.year.unique():
    leap_ = calendar.isleap(a_year)

    for fip in county_gridmet_mean_indices.county_fips.unique():
        curr_df = county_gridmet_mean_indices[
            (county_gridmet_mean_indices.year == a_year)
            & (county_gridmet_mean_indices.county_fips == fip)
        ]

        annual_temp.loc[wide_pointer, "county_fips"] = fip
        annual_temp.loc[wide_pointer, "year"] = a_year

        if leap_:
            annual_temp.loc[wide_pointer, "annual_avg_tavg"] = (
                curr_df.sum_tavg.sum() / 366
            )
        else:
            annual_temp.loc[wide_pointer, "annual_avg_tavg"] = (
                curr_df.sum_tavg.sum() / 365
            )
        wide_pointer += 1

annual_temp.head(2)

# %%
annual_temp[annual_temp.county_fips == 104001]

# %%
annual_temp["annual_avg_tavg"] = annual_temp["annual_avg_tavg"].astype(float)
annual_temp = annual_temp.round(decimals=2)
annual_temp = rc.correct_Mins_county_6digitFIPS(df=annual_temp, col_="county_fips")
annual_temp.head(5)

# %%
filename = reOrganized_dir + "county_annual_avg_tavg.sav"

export_ = {
    "annual_temp": annual_temp,
    "source_code": "county_monthly_SW_2_seasonal",
    "Author": "HN",
    "Min_file_used": "county_gridmet_mean_indices.csv",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %%
annual_temp

# %%

# %%

# %% [markdown]
# # Min's inconsistency

# %%
cnty_annual_GPP_NPP_prod = pd.read_csv(
    reOrganized_dir + "county_annual_GPP_NPP_productivity.csv"
)
cnty_annual_GPP_NPP_prod.head(5)

# %%
NPP = pd.read_csv(Min_data_base + "county_annual_MODIS_NPP.csv")
GPP = pd.read_csv(Min_data_base + "county_annual_MODIS_GPP.csv")
prod = pd.read_csv(Min_data_base + "county_annual_productivity.csv")

# %%
print(101005 in list(NPP.county.unique()))
print(101005 in list(GPP.county.unique()))
print(101005 in list(prod.county.unique()))

# %%
print(117000 in list(NPP.county.unique()))
print(117000 in list(GPP.county.unique()))
print(117000 in list(prod.county.unique()))

# %%
col_ = "MODIS_NPP"
print(cnty_annual_GPP_NPP_prod[cnty_annual_GPP_NPP_prod[col_].isna()].shape)
print(
    len(cnty_annual_GPP_NPP_prod[cnty_annual_GPP_NPP_prod[col_].isna()].county.unique())
)
cnty_annual_GPP_NPP_prod[cnty_annual_GPP_NPP_prod[col_].isna()]

# %%
col_ = "MODIS_GPP"
print(cnty_annual_GPP_NPP_prod[cnty_annual_GPP_NPP_prod[col_].isna()].shape)
print(
    len(cnty_annual_GPP_NPP_prod[cnty_annual_GPP_NPP_prod[col_].isna()].county.unique())
)
cnty_annual_GPP_NPP_prod[cnty_annual_GPP_NPP_prod[col_].isna()]

# %%
col_ = "productivity"
print(cnty_annual_GPP_NPP_prod[cnty_annual_GPP_NPP_prod[col_].isna()].shape)
print(
    len(cnty_annual_GPP_NPP_prod[cnty_annual_GPP_NPP_prod[col_].isna()].county.unique())
)
cnty_annual_GPP_NPP_prod[cnty_annual_GPP_NPP_prod[col_].isna()]

# %%

# %%
