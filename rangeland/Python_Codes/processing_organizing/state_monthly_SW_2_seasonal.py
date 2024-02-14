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
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI = abb_dict['SoI']
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %% [markdown]
# ### county_fips

# %%
county_fips = pd.read_pickle(reOrganized_dir + "county_fips.sav")
county_fips = county_fips["county_fips"]


L=len(county_fips[county_fips.state == "SD"])
print ("number of counties in SD is {}".format(L))
county_fips = county_fips[county_fips.state.isin(SoI_abb)]
county_fips.head(2)

# %%
county_id_name_fips = pd.read_csv(Min_data_base + "county_id_name_fips.csv")

L = len(county_id_name_fips[county_id_name_fips.STATE == "SD"])
print ("number of counties in SD is {}".format(L))

# %%
# Min's file ("county_id_name_fips.csv") has a missing county in it county_fips = 46102

# %%
gridmet_mean_indices = pd.read_csv(Min_data_base + "statefips_gridmet_mean_indices.csv")
gridmet_mean_indices.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
gridmet_mean_indices.rename(columns={"statefips": "state_fips"}, inplace=True)

gridmet_mean_indices.state_fips = gridmet_mean_indices.state_fips.astype(str)
gridmet_mean_indices.state_fips = gridmet_mean_indices.state_fips.str.slice(1, 3)

gridmet_mean_indices = gridmet_mean_indices[gridmet_mean_indices.state_fips.isin(county_fips.state_fips.unique())]

print (gridmet_mean_indices.shape)
print (f"{len(gridmet_mean_indices.state_fips.unique()) = }")
gridmet_mean_indices.head(2)

# %%
sorted(gridmet_mean_indices.columns)

# %%
needed_cols = ["year", "state_fips", "month", "tavg_avg", "ppt", 'danger', 'emergency']
gridmet_mean_indices = gridmet_mean_indices[needed_cols]
gridmet_mean_indices.head(2)

# %% [markdown]
# ## Tonsor Seasons
#  - January–March 
#  - April–July
#  - August – September 
#  - October – December

# %%
tonsor_seasons = {"season_1": [1, 2, 3],
                  "season_2" : [4, 5, 6, 7],
                  "season_3": [8, 9],
                  "season_4" : [10, 11, 12]}

days_per_month = {"1":31,
                  "2":28,
                  "3":31,
                  "4":30,
                  "5":31,
                  "6":30,
                  "7":31,
                  "8":31,
                  "9":30,
                  "10":31,
                  "11":30,
                  "12":31
                 }

no_days_in_each_season = {"season_1":90,
                          "season_2":122,
                          "season_3":61,
                          "season_4":92}

# %%
gridmet_mean_indices["sum_tavg"] = 666
gridmet_mean_indices.head(2)

# %%
# %%time
for a_year in gridmet_mean_indices.year.unique():
    leap_ = calendar.isleap(a_year)
    for a_month in gridmet_mean_indices.month.unique():
        curr_df = gridmet_mean_indices[(gridmet_mean_indices.year == a_year) &\
                                       (gridmet_mean_indices.month == a_month)]
        curr_locs = curr_df.index
        if leap_:
            if a_month == 2:
                gridmet_mean_indices.loc[curr_locs, "sum_tavg"] = gridmet_mean_indices.loc[
                                                                            curr_locs, "tavg_avg"]*\
                                                                               (days_per_month[str(a_month)]+1)
        else:
            gridmet_mean_indices.loc[curr_locs, "sum_tavg"] = gridmet_mean_indices.loc[
                                                                        curr_locs, "tavg_avg"]*\
                                                                         days_per_month[str(a_month)]
            
gridmet_mean_indices.head(5)

# %%

# %%
# %%time

needed_cols = ['state_fips', 'year', 
               'S1_stateMean_total_precip',    'S2_stateMean_total_precip', 
               'S3_stateMean_total_precip',    'S4_stateMean_total_precip', 
               'S1_stateMean_avg_Tavg',        'S2_stateMean_avg_Tavg', 
               'S3_stateMean_avg_Tavg',        'S4_stateMean_avg_Tavg',
               'S1_stateMean_total_danger',    'S2_stateMean_total_danger', 
               'S3_stateMean_total_danger',    'S4_stateMean_total_danger',
               'S1_stateMean_total_emergency', 'S2_stateMean_total_emergency', 
               'S3_stateMean_total_emergency', 'S4_stateMean_total_emergency']

nu_rows = len(gridmet_mean_indices.year.unique()) * len(gridmet_mean_indices.state_fips.unique())
seasonal = pd.DataFrame(columns = needed_cols, index=range(nu_rows))

wide_pointer = 0
for a_year in gridmet_mean_indices.year.unique():
    leap_ = calendar.isleap(a_year)
    
    for fip in gridmet_mean_indices.state_fips.unique():
        curr_df = gridmet_mean_indices[(gridmet_mean_indices.year == a_year) &\
                                              (gridmet_mean_indices.state_fips == fip)]
        
        curr_df_S1 = curr_df[curr_df.month.isin(tonsor_seasons["season_1"])]
        curr_df_S2 = curr_df[curr_df.month.isin(tonsor_seasons["season_2"])]
        curr_df_S3 = curr_df[curr_df.month.isin(tonsor_seasons["season_3"])]
        curr_df_S4 = curr_df[curr_df.month.isin(tonsor_seasons["season_4"])]
        
        seasonal.loc[wide_pointer, "state_fips"] = fip
        seasonal.loc[wide_pointer, "year"] = a_year

        # precipitation
        seasonal.loc[wide_pointer, "S1_stateMean_total_precip"] = curr_df_S1.ppt.sum()
        seasonal.loc[wide_pointer, "S2_stateMean_total_precip"] = curr_df_S2.ppt.sum()
        seasonal.loc[wide_pointer, "S3_stateMean_total_precip"] = curr_df_S3.ppt.sum()
        seasonal.loc[wide_pointer, "S4_stateMean_total_precip"] = curr_df_S4.ppt.sum()
        
        # danger
        seasonal.loc[wide_pointer, "S1_stateMean_total_danger"] = curr_df_S1.danger.sum()
        seasonal.loc[wide_pointer, "S2_stateMean_total_danger"] = curr_df_S2.danger.sum()
        seasonal.loc[wide_pointer, "S3_stateMean_total_danger"] = curr_df_S3.danger.sum()
        seasonal.loc[wide_pointer, "S4_stateMean_total_danger"] = curr_df_S4.danger.sum()
        
        # emergency
        seasonal.loc[wide_pointer, "S1_stateMean_total_emergency"] = curr_df_S1.emergency.sum()
        seasonal.loc[wide_pointer, "S2_stateMean_total_emergency"] = curr_df_S2.emergency.sum()
        seasonal.loc[wide_pointer, "S3_stateMean_total_emergency"] = curr_df_S3.emergency.sum()
        seasonal.loc[wide_pointer, "S4_stateMean_total_emergency"] = curr_df_S4.emergency.sum()

        
        if leap_:
            seasonal.loc[wide_pointer, "S1_stateMean_avg_Tavg"] = \
                                        curr_df_S1.sum_tavg.sum() / (no_days_in_each_season["season_1"]+1)
        else:
            seasonal.loc[wide_pointer, "S1_stateMean_avg_Tavg"] = \
                                        curr_df_S1.sum_tavg.sum() / no_days_in_each_season["season_1"]
            
        seasonal.loc[wide_pointer, "S2_stateMean_avg_Tavg"] = \
                                        curr_df_S2.sum_tavg.sum() / no_days_in_each_season["season_2"]
        
        
        seasonal.loc[wide_pointer, "S3_stateMean_avg_Tavg"] = \
                                        curr_df_S3.sum_tavg.sum() / no_days_in_each_season["season_3"]
        
        
        seasonal.loc[wide_pointer, "S4_stateMean_avg_Tavg"] = \
                                        curr_df_S4.sum_tavg.sum() / no_days_in_each_season["season_4"]
        wide_pointer += 1
        
        del(curr_df, curr_df_S1, curr_df_S2, curr_df_S3, curr_df_S4)
        
seasonal.head(5)

# %%
fip="01"
a_year = 1979

curr_df = gridmet_mean_indices[(gridmet_mean_indices.year == a_year) &\
                                          (gridmet_mean_indices.state_fips == fip)]

curr_df_S1 = curr_df[curr_df.month.isin(tonsor_seasons["season_1"])]
curr_df_S2 = curr_df[curr_df.month.isin(tonsor_seasons["season_2"])]
curr_df_S3 = curr_df[curr_df.month.isin(tonsor_seasons["season_3"])]
curr_df_S4 = curr_df[curr_df.month.isin(tonsor_seasons["season_4"])]


# %%
print (curr_df_S1.danger.sum())
print (curr_df_S2.danger.sum())
print (curr_df_S3.danger.sum())
print (curr_df_S4.danger.sum())
print ()
# emergency
print (curr_df_S1.emergency.sum())
print (curr_df_S2.emergency.sum())
print (curr_df_S3.emergency.sum())
print (curr_df_S4.emergency.sum())

# %%
curr_df_S2

# %%

# %%

# %%

# %%
for a_col in needed_cols[2:]:
    seasonal[a_col] = seasonal[a_col].astype(float)

seasonal = seasonal.round(decimals=2)
# seasonal = rc.correct_Mins_county_6digitFIPS(df=seasonal, col_="county_fips")
seasonal.head(5)

# %%

# %%
import pickle
from datetime import datetime

filename = reOrganized_dir + "state_seasonal_temp_ppt_weighted.sav"

desc = "The mean in the column names such as " + \
       "S3_stateMean_total_emergency (stateMean) refers "+\
       "to the file name that has the word (mean) in it"

export_ = {"seasonal": seasonal, 
           "source_code" : "state_monthly_SW_2_seasonal",
           "Author": "HN",
           "Min_file_used" : "statefips_gridmet_mean_indices.csv",
           "column_name_description" : desc,
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
seasonal.head(2)

# %%
county_fips.head(2)

# %%
seasonal = pd.merge(seasonal, county_fips, on=["state_fips"], how="left")
seasonal.head(2)

# %%
len(seasonal.state.unique())

# %%
gridmet_mean_indices.head(5)

# %% [markdown]
# # On Jan 12. HN, KR, MB 
# had a meeting and we wanted to model one year (snapshot) as function of long run averages.
# Here I am doing annual temp!!!!!

# %%
# %%time

needed_cols = ['state_fips', 'year', 'annual_avg_Tavg']

nu_rows = len(gridmet_mean_indices.year.unique()) * len(gridmet_mean_indices.state_fips.unique())
annual_temp = pd.DataFrame(columns = needed_cols, index=range(nu_rows))

wide_pointer = 0
for a_year in gridmet_mean_indices.year.unique():
    leap_ = calendar.isleap(a_year)
    
    for fip in gridmet_mean_indices.state_fips.unique():
        curr_df = gridmet_mean_indices[(gridmet_mean_indices.year == a_year) &\
                                              (gridmet_mean_indices.state_fips == fip)]

        annual_temp.loc[wide_pointer, "state_fips"] = fip
        annual_temp.loc[wide_pointer, "year"] = a_year

        if leap_:
            annual_temp.loc[wide_pointer, "annual_avg_Tavg"] = curr_df.sum_tavg.sum() / 366
        else:
            annual_temp.loc[wide_pointer, "annual_avg_Tavg"] = curr_df.sum_tavg.sum() / 365
        wide_pointer += 1

annual_temp.head(2)

# %%
annual_temp["annual_avg_Tavg"] = annual_temp["annual_avg_Tavg"].astype(float)
annual_temp = annual_temp.round(decimals=2)
# annual_temp = rc.correct_Mins_county_6digitFIPS(df=annual_temp, col_="county_fips")
annual_temp.head(5)

# %%
print (len(annual_temp.state_fips.unique()) * len(annual_temp.year.unique()))
print (len(annual_temp))

# %%
filename = reOrganized_dir + "state_annual_avg_Tavg.sav"

export_ = {"annual_temp": annual_temp, 
           "source_code" : "state_monthly_SW_2_seasonal",
           "Author": "HN",
           "Min_file_used" : "statefips_gridmet_mean_indices.csv",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
annual_temp.head(2)

# %%
len(annual_temp.state_fips.unique())

# %%
d_ = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
A_csv = pd.read_csv(d_ + "county_adjacency_cleaned_corrected.csv")
Adj_25 = pd.read_pickle(d_ + "adjacency_binary_matrix_strict25states.sav")
Adj = pd.read_pickle(d_ + "county_adjacency_binary_matrix.sav")

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
