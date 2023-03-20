# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# ### Tutorial from 
# https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/

# %%
import pandas as pd
import csv
import numpy as np
import os, os.path
import sys

import collections # to count frequency of elements of an array
# to move files from one directory to another
import shutil

# %%
ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
test20 = pd.read_csv(ML_data_folder+"test20_split_2Bconsistent_Oct17.csv")

# %%
test20.head(2)

# %% [markdown]
# # EVI - SG

# %%
VI_idx = "EVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"
smooth_type = "SG"
file_names = [smooth_type + "_Walla2015_" + VI_idx + "_JFD.csv", 
              smooth_type + "_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              smooth_type + "_Grant2017_" + VI_idx + "_JFD.csv", 
              smooth_type + "_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

data=pd.DataFrame()

for file in file_names:
    curr_file=pd.read_csv(data_dir + file)
    curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])
    
    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years)==3 or len(all_years)==2:
        proper_year = all_years[1]
    elif len(all_years)==1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
    data=pd.concat([data, curr_file])

data.reset_index(drop=True, inplace=True)
data.head(2)

GT_TS = data[data.ID.isin(list(test20.ID.unique()))].copy()
print (len(GT_TS.ID.unique()))

# %%
EVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + EVI_colnames
GT_wide = pd.DataFrame(columns=columnNames, 
                        index=range(len(GT_TS.ID.unique())))
GT_wide["ID"] = GT_TS.ID.unique()

for an_ID in GT_TS.ID.unique():
    curr_df = GT_TS[GT_TS.ID==an_ID]
    
    GT_wide_indx = GT_wide[GT_wide.ID==an_ID].index
    GT_wide.loc[GT_wide_indx, VI_idx+"_1":VI_idx+"_36"] = curr_df.EVI.values[:36]

print (len(GT_wide.ID.unique()))
GT_wide.head(2)

print (len(test20.ID.unique()))
print (len(test20.ID))

print ((list(test20.ID)) == (list(GT_wide.ID)))
sorted(list(test20.ID)) == sorted(list(GT_wide.ID))

# %%
test20.head(2)
test20.sort_values(by=["ID"], inplace=True)
GT_wide.sort_values(by=["ID"], inplace=True)

test20.reset_index(drop=True, inplace=True)
GT_wide.reset_index(drop=True, inplace=True)

(list(test20.ID)) == (list(GT_wide.ID))

# %%
GT_wide = pd.merge(GT_wide, test20, on=['ID'], how='left')
out_name = ML_data_folder + "widen_test_TS/" +  VI_idx + "_" + smooth_type + \
          "_wide_test20_split_2Bconsistent_Oct17.csv"
GT_wide.to_csv(out_name, index = False)

# %% [markdown]
# # EVI - Regular

# %%
VI_idx = "EVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/04_regularized_TS/"
smooth_type = "regular"
file_names = [smooth_type + "_Walla2015_" + VI_idx + "_JFD.csv", 
              smooth_type + "_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              smooth_type + "_Grant2017_" + VI_idx + "_JFD.csv", 
              smooth_type + "_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

data=pd.DataFrame()

for file in file_names:
    curr_file=pd.read_csv(data_dir + file)
    curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])
    
    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years)==3 or len(all_years)==2:
        proper_year = all_years[1]
    elif len(all_years)==1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
    data=pd.concat([data, curr_file])

data.reset_index(drop=True, inplace=True)
data.head(2)

GT_TS = data[data.ID.isin(list(test20.ID.unique()))].copy()
print (len(GT_TS.ID.unique()))

# %%
EVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + EVI_colnames
GT_wide = pd.DataFrame(columns=columnNames, 
                        index=range(len(GT_TS.ID.unique())))
GT_wide["ID"] = GT_TS.ID.unique()

for an_ID in GT_TS.ID.unique():
    curr_df = GT_TS[GT_TS.ID==an_ID]
    
    GT_wide_indx = GT_wide[GT_wide.ID==an_ID].index
    GT_wide.loc[GT_wide_indx, VI_idx+"_1":VI_idx+"_36"] = curr_df.EVI.values[:36]

print (len(GT_wide.ID.unique()))
GT_wide.head(2)

print (len(test20.ID.unique()))
print (len(test20.ID))

print ((list(test20.ID)) == (list(GT_wide.ID)))
sorted(list(test20.ID)) == sorted(list(GT_wide.ID))

# %%
test20.head(2)
test20.sort_values(by=["ID"], inplace=True)
GT_wide.sort_values(by=["ID"], inplace=True)

test20.reset_index(drop=True, inplace=True)
GT_wide.reset_index(drop=True, inplace=True)

(list(test20.ID)) == (list(GT_wide.ID))

# %%
GT_wide = pd.merge(GT_wide, test20, on=['ID'], how='left')

out_name = ML_data_folder + "widen_test_TS/" +  VI_idx +"_" + smooth_type + \
          "_wide_test20_split_2Bconsistent_Oct17.csv"
GT_wide.to_csv(out_name, index = False)

# %% [markdown]
# # NDVI - SG

# %%
VI_idx = "NDVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/05_SG_TS/"
smooth_type = "SG"
file_names = [smooth_type + "_Walla2015_" + VI_idx + "_JFD.csv", 
              smooth_type + "_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              smooth_type + "_Grant2017_" + VI_idx + "_JFD.csv", 
              smooth_type + "_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

data=pd.DataFrame()

for file in file_names:
    curr_file=pd.read_csv(data_dir + file)
    curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])
    
    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years)==3 or len(all_years)==2:
        proper_year = all_years[1]
    elif len(all_years)==1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
    data=pd.concat([data, curr_file])

data.reset_index(drop=True, inplace=True)
data.head(2)

GT_TS = data[data.ID.isin(list(test20.ID.unique()))].copy()
print (len(GT_TS.ID.unique()))

# %%
EVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + EVI_colnames
GT_wide = pd.DataFrame(columns=columnNames, 
                        index=range(len(GT_TS.ID.unique())))
GT_wide["ID"] = GT_TS.ID.unique()

for an_ID in GT_TS.ID.unique():
    curr_df = GT_TS[GT_TS.ID==an_ID]
    
    GT_wide_indx = GT_wide[GT_wide.ID==an_ID].index
    GT_wide.loc[GT_wide_indx, VI_idx+"_1":VI_idx+"_36"] = curr_df.NDVI.values[:36]

print (len(GT_wide.ID.unique()))
GT_wide.head(2)

print (len(test20.ID.unique()))
print (len(test20.ID))

print ((list(test20.ID)) == (list(GT_wide.ID)))
sorted(list(test20.ID)) == sorted(list(GT_wide.ID))

# %%
test20.head(2)
test20.sort_values(by=["ID"], inplace=True)
GT_wide.sort_values(by=["ID"], inplace=True)

test20.reset_index(drop=True, inplace=True)
GT_wide.reset_index(drop=True, inplace=True)

(list(test20.ID)) == (list(GT_wide.ID))

# %%
GT_wide = pd.merge(GT_wide, test20, on=['ID'], how='left')

out_name = ML_data_folder + "widen_test_TS/" +  VI_idx +"_" + smooth_type + \
          "_wide_test20_split_2Bconsistent_Oct17.csv"
GT_wide.to_csv(out_name, index = False)

# %% [markdown]
# # NDVI - Regular

# %%
VI_idx = "NDVI"
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/04_regularized_TS/"
smooth_type = "regular"
file_names = [smooth_type + "_Walla2015_" + VI_idx + "_JFD.csv", 
              smooth_type + "_AdamBenton2016_" + VI_idx + "_JFD.csv", 
              smooth_type + "_Grant2017_" + VI_idx + "_JFD.csv", 
              smooth_type + "_FranklinYakima2018_"+ VI_idx +"_JFD.csv"]

data=pd.DataFrame()

for file in file_names:
    curr_file=pd.read_csv(data_dir + file)
    curr_file['human_system_start_time'] = pd.to_datetime(curr_file['human_system_start_time'])
    
    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years)==3 or len(all_years)==2:
        proper_year = all_years[1]
    elif len(all_years)==1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year==proper_year]
    data=pd.concat([data, curr_file])

data.reset_index(drop=True, inplace=True)
data.head(2)

GT_TS = data[data.ID.isin(list(test20.ID.unique()))].copy()
print (len(GT_TS.ID.unique()))

# %%
EVI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37) ]
columnNames = ["ID"] + EVI_colnames
GT_wide = pd.DataFrame(columns=columnNames, 
                        index=range(len(GT_TS.ID.unique())))
GT_wide["ID"] = GT_TS.ID.unique()

for an_ID in GT_TS.ID.unique():
    curr_df = GT_TS[GT_TS.ID==an_ID]
    
    GT_wide_indx = GT_wide[GT_wide.ID==an_ID].index
    GT_wide.loc[GT_wide_indx, VI_idx+"_1":VI_idx+"_36"] = curr_df.NDVI.values[:36]

print (len(GT_wide.ID.unique()))
GT_wide.head(2)

print (len(test20.ID.unique()))
print (len(test20.ID))

print ((list(test20.ID)) == (list(GT_wide.ID)))
sorted(list(test20.ID)) == sorted(list(GT_wide.ID))

# %%
test20.head(2)
test20.sort_values(by=["ID"], inplace=True)
GT_wide.sort_values(by=["ID"], inplace=True)

test20.reset_index(drop=True, inplace=True)
GT_wide.reset_index(drop=True, inplace=True)

(list(test20.ID)) == (list(GT_wide.ID))

# %%
GT_wide = pd.merge(GT_wide, test20, on=['ID'], how='left')

out_name = ML_data_folder + "widen_test_TS/" +  VI_idx +"_" + smooth_type + \
          "_wide_test20_split_2Bconsistent_Oct17.csv"
GT_wide.to_csv(out_name, index = False)

# %%
GT_wide.head(2)

# %%

# %%
