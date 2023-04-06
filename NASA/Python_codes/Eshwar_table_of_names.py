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
# import warnings
# warnings.filterwarnings("ignore")

import csv
import numpy as np
import pandas as pd
import scipy

import os, os.path
import sys

# to move files from one directory to another
import shutil

# %%
im_dir = "/Users/hn/Documents/01_research_data/NASA/snapshots/TS/06_snapshot_flat_PNG/"

# %%
a_dir = "/Users/hn/Documents/01_research_data/NASA/"
param_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
evaluation_set = pd.read_csv(param_dir + "evaluation_set.csv", low_memory=False)

# %%
evaluation_set.head(2)

# %%
all_files = [x for x in os.listdir(im_dir) if x.endswith(".png")]

# %%
len(all_files)/3

# %%
TOA_files = [x for x in all_files if ("TOA" in x)]
corrected_files = [x for x in all_files if ("corrected" in x)]

NDVIs = [x for x in all_files if not("corrected" in x)]
NDVIs = [x for x in NDVIs if not("TOA" in x)]

# %%
df = pd.DataFrame(index=range(len(NDVIs)), 
                 columns=["ID", 'NDVI_TS_Name', 'corrected_RGB', 
                          'TOA_RGB', "latitude", "longitude",
                          "Question Text", "Text box Text"])

df["Question Text"] = "How would you label this field?"
df["Text box Text"] = "Notes if you want to add any."

for counter, file in enumerate(NDVIs):
    L = file.split("_")
    L.pop(0)
    field_ID = "_".join(L)[:-4]
    # print (file, counter, field_ID, sep = ", ")
    
    # pick corrected name
    TOA_file_name = [x for x in TOA_files if field_ID in x]
    correct_file_name = [x for x in corrected_files if field_ID in x]
    
    df.loc[counter, "ID"] = field_ID
    df.loc[counter, "NDVI_TS_Name"] = file
    df.loc[counter, "corrected_RGB"] = correct_file_name[0]
    df.loc[counter, "TOA_RGB"] = TOA_file_name[0]
    
    df.loc[counter, "longitude"] = TOA_file_name[0].split("_")[5]
    df.loc[counter, "latitude"] = TOA_file_name[0].split("_")[4]
    


# %%
df.head(2)

# %%
result = pd.merge(df, evaluation_set, on="ID")
result.head(2)

# %%
result.sort_values(by=['CropTyp', 'county', "ID"], inplace=True)
result.reset_index(drop=True, inplace=True)
result.head(2)

# %%
eshwar_limited = result[["ID", 'NDVI_TS_Name', 'corrected_RGB', 
                         'TOA_RGB', "latitude", "longitude", 
                         "Question Text", "Text box Text"]].copy()
eshwar_limited.reset_index(drop=True, inplace=True)
eshwar_limited.head(2)

# %%
len(result.ID.unique())

# %%
"100010_WSDA_SF_2017" in result.ID.unique()

# %%
result[result.ID=="100010_WSDA_SF_2017"]

# %%

# %%
out_name = param_dir + "Eshwar_Extensive.csv"
result.index += 1
result = result.reset_index().rename({'index':'Question_overall'}, axis = 'columns')
result.to_csv(out_name, index = False)

out_name = param_dir + "Eshwar_limited.csv"
eshwar_limited.index += 1
eshwar_limited = eshwar_limited.reset_index().rename({'index':'Question_overall'}, axis = 'columns')
eshwar_limited.to_csv(out_name, index = False)

# %%
eshwar_limited.head(2)

# %% [markdown]
# # Break into small files

# %%
# result = result.reset_index().rename({'index':'Question_overall'}, axis = 'columns')
# eshwar_limited = eshwar_limited.reset_index().rename({'index':'Question_overall'}, axis = 'columns')
eshwar_limited.head(2)

# %%
no_questions = 60
if eshwar_limited.shape[0] % no_questions != 0:
    no_dfs = eshwar_limited.shape[0] // no_questions + 1
else:
    no_dfs = eshwar_limited.shape[0] // no_questions

# %%
broken_extended_dir = param_dir + "Eshwar_extended_broken/"
broken_limited_dir = param_dir + "Eshwar_limited_broken/"

os.makedirs(broken_extended_dir, exist_ok=True)
os.makedirs(broken_limited_dir, exist_ok=True)

writer_limited = pd.ExcelWriter(param_dir + 'all_limiteds.xlsx', engine='xlsxwriter')
writer_extended = pd.ExcelWriter(param_dir + 'all_extended.xlsx', engine='xlsxwriter')


for ii in range(no_dfs):
    curr_eshwar_limited = eshwar_limited.loc[(ii*no_questions): ((ii+1) * no_questions) - 1, ]
    curr_result = result.loc[(ii*no_questions): ((ii+1) * no_questions) - 1, ]
    
    curr_eshwar_limited.reset_index(drop=True, inplace=True)
    curr_result.reset_index(drop=True, inplace=True)
    
    curr_eshwar_limited.index += 1
    curr_result.index += 1
    
    curr_eshwar_limited = curr_eshwar_limited.reset_index().rename({'index':'Question_in_set'}, axis = 'columns')
    curr_result = curr_result.reset_index().rename({'index':'Question_in_set'}, axis = 'columns')
    
    out_name = broken_limited_dir + "question_set_limited_" + str(ii+1) + ".csv"
    curr_eshwar_limited.to_csv(out_name, index = False)
    
    out_name = broken_extended_dir + "question_set_extended_" + str(ii+1) + ".csv"
    curr_result.to_csv(out_name, index = False)
    
    
    curr_eshwar_limited.to_excel(writer_limited, sheet_name= "limited_" + str(ii+1), index=False)
    curr_result.to_excel(writer_extended, sheet_name= "extended_" + str(ii+1), index=False)

writer_limited.save()
writer_extended.save()

# %%

# %%

# %%

# %%

# %%

# %%
