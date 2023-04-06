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

a_dir = "/Users/hn/Documents/01_research_data/NASA/"
param_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"

# %%
# writer_limited = pd.ExcelWriter(param_dir + 'all_limiteds.xlsx', engine='xlsxwriter')

# %%
perrys_choice = pd.read_excel(io=param_dir + "PerrysChoices.xlsx", sheet_name=0)
perrys_choice.head(3)

# %%
perrys_choice.columns = perrys_choice.iloc[1]

# %%
perrys_choice = perrys_choice.drop(labels=[0, 1], axis='index')
perrys_choice.reset_index(drop=True, inplace=True)
perrys_choice.head(2)

# %%
people_names = ["HN", "MB", "S", "K"]

chosen_metadata = pd.DataFrame()

for a_person in people_names:
    for row in range(perrys_choice.shape[0]):
        col_name = a_person + "Form #"
        if not pd.isna(perrys_choice.loc[row, col_name]):
            curr_sheet_name = "limited_" + str(int(perrys_choice.loc[row, col_name]))
            curr_sheet = pd.read_excel(io=param_dir + "PerrysChoices.xlsx", sheet_name=curr_sheet_name)

            curr_question_column = a_person + "Question #"

            curr_question_number = str(int(perrys_choice.loc[row, curr_question_column]))

            chosen_row = curr_sheet[curr_sheet.Question_in_set == int(curr_question_number)].copy()
            chosen_row.reset_index(drop=True, inplace=True)

            chosen_metadata = pd.concat([chosen_metadata, chosen_row])
chosen_metadata.head(1)

# %%
chosen_metadata.drop(labels=["Question Text", "Text box Text", "Question_overall", "Question_in_set"], 
                                 axis="columns", inplace=True)

# %%
extensive_dt = pd.read_csv(param_dir + "Eshwar_Extensive.csv")
extensive_dt.head(1)

# %%
extensive_dt_chosen_subset = extensive_dt[extensive_dt.ID.isin(chosen_metadata.ID)].copy()
extensive_dt_chosen_subset.reset_index(drop=True, inplace=True)
extensive_dt_chosen_subset.head(2)

# %%
extensive_dt_chosen_subset.drop(labels=["Question Text", "Text box Text", "Question_overall"], 
                                 axis="columns", inplace=True)

# %%
#sorted(extensive_dt_chosen_subset.CropTyp.unique())

# %%
desired_order = ['corn, field', 'corn, sweet', 'corn seed', 
                 'wheat', 'wheat fallow', 'potato', 'bean, dry', 'bean, green',
                 'onion', 'pea seed', 'pea, dry','pea, green',
                 'barley', 'barley hay', 'buckwheat', 
                 'canola', 'carrot', 'carrot seed', 'market crops',
                 'mint', 'oat hay', 'triticale', 'triticale hay',
                 'yellow mustard', 
                 'alfalfa seed', 'grass seed', 'bluegrass seed',
                 'alfalfa hay', 'grass hay']


# %%
chosen_subset_myOrder = extensive_dt_chosen_subset.copy()
chosen_subset_myOrder.CropTyp = chosen_subset_myOrder.CropTyp.astype("category")
chosen_subset_myOrder['CropTyp'] = chosen_subset_myOrder['CropTyp'].cat.set_categories(desired_order)
chosen_subset_myOrder = chosen_subset_myOrder.sort_values(["CropTyp"])
chosen_subset_myOrder.reset_index(drop=True, inplace=True)
chosen_subset_myOrder.head(2)

# %%
alfalfa_hay = chosen_subset_myOrder[chosen_subset_myOrder.CropTyp=="grass seed"]
alfalfa_hay.shape

# %%
# remove Grass hay and alfalf hay

# %%
chosen_subset_myOrder = chosen_subset_myOrder[chosen_subset_myOrder.CropTyp != "alfalfa hay"]
chosen_subset_myOrder.shape

# %% [markdown]
# # Break and save into 50 question sheets

# %%
no_questions = 50
if chosen_subset_myOrder.shape[0] % no_questions != 0:
    no_dfs = chosen_subset_myOrder.shape[0] // no_questions + 1
else:
    no_dfs = chosen_subset_myOrder.shape[0] // no_questions

# %%
chosen_subset_myOrder.head(1)

# %%
chosen_subset_myOrder.index += 1
chosen_subset_myOrder = chosen_subset_myOrder.reset_index().rename({'index':'Question_overall'}, 
                                                                   axis = 'columns')

# %%
writer_extended = pd.ExcelWriter(param_dir + 'set1_PerryandCo.xlsx', engine='xlsxwriter')

for ii in range(no_dfs):
    # curr_eshwar_limited = eshwar_limited.loc[(ii*no_questions): ((ii+1) * no_questions) - 1, ]
    curr_result = chosen_subset_myOrder.loc[(ii*no_questions): ((ii+1) * no_questions) - 1, ]
    
    # curr_eshwar_limited.reset_index(drop=True, inplace=True)
    curr_result.reset_index(drop=True, inplace=True)
    
    # curr_eshwar_limited.index += 1
    curr_result.index += 1
    
    # curr_eshwar_limited = curr_eshwar_limited.reset_index().rename({'index':'Question_in_set'}, axis = 'columns')
    curr_result = curr_result.reset_index().rename({'index':'Question_in_set'}, axis = 'columns')
    
    # out_name = broken_limited_dir + "question_set_limited_" + str(ii+1) + ".csv"
    # curr_eshwar_limited.to_csv(out_name, index = False)
    
    out_name = param_dir + "set1_Perry_" + str(ii+1) + ".csv"
    # curr_result.to_csv(out_name, index = False)
    
    
    # curr_eshwar_limited.to_excel(writer_limited, sheet_name= "limited_" + str(ii+1), index=False)
    curr_result.to_excel(writer_extended, sheet_name= "extended_" + str(ii+1), index=False)

# writer_limited.save()
writer_extended.save()

# %%

# %%
