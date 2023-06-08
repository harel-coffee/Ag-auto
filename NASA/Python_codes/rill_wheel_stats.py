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
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc

# %%
data_dir = "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/"
output_dir = "/Users/hn/Documents/01_research_data/NASA/"
param_dir = output_dir + "/parameters/"

# double_crop_potens = pd.read_csv(param_dir + "double_crop_potential_plants.csv")

# %%

# %%
f_names = ["AdamBenton2016.csv",
           "FranklinYakima2018.csv",
           "Grant2017.csv",
           "Walla2015.csv"]

# %% [markdown]
# # List of unwanted fields

# %%
unwanted_crop = ["christmas tree", "conifer seed", 
                 "crp/conservation", "dahlia", "dandelion", "developed", 
                 "driving range", "flowers, nursery and christmas tree farms",
                 "golf course", "greenhouse", "iris", "miscellaneous deciduous", 
                 "nursery, caneberry", "nursery, greenhouse", "nursery, holly",
                 "nursery, lavender", "nursery, lilac", "nursery, orchard/vineyard",
                 "nursery, ornamental", "nursery, silvaculture", "nursery, silviculture",
                 "peony", "reclamation seed",
                 "research station", "shellfish", 
                 "silvaculture", "silviculture", "tulip"]

# %% [markdown]
# # Read Files

# %%
all_fields = pd.DataFrame(data=None, index=None)
all_fields_correct_year = pd.DataFrame(data=None, index=None)
for file in f_names:
    curr_table = pd.read_csv(data_dir + file, low_memory=False)
    all_fields = pd.concat([all_fields, curr_table]).reset_index(drop=True)
    
    # pick proper year here. currently it is in the function 
    # generate_training_set_important_counties(.)
    # or you want to keep it in the function?
    curr_table = nc.filter_by_lastSurvey(curr_table, file[-8:-4])
    all_fields_correct_year = pd.concat([all_fields_correct_year, curr_table]).reset_index(drop=True)
    

all_fields['CropTyp'] = all_fields['CropTyp'].str.lower()
all_fields.drop_duplicates(inplace=True) # sanity check


all_fields_correct_year['CropTyp'] = all_fields_correct_year['CropTyp'].str.lower()
all_fields_correct_year.drop_duplicates(inplace=True) # sanity check

# %%
print ("all_fields.shape is: ", all_fields.shape) 
print ("all_fields_correct_year.shape is: ", all_fields_correct_year.shape) 

print ("No. unique crops in all_fields is:", len(all_fields.CropTyp.unique()))
print ("No. unique crops in all_fields_correct_year is:", len(all_fields.CropTyp.unique()))

# %% [markdown]
# # Filters 
#   - NASS, last survey date, Irrigated fields, and unwanted locations such as research station.

# %% [markdown]
# # Filter Irrigated Fields

# %%
# pick up irrigated
all_fields_irr = nc.filter_out_nonIrrigated(all_fields)
all_fields_correct_year_irr = nc.filter_out_nonIrrigated(all_fields_correct_year)


print ("all_fields.shape:", all_fields.shape)
print ("all_fields_irr.shape:", all_fields_irr.shape)
print ("")
print ("all_fields_correct_year.shape:", all_fields_correct_year.shape)
print ("all_fields_correct_year_irr.shape", all_fields_correct_year_irr.shape)

# %%
del(all_fields, all_fields_correct_year)

# %%
all_fields_correct_year_irr.shape

# %%
all_fields_correct_year_irr.head(2)

# %% [markdown]
# # Toss NASS

# %%
print (all_fields_correct_year_irr.shape)
all_fields_correct_year_irr_noNass = nc.filter_out_NASS(all_fields_correct_year_irr)
print (all_fields_correct_year_irr_noNass.shape)

# %% [markdown]
# # Drop stupid Columns

# %%
all_fields_correct_year_irr_noNass.drop(labels=["CropGrp", "CropTyp", "IntlSrD", "Notes", "TRS",
                                                "RtCrpTy", "Shp_Lng", "Shap_Ar", "ExctAcr"], 
                                        axis='columns',
                                       inplace = True)

# %%
all_correct_year_irr_noNass_count = all_fields_correct_year_irr_noNass.groupby(['county', 'Irrigtn']).count()
all_correct_year_irr_noNass_count.reset_index(inplace=True)
all_correct_year_irr_noNass_count.sort_values(by=['county', 'Irrigtn'], inplace=True)

# %%
all_correct_year_irr_noNass_count.head(5)

# %%
all_correct_year_irr_noNass_count.drop(labels=["Acres", "LstSrvD", "DataSrc"], 
                                       axis='columns',
                                       inplace = True)
all_correct_year_irr_noNass_count.rename(columns={"ID": "count"}, inplace=True)

all_correct_year_irr_noNass_count.head(2)

# %%

# %%
all_correct_year_irr_noNass_Acr = all_fields_correct_year_irr_noNass.groupby(['county', 'Irrigtn']).sum()
all_correct_year_irr_noNass_Acr.reset_index(inplace=True)
all_correct_year_irr_noNass_Acr.sort_values(by=['county', 'Irrigtn'], inplace=True)

# %%
all_correct_year_irr_noNass_Acr.head(2)

# %%
print (all_correct_year_irr_noNass_Acr.shape)
print (all_correct_year_irr_noNass_count.shape)

# %%
stats = pd.merge(all_correct_year_irr_noNass_count, all_correct_year_irr_noNass_Acr, 
                 on=['county', "Irrigtn"], how='left')
stats.head(2)

# %%
out_name = output_dir + "rill_wheel_stat_allFields_CountyIrrType.csv"
stats.to_csv(out_name, index = False)
del(stats)

# %%

# %% [markdown]
# # Group by irrigation type. No counties here

# %%
all_correct_year_irr_noNass_count = all_fields_correct_year_irr_noNass.groupby(['Irrigtn']).count()
all_correct_year_irr_noNass_count.reset_index(inplace=True)
all_correct_year_irr_noNass_count.sort_values(by=['Irrigtn'], inplace=True)

all_correct_year_irr_noNass_count.drop(labels=["Acres", "LstSrvD", "DataSrc", "county", "ExctAcr"], 
                                       axis='columns',
                                       inplace = True)
all_correct_year_irr_noNass_count.rename(columns={"ID": "count"}, inplace=True)

all_correct_year_irr_noNass_count.head(2)

# %%
all_correct_year_irr_noNass_Acr = all_fields_correct_year_irr_noNass.groupby(['Irrigtn']).sum()
all_correct_year_irr_noNass_Acr.reset_index(inplace=True)
all_correct_year_irr_noNass_Acr.sort_values(by=['Irrigtn'], inplace=True)

all_correct_year_irr_noNass_Acr.head(2)

# %%
stats = pd.merge(all_correct_year_irr_noNass_count, all_correct_year_irr_noNass_Acr, 
                 on=["Irrigtn"], how='left')
stats.head(2)

# %%
out_name = output_dir + "rill_wheel_stat_allFields_justIrrType.csv"
stats.to_csv(out_name, index = False)

# %%
del(all_correct_year_irr_noNass_Acr, 
    all_correct_year_irr_noNass_count, 
    all_fields_correct_year_irr,
    all_fields_irr, all_fields_correct_year_irr_noNass, stats)

# %%

# %% [markdown]
# # Do the same for the damn 6000 training set

# %%
adir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
fucking_train = pd.read_csv(adir + "evaluation_set.csv")

# %%
fucking_train.drop(labels=['ID', "ExctAcr", "LstSrvD", "DataSrc", "CropTyp"], 
                   axis="columns", 
                   inplace=True)
fucking_train.head(2)

# %%
fucking_train_count = fucking_train.groupby(['county', 'Irrigtn']).count()
fucking_train_count.reset_index(inplace=True)
fucking_train_count.sort_values(by=['county', 'Irrigtn'], inplace=True)
fucking_train_count.rename(columns={"Acres": "count"}, inplace=True)
fucking_train_count.head(2)

# %%
fucking_train_Acr = fucking_train.groupby(['county', 'Irrigtn']).sum()
fucking_train_Acr.reset_index(inplace=True)
fucking_train_Acr.sort_values(by=['county', 'Irrigtn'], inplace=True)
fucking_train_Acr.head(2)

# %%
stats = pd.merge(fucking_train_count, fucking_train_Acr, on=['county', "Irrigtn"], how='left')
stats.head(2)

# %%
out_name = output_dir + "rill_wheel_stat_trainSet_CountyIrrType.csv"
stats.to_csv(out_name, index = False)
del(stats)

# %% [markdown]
# # No county here

# %%
fucking_train_count = fucking_train.groupby(['Irrigtn']).count()
fucking_train_count.reset_index(inplace=True)
fucking_train_count.sort_values(by=['Irrigtn'], inplace=True)
fucking_train_count.drop(labels=["county"], axis="columns", inplace=True)

fucking_train_count.rename(columns={"Acres": "count"}, inplace=True)
fucking_train_count.head(2)

# %%
fucking_train_Acr = fucking_train.groupby(['Irrigtn']).sum()
fucking_train_Acr.reset_index(inplace=True)
fucking_train_Acr.sort_values(by=['Irrigtn'], inplace=True)
fucking_train_Acr.head(2)

# %%
stats = pd.merge(fucking_train_count, fucking_train_Acr, on=["Irrigtn"], how='left')
stats.head(2)

# %%
out_name = output_dir + "rill_wheel_stat_trainSet_JustIrrType.csv"
stats.to_csv(out_name, index = False)

# %%

# %%
del(fucking_train_count, fucking_train_Acr, stats)

# %% [markdown]
# # Do the same fucking shit for 250 experts' fields

# %%
import pandas as pd
import csv

import os, os.path
import sys

# to move files from one directory to another
import shutil

# %%
perry_dir = "/Users/hn/Documents/01_research_data/NASA/Perry_and_Co/"

choices_set_1_xl = pd.ExcelFile(perry_dir + "set1_PerryandCo.xlsx")
choices_set_1_sheet_names = choices_set_1_xl.sheet_names  # see all sheet names
print (choices_set_1_sheet_names)


# %%
experts_250 = pd.DataFrame()

for a_sheet in choices_set_1_sheet_names:
    a_choice_sheet = choices_set_1_xl.parse(a_sheet)
    experts_250 = pd.concat([experts_250, a_choice_sheet]).reset_index(drop=True)

# %%
experts_250.drop(labels = ["Question_in_set", "Question_overall", "ID", 
                           "NDVI_TS_Name", "corrected_RGB", "TOA_RGB", "latitude",
                           "longitude", "CropTyp", "DataSrc", "ExctAcr", "LstSrvD"],
                 axis="columns",
                inplace=True)
experts_250.head(2)

# %%
experts_250_count = experts_250.groupby(['county', 'Irrigtn']).count()
experts_250_count.reset_index(inplace=True)
experts_250_count.sort_values(by=['county', 'Irrigtn'], inplace=True)
experts_250_count.rename(columns={"Acres": "count"}, inplace=True)
print (experts_250_count.head(2))

experts_250_Acr = experts_250.groupby(['county', 'Irrigtn']).sum()
experts_250_Acr.reset_index(inplace=True)
experts_250_Acr.sort_values(by=['county', 'Irrigtn'], inplace=True)
experts_250_Acr.head(2)

# %%
stats = pd.merge(experts_250_Acr, experts_250_count, on=["county", "Irrigtn"], how='left')

out_name = output_dir + "rill_wheel_stat_expertsSet1_CountyIrrType.csv"
stats.to_csv(out_name, index = False)
del(stats)

# %%
experts_250_count = experts_250.groupby(['Irrigtn']).count()
experts_250_count.reset_index(inplace=True)
experts_250_count.drop(labels=["county"], axis="columns", inplace=True)
experts_250_count.sort_values(by=['Irrigtn'], inplace=True)
experts_250_count.rename(columns={"Acres": "count"}, inplace=True)
print (experts_250_count.head(2))

experts_250_Acr = experts_250.groupby(['Irrigtn']).sum()
experts_250_Acr.reset_index(inplace=True)
experts_250_Acr.sort_values(by=['Irrigtn'], inplace=True)
experts_250_Acr.head(2)

# %%
stats = pd.merge(experts_250_Acr, experts_250_count, on=["Irrigtn"], how='left')

out_name = output_dir + "rill_wheel_stat_expertsSet1_JustIrrType.csv"
stats.to_csv(out_name, index = False)
del(stats)
