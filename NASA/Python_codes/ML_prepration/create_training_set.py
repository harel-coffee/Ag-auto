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


# %% [markdown]
# # Toss unwanted crops: xmass-tree and such

# %%
# toss anything with "nursery" in crop-type
all_fields_irr = all_fields_irr[~(all_fields_irr['CropTyp'].str.contains('nursery'))]
all_fields_correct_year_irr = \
                  all_fields_correct_year_irr[~(all_fields_correct_year_irr['CropTyp'].str.contains('nursery'))]


# toss unwanted crops
all_fields_irr = all_fields_irr[~(all_fields_irr.CropTyp.isin(unwanted_crop))]
all_fields_correct_year_irr = all_fields_correct_year_irr[~(all_fields_correct_year_irr.CropTyp.isin(unwanted_crop))]


LL = len(all_fields_irr.CropTyp.unique())
print ("After tossing unwanted crops, # unique crops in all_fields is [%(ncrops)d]." % {"ncrops": LL}) 

LL = len(all_fields_correct_year_irr.CropTyp.unique())
print ("After tossing unwanted crops, # unique crops in all_fields_correct_year is [%(ncrops)d]." % {"ncrops": LL}) 

# %% [markdown]
# # Detect crop-types with less than 10 fields!
#
# - No filter for ```NASS``` or ```last-survey-date```: checkpoint
#
# Except we have filtered correct year in ```all_fields_correct_year``` which we are not using for the purpose of low-frequency crop-types!

# %%
print (np.sort(all_fields_irr.LstSrvD.unique())[1:10])
print ("")
print (all_fields_irr.DataSrc.unique())

# %% [markdown]
# # Toss crop types with less than 10 fields growing them.
#
# These fields are found by looking into ```all_fields```. We count number of these fields
# before filtering by ```last-survey-date```!

# %%
all_fields_irr_narrow = all_fields_irr[["ID", "CropTyp"]].copy()
all_fields_irr_narrow = all_fields_irr_narrow.groupby(['CropTyp']).count()
all_fields_irr_narrow = all_fields_irr_narrow[all_fields_irr_narrow.ID <= 10].copy()
all_fields_irr_narrow.reset_index(inplace=True)

out_name = output_dir + "cropTypes_lessThan10_6counties_irr.csv"
# all_fields_irr_narrow.to_csv(out_name, index = False)

all_fields_irr_narrow.head(2)

not_important_crops = all_fields_irr_narrow.CropTyp.unique()

# %%
# del(all_fields_irr_narrow, all_fields_irr)

# %%
LL = len(all_fields_correct_year_irr.CropTyp.unique())
print ("Before tossing low-count-fields, # unique crops is [%(ncrops)d]." % {"ncrops": LL}) 

all_fields_correct_year_irr = \
                     all_fields_correct_year_irr[~(all_fields_correct_year_irr.CropTyp.isin(not_important_crops))]

LL = len(all_fields_correct_year_irr.CropTyp.unique())
print ("After tossing low-count-fields, # unique crops is [%(ncrops)d]." % {"ncrops": LL}) 

# %%
# ryegrass_seed = all_fields_correct_year_irr[all_fields_correct_year_irr.CropTyp == "ryegrass seed"]

# print (ryegrass_seed.shape)

# pepper = all_fields_correct_year_irr[all_fields_correct_year_irr.CropTyp == "pepper"]
# print (pepper.shape)

# %%

# %%

# %%

# %% [markdown]
# # TOSS NASS

# %%
print (all_fields_correct_year_irr.shape)

all_fields_correct_year_irr_noNass = nc.filter_out_NASS(all_fields_correct_year_irr)

print (all_fields_correct_year_irr_noNass.shape)

# %%
del(all_fields_correct_year_irr)

# %%
# print ("No. unique crop type is [%(ncrops)d]." % {"ncrops":len(all_fields.CropTyp.unique())}) 
# print ("")
# print (np.sort(all_fields.CropTyp.unique()))

# %%
# %who

# %% [markdown]
# # Choose 10% of the fields, randomly

# %%
# number_of_fields_to_pick = all_fields_correct_year_irr_noNass.shape[0] // 10
number_of_fields_to_pick = all_fields_irr.shape[0] // 10
min_count = 50
unique_crops = all_fields_correct_year_irr_noNass.CropTyp.unique()
all_fields_correct_year_irr_noNass.reset_index(inplace=True, drop=True)

# %%
number_of_fields_to_pick

# %%
import random
random.seed(10)
np.random.seed(10)

unique_fields = all_fields_correct_year_irr_noNass.ID.unique()
unique_crops = all_fields_correct_year_irr_noNass.CropTyp.unique()
randomly_chosen_fields = list(np.random.choice(unique_fields, number_of_fields_to_pick, replace=False))

randomly_chosen_dt = all_fields_correct_year_irr_noNass[\
                                all_fields_correct_year_irr_noNass.ID.isin(randomly_chosen_fields)].copy()

not_chosen_dt = all_fields_correct_year_irr_noNass[\
                                ~(all_fields_correct_year_irr_noNass.ID.isin(randomly_chosen_fields))].copy()

# %%
# not_chosen_pepper = not_chosen_dt[not_chosen_dt.CropTyp == "pepper"]
# print (not_chosen_pepper.shape)

# chosen_peppers = randomly_chosen_dt[randomly_chosen_dt.CropTyp == "pepper"]
# print (chosen_peppers.shape)

# %%

# %%

# %% [markdown]
# # Go through crops and make sure 50 of each is chosen!!!

# %%
for a_crop in unique_crops:
    curr_chose_size = randomly_chosen_dt[randomly_chosen_dt.CropTyp == a_crop].shape[0]
    if (curr_chose_size < min_count):
        not_chosen_dt_a_crop = not_chosen_dt[not_chosen_dt.CropTyp == a_crop]
        """
          we need extra fields to reach min_count. But we may have less than what we need.
          So, we settle with whatever we have!
        """
        need_more_count = min_count - curr_chose_size
        # print ("curr_chose_size [%(curr_chose_size)d]." % {"curr_chose_size": curr_chose_size}) 
        # print ("need_more_count [%(need_more_count)d]." % {"need_more_count": need_more_count}) 
        # print ("not_chosen_dt_a_crop.shape[0] [%(AA)d]." % {"AA": not_chosen_dt_a_crop.shape[0]}) 

        need_more_count = min(need_more_count, not_chosen_dt_a_crop.shape[0])
        # print ("need_more_count [%(need_more_count)d]." % {"need_more_count": need_more_count}) 
        # print()
        
        
        additional_rand_choice = list(np.random.choice(not_chosen_dt_a_crop.ID.unique(), 
                                                       need_more_count, replace=False))

        additional_dt = not_chosen_dt_a_crop[not_chosen_dt_a_crop.ID.isin(additional_rand_choice)].copy()
        randomly_chosen_dt = pd.concat([randomly_chosen_dt, additional_dt]).reset_index(drop=True)
        
randomly_chosen_dt.sort_values(by=['CropTyp', 'ID'], inplace=True)

needed_columns = ['ID', 'CropTyp', 'Irrigtn', 'DataSrc', 
                  'Acres', 'ExctAcr', 'LstSrvD','county']
   
randomly_chosen_dt = randomly_chosen_dt[needed_columns]

# %%
print (randomly_chosen_dt.shape)
randomly_chosen_dt.head(2)

# %%
randomly_chosen_dt.sort_values(by=['CropTyp', 'ID', 'county'], inplace=True)

out_name = output_dir + "evaluation_set.csv"
randomly_chosen_dt.to_csv(out_name, index = False)

# %%

# %%

# %%

# %%

# %% [markdown]
# # Move the randomly chosen plots to new directory for labeling

# %%
file_prefix =  "training_set_"
file_post_fix = "_NASSOut_JustIrr_PereOut_LastSurveyFiltered_10Perc.csv"

dir_base = "/Users/hn/Documents/01_research_data/remote_sensing/01_NDVI_TS/70_Cloud/00_Eastern_WA_withYear/2Years/"
file_directory = dir_base + "ground_truth_tables/"

plot_directory_base = dir_base + "confusions_plots/plots/plots_fine_granularity/ALLYCF_plots_fine_gran/"
plot_directory_postfix = "_regular_irrigated_only_EVI_SOS3_EOS3/"



# %%
years = [2018] # 
for year in years:
    an_f_name = file_prefix + str(year) + file_post_fix    
    ground_truth_table = pd.read_csv(file_directory + an_f_name, low_memory=False)
    curr_plot_dir_base = plot_directory_base + str(year) + plot_directory_postfix
    
    for ii in np.arange(len(ground_truth_table.index)):
        crop_type = ground_truth_table.CropTyp[ii]
        crop_type = crop_type.replace(", ", "_")
        crop_type = crop_type.replace(" ", "_")

        curr_plot_dir = curr_plot_dir_base + crop_type + "/"
        
        trainint_path = curr_plot_dir + "ground_truth/"
        os.makedirs(trainint_path, exist_ok=True)
        
        curr_file_to_move = ground_truth_table.county[ii].replace(" ", "_") + "_" + \
                              crop_type + "_SF_year_" + str(year) + "_" + ground_truth_table.ID[ii] + ".png"

        try:
            shutil.move(curr_plot_dir + curr_file_to_move, trainint_path + curr_file_to_move)
        except:
            print ("no such a file")
            print (curr_plot_dir + curr_file_to_move)


# %%
ground_truth_table.county[ii].replace(" ", "_")
