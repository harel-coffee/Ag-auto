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
a_dir = "/Users/hn/Documents/01_research_data/NASA/"

VI_TS_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/06_SOS_plots/train_plots_JFD/"

out_dir_base = "/Users/hn/Documents/01_research_data/NASA/snapshots/TS/06_snapshot_plots_exactEval/"
flat_dir = "/Users/hn/Documents/01_research_data/NASA/snapshots/TS/06_snapshot_flat/"
# double_crop_potens = pd.read_csv(param_dir + "double_crop_potential_plants.csv")

# %% [markdown]
# # Move the randomly chosen plots to new directory for labeling

# %%
evaluation_set = pd.read_csv(a_dir + "evaluation_set.csv", low_memory=False)

# %%
evaluation_set.head(2)

# %%
"104119_WSDA_SF_2016" in list(evaluation_set.ID)

# %%
evaluation_set[evaluation_set.ID ==  "100808_WSDA_SF_2018"]

# %%
evaluation_set.county.unique()

# %%
for a_row in range(evaluation_set.shape[0]):
    plant = evaluation_set.loc[a_row, 'CropTyp'].lower().replace(" ", "_").replace(",", "").replace("/", "_")
    sub_dir = VI_TS_dir + plant + "/"
    
    county = evaluation_set.loc[a_row, 'county']
    curr_ID = evaluation_set.loc[a_row, 'ID']
    
    if county == "Grant":
        county_name = "Grant2017"
    elif county in ["Adams", "Benton"]:
        county_name = "AdamBenton2016"
    elif county in ["Franklin", "Yakima"]:
        county_name = "FranklinYakima2018"
    elif county == "Walla Walla":
        county_name = "Walla2015"
    curr_filename = county_name + "_" + curr_ID + ".pdf"    
    output_dir = out_dir_base + plant + "/"
    try:
        shutil.copy(sub_dir + curr_filename, output_dir + curr_filename)
    except:
        print ("a_row", a_row)
        print (curr_filename)
        print (county_name)
        print (curr_ID)
        print ("_____________________________________")

# %%
evaluation_set.shape


# %%
def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

# Run the above function and store its results in a variable.   
full_file_paths = get_filepaths(out_dir_base)

# %%
for file in full_file_paths:
    try:
        shutil.copy(file, flat_dir)
    except:
        print ("file", file)
        print ("_____________________________________")

# %%

# %%

# %%
