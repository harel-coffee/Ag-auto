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
# Oct 17, 2022
#
# In this notebook we want to look at crop types, number of fields, etc. to present in the paper.

# %%
import pandas as pd
import csv
import numpy as np
import os, os.path
import sys

import collections # to count frequency of elements of an array
# to move files from one directory to another
import shutil
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc
import NASA_plot_core as ncp

# %%
database_dir = "/Users/hn/Documents/01_research_data/NASA/"
ML_data_dir  = database_dir + "ML_data_Oct17/"
params_dir   = database_dir + "parameters/"
perry_dir    = database_dir + "Perry_and_Co/"

# %%
SF_data_dir = "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/"
csv_files = [x for x in os.listdir(SF_data_dir) if x.endswith(".csv")]
print (csv_files)
all_data=pd.DataFrame()
for a_file in csv_files:
    curr_file = pd.read_csv(SF_data_dir+a_file)
    all_data=pd.concat([all_data, curr_file])
    
print (all_data.shape)
all_data.head(2)

# %%
train_labels = pd.read_csv(ML_data_dir + "groundTruth_labels_Oct17_2022.csv")
evaluation_set = pd.read_csv(params_dir+"evaluation_set.csv")
print (f"{evaluation_set.shape=}")
print (f"{len(evaluation_set.CropTyp.unique())=}")

# %%
AnnualPerennialToss = pd.read_csv(params_dir + "AnnualPerennialTossMay122023.csv")
print (f"{AnnualPerennialToss.shape=}")

AnnualPerennialToss.rename(columns={"Crop_Type": "CropTyp"}, inplace=True)
print (AnnualPerennialToss.potential.unique())

# %%
AnnualPerennialToss.head(2)

# %%
toss_crops = AnnualPerennialToss[AnnualPerennialToss.potential=="toss"].CropTyp.unique()
perennial_crops = AnnualPerennialToss[AnnualPerennialToss.potential=="n"].CropTyp.unique()
annual_crops = AnnualPerennialToss[AnnualPerennialToss.potential=="y"].CropTyp.unique()
yn_crops = AnnualPerennialToss[AnnualPerennialToss.potential=="yn"].CropTyp.unique()

print (f"{len(toss_crops)=}")
print (f"{len(perennial_crops)=}")
print (f"{len(annual_crops)=}")
print (f"{len(yn_crops)=}")

# %%
train_labels.head(2)

# %%
train_labels = pd.merge(train_labels, evaluation_set, on=['ID'], how='left')
train_labels.head(2)

# %%
print (train_labels.shape)
print (len(train_labels.ID.unique()))

# %%
print (f"{len(train_labels.CropTyp.unique())=}")
print (f"{len(AnnualPerennialToss.CropTyp.unique())=}")
print (f"{len(evaluation_set.CropTyp.unique())=}")

# %%
# [x for x in train_labels.CropTyp.unique() if not(x in annual_crops)]

# %%

# %%
[x for x in evaluation_set.CropTyp.unique() if x not in train_labels.CropTyp.unique()]

# %%
alfalfa_grass = evaluation_set[evaluation_set.CropTyp=="alfalfa/grass hay"]
print (alfalfa_grass.shape)
alfalfa_grass_big=alfalfa_grass[alfalfa_grass.ExctAcr>10]
print (alfalfa_grass_big.shape)

# %%
evaluation_set[evaluation_set.CropTyp=="alkali bee bed"].shape

# %%
evaluation_set[evaluation_set.CropTyp=="caneberry"].shape

# %%
evaluation_set[evaluation_set.CropTyp=="walnut"].shape

# %%
len(list(evaluation_set.CropTyp.unique()))

# %%
# sorted(list(evaluation_set.CropTyp.unique()))

# %%
# train_labels.groupby(['CropTyp'])['CropTyp'].count()

# %%
# evaluation_set_big = evaluation_set[evaluation_set.ExctAcr>10]
# evaluation_set_big.groupby(['CropTyp'])['CropTyp'].count()

# %% [markdown]
# ### Test and train 20-80 stats here

# %%
train80 = pd.read_csv(ML_data_dir+"train80_split_2Bconsistent_Oct17.csv")
test20 = pd.read_csv(ML_data_dir+"test20_split_2Bconsistent_Oct17.csv")

# %%
test20.head(2)

# %%
test20.groupby(['Vote'])['Vote'].count()

# %%
573+59

# %%
train80.groupby(['Vote'])['Vote'].count()

# %%
2292 + 573

# %%
59+236

# %%

# %%
train_labels.shape

# %%
SF_data_dir = "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/"

csv_files = ["FranklinYakima2018", "Walla2015", "AdamBenton2016", "Grant2017"]
all_data=pd.DataFrame()
for a_file in csv_files:
    # print (a_file)
    curr_file = pd.read_csv(SF_data_dir + a_file + ".csv")
    year = int(a_file[-4:])
    curr_file =  nc.filter_by_lastSurvey(curr_file, year)
    curr_file["year"] = year
    all_data=pd.concat([all_data, curr_file])

all_data = nc.filter_out_nonIrrigated(all_data)
all_data = all_data[all_data.ExctAcr > 10]
all_data["year_2"] = all_data.LstSrvD.str.split(pat = "/", expand=True)[0]
print (all_data.shape)
all_data.head(2)

# %%
all_data_noNASS = all_data[all_data.DataSrc != "nass"]
all_data_noNASS = all_data_noNASS[all_data_noNASS.CropTyp.isin(list(train_labels.CropTyp.unique()))].copy()

# %%
all_data_areas = all_data_noNASS.groupby(['CropTyp'])['Acres'].sum()
all_data_areas = pd.DataFrame(all_data_areas)
all_data_areas.reset_index(inplace=True)

all_data_areas.rename(columns={"Acres": "all_data_Acres"}, inplace=True)

GT_areas = train_labels.groupby(['CropTyp'])['Acres'].sum()
GT_areas = pd.DataFrame(GT_areas)
GT_areas.reset_index(inplace=True)
GT_areas.rename(columns={"Acres": "GT_Acres"}, inplace=True)

# %%
train80_labels = train_labels[train_labels.ID.isin(list(train80.ID))]
test20_labels = train_labels[train_labels.ID.isin(list(test20.ID))]

# %%
train80_areas = train80_labels.groupby(['CropTyp'])['Acres'].sum()
test20_areas = test20_labels.groupby(['CropTyp'])['Acres'].sum()

train80_areas = pd.DataFrame(train80_areas)
train80_areas.reset_index(inplace=True)

test20_areas = pd.DataFrame(test20_areas)
test20_areas.reset_index(inplace=True)

train80_areas.rename(columns={"Acres": "train80_areas"}, inplace=True)
test20_areas.rename(columns={"Acres": "test20_areas"}, inplace=True)

# %%
all_data_areas = pd.merge(all_data_areas, GT_areas, on=['CropTyp'], how='left')
all_data_areas = pd.merge(all_data_areas, train80_areas, on=['CropTyp'], how='left')
all_data_areas = pd.merge(all_data_areas, test20_areas, on=['CropTyp'], how='left')

# %%
all_data_areas['test20_areas'].fillna(0, inplace=True)
all_data_areas.test20_areas = all_data_areas.test20_areas.astype("int")

# %%
for idx in all_data_areas.index:
    print (all_data_areas.loc[idx, ].values)

# %%

# %%

# %%

# %%

# %%
perry_dir = "/Users/hn/Documents/01_research_data/NASA/Perry_and_Co/"
set1_fields = pd.read_csv(perry_dir + "set_1_experts_stats_extended_sortOpinionCrop.csv")
print (set1_fields.shape)
set1_fields.head(2)

# %%
set1_fields[set1_fields.ExctAcr<10].shape

# %%

# %%
set2_all_responses = pd.read_csv(perry_dir + "set2_all_responses.csv")
set2_noRepetition = pd.read_csv(perry_dir + "set2_all_responases_noRepetition.csv")

# %%
set2_all_responses.shape

# %%
set2_noRepetition.shape

# %%
# SVM Kappa
round(((632 * (562 + 55)) - (573*566 + 59*66)) / (632**2 - (573*566 + 59*66)), 3)

# %%
# DL Kappa
round(((632 * (568 + 55)) - (573*572 + 59*60)) / (632**2 - (573*572 + 59*60)), 3)

# %%
# kNN Kappa
round(((632 * (559 + 48)) - (573*570 + 59*62)) / (632**2 - (573*570 + 59*62)), 3)

# %%
# RF Kappa
round(((632 * (565 + 44)) - (573*580 + 59*52)) / (632**2 - (573*580 + 59*52)), 3)

# %%
# NDVI-ratio Kappa
round(((632 * (499 + 8)) - (573*550 + 59*82)) / (632**2 - (573*550 + 59*82)), 3)

# %%
# DL User accuracies
# class 1
print (f"User, DL, Class 1: {round(568/572, 2)}")
# class 2
print (f"User, DL, Class 2: {round(55/60, 2)}")

print ("==================================================")

# class 1
print (f"Producer, DL, Class 1: {round(568/573, 2)}" )

# class 2
print (f"Producer, DL, Class 2: {round(55/59, 2)}")

# %%
# DL User accuracies
print (f"User, SVM, Class 2: {round(55/66, 2)}")
print (f"User, DL, Class 2: {round(55/60, 2)}")
print (f"User, kNN, Class 2: {round(48/62, 2)}")
print (f"User, RF, Class 2: {round(44/52, 2)}")
print (f"User, NDVI-ratio, Class 2: {round(8/82, 3)}")
print ("==================================================")
# class 2
print (f"Producer, SVM, Class 2: {round(55/59, 2)}")
print (f"Producer, DL, Class 2: {round(55/59, 2)}")
print (f"Producer, kNN, Class 2: {round(48/59, 2)}")
print (f"Producer, RF, Class 2: {round(44/59, 2)}")
print (f"Producer, NDVI-ratio, Class 2: {round(8/59, 2)}")

# %%
