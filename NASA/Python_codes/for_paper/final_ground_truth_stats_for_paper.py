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

# %%
database_dir = "/Users/hn/Documents/01_research_data/NASA/"
ML_data_dir  = database_dir + "ML_data_Oct17/"
params_dir   = database_dir + "parameters/"
perry_dir    = database_dir + "Perry_and_Co/"

# %%
train_labels = pd.read_csv(ML_data_dir + "groundTruth_labels_Oct17_2022.csv")

evaluation_set=pd.read_csv(params_dir+"evaluation_set.csv")

# %%
train_labels.head(2)

# %%
train_labels = pd.merge(train_labels, evaluation_set, on=['ID'], how='left')
train_labels.head(2)

# %%
print (train_labels.shape)
print (len(train_labels.ID.unique()))

# %%
print (len(train_labels.CropTyp.unique()))

# %%
sorted(list(train_labels.CropTyp.unique()))

# %%
print (len(evaluation_set.CropTyp.unique()))

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
train80.groupby(['Vote'])['Vote'].count()

# %%

# %%

# %%
