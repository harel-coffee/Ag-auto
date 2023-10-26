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

# %% [markdown]
# Test time series and see of correct labels are assigned to correct fields in the new split!!!

# %%
import numpy as np
import pandas as pd
import scipy, scipy.signal

from datetime import date
import os, os.path, shutil, sys, time

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

# %% [markdown]
# # Directories

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
VI_idx = "NDVI"
smooth_type = "SG"

data_base_ = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
overSamples_data_folder = data_base_ + "overSamples/"

# %%

# %%
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta.head(2)

# %%
GT_labels = pd.read_csv(data_base_ + "groundTruth_labels_Oct17_2022.csv")

GT_labels.rename(columns={"Vote": "label"}, inplace=True)
GT_labels.head(2)

# %% [markdown]
# ### SR 3

# %%
sr = 3

# %%
f_name = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample" + str(sr) + ".csv"

# train80_GT_wide:
SG_wide = pd.read_csv(overSamples_data_folder + f_name)

# %%
TR = 1 # split 1

SG_wide_TR1 = pd.read_csv(overSamples_data_folder + "train_test_DL_" + str(1) + "/" + f_name)

# %%
SG_wide.head(2)

# %%
SG_wide_TR1.head(2)

# %% [markdown]
# ### Check if labels are correct

# %%
SG_wide_TR1_copy = SG_wide_TR1.copy()
SG_wide_TR1_copy = SG_wide_TR1_copy.merge(GT_labels, how='left', on='ID')
SG_wide_TR1_copy.head(5)

# %%
sum(SG_wide_TR1_copy.Vote - SG_wide_TR1_copy.label)

# %%
Id = "148385_WSDA_SF_2015"
GT_labels[GT_labels.ID == Id]

# %%
SG_wide_TR1[SG_wide_TR1.ID == Id]

# %%

# %% [markdown]
# # test original oversamples

# %%
d_ = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/overSamples/"

EVI_SG_ = pd.read_csv(d_ + "EVI_SG_wide_train80_split_2Bconsistent_Oct17_overSample4.csv")
EVI_SG_.head(2)

# %%
EVI_SG_ = EVI_SG_.merge(GT_labels, how='left', on='ID')
EVI_SG_.head(2)

# %%
sum(EVI_SG_.Vote - EVI_SG_.label)

# %%
