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
# Here we want to see how much each file size would be if we eliminate NDVI

# %%
import csv
import numpy as np
import pandas as pd

import datetime
from datetime import date
import datetime
import time

import scipy
import os, os.path
from os import listdir
from os.path import isfile, join
import sys

import re
# from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as npc

# %% [markdown]
# ### Set up directories

# %%
data_dir = "/Users/hn/Documents/01_research_data/NASA/VI_TS/sixth_investig_intersected/"

# %%
L5 = pd.read_csv(data_dir + "L5_T1C2L2_Scaled_intGrant_2008-01-01_2012-05-05.csv")
L7 = pd.read_csv(data_dir + "L7_T1C2L2_Scaled_intGrant_2008-01-01_2021-09-23.csv")
L8 = pd.read_csv(data_dir + "L8_T1C2L2_Scaled_intGrant_2008-01-01_2021-10-14.csv")

# %%
L5.drop(['NDVI'], axis=1, inplace=True)
L7.drop(['NDVI'], axis=1, inplace=True)
L8.drop(['NDVI'], axis=1, inplace=True)
L8.head(2)

# %%
L578 = pd.concat([L5, L7, L8])


# %%
L578.head(2)

# %%
output_dir = data_dir + "/noNDVI/"
os.makedirs(output_dir, exist_ok=True)

out_name = output_dir + "GrantL578.csv"
L578.to_csv(out_name, index = False)

out_name = output_dir + "GrantL5.csv"
L5.to_csv(out_name, index = False)

out_name = output_dir + "GrantL7.csv"
L7.to_csv(out_name, index = False)


out_name = output_dir + "GrantL8.csv"
L8.to_csv(out_name, index = False)


# %%
