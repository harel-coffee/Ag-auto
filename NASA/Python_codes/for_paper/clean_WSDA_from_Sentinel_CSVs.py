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
import pandas as pd
import numpy as np
import datetime
from datetime import date
import datetime
import time

import os, os.path
from os import listdir
from os.path import isfile, join

import re
# from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys

# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as npc

# %%
sentinel_dir = "/Users/hn/Documents/01_research_data/remote_sensing/01_NDVI_TS/"+\
               "70_Cloud/00_Eastern_WA_withYear/2Years/"

sentinel_fNames=[x for x in os.listdir(sentinel_dir) if x.startswith("Eastern_WA")]
sentinel_fNames=sorted(sentinel_fNames)
sentinel_fNames

# %%
for fName in sentinel_fNames:
    curr = pd.read_csv(sentinel_dir+fName)
    curr=curr[["ID", "EVI", "NDVI", "system_start_time"]]
    out_name = sentinel_dir + "clean_" + fName
    curr.to_csv(out_name, index = False)

# %%

# %%
A = pd.read_csv(sentinel_dir + "buckwheat_challenge_sentinel_DF.csv")
A.head(2)

# %%

# %%

# %%
