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
import pandas as pd
import numpy as np
from datetime import datetime
import os, pickle, sys

# %%
dir_ = "/Users/hn/Desktop/kirti_RL/"
nass = pd.read_csv(dir_ + "NASS.csv")
mine = pd.read_csv(dir_ + "mine.csv")
mine.head(2)

# %%
my_Autauga = mine[(mine.State == "ALABAMA") & (mine.County == "AUTAUGA") & (mine.Year == 2017)].copy()
my_Autauga.head(2)

# %%
NASS_Autauga = nass[(nass.State == "ALABAMA") & (nass.County == "AUTAUGA") & (nass.Year == 2017)].copy()
NASS_Autauga.head(10)

# %% [markdown]
# ### my_Autauga includes non-numbers.
# They have divided fields into different categories based on sizes.
#

# %%
my_Autauga.Value.unique()

# %%
my_Autauga = my_Autauga[my_Autauga.Value != " (D)"]
my_Autauga.Value.unique()

# %%
my_Autauga.Value = my_Autauga.Value.replace(',', '', regex=True)
my_Autauga.Value = my_Autauga.Value.astype(int)
my_Autauga.Value.unique()

# %%
my_Autauga.head(2)

# %%
my_Autauga[["County", "Value", "Data Item"]].groupby(["County", "Data Item"]).sum().reset_index()

# %%
NASS_Autauga

# %%

# %%
