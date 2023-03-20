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
import os

# %%
data_dir = "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/"

# %%
csv_files = [x for x in os.listdir(data_dir) if x.endswith(".csv")]

# %%
all_data=pd.DataFrame()
for a_file in csv_files:
    curr_file = pd.read_csv(data_dir+a_file)
    all_data=pd.concat([all_data, curr_file])

# %%
all_data.head(2)

# %%
all_data.dropna(subset=['county'], inplace=True) 

# %%
len(all_data.CropTyp.unique())

# %%
sorted(all_data.CropTyp.unique())

# %%
all_data.ExctAcr.sum()

# %%
sorted(all_data.county.unique())

# %%
