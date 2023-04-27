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

# %%
import pandas as pd

# %%

# %%

# %%
dir_base = "/Users/hn/Documents/01_research_data/NASA/VI_TS/"
data_dir = dir_base + "/8th_intersected_2008_2018_EastIrr/02_outliers_removed/"


SF_data_dir = "/Users/hn/Documents/01_research_data/NASA/shapefiles/10_intersect_East_Irr_2008_2018_2cols/"

# %%
# %%time
f_name = "noOutlier_L8_T1C2L2_inters_2008_2018_EastIrr_2008-01-01_2022-01-01_EVI.csv"
noOutlier_L8 = pd.read_csv(data_dir + f_name)

data_part_fname = "10_intersect_East_Irr_2008_2018_2cols_data_part.csv"
SF_data_IDs = pd.read_csv(SF_data_dir + data_part_fname)

# %%
print (f"{noOutlier_L8.shape=}")
noOutlier_L8.head(2)

# %%
SF_data_IDs.head(2)

# %%
SF_data_IDs.sort_values(by=["ID"], inplace=True)
SF_data_IDs.reset_index(drop=True, inplace=True)

# %%
print (f"{len(SF_data_IDs.ID.unique())=}")
print (f"{SF_data_IDs.shape=}")

# %%
SF_data_IDs[SF_data_IDs.acreage<10].shape

# %%
SF_data_IDs[SF_data_IDs.acreage<10].shape
