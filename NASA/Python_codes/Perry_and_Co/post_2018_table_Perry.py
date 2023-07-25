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
# Here we want to pick proper model (DL, NDVI, SG) with proper probability thresholds and write it to the disk and merge it with shapefile and hand it to Perry. 

# %%
import pandas as pd
import os, sys

# %%
data_base_dir = "/Users/hn/Documents/01_research_data/NASA/"
data_dir = data_base_dir + "merged_trend_ML_preds/"
meta_dir = data_base_dir + "/shapefiles/10_intersect_East_Irr_2008_2018_2cols/"

# %%
NDVI_SG_preds = pd.read_csv(data_dir + "NDVI_SG_preds_intersect.csv")

# %%
prob_NDVI = 0.9
colName = "NDVI_SG_DL_p9"
NDVI_SG_preds[colName] = -1
NDVI_SG_preds.loc[NDVI_SG_preds.NDVI_SG_DL_p_single < prob_NDVI, colName] = 2
NDVI_SG_preds.loc[NDVI_SG_preds.NDVI_SG_DL_p_single >= prob_NDVI, colName] = 1
NDVI_SG_preds.drop(['NDVI_SG_DL_p_single'], axis=1, inplace=True)


# %%
NDVI_SG_preds.head(2)

# %%
NDVI_SG_preds = NDVI_SG_preds[["ID", "year", "NDVI_SG_DL_p9"]]

# %%
out_drive = data_base_dir + "Perry_2008_2018/"
output_name = "Perry_2008_2018_preds.csv"
NDVI_SG_preds.to_csv(out_drive + output_name, index = False)

# %%
