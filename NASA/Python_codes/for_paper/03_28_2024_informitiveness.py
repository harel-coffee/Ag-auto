# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

# %%
import pandas as pd

# %%
dir_ = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"

# %%
eval_ = pd.read_csv('/Users/hn/Documents/01_research_data/NASA/parameters/evaluation_set.csv')
GT = pd.read_csv(dir_ + 'groundTruth_labels_Oct17_2022.csv')

# %%
print (eval_.shape)
eval_.head(2)

# %%
print (GT.shape)
GT.head(2)

# %%
GT = pd.merge(GT, eval_, how="left", on="ID")
GT.head(2)

# %%
count_df = GT.groupby(["CropTyp"])[["ID"]].count()
count_df.reset_index(inplace=True)
count_df.shape

# %%
area_df = GT.groupby(["CropTyp"])[["ExctAcr"]].sum()
area_df.reset_index(inplace=True)
area_df = area_df.round()
area_df["ExctAcr"] = area_df["ExctAcr"].astype(int)
print (area_df.shape)
area_df.head(2)

# %%

# %%
count_area = pd.merge(count_df, area_df, how="left", on="CropTyp")

# %%
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
    print(count_area)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
