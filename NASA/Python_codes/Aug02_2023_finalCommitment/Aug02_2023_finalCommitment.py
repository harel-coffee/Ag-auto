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
# This is for Aug 02, 2023 meeting. Final commitment presentation.

# %%
import pandas as pd
import os, sys

# %%
data_base_dir = "/Users/hn/Documents/01_research_data/NASA/"
RegionalStat_dir = data_base_dir + "RegionalStatData/"
meta_dir = data_base_dir + "/parameters/"

# %%
all_preds_overSample = pd.read_csv(RegionalStat_dir + "all_preds_overSample.csv")
all_preds_overSample = all_preds_overSample[all_preds_overSample.ExctAcr > 10]
sorted(list(all_preds_overSample.Irrigtn.unique()))

# %%
all_preds_overSample.head(2)

# %%
sorted(list(all_preds_overSample.CropTyp.unique()))

# %%
AnnualPerennialToss = pd.read_csv(meta_dir + "AnnualPerennialTossMay122023.csv")
AnnualPerennialToss.rename(columns={"Crop_Type": "CropTyp"}, inplace=True)
print (f"{AnnualPerennialToss.shape=}")
print (AnnualPerennialToss.potential.unique())

# %%
badCrops = list(AnnualPerennialToss.loc[AnnualPerennialToss.potential=="toss", "CropTyp"])

AnnualPerennialToss = AnnualPerennialToss[AnnualPerennialToss.potential!="toss"]
AnnualPerennialToss.reset_index(drop=True, inplace=True)
print (f"{AnnualPerennialToss.shape=}")
print (AnnualPerennialToss.potential.unique())
AnnualPerennialToss.head(2)

# %%
AnnualPerennialToss.potential.unique()

# %%
AnnualPerennialToss[AnnualPerennialToss.potential=="n"].head(5)

# %%
all_preds_overSample = all_preds_overSample[all_preds_overSample.CropTyp.isin(list(AnnualPerennialToss.CropTyp))]
all_preds_overSample.reset_index(drop=True, inplace=True)

# %%
hay_crops = list(AnnualPerennialToss.loc[AnnualPerennialToss.potential == "yn", "CropTyp"])
all_preds_overSample_noHay = all_preds_overSample[~all_preds_overSample.CropTyp.isin(hay_crops)].copy()
all_preds_overSample_noHay.reset_index(drop=True, inplace=True)

# %%

# %% [markdown]
# ### Order Crops by count of double-crop (including hay)

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ###  Order Crops by acreage of double-crop (including hay)

# %%

# %%

# %%

# %%

# %% [markdown]
# ###  Order Crops by count of double-crop (excluding hay)

# %%

# %%

# %%

# %%

# %% [markdown]
# ###  Order Crops by acreage of double-crop (excluding hay)

# %%

# %%

# %%

# %%

# %% [markdown]
# ###  Order counties by count of double-crop (including hay)

# %%

# %%

# %%

# %%

# %% [markdown]
# ###  Order counties by acreage of double-crop (including hay)

# %%

# %%

# %%

# %%

# %% [markdown]
# ###  Order counties by count of double-crop (excluding hay)

# %%

# %%

# %%

# %%

# %% [markdown]
# ###  Order counties by acreage of double-crop (excluding hay)

# %%

# %%

# %%

# %%

# %%
