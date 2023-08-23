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

# %%
dir_ = "/Users/hn/Desktop/for_sid_2cropOnMap/"
all_preds_overSample = pd.read_csv(dir_ + "all_preds_overSample.csv")
all_eastern_centroid = pd.read_csv(dir_ + "all_eastern_centroid.csv")

print (f"{all_preds_overSample.shape=}")
print (f"{all_eastern_centroid.shape=}")

# %%
all_preds_overSample = pd.merge(all_preds_overSample, all_eastern_centroid, how="left", on="ID")

# %%
out_name = dir_ + "all_preds_overSample_withCentroids.csv"
all_preds_overSample.to_csv(out_name, index = False)

# %%
