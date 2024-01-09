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
rangeland_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"

# %%
grids_25states = pd.read_csv(rangeland_dir_base + "grids_25states.csv")

# %%
grids_25states.shape

# %%
grids_25states

# %%
sorted(list(grids_25states.state.unique()))

# %%
