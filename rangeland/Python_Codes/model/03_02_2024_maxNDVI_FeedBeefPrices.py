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

# %% [markdown]
# In the group meeting it seemed we want to model changes/deltas as a function of max of NDVI and dummy seasons.
#
# But at the time of coding questions rise up:
#   - dummy season changes will be 16 cases: season 1 to season 2, season 1 to season 3, and so on.
#   - Time-Invariant variables should not be differenced.
#
# Then on March 1st, 2024, Mike and I met on zoom and decided that we want:
#   - state-level modeling of changes/deltas. If not a significant discovery is here, then we will switch to county-level and perhaps high frequency data such as slaughter.
#
#   - We want independent variables to be NPP/NDVI, as well as beef and feed prices. Other variables are secondary.
#   Questions will be there for Matt and Shannon about beef and feed prices.

# %% [markdown]
# <p style="font-family: Arial; font-size:1.4em;color:gold;"> Golden </p>

# %%
import shutup

shutup.please()

import pandas as pd
import numpy as np
import os, os.path, pickle, sys
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

from sklearn import preprocessing
import statistics
import statsmodels.api as sm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

from datetime import datetime, date

current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
param_dir = data_dir_base + "parameters/"

Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
mike_dir = data_dir_base + "Mike/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %% [markdown]
# ### Read beef and hay prices

# %%
# filename = reOrganized_dir + "beef_hay_cost_fromMikeLinkandFile.sav"
# beef_hay_cost = pd.read_pickle(filename)
# beef_hay_cost.keys()

# beef_hay_cost_deltas = beef_hay_cost["beef_hay_costDeltas_MikeLinkandFile"]
# beef_hay_cost = beef_hay_cost["beef_hay_cost_MikeLinkandFile"]

# %%

# %%
state_USDA_ShannonCattle = pd.read_pickle(reOrganized_dir + "state_USDA_ShannonCattle.sav")
state_USDA_ShannonCattle.keys()

# %% [markdown]
# ## Read inventory deltas

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
abb_dict.keys()

# %%
f_ = "state_USDA_ShannonCattle.sav"
state_USDA_ShannonCattle = pd.read_pickle(reOrganized_dir + f_)
list(state_USDA_ShannonCattle.keys())

# %%

# %%
Shannon_CATINV_deltas = state_USDA_ShannonCattle["shannon_invt_deltas_tall"]
Shannon_CATINV_ratios = state_USDA_ShannonCattle["shannon_invt_ratios_tall"]
print(f"{Shannon_CATINV_deltas.shape = }")
print(f"{Shannon_CATINV_ratios.shape = }")

## pick up states of interest
Shannon_CATINV_deltas = Shannon_CATINV_deltas[Shannon_CATINV_deltas.state.isin(SoI_abb)]
Shannon_CATINV_ratios = Shannon_CATINV_ratios[Shannon_CATINV_ratios.state.isin(SoI_abb)]

Shannon_CATINV_deltas.reset_index(drop=True, inplace=True)
Shannon_CATINV_ratios.reset_index(drop=True, inplace=True)

Shannon_CATINV_deltas.inventory_delta = Shannon_CATINV_deltas.inventory_delta.astype(np.float32)
Shannon_CATINV_ratios.inventory_ratio = Shannon_CATINV_ratios.inventory_ratio.astype(np.float32)

print(f"{Shannon_CATINV_deltas.shape = }")
print(f"{Shannon_CATINV_ratios.shape = }")

Shannon_CATINV_deltas.head(2)

# %%
Shannon_CATINV_ratios.head(2)

# %% [markdown]
# ## Pick the years that they intersect

# %%
min_yr = max(beef_hay_cost_deltas.year.min(), Shannon_CATINV_deltas.year.min())
max_yr = min(beef_hay_cost_deltas.year.max(), Shannon_CATINV_deltas.year.max())
print(f"{min_yr = }")
print(f"{max_yr = }")

# %%
Shannon_CATINV_deltas = Shannon_CATINV_deltas[
    (Shannon_CATINV_deltas.year >= min_yr) & (Shannon_CATINV_deltas.year <= max_yr)
]

beef_hay_cost_deltas = beef_hay_cost_deltas[
    (beef_hay_cost_deltas.year >= min_yr) & (beef_hay_cost_deltas.year <= max_yr)
]
#############################################
Shannon_CATINV_ratios = Shannon_CATINV_ratios[
    (Shannon_CATINV_ratios.year >= min_yr) & (Shannon_CATINV_ratios.year <= max_yr)
]

# %%

# %%
inventoryDelta_beefHayCostDeltas = pd.merge(
    Shannon_CATINV_deltas, beef_hay_cost_deltas, on=["year"], how="left"
)
inventoryRatio_beefHayCostDeltas = pd.merge(
    Shannon_CATINV_ratios, beef_hay_cost_deltas, on=["year"], how="left"
)

print(f"{inventoryDelta_beefHayCostDeltas.shape=}")
print(f"{inventoryRatio_beefHayCostDeltas.shape=}")
inventoryDelta_beefHayCostDeltas.head(3)

# %% [markdown]
# ## It seems hay prices and beef prices have had non-overlapping years.
# Lets drop ```NA```s then

# %%
inventoryDelta_beefHayCostDeltas.dropna(how="any", inplace=True)
inventoryDelta_beefHayCostDeltas.reset_index(drop=True, inplace=True)
print(f"{inventoryDelta_beefHayCostDeltas.shape = }")
inventoryDelta_beefHayCostDeltas.head(2)

# %%
inventoryRatio_beefHayCostDeltas.dropna(how="any", inplace=True)
inventoryRatio_beefHayCostDeltas.reset_index(drop=True, inplace=True)
print(f"{inventoryRatio_beefHayCostDeltas.shape = }")
inventoryRatio_beefHayCostDeltas.head(2)

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharey=False)
axes.grid(axis="y", which="both")
axes.set_axisbelow(True)  # sends the grids underneath the plot

plt.hist((inventoryRatio_beefHayCostDeltas["inventory_ratio"]), bins=200)
# plt.xticks(np.arange(0, 102, 2), rotation ='vertical');

axes.title.set_text("allHay_wtAvg_2_$perTon_calendar_yr")
axes.set_xlabel("allHay_wtAvg_2_$perTon_calendar_yr")
axes.set_ylabel("count")
fig.tight_layout()

# fig_name = plots_dir + "xxxxx.pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
filename = reOrganized_dir + "state_delta_and_normalDelta_OuterJoined.sav"
state_data = pd.read_pickle(filename)
list(state_data.keys())

# %%
delta_data = state_data["delta_data"]
delta_data_normal = state_data["delta_data_normal"]
delta_data_normal.head(2)

# %% [markdown]
# <span style='color:red; font-size:30pt'>Fix</span>
#
# This in the notebook ```state_vars_oneFile_outerjoin.ipynb```

# %%
v = delta_data_normal["year"].copy()
w = v - 1
delta_data_normal["year"] = v.astype(str) + "_" + w.astype(str)

delta_data_normal.head(2)

# %%
list(delta_data_normal.columns)

# %% [markdown]
# # Model
#
# We are using national beef and hay prices for all states! if it's the same for all states, what's the diff between including and excluding them.

# %%
inventoryRatio_beefHayCostDeltas.head(3)

# %%
delta_data_normal.head(2)

# %%
all_data = pd.merge(
    inventoryRatio_beefHayCostDeltas,
    delta_data_normal[["year", "state_fips", "state_total_npp"]],
    on=["year", "state_fips"],
    how="left",
)

all_data.dropna(how="any", inplace=True)
all_data.reset_index(drop=True, inplace=True)
all_data.head(2)

# %%

# %%
list(inventoryRatio_beefHayCostDeltas.columns)

# %%
indp_vars = [
    "state_total_npp",
    "allHay_wtAvg_2_$perTon_calendar_yr_normal",
    "steers_medium_large_600_650lbs_$_cwt_yrAvg_normal",
]
y_var = "inventory_ratio"

#################################################################
X = all_data[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = np.log(all_data[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%
indp_vars = ["state_total_npp"]
y_var = "inventory_ratio"

#################################################################
X = all_data[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = np.log(all_data[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%
