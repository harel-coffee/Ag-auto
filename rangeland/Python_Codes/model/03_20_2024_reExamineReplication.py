# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

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

# %%
from statsmodels.formula.api import ols

# fit = ols('inventory ~ C(state_dummy_int) + max_ndvi_in_year_modis + beef_price_at_1982 + hay_price_at_1982',
#           data=all_df_normalized_needed).fit()

# fit.summary()

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

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_fips = abb_dict["state_fips"]
state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
print(len(state_fips_SoI))
state_fips_SoI.head(2)

# %% [markdown]
# ### Read beef and hay prices
#
# #### non-normalized

# %%
# state_USDA_ShannonCattle = pd.read_pickle(reOrganized_dir + "state_USDA_ShannonCattle.sav")
# list(state_USDA_ShannonCattle.keys())

# shannon_invt_tall = state_USDA_ShannonCattle["shannon_invt_tall"]

# ## pick up states of interest
# shannon_invt_tall = shannon_invt_tall[shannon_invt_tall.state.isin(SoI_abb)]
# shannon_invt_tall.reset_index(drop=True, inplace=True)
# shannon_invt_tall.inventory = shannon_invt_tall.inventory.astype(np.float32)
# print(f"{shannon_invt_tall.shape = }")
# shannon_invt_tall.head(2)

# beef_price = state_USDA_ShannonCattle["beef_price_at_1982"]
# chicken_price = state_USDA_ShannonCattle["chicken_price_at_1982"]
# hay_price  = state_USDA_ShannonCattle["HayPrice_Q1_at_1982"]

# beef_chicken_hay_price_at_1982 = state_USDA_ShannonCattle["beef_chicken_hay_price_at_1982"]
# print (beef_chicken_hay_price_at_1982.shape)
# beef_chicken_hay_price_at_1982.head(2)


# # Pick the years that they intersect

# min_yr = max(beef_chicken_hay_price_at_1982.year.min(), shannon_invt_tall.year.min())
# max_yr = min(beef_chicken_hay_price_at_1982.year.max(), shannon_invt_tall.year.max())
# print(f"{min_yr = }")
# print(f"{max_yr = }")

# shannon_invt_tall = shannon_invt_tall[(shannon_invt_tall.year >= min_yr) & (shannon_invt_tall.year <= max_yr)]

# beef_chicken_hay_price_at_1982 = beef_chicken_hay_price_at_1982[
#     (beef_chicken_hay_price_at_1982.year >= min_yr) & (beef_chicken_hay_price_at_1982.year <= max_yr)]

# invt_BH_cost = pd.merge(shannon_invt_tall, beef_chicken_hay_price_at_1982,
#                         on=["year", "state_fips"], how="left")
# print(f"{len(invt_BH_cost.state_fips.unique())=}")
# print(f"{invt_BH_cost.shape=}")
# invt_BH_cost.head(3)

# %%
filename_old = (
    reOrganized_dir + "old/state_data_and_deltas_and_normalDelta_OuterJoined.sav"
)
all_data_dict_old = pd.read_pickle(filename_old)

# %%
filename = reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
all_data_dict = pd.read_pickle(filename)
print(all_data_dict["Date"])
list(all_data_dict.keys())

# %%
# all_df_old = all_data_dict_old["all_df_outerjoined"]
# all_df_normal_old = all_data_dict_old["all_df_outerjoined_normalized"]

# %%
all_df = all_data_dict["all_df_outerjoined"]
all_df_normal = all_data_dict["all_df_outerjoined_normalized"]
print([x for x in list(all_df.columns) if "npp" in x])

# %%
all_df_normal.dropna(how="all", inplace=True)
all_df_normal.shape

# %%
print([x for x in list(all_df_normal.columns) if "npp" in x])
print([x for x in list(all_df.columns) if "npp" in x])

# %%
len(all_df.state_fips.unique())

# %%
all_df_normal.dropna(how="all", inplace=False)

# %%
# cc = [x for x in list(all_df_old.columns) if not(x in list(all_df.columns))]
# all_df_old.drop(columns=cc, inplace=True)

# cc = [x for x in list(all_df.columns) if not(x in list(all_df_old.columns))]
# all_df.drop(columns=cc, inplace=True)

# %%
print(all_df.shape)
# print (all_df_old.shape)

# %%
all_df.head(2)

# %%
print([x for x in list(all_df.columns) if "npp" in x])

# %%
all_df = all_df[all_df.state_fips.isin(list(state_fips_SoI.state_fips))].copy()
all_df_normal = all_df_normal[
    all_df_normal.state_fips.isin(list(state_fips_SoI.state_fips))
].copy()

all_df.reset_index(drop=True, inplace=True)
all_df_normal.reset_index(drop=True, inplace=True)


# all_df_old = all_df_old[all_df_old.state_fips.isin(list(state_fips_SoI.state_fips))].copy()
# all_df_normal_old = all_df_normal_old[all_df_normal_old.state_fips.isin(list(state_fips_SoI.state_fips))].copy()

# all_df_old.reset_index(drop=True, inplace=True)
# all_df_normal_old.reset_index(drop=True, inplace=True)

# %%
print(all_df.shape)
# print (all_df_old.shape)

# %%
all_df.head(2)

# %%
# all_df.fillna(0, inplace=True)
# all_df_old.fillna(0, inplace=True)


cc = ["max_ndvi_in_year_modis", "beef_price_at_1982", "hay_price_at_1982"]
# all_df.equals(all_df_old)

# %%
print([x for x in list(all_df_normal.columns) if "npp" in x])

# %%
# cc = [x for x in list(all_df_normal_old.columns) if not(x in list(all_df_normal.columns))]
# all_df_normal_old.drop(columns=cc, inplace=True)

# cc = [x for x in list(all_df_normal.columns) if not(x in list(all_df_normal_old.columns))]
# all_df_normal.drop(columns=cc, inplace=True)

# %%
print([x for x in list(all_df_normal.columns) if "npp" in x])

# %%
# all_df_normal.fillna(0, inplace=True)
# all_df_normal_old.fillna(0, inplace=True)


cc = ["max_ndvi_in_year_modis", "beef_price_at_1982", "hay_price_at_1982"]
# all_df_normal_old.equals(all_df_normal)

# %%
# not_equal_cols = []
# for a_col in all_df_normal_old.columns:
#     if ((all_df_normal_old[a_col] == all_df_normal[a_col]).sum() != len(all_df_normal_old)):
#         not_eq_indx = np.where(all_df_normal_old[a_col] != all_df_normal[a_col])[0]
#         if (all_df_normal_old.loc[not_eq_indx, a_col] -
#             all_df_normal.loc[not_eq_indx, a_col]).sum() > 0.1:
#             not_equal_cols = not_equal_cols + [a_col]

# %%
print([x for x in list(all_df_normal.columns) if "npp" in x])
print([x for x in list(all_df.columns) if "npp" in x])

# %%
# cc = ["year", "state_fips"] + not_equal_cols
# check_old = all_df_normal_old[cc].copy()
# check = all_df_normal[cc].copy()

# %%
# check_old.head(2)

# %%
# check.head(2)

# %%
# a_col = "herb_area_acr"
# not_eq_indx = np.where(check_old[a_col] != check[a_col])[0]
# check_old.loc[not_eq_indx, a_col]

# %%
# check.loc[not_eq_indx, a_col]

# %%

# %%

# %% [markdown]
# ### Convert inventory to log

# %%
all_df.dropna(how="all", inplace=True)
all_df.reset_index(drop=True, inplace=True)

all_df_normal.dropna(how="all", inplace=True)
all_df_normal.reset_index(drop=True, inplace=True)

# %%

# %%
all_df["inventory"] = np.log(all_df["inventory"])
all_df_normal["inventory"] = np.log(all_df_normal["inventory"])

# %%
print([x for x in list(all_df_normal.columns) if "npp" in x])

# %% [markdown]
# # Subset to states of interest
#
# **EW_meridian** exist only for the 29 states of interest. So, in the cell above we are automatically subseting.

# %%
all_df = all_df[all_df.state_fips.isin(list(state_fips_SoI.state_fips))].copy()
all_df_normal = all_df_normal[
    all_df_normal.state_fips.isin(list(state_fips_SoI.state_fips))
].copy()

all_df.reset_index(drop=True, inplace=True)
all_df_normal.reset_index(drop=True, inplace=True)

# %%
all_df_normal.head(2)

# %%
[x for x in list(all_df_normal.columns) if "npp" in x]

# %%
print([x for x in list(all_df.columns) if "npp" in x])

# %%
# EW_meridian exist only for the 29 states of interest. So, here we are automatically subseting
needed_cols = [
    "year",
    "state_fips",
    "inventory",
    "total_matt_npp",
    "unit_matt_npp",
    "rangeland_acre",
    "max_ndvi_in_year_modis",
    "beef_price_at_1982",
    "hay_price_at_1982",
    "EW_meridian",
    "state_dummy_int",
]

inv_prices_ndvi_npp = all_df[needed_cols].copy()

# %%
inv_prices_ndvi_npp_normal = all_df_normal[needed_cols].copy()

# %%
inv_prices_ndvi_npp_normal

# %%
inv_prices_ndvi_npp.dropna(how="any", inplace=True)
inv_prices_ndvi_npp.reset_index(drop=True, inplace=True)

inv_prices_ndvi_npp_normal.dropna(how="any", inplace=True)
inv_prices_ndvi_npp_normal.reset_index(drop=True, inplace=True)

# %%
inv_prices_ndvi_npp_normal

# %%
# non-normal data
fit = ols(
    "inventory ~ max_ndvi_in_year_modis + beef_price_at_1982 + hay_price_at_1982",
    data=inv_prices_ndvi_npp,
).fit()

fit.summary()

# %%
# normalized data

fit = ols(
    "inventory ~ max_ndvi_in_year_modis + beef_price_at_1982 + hay_price_at_1982",
    data=inv_prices_ndvi_npp_normal,
).fit()

fit.summary()

# %%

# %%
fit = ols(
    "inventory ~ max_ndvi_in_year_modis + beef_price_at_1982 + hay_price_at_1982 + C(state_dummy_int)",
    data=inv_prices_ndvi_npp,
).fit()

fit.summary()

# %%
fit = ols(
    "inventory ~ max_ndvi_in_year_modis + beef_price_at_1982 + hay_price_at_1982 + C(state_dummy_int)",
    data=inv_prices_ndvi_npp_normal,
).fit()

fit.summary()

# %%
fit = ols(
    "inventory ~ max_ndvi_in_year_modis + beef_price_at_1982 + hay_price_at_1982 + C(EW_meridian)",
    data=inv_prices_ndvi_npp,
).fit()

fit.summary()

# %%
fit = ols(
    "inventory ~ max_ndvi_in_year_modis + beef_price_at_1982 + hay_price_at_1982 + C(EW_meridian)",
    data=inv_prices_ndvi_npp_normal,
).fit()

fit.summary()

# %% [markdown]
# ### West of Meridian

# %%
fit = ols(
    "inventory ~ max_ndvi_in_year_modis + beef_price_at_1982 + hay_price_at_1982 + C(EW_meridian)",
    data=inv_prices_ndvi_npp_normal[inv_prices_ndvi_npp_normal.EW_meridian == "W"],
).fit()

fit.summary()

# %% [markdown]
# ### East of Meridian

# %%
fit = ols(
    "inventory ~ max_ndvi_in_year_modis + beef_price_at_1982 + hay_price_at_1982",
    data=inv_prices_ndvi_npp_normal[inv_prices_ndvi_npp_normal.EW_meridian == "E"],
).fit()

fit.summary()

# %% [markdown]
# # Square terms

# %%
inv_prices_ndvi_npp_normal.head(2)

# %%

# %%
inv_prices_ndvi_npp_normal["max_ndvi_in_year_modis_sq"] = (
    inv_prices_ndvi_npp_normal["max_ndvi_in_year_modis"] ** 2)
inv_prices_ndvi_npp_normal["beef_price_at_1982_sq"] = (inv_prices_ndvi_npp_normal["beef_price_at_1982"] ** 2)
inv_prices_ndvi_npp_normal["hay_price_at_1982_sq"] = (inv_prices_ndvi_npp_normal["hay_price_at_1982"] ** 2)

# %%
### aggregate. Everything in there, no dummy
fit = ols(
    "inventory ~ max_ndvi_in_year_modis + max_ndvi_in_year_modis_sq + \
                       beef_price_at_1982 + beef_price_at_1982_sq + \
                       hay_price_at_1982 + hay_price_at_1982_sq",
    data=inv_prices_ndvi_npp_normal,
).fit()

fit.summary()

# %%
### state dummy
fit = ols(
    "inventory ~ max_ndvi_in_year_modis + max_ndvi_in_year_modis_sq + \
                       beef_price_at_1982 + beef_price_at_1982_sq + \
                       hay_price_at_1982 + hay_price_at_1982_sq + \
                       C(state_dummy_int)",
    data=inv_prices_ndvi_npp_normal,
).fit()

fit.summary()

# %%
### Meridian dummy
fit = ols(
    "inventory ~ max_ndvi_in_year_modis + max_ndvi_in_year_modis_sq + \
                       beef_price_at_1982 + beef_price_at_1982_sq + \
                       hay_price_at_1982 + hay_price_at_1982_sq + \
                       C(EW_meridian)",
    data=inv_prices_ndvi_npp_normal,
).fit()

fit.summary()

# %%
### west of Meridian
fit = ols(
    "inventory ~ max_ndvi_in_year_modis + max_ndvi_in_year_modis_sq + \
                       beef_price_at_1982 + beef_price_at_1982_sq + \
                       hay_price_at_1982 + hay_price_at_1982_sq + \
                       C(EW_meridian)",
    data=inv_prices_ndvi_npp_normal[inv_prices_ndvi_npp_normal.EW_meridian == "W"],
).fit()

fit.summary()

# %%
### east of Meridian
fit = ols(
    "inventory ~ max_ndvi_in_year_modis + max_ndvi_in_year_modis_sq + \
                       beef_price_at_1982 + beef_price_at_1982_sq + \
                       hay_price_at_1982 + hay_price_at_1982_sq + \
                       C(EW_meridian)",
    data=inv_prices_ndvi_npp_normal[inv_prices_ndvi_npp_normal.EW_meridian == "E"],
).fit()

fit.summary()

# %%
print(
    len(
        inv_prices_ndvi_npp_normal[
            inv_prices_ndvi_npp_normal.EW_meridian == "E"
        ].state_fips.unique()
    )
)
print(
    len(
        inv_prices_ndvi_npp_normal[
            inv_prices_ndvi_npp_normal.EW_meridian == "W"
        ].state_fips.unique()
    )
)

# %%
state_fips_SoI[state_fips_SoI.state_fips == "47"]

# %%
len(inv_prices_ndvi_npp_normal.state_fips.unique())

# %%
inv_prices_ndvi_npp_normal.columns

# %%

# %%

# %%
### west of Meridian
fit = ols(
    "inventory ~ total_matt_npp + beef_price_at_1982 + hay_price_at_1982 + C(EW_meridian)",
    data=inv_prices_ndvi_npp_normal[inv_prices_ndvi_npp_normal.EW_meridian == "W"],
).fit()

fit.summary()

# %%
### west of Meridian
fit = ols(
    "inventory ~ total_matt_npp + beef_price_at_1982 + hay_price_at_1982 + C(EW_meridian)",
    data=inv_prices_ndvi_npp_normal[inv_prices_ndvi_npp_normal.EW_meridian == "E"],
).fit()

fit.summary()

# %%
inv_prices_ndvi_npp.total_matt_npp.mean()

# %%
26, 786, 792, 214

# %%

# %%
np.exp(inv_prices_ndvi_npp.inventory.mean())

# %%
755, 465

# %%
26786792214 / 755465

# %%
### west of Meridian
fit = ols(
    "inventory ~ total_matt_npp + beef_price_at_1982 + hay_price_at_1982 + C(EW_meridian)",
    data=inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "W"],
).fit()

fit.summary()

# %%
### west of Meridian
inv_prices_ndvi_npp.total_matt_npp = np.log(inv_prices_ndvi_npp.total_matt_npp)

fit = ols(
    "inventory ~ total_matt_npp + beef_price_at_1982 + hay_price_at_1982 + C(EW_meridian)",
    data=inv_prices_ndvi_npp,
).fit()

inv_prices_ndvi_npp.total_matt_npp = np.exp(inv_prices_ndvi_npp.total_matt_npp)
fit.summary()

# %%
### west of Meridian
inv_prices_ndvi_npp.total_matt_npp = np.log(inv_prices_ndvi_npp.total_matt_npp)

fit = ols(
    "inventory ~ total_matt_npp + beef_price_at_1982 + hay_price_at_1982 + C(EW_meridian)",
    data=inv_prices_ndvi_npp[inv_prices_ndvi_npp.state_fips == "53"],
).fit()

inv_prices_ndvi_npp.total_matt_npp = np.exp(inv_prices_ndvi_npp.total_matt_npp)
fit.summary()

# %%
# ### west of Meridian
# inv_prices_ndvi_npp.total_matt_npp = np.log(inv_prices_ndvi_npp.total_matt_npp)

# fit = ols('inventory ~ total_matt_npp + beef_price_at_1982 + hay_price_at_1982 + C(EW_meridian)',
#           data = inv_prices_ndvi_npp[inv_prices_ndvi_npp.state_fips=="53"]).fit()

# inv_prices_ndvi_npp.total_matt_npp = np.exp(inv_prices_ndvi_npp.total_matt_npp)
# fit.summary()

# %%
print(all_df[all_df.state_fips == "53"].total_matt_npp.min())
print(all_df[all_df.state_fips == "53"].total_matt_npp.max())

# %%
6, 451, 447, 773
15, 454, 643, 636

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
axs.grid(axis="y", which="both")

df = all_df.copy()
df = df[["year", "total_matt_npp", "state_fips"]]

df = df[df.state_fips == "53"]
df.dropna(subset=["year"], inplace=True)
df.dropna(subset=["total_matt_npp"], inplace=True)
axs.plot(df.year, df.total_matt_npp)

plots_dir = data_dir_base + "00_plots/"

# %%
df.year.unique()

# %%
inv_prices_ndvi_npp.columns

# %%
# fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})

# axs.grid(axis="y", which="both")

# axs.scatter(all_df_normal.inventory, total_matt_npp.unit_matt_npp,
#             s = 20, c="r", marker="x");

# axs.set_xlabel("max NDVI in a year (normalized)")
# axs.set_ylabel("unit Matt's NPP in a year (normalized)")

# plots_dir = data_dir_base + "00_plots/"

# %%
