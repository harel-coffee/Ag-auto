# ---
# jupyter:
#   jupytext:
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

# %% [markdown]
# # Wed. 29th or Tues 28 of Nov
#
# Mike and I met in my office. Looked at annual plots of inventories.
# Then saw a big dip in annual data that will be missed on cencus data.
# So, here we model annual data.
#
#
# On Dec 5. Mike and I had a follow up and talked about the next step.
#
# - **Hypothesis** Decline in inventory from time $t$ to $t+1$ if ```NPP``` at $t$ was below average.
#
# What about lag tho? He had mentioned earlier maybe people make changes 3 years after a drought.
#
# - Annual State Level
# - Add Washington, Utah, Arizona, Nevada
# - Find some examples that inventory goes down sharply at time $t+1$ and look at NPP at time $t$.
#
# - $y-$variable should be deltas: $y_{t+1} = I_{t+1} - I_t$ where $I$ is for inventory.
# Under this scenario independent variables can be also deltas or $x_t$ corresponds to $y_{t+1}$. In this notebook
# we will go with the latter scenario.
#

# %%
import shutup

shutup.please()

import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

sys.path.append("/Users/hn/Documents/00_GitHub/Rangeland/Python_Codes/")
import rangeland_core as rc

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %% [markdown]
# # Read

# %%
abb_dict = pd.read_pickle(param_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
county_id_name_fips = pd.read_csv(Min_data_base + "county_id_name_fips.csv")
county_id_name_fips.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

county_id_name_fips = county_id_name_fips[
    county_id_name_fips.state.isin(SoI_abb)
].copy()

county_id_name_fips.sort_values(by=["state", "county"], inplace=True)

county_id_name_fips = rc.correct_Mins_FIPS(df=county_id_name_fips, col_="county")
county_id_name_fips.rename(columns={"county": "county_fips"}, inplace=True)


county_id_name_fips.reset_index(drop=True, inplace=True)
print(len(county_id_name_fips.state.unique()))
county_id_name_fips.head(2)

# %%
county_id_name_fips["state_fip"] = county_id_name_fips.county_fips.str.slice(0, 2)

county_id_name_fips = county_id_name_fips.drop(
    columns=["county_name", "county_fips", "fips"]
)
county_id_name_fips.drop_duplicates(inplace=True)
county_id_name_fips.reset_index(drop=True, inplace=True)

county_id_name_fips.head(2)

# %%
state_SoI_fip = county_id_name_fips.state_fip.unique()
len(state_SoI_fip)

# %%
# herb = pd.read_csv(data_dir_base + "Supriya/Nov30_herb/state_herb_ratio.csv")
# herb = rc.correct_state_int_fips_to_str(df=herb, col_="state_fip")
# herb.sort_values(by=["state_fip"], inplace=True)
herb = pd.read_pickle(data_dir_base + "Supriya/Nov30_HerbRatio/state_herb_ratio.sav")
print(herb.keys())
herb = herb["state_herb_ratio"]
herb = herb[herb.state_fip.isin(state_SoI_fip)]
# herb.dropna(how="any", inplace=True)
herb.head(3)

# %%
herb.dropna(how="any", inplace=True)
herb.reset_index(drop=True, inplace=True)
herb.head(3)

# %%
NPP = pd.read_csv(Min_data_base + "statefips_annual_MODIS_NPP.csv")
NPP.rename(columns={"NPP": "modis_npp", "statefips90m": "state_fip"}, inplace=True)
NPP = rc.correct_3digitStateFips_Min(NPP, "state_fip")

NPP = NPP[NPP.state_fip.isin(state_SoI_fip)]
NPP.reset_index(drop=True, inplace=True)

NPP.head(2)

# %%
prod = pd.read_csv(Min_data_base + "statefips_annual_productivity.csv")
prod.rename(columns={"statefips90m": "state_fip"}, inplace=True)
prod = rc.correct_3digitStateFips_Min(prod, "state_fip")

prod = prod[prod.state_fip.isin(state_SoI_fip)]
prod.reset_index(drop=True, inplace=True)

prod.head(2)

# %%
# Rangeland area and Total area:
state_RA = pd.read_pickle(reOrganized_dir + "state_RA_area.sav")
state_RA = state_RA["state_RA_area"]
state_RA = state_RA[state_RA.state_fip.isin(state_SoI_fip)]
state_RA.head(2)

# %%
print(len(NPP.state_fip.unique()))
print(len(state_RA.state_fip.unique()))
print(len(prod.state_fip.unique()))
print(len(herb.state_fip.unique()))

# %%
print([x for x in NPP.state_fip.unique() if not (x in prod.state_fip.unique())])
print([x for x in state_RA.state_fip.unique() if not (x in prod.state_fip.unique())])
print([x for x in herb.state_fip.unique() if not (x in prod.state_fip.unique())])

# %%
print([x for x in prod.state_fip.unique() if not (x in NPP.state_fip.unique())])
print([x for x in prod.state_fip.unique() if not (x in state_RA.state_fip.unique())])
print([x for x in prod.state_fip.unique() if not (x in herb.state_fip.unique())])

# %%
county_id_name_fips[county_id_name_fips.state_fip == "21"].state.unique()

# %%
prod_test = pd.read_csv(Min_data_base + "statefips_annual_productivity.csv")
prod_test.statefips90m = prod_test.statefips90m.astype(str)
prod_test["state_fip"] = prod_test.statefips90m.str.slice(0, 2)
print(sorted(prod_test.state_fip.unique()))

# %%
county_id_name_fips[county_id_name_fips.state == "TN"]

# %%

# %%
state_RA.head(2)

# %%
state_NPP_Ra = pd.merge(NPP, state_RA, on=["state_fip"], how="left")
state_NPP_Ra.head(2)

# %%
state_NPP_Ra = rc.covert_unitNPP_2_total(
    NPP_df=state_NPP_Ra,
    npp_col_="modis_npp",
    area_col_="rangeland_acre",
    new_col_="state_rangeland_npp",
)

### Security check to not make mistake later:
state_NPP_Ra.drop(columns=["modis_npp"], inplace=True)
state_NPP_Ra.head(2)

# %%
herb.head(2)

# %%
state_NPP_Ra_herb = pd.merge(
    state_NPP_Ra, herb[["state_fip", "herb_avg"]], on=["state_fip"], how="left"
)

state_NPP_Ra_herb.head(2)

# %%
print(sorted(state_NPP_Ra_herb.year.unique()))

# %% [markdown]
# # Read inventory deltas

# %%
Shannon_Beef_Cows_fromCATINV_deltas = pd.read_pickle(
    reOrganized_dir + "state_USDA_ShannonCattle.sav"
)
Shannon_CATINV_deltas = Shannon_Beef_Cows_fromCATINV_deltas[
    "shannon_annual_inventory_deltas_tall"
]
Shannon_CATINV_deltas = Shannon_CATINV_deltas[
    Shannon_CATINV_deltas.state_fip.isin(state_SoI_fip)
]
Shannon_CATINV_deltas.inventory_delta = Shannon_CATINV_deltas.inventory_delta.astype(
    np.float32
)
Shannon_CATINV_deltas.head(2)

# %%

# %%
print(f"{Shannon_CATINV_deltas.shape = }")
desired_delta_years = [(str(x) + "-" + str(x - 1)) for x in np.arange(2002, 2022)]

Shannon_CATINV_deltas = Shannon_CATINV_deltas[
    Shannon_CATINV_deltas.year.isin(desired_delta_years)
]

Shannon_CATINV_deltas.sort_values(by=["state", "year"], inplace=True)
Shannon_CATINV_deltas.reset_index(drop=True, inplace=True)

print(f"{Shannon_CATINV_deltas.shape = }")

Shannon_CATINV_deltas.drop(columns=["state"], inplace=True)
Shannon_CATINV_deltas.head(2)

# %%
state_NPP_Ra_herb.head(2)

# %% [markdown]
# ### Change years' labels in state_NPP_Ra_herb
#
# So that we can merge dataframes. But do not forget what you are doing.

# %%
state_NPP_Ra_herb.year = state_NPP_Ra_herb.year.astype(str)

# %%
for idx_ in state_NPP_Ra_herb.index:
    yr = state_NPP_Ra_herb.loc[idx_, "year"]
    state_NPP_Ra_herb.loc[idx_, "year"] = str(int(yr) + 1) + "-" + str(int(yr))

state_NPP_Ra_herb.sort_values(by=["state_fip", "year"], inplace=True)
state_NPP_Ra_herb.reset_index(drop=True, inplace=True)

# %%
state_NPP_Ra_herb.tail(3)

# %%
Shannon_CATINV_deltas.tail(3)

# %%
ccc_ = ["state_fip", "year", "rangeland_acre", "state_rangeland_npp", "herb_avg"]

state_NPP_Ra_herb_InvenDelta = pd.merge(
    state_NPP_Ra_herb[ccc_],
    Shannon_CATINV_deltas,
    on=["state_fip", "year"],
    how="left",
)
state_NPP_Ra_herb_InvenDelta.head(2)

# %% [markdown]
# ### "normalize" NPP
#
# Let $\mu = \text{NPP}_{avg}$ then compute $(\text{NPP} - \mu) / \mu$

# %%
npp_mean = state_NPP_Ra_herb_InvenDelta.state_rangeland_npp.mean()
state_NPP_Ra_herb_InvenDelta["state_normal_npp"] = (
    state_NPP_Ra_herb_InvenDelta["state_rangeland_npp"] - npp_mean
) / npp_mean

state_NPP_Ra_herb_InvenDelta.head(2)

# %%
state_NPP_Ra_herb_InvenDelta = pd.merge(
    state_NPP_Ra_herb_InvenDelta, county_id_name_fips, on=["state_fip"], how="left"
)
state_NPP_Ra_herb_InvenDelta.head(2)

# %%
new_order = [
    "state",
    "state_fip",
    "year",
    "state_normal_npp",
    "rangeland_acre",
    "herb_avg",
    "inventory_delta",
    "state_rangeland_npp",
]

state_NPP_Ra_herb_InvenDelta = state_NPP_Ra_herb_InvenDelta[new_order]
state_NPP_Ra_herb_InvenDelta.head(2)

# %%
len(state_NPP_Ra_herb_InvenDelta.state.unique())

# %%
tick_legend_FontSize = 8

params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.5,
    "axes.titlesize": tick_legend_FontSize * 1.3,
    "xtick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
    "axes.titlepad": 10,
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharey=False, sharex=False)
(ax1, ax2) = axes
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")

var = "state_normal_npp"
sns.histplot(
    data=state_NPP_Ra_herb_InvenDelta[var], kde=True, bins=200, color="darkblue", ax=ax1
)

# ax1.title("Linear graph")
ax1.title.set_text(var.replace("_", " ") + " density")

A = state_NPP_Ra_herb_InvenDelta[state_NPP_Ra_herb_InvenDelta[var] > 0]
sns.histplot(data=A[var], kde=True, bins=200, color="darkblue", ax=ax2)
ax2.title.set_text("positive " + var.replace("_", " ") + " density")

ax1.set_xlabel("")
ax2.set_xlabel(var.replace("_", " "))
fig.tight_layout()

# %%
state_NPP_Ra_herb_InvenDelta[
    state_NPP_Ra_herb_InvenDelta.state_normal_npp < 0
].state.unique()

# %%
state_NPP_Ra_herb_InvenDelta[
    state_NPP_Ra_herb_InvenDelta.state_normal_npp > 1
].state.unique()

# %%
fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharey=False, sharex=False)
(ax1, ax2) = axes
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")

var = "state_normal_npp"
sns.histplot(
    data=state_NPP_Ra_herb_InvenDelta[var], kde=True, bins=200, color="darkblue", ax=ax1
)

# ax1.title("Linear graph")
ax1.title.set_text(var.replace("_", " ") + " density")

sns.histplot(
    data=1 / (state_NPP_Ra_herb_InvenDelta[var]),
    kde=True,
    bins=200,
    color="darkblue",
    ax=ax2,
)
ax2.title.set_text("1/(" + var.replace("_", " ") + ")" + " density")

ax1.set_xlabel("")
ax2.set_xlabel(var.replace("_", " "))
fig.tight_layout()

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharey=False, sharex=False)
axes.grid(axis="y", which="both")

var = "inventory_delta"
sns.histplot(
    data=state_NPP_Ra_herb_InvenDelta[var],
    kde=True,
    bins=200,
    color="darkblue",
    ax=axes,
)
axes.title.set_text(var.replace("_", " ") + " density")

axes.set_xlabel(var.replace("_", " "))

# %% [markdown]
# ## Model

# %%
state_NPP_Ra_herb_InvenDelta.head(2)

# %%
print(state_NPP_Ra_herb_InvenDelta.year.unique().min())
yr_max = state_NPP_Ra_herb_InvenDelta.year.unique().max()
print(yr_max)

# %%
train_df = state_NPP_Ra_herb_InvenDelta[
    state_NPP_Ra_herb_InvenDelta.year < yr_max
].copy()
test_df = state_NPP_Ra_herb_InvenDelta[
    state_NPP_Ra_herb_InvenDelta.year == yr_max
].copy()

# %%
test_df.year.unique()

# %%
train_df.year.unique().max()

# %%
train_df.head(2)

# %%
indp_vars_ = ["state_normal_npp", "rangeland_acre", "herb_avg"]
train_A = train_df[indp_vars_].values
train_A = np.hstack([train_A, np.ones(len(train_A)).reshape(-1, 1)])
print(train_A.shape)

y_var = "inventory_delta"

train_y = train_df[y_var].values.reshape(-1).astype("float")

# %%
NPP_sol, NPP_RSS, NPP_rank, NPP_singular_vals = np.linalg.lstsq(train_A, train_y)

# %%
NPP_sol

# %%
NPP_coef, Ra_coef, herb_coef, intercept = NPP_sol[0], NPP_sol[1], NPP_sol[2], NPP_sol[3]

# %%
NPP_RSS

# %% [markdown]
# ### Apply to test set

# %%
test_A = test_df[indp_vars_].values
test_A = np.hstack([test_A, np.ones(len(test_A)).reshape(-1, 1)])
y_test = test_df[[y_var]].values.reshape(-1)

# %%
yhat_test = test_A @ NPP_sol

NPP_test_res = y_test - yhat_test
NPP_RSS_test = np.dot(NPP_test_res, NPP_test_res)
NPP_MSE_test = NPP_RSS_test / len(y_test)
NPP_RSE_test = np.sqrt(NPP_MSE_test)

# %%
print("    RSS_test = {0:.0f}.".format(NPP_RSS_test))
print("    MSE_test = {0:.0f}.".format(NPP_MSE_test))
print("    RSE =  {0:.0f}.".format(NPP_RSE_test))

# %% [markdown]
# ### Transform for nomal distribution

# %%
state_NPP_Ra_herb_InvenDelta.head(2)

# %%
state_NPP_Ra_herb_InvenDelta["inverse_state_normal_npp"] = (
    1 / state_NPP_Ra_herb_InvenDelta["state_normal_npp"]
)
state_NPP_Ra_herb_InvenDelta.head(2)

# %%
print(state_NPP_Ra_herb_InvenDelta.inverse_state_normal_npp.min().round(2))
print(state_NPP_Ra_herb_InvenDelta.inverse_state_normal_npp.max().round(2))

# %%
print(state_NPP_Ra_herb_InvenDelta.rangeland_acre.min())
print(state_NPP_Ra_herb_InvenDelta.rangeland_acre.max())

# %%
fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharey=False, sharex=False)
(ax1, ax2) = axes
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")

var = "rangeland_acre"
sns.histplot(
    data=state_NPP_Ra_herb_InvenDelta[var],
    kde=True,
    bins=200,
    color="darkblue",
    ax=ax1,
)

# ax1.title("Linear graph")
ax1.title.set_text(var.replace("_", " ") + " density")

sns.histplot(
    data=np.log10(state_NPP_Ra_herb_InvenDelta[var]),
    kde=True,
    bins=200,
    color="darkblue",
    ax=ax2,
)
ax2.title.set_text("log10(" + var.replace("_", " ") + ")" + " density")

ax1.set_xlabel("")
ax2.set_xlabel(var.replace("_", " "))
fig.tight_layout()

# %%
state_NPP_Ra_herb_InvenDelta["log_rangeland_acre"] = np.log10(
    state_NPP_Ra_herb_InvenDelta["rangeland_acre"]
)
state_NPP_Ra_herb_InvenDelta.head(2)

# %%
print(state_NPP_Ra_herb_InvenDelta.log_rangeland_acre.min().round(2))
print(state_NPP_Ra_herb_InvenDelta.log_rangeland_acre.max().round(2))
print()
print(state_NPP_Ra_herb_InvenDelta.inverse_state_normal_npp.min().round(2))
print(state_NPP_Ra_herb_InvenDelta.inverse_state_normal_npp.max().round(2))

print()
print(state_NPP_Ra_herb_InvenDelta.herb_avg.min().round(2))
print(state_NPP_Ra_herb_InvenDelta.herb_avg.max().round(2))
print()
print(state_NPP_Ra_herb_InvenDelta.inventory_delta.min().round(2))
print(state_NPP_Ra_herb_InvenDelta.inventory_delta.max().round(2))

# %% [markdown]
# # Normalize other columns as well?
#
#  - inventory delta loos like long tail normal. No transformation. Just normalize for scale purposes.

# %%
state_NPP_Ra_herb_InvenDelta.head(2)

# %%
inv_delta_mean = state_NPP_Ra_herb_InvenDelta["inventory_delta"].mean()
state_NPP_Ra_herb_InvenDelta["normal_inventory_delta"] = (
    state_NPP_Ra_herb_InvenDelta["inventory_delta"] - inv_delta_mean
) / inv_delta_mean

state_NPP_Ra_herb_InvenDelta.head(2)

# %%
print(state_NPP_Ra_herb_InvenDelta.inventory_delta.min())
print(state_NPP_Ra_herb_InvenDelta.inventory_delta.max())

print(state_NPP_Ra_herb_InvenDelta.normal_inventory_delta.min())
print(state_NPP_Ra_herb_InvenDelta.normal_inventory_delta.max())

# %%
RA_mean = state_NPP_Ra_herb_InvenDelta["rangeland_acre"].mean()
state_NPP_Ra_herb_InvenDelta["normal_RA"] = (
    state_NPP_Ra_herb_InvenDelta["rangeland_acre"] - RA_mean
) / RA_mean
state_NPP_Ra_herb_InvenDelta.head(2)

# %%
state_NPP_Ra_herb_InvenDelta.drop(columns=["normal_RA"], inplace=True)

# %%
state_NPP_Ra_herb_InvenDelta.head(2)

# %%
print("state_normal_npp")
print(state_NPP_Ra_herb_InvenDelta.state_normal_npp.min().round(2))
print(state_NPP_Ra_herb_InvenDelta.state_normal_npp.max().round(2))
print()
print("inverse_state_normal_npp")
print(state_NPP_Ra_herb_InvenDelta.inverse_state_normal_npp.min().round(2))
print(state_NPP_Ra_herb_InvenDelta.inverse_state_normal_npp.max().round(2))
print()
print("log_rangeland_acre")
print(state_NPP_Ra_herb_InvenDelta.log_rangeland_acre.min().round(2))
print(state_NPP_Ra_herb_InvenDelta.log_rangeland_acre.max().round(2))
print()
print("herb_avg")
print(state_NPP_Ra_herb_InvenDelta.herb_avg.min().round(2))
print(state_NPP_Ra_herb_InvenDelta.herb_avg.max().round(2))
print()
print("inventory_delta")
print(state_NPP_Ra_herb_InvenDelta.inventory_delta.min().round(2))
print(state_NPP_Ra_herb_InvenDelta.inventory_delta.max().round(2))
print()
print("normal_inventory_delta")
print(state_NPP_Ra_herb_InvenDelta.normal_inventory_delta.min().round(2))
print(state_NPP_Ra_herb_InvenDelta.normal_inventory_delta.max().round(2))

# %%
print(NPP.modis_npp.min().round(2))
print(NPP.modis_npp.max().round(2))

# %% [markdown]
# ## New Model
#
# Before we did
#
# ```indp_vars_ = ["normal_npp", "rangeland_acre", "herb_avg"]```
#
# **Now**
#
# ```indp_vars_ = ["normal_npp", "log_rangeland_acre", "herb_avg"]```

# %%
train_df = state_NPP_Ra_herb_InvenDelta[
    state_NPP_Ra_herb_InvenDelta.year < yr_max
].copy()
test_df = state_NPP_Ra_herb_InvenDelta[
    state_NPP_Ra_herb_InvenDelta.year == yr_max
].copy()

# %%
indp_vars_ = ["state_normal_npp", "log_rangeland_acre", "herb_avg"]

train_A = train_df[indp_vars_].values
train_A = np.hstack([train_A, np.ones(len(train_A)).reshape(-1, 1)])
print(train_A.shape)

y_var = "inventory_delta"

train_y = train_df[y_var].values.reshape(-1).astype("float")

# %%
NPP_sol, NPP_RSS, NPP_rank, NPP_singular_vals = np.linalg.lstsq(train_A, train_y)

# %% [markdown]
# ### Apply to test set

# %%
test_A = test_df[indp_vars_].values
test_A = np.hstack([test_A, np.ones(len(test_A)).reshape(-1, 1)])
y_test = test_df[[y_var]].values.reshape(-1)

# %%
yhat_test = test_A @ NPP_sol


# %%
NPP_test_res = y_test - yhat_test
NPP_RSS_test = np.dot(NPP_test_res, NPP_test_res)
NPP_MSE_test = NPP_RSS_test / len(y_test)
NPP_RSE_test = np.sqrt(NPP_MSE_test)

print("    RSS_test = {0:.0f}.".format(NPP_RSS_test))
print("    MSE_test = {0:.0f}.".format(NPP_MSE_test))
print("    RSE =  {0:.0f}.".format(NPP_RSE_test))

# %%
print(state_NPP_Ra_herb_InvenDelta.state_normal_npp.min().round(2))
print(state_NPP_Ra_herb_InvenDelta.state_normal_npp.max().round(2))

# %%
state_NPP_Ra_herb_InvenDelta.head(2)

# %%

# %%
