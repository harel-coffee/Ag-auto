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
# **April 13. 2024**
#
# \begin{equation}
# \mathbf{y}_{it} = \mathbf{x}^T_{it} \mathbf{\beta} +\varepsilon_{it} + u_{i}
# \end{equation}
#
# The common $u_{i}$ is the usual unit (e.g., country) effect. The correlation across space is implied by the spatial autocorrelation structure
#
# \begin{equation}
# \varepsilon_{it} = \lambda \sum_{j=1}^n W_{ij} \varepsilon_{jt} + v_t
# \end{equation}
#
# ```What is```$v_t$```?```
#
# \begin{equation}
# \mathbf{\varepsilon}_t = (\mathbf{I}_n - \lambda \mathbf{W})^{-1} \mathbf{v}_t,~~~ \mathbf{v}_t =v_t \mathbf{i}
# \end{equation}
#
#
# For when there is spatial autocorrelation in the panel data and when $|\lambda| < 1$ and $\mathbf{I} - \lambda \mathbf{W}$ is non-singular we can write (for $n$ units at time $t$)
#
# \begin{equation}
# \mathbf{y}_t = \mathbf{X}_t \mathbf{\beta} + (\mathbf{I}_n - \lambda \mathbf{W})^{-1} \mathbf{v}_t + \mathbf{u}
# \end{equation}
#
# Assumptions $v_t$ and $u_t$ have mean zero and variances $\sigma^2_v$ and $\sigma_u^2$ and are independent
# across countries and of each other.
#
# The Greene's book: "There is no natural residual based estimator of $\lambda$". It refers to another paper on how to estimate $\lambda$ (Example 11.12). Kelejian and Prucha (1999) have developed a moment-based estimator for $\lambda$ that helps to alleviate the problem (related to the matrix $\mathbf{I}_n - \lambda \mathbf{W}$ and its singularity). Once the estimate of $\lambda$ is in hand, estimation of the spatial autocorrelation model is done by FGLS
#
# The following code is based on *A generalized moments estimator for the autoregressive parameter in a spatial model: 1999* which is cited in Greene's book.

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
from scipy.linalg import inv

current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)

# %%
from pysal.lib import weights
from pysal.model import spreg
from pysal.explore import esda
import geopandas, contextily

from scipy.stats import ttest_ind

# %%
tick_legend_FontSize = 8

params = {"legend.fontsize": tick_legend_FontSize,  # medium, large
          "axes.labelsize": tick_legend_FontSize * 1.5,
          "axes.titlesize": tick_legend_FontSize * 1.3,
          "xtick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
          "ytick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
          "axes.titlepad": 10}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

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
abb_dict["Date"]

# %%
state_fips = abb_dict["state_fips"]
state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
print (len(state_fips_SoI))
state_fips_SoI.head(2)

# %% [markdown]
# ## Adjacency matrix

# %%
filename = reOrganized_dir + "state_adj_dfs.sav"
state_adj_dfs = pd.read_pickle(filename)
list(state_adj_dfs.keys())

# %%

# %%
# Adjacency matrices:

adj_df_fips = state_adj_dfs["adj_df_fips_rowNormalized"]
adj_df_fips_SoI = state_adj_dfs["adj_df_fips_SoI_rowNormalized"]
adj_df_SoI = state_adj_dfs["adj_df_SoI_rowNormalized"]

# %% [markdown]
# ### Read beef and hay prices
#
# #### non-normalized

# %%
filename = reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
all_data_dict = pd.read_pickle(filename)
print (all_data_dict["Date"])
list(all_data_dict.keys())

# %%
all_df = all_data_dict["all_df_outerjoined"]
all_df_normal = all_data_dict["all_df_outerjoined_normalized"]
print ([x for x in list(all_df.columns) if "npp" in x])

# %%
all_df = rc.convert_lb_2_kg(df=all_df, 
                            matt_total_npp_col="total_matt_npp", 
                            new_col_name="metric_total_matt_npp")

# %%
print (len(all_df.state_fips.unique()))
all_df.head(2)

# %%
all_df = all_df[all_df.state_fips.isin(list(state_fips_SoI.state_fips))].copy()
all_df.reset_index(drop=True, inplace=True)

print (all_df.shape)
# print (all_df_old.shape)

# %% [markdown]
# # Subset to states of interest
#
# **EW_meridian** exist only for the 29 states of interest. So, in the cell above we are automatically subseting.

# %%
all_df = all_df[all_df.state_fips.isin(list(state_fips_SoI.state_fips))].copy()
all_df.reset_index(drop=True, inplace=True)

all_df.head(2)

# %%
all_df["log_inventory"] = np.log(all_df["inventory"])
all_df["inventoryDiv1000"] = all_df["inventory"]/1000
all_df["log_total_matt_npp"] = np.log(all_df["total_matt_npp"])
all_df["total_matt_nppDiv1B"] = all_df["total_matt_npp"] / 1000000000

all_df["log_metric_total_matt_npp"] = np.log(all_df["metric_total_matt_npp"])
all_df["metric_total_matt_nppDiv10M"] = all_df["metric_total_matt_npp"] / 10000000
all_df["metric_total_matt_nppDiv500K"] = all_df["metric_total_matt_npp"] / 500000

all_df.head(2)

# %%
all_df.sort_values(by=['state_fips', 'year'], inplace=True)

# %%
# EW_meridian exist only for the 29 states of interest. So, here we are automatically subseting
needed_cols = ["year", "state_fips", 
               "metric_total_matt_npp", "metric_total_matt_nppDiv500K", "metric_total_matt_nppDiv10M",
               "log_metric_total_matt_npp",
               "total_matt_npp", "log_total_matt_npp", "total_matt_nppDiv1B",
               "inventory", "log_inventory", "inventoryDiv1000",
               "unit_matt_npp", "rangeland_acre",
               "max_ndvi_in_year_modis", 
               "beef_price_at_1982", "hay_price_at_1982",
               "EW_meridian", "state_dummy_int"]

inv_prices_ndvi_npp = all_df[needed_cols].copy()

# %%
inv_prices_ndvi_npp.dropna(how="any", inplace=True)
inv_prices_ndvi_npp.reset_index(drop=True, inplace=True)
len(inv_prices_ndvi_npp.state_fips.unique())

# %%
[x for x in sorted(all_df.state_fips.unique()) if not (x in sorted(inv_prices_ndvi_npp.state_fips.unique()))]

# %%
state_fips[state_fips["state_fips"]=="21"]

# %% [markdown]
# # NPP Models

# %%
list(inv_prices_ndvi_npp.columns)

# %%
inv_prices_ndvi_npp.year.unique()

# %% [markdown]
# ### A cross-sectional model

# %%
df_2017 = inv_prices_ndvi_npp[inv_prices_ndvi_npp["year"] == 2017].copy()
df_2017.reset_index(drop=True, inplace=True)

df_2017_west = df_2017[df_2017.EW_meridian == "W"].copy()
df_2017_west.reset_index(drop=True, inplace=True)

# %%
fit = ols('inventory ~ total_matt_npp',
          data = df_2017_west).fit() 

print (f"{fit.pvalues['total_matt_npp'] = }")

fit.summary()

# %%
west_2017_residuals = fit.resid

# %%
adj_df_fips_SoI.head(2)

# %%
west_state_fips = list(state_fips_SoI[state_fips_SoI["EW_meridian"] == "W"]["state_fips"])
west_state_fips[0:4]

# %%
adj_df_fips_SoI_west = adj_df_fips_SoI.copy()

# pick columns that are on the west side
adj_df_fips_SoI_west = adj_df_fips_SoI_west[west_state_fips].copy()
adj_df_fips_SoI_west = adj_df_fips_SoI_west.loc[adj_df_fips_SoI_west.index.isin(west_state_fips)]
# adj_df_fips_SoI_west.reset_index(drop=True, inplace=True)
print (adj_df_fips_SoI_west.shape)
adj_df_fips_SoI_west.head(2)


# %%
# Sort adjacency matrix according to df_2017_west so that things are correct
# seems they are ordered increasingly
df_2017_west.head(2)

# %%
df_2017_west.state_fips

# %%
adj_df_fips_SoI_west = adj_df_fips_SoI_west[list(df_2017_west.state_fips)]
adj_df_fips_SoI_west.sort_index(inplace=True)

print (list(df_2017_west.state_fips))
print (list(adj_df_fips_SoI_west.columns))
print (list(adj_df_fips_SoI_west.index))
adj_df_fips_SoI_west.head(2)

# %%
list(state_adj_dfs.keys())

# %%
state_adj_dfs["source_code"]

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
# # check with Shannon and see if years are shifted (Jan 1 recording problem)

# pd.merge(inv_prices_ndvi_npp[["year", "state_fips", "inventory"]], state_fips_SoI[["state", "state_fips"]],
#          how="left", on="state_fips")

# %%
# Matt NPP goes up to 2021. That's why, inventory also goes up to there
# since we dropped NAs.
A = all_data_dict["all_df_outerjoined"].copy()
A = A[["year", "inventory", "state_fips", "total_matt_npp"]]
A.dropna(how="any", inplace=True)
print (A.year.max())
A.head(2)

# %%

# %%
