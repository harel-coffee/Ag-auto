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

# %% [markdown]
# This is a coopy of ```04_02_2024_NonNormalModelsInterpret.ipynb```. Reason: In that notebook we have plots. I wanted this notebook be shorter.
#
#
# The link Mike sent about [spatial correlation](https://geographicdata.science/book/notebooks/11_regression.html)
#
# https://scholar.google.com/scholar?hl=en&as_sdt=0%2C48&q=Conley+T%2C+Spatial+econometrics.New+Palgrave+Dictionary+of+Economic&btnG=
#
# https://www.nber.org/system/files/working_papers/t0055/t0055.pdf
#
# https://pysal.org/libpysal/generated/libpysal.weights.KNN.html

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

# %% [markdown]
# #### Fit OLS model with Spreg
#
# \begin{equation}
# m1 = spreg.OLS(db[["log_price"]].values, #Dependent variable
#                     db[variable_names].values, # Independent variables
#                name_y="log_price", # Dependent variable name
#                name_x=variable_names # Independent variable name
#                )
# \end{equation}

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
print (len(state_fips_SoI))
state_fips_SoI.head(2)

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

# %%
all_df["log_inventory"] = np.log(all_df["inventory"])
all_df["inventoryDiv1000"] = all_df["inventory"]/1000
all_df["log_total_matt_npp"] = np.log(all_df["total_matt_npp"])
all_df["total_matt_nppDiv1B"] = all_df["total_matt_npp"] / 1000000000

all_df["log_metric_total_matt_npp"] = np.log(all_df["metric_total_matt_npp"])
all_df["metric_total_matt_nppDiv10M"] = all_df["metric_total_matt_npp"] / 10000000
all_df["metric_total_matt_nppDiv500K"] = all_df["metric_total_matt_npp"] / 500000

# %%
print (all_df.metric_total_matt_npp.min())
print (all_df.metric_total_matt_npp.max())
print()
print (all_df.inventory.min())
print (all_df.inventory.max())
print("---------------------------------------------------------------------------")
print (all_df.metric_total_matt_nppDiv10M.min())
print (all_df.metric_total_matt_nppDiv10M.max())
print()
print (all_df.metric_total_matt_nppDiv500K.min())
print (all_df.metric_total_matt_nppDiv500K.max())
print()
print (all_df.inventoryDiv1000.min())
print (all_df.inventoryDiv1000.max())

# %%
all_df.head(2)

# %% [markdown]
# # Sort 
# maybe it'll be useful for weighted LS

# %%
all_df.sort_values(by=['state_fips', 'year'], inplace=True)

# %%

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
len(inv_prices_ndvi_npp.state_fips.unique())

# %%
inv_prices_ndvi_npp.dropna(how="any", inplace=True)
inv_prices_ndvi_npp.reset_index(drop=True, inplace=True)
len(inv_prices_ndvi_npp.state_fips.unique())

# %%
[x for x in sorted(all_df.state_fips.unique()) if not (x in sorted(inv_prices_ndvi_npp.state_fips.unique()))]

# %%
all_df[all_df.state_fips == "21"].unit_matt_npp.unique()

# %%
state_fips = abb_dict["state_fips"]
state_fips[state_fips["state_fips"]=="21"]

# %%
print (len(inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "E"].state_fips.unique()))
print (len(inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "W"].state_fips.unique()))

# %% [markdown]
# ## NPP Models

# %%
print (inv_prices_ndvi_npp.beef_price_at_1982.min())
print (inv_prices_ndvi_npp.beef_price_at_1982.max())
print ()
print (inv_prices_ndvi_npp.log_total_matt_npp.min())
print (inv_prices_ndvi_npp.log_total_matt_npp.max())
print ()
print (inv_prices_ndvi_npp.hay_price_at_1982.min())
print (inv_prices_ndvi_npp.hay_price_at_1982.max())

# %%
inv_prices_ndvi_npp[inv_prices_ndvi_npp.hay_price_at_1982 == inv_prices_ndvi_npp.hay_price_at_1982.max()]

# %%
[x for x in inv_prices_ndvi_npp.columns if "npp" in x]

# %%
[x for x in inv_prices_ndvi_npp.columns if "inventory" in x]

# %%
depen_var_name = "inventoryDiv1000"
indp_vars = ["metric_total_matt_nppDiv10M", "beef_price_at_1982", "hay_price_at_1982"]
m5 = spreg.OLS_Regimes(y = inv_prices_ndvi_npp[depen_var_name].values, # Dependent variable
                       x = inv_prices_ndvi_npp[indp_vars].values, # Independent variables

                       # Variable specifying neighborhood membership
                       regimes = inv_prices_ndvi_npp["EW_meridian"].tolist(),
              
                       # Variables to be allowed to vary (True) or kept
                       # constant (False). Here we set all to False
                       # cols2regi=[False] * len(indp_vars),
                        
                       # Allow the constant term to vary by group/regime
                       constant_regi="many",
                        
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y=depen_var_name, # Dependent variable name
                       name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

m5_results

# West regime
## Extract variables for the west side regime
west_m = [i for i in m5_results.index if "W_" in i]

## Subset results to west side and remove the W_
west = m5_results.loc[west_m, :].rename(lambda i: i.replace("W_", ""))
## Build multi-index column names
west.columns = pd.MultiIndex.from_product([["West Meridian"], west.columns])

# East model
## Extract variables for the eastern regime
east_m = [i for i in m5_results.index if "E_" in i]
east = m5_results.loc[east_m, :].rename(lambda i: i.replace("E_", ""))
## Build multi-index column names
east.columns = pd.MultiIndex.from_product([["East Meridian"], east.columns])
# Concat both models
pd.concat([west, east], axis=1)

# %%
m5.chow.joint

# %% [markdown]
# The next step then is to check whether each of the coefficients in our model differs across regimes. For this, we can pull them out into a table:

# %%
pd.DataFrame(m5.chow.regi, # Chow results by variable
             index = m5.name_x_r, # Name of variables
             columns = ["Statistic", "P-value"])

# %%
### west of Meridian
# + C(state_dummy_int)
# + C(EW_meridian)
# [inv_prices_ndvi_npp.EW_meridian == "W"]
# + beef_price_at_1982 + hay_price_at_1982

fit = ols('inventoryDiv1000 ~ metric_total_matt_nppDiv10M',
          data = inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "W"]).fit() 

print (f"{fit.pvalues['metric_total_matt_nppDiv10M'] = }")

fit.summary()

# %%
short_summ =pd.DataFrame({"Coeff.": fit.params.values,          
                          "Std. Error": fit.bse.values.round(2),
                          "t": fit.tvalues.values,
                          "P-Value": fit.pvalues.values},
                          index = list(fit.params.index))
short_summ

# %%
print (inv_prices_ndvi_npp.shape)
inv_prices_ndvi_npp_west = inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "W"].copy()
print (inv_prices_ndvi_npp_west.shape)
inv_prices_ndvi_npp_west_noTexas = inv_prices_ndvi_npp_west[inv_prices_ndvi_npp_west.state_fips != "48"].copy()
print (inv_prices_ndvi_npp_west_noTexas.shape)

# %%
df_ = inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "W"].copy()
print (df_[df_["inventoryDiv1000"]>3000].state_fips.unique())
del(df_)

# %%
state_fips_SoI[state_fips_SoI.state_fips=="48"]

# %%
inv_prices_ndvi_npp.head(2)

# %%
list(inv_prices_ndvi_npp.columns)

# %%
inv_prices_ndvi_npp["W_meridian_bool"] = True
inv_prices_ndvi_npp.loc[inv_prices_ndvi_npp.EW_meridian=="E", "W_meridian_bool"] = False

inv_prices_ndvi_npp.W_meridian_bool.head(2)

# %%
inv_prices_ndvi_npp.head(2)

# %% [markdown]
# # Weighted LS
#
# - Weight Matrix is adjacency matrix here (queen contiguity)
# - Weight Matrix is nonnegative
# - Diagonals are zero
# - non-nilpotent
#
# We also have a version that the adj. matrix is divided by its largest eigenvalue, so the new normalized matrix's largest eigenvalue is 1. This was used in Tonsor, however, they used the weight matrix in a different way: lagged ```y``` or sth.
#
# Tonsor's equation is 
#
# \begin{equation} 
# \mathbf{y} = \rho \mathbf{Wy}  + \beta \mathbf{X} + \varepsilon
# \end{equation}
#
# What about Distance-based weights?

# %%
filename = reOrganized_dir + "state_adj_dfs.sav"
state_adj_dfs = pd.read_pickle(filename)
list(state_adj_dfs.keys())

# %%
# Adjacency matrices:

adj_df_fips = state_adj_dfs["adj_df_fips"]
adj_df_fips_SoI = state_adj_dfs["adj_df_fips_SoI"]
adj_df_SoI = state_adj_dfs["adj_df_SoI"]

# %%
inv_prices_ndvi_npp.shape

# %%
adj_df_SoI.shape

# %%
fit = ols('inventoryDiv1000 ~ metric_total_matt_nppDiv10M', data = inv_prices_ndvi_npp).fit()
print (len(fit.resid))
print (len(fit.fittedvalues))

# %%
# df = pd.DataFrame({'hours': [1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8],
#                    'score': [48, 78, 72, 70, 66, 92, 93, 75, 75, 80, 95, 97,
#                              90, 96, 99, 99]})

# #######################################################################
# #######################################################################
# #######################################################################

# # define predictor and response variables
# y = df['score']
# X = df['hours']

# # add constant to predictor variables
# X = sm.add_constant(X)

# # fit linear regression model
# print (f"{X.shape = }")
# print (f"{y.shape = }")
# fit = sm.OLS(y, X).fit()

# #######################################################################
# #######################################################################
# #######################################################################
# wt = 1 / ols('fit.resid.abs() ~ fit.fittedvalues', data=df).fit().fittedvalues**2

# # fit weighted least squares regression model
# fit_wls = sm.WLS(y, X, weights=wt).fit()

# # view summary of weighted least squares regression model
# print(fit_wls.summary())

# %%

# %%
# d = {"resid_": fit.resid.abs(), "preds_" : fit.fittedvalues}
# df_2 = pd.DataFrame(d)
# df_2.head(3)

# y = df_2['resid_']
# X = df_2['preds_']

# # add constant to predictor variables
# X = sm.add_constant(X)

# # fit linear regression model
# fit_2 = sm.OLS(y, X).fit()

# #view model summary
# print(fit_2.summary())

# wt_2 = 1 / fit_2.fittedvalues**2
# wt_2

# %%
fit = ols('inventoryDiv1000 ~ metric_total_matt_nppDiv10M',
          data = inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "W"]).fit()
fit.summary()


# %%
block_diag_weights = rc.create_adj_weight_matrix(data_df=inv_prices_ndvi_npp, 
                                                 adj_df=adj_df_fips, 
                                                 fips_var="state_fips")

block_diag_weights.head(2)

# %%
y_var = "inventoryDiv1000"
X_vars = ["metric_total_matt_nppDiv10M"]

y = inv_prices_ndvi_npp[y_var]
X = inv_prices_ndvi_npp[X_vars]
X = sm.add_constant(X)


theta = inv(X.T.values @ block_diag_weights.values @ X.values) @ X.T.values @ block_diag_weights.values @ y
theta

# %%
list(X.columns)

# %%

# %%

# %%

# %%
