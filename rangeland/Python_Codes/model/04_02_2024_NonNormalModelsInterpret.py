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
# https://geographicdata.science/book/notebooks/11_regression.html

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
len(all_df.state_fips.unique())

# %%
all_df.head(2)

# %%
all_df = all_df[all_df.state_fips.isin(list(state_fips_SoI.state_fips))].copy()
all_df.reset_index(drop=True, inplace=True)

# %%
print (all_df.shape)
# print (all_df_old.shape)

# %% [markdown]
# # Subset to states of interest
#
# **EW_meridian** exist only for the 29 states of interest. So, in the cell above we are automatically subseting.

# %%
all_df = all_df[all_df.state_fips.isin(list(state_fips_SoI.state_fips))].copy()
all_df.reset_index(drop=True, inplace=True)

# %%
all_df.head(2)

# %%
all_df["log_inventory"] = np.log(all_df["inventory"])
all_df["inventoryDiv1000"] = all_df["inventory"]/1000
all_df["log_total_matt_npp"] = np.log(all_df["total_matt_npp"])
all_df["total_matt_nppDiv1B"] = all_df["total_matt_npp"] / 1000000000


all_df["log_metric_total_matt_npp"] = np.log(all_df["metric_total_matt_npp"])
all_df["metric_total_matt_nppDiv10M"] = all_df["metric_total_matt_npp"] / 10000000
all_df["metric_total_matt_nppDiv500K"] = all_df["metric_total_matt_npp"] / 500000

# %%

# %%

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
453,592,370

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
# non-normal data

# log_inventory, inventoryDiv1000, C(EW_meridian)
# [inv_prices_ndvi_npp.EW_meridian=="E"]
fit = ols('inventory ~ max_ndvi_in_year_modis + beef_price_at_1982 + hay_price_at_1982', 
           data=inv_prices_ndvi_npp).fit()

fit.summary()

# %%
inv_prices_ndvi_npp.columns

# %%
print (len(inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "E"].state_fips.unique()))
print (len(inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "W"].state_fips.unique()))

# %%
state_fips_SoI[state_fips_SoI.state_fips == "47"]

# %%
len(inv_prices_ndvi_npp.state_fips.unique())

# %% [markdown]
# ## NPP Models

# %%
inv_prices_ndvi_npp.columns

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
inv_prices_ndvi_npp[inv_prices_ndvi_npp.year==2011].beef_price_at_1982.unique()

# %%
state_fips_SoI[state_fips_SoI.state_fips=="35"]

# %%
[x for x in inv_prices_ndvi_npp.columns if "npp" in x]

# %%
[x for x in inv_prices_ndvi_npp.columns if "inventory" in x]

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

# %%
fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
axs.grid(axis="y", which="both")

x_col = "metric_total_matt_nppDiv10M"
y_col = "inventoryDiv1000"

x = inv_prices_ndvi_npp.loc[inv_prices_ndvi_npp.EW_meridian == "W", x_col]
y = inv_prices_ndvi_npp.loc[inv_prices_ndvi_npp.EW_meridian == "W", y_col]

axs.scatter(x, y, s = 20, c="dodgerblue", marker="x");

x_line = np.arange(min(x), max(x), 0.01)
y_line = fit.predict(pd.DataFrame(x_line).rename(columns = {0:x_col}))
axs.plot(x_line, y_line, color="r", linewidth=4)

plt.text(min(x), max(y)-.2, 'West of meridian.')

axs.set_xlabel(x_col);
axs.set_ylabel(y_col);

plots_dir = data_dir_base + "00_plots/"
fig_name = plots_dir + y_col + "_" + x_col + "_WestMeridian.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
df_ = inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "W"].copy()
df_[df_["inventoryDiv1000"]>3000].state_fips.unique()

# %%
state_fips_SoI[state_fips_SoI.state_fips=="48"]

# %%

# %%
### Remove Texas for ```inventoryDiv1000``` against ```metric_total_matt_nppDiv10M```


### west of Meridian
# + C(state_dummy_int)
# + C(EW_meridian)
# [inv_prices_ndvi_npp.EW_meridian == "W"]
# + beef_price_at_1982 + hay_price_at_1982
df_ = inv_prices_ndvi_npp.copy()
df_ = df_[df_.EW_meridian == "W"]
df_ = df_[df_["state_fips"]!="48"]

x_col = "metric_total_matt_nppDiv10M"
y_col = "inventoryDiv1000"
df_ = df_[[y_col, x_col]]

fit_noTexas = ols('inventoryDiv1000 ~ metric_total_matt_nppDiv10M', data = df_).fit() 
print (f"{fit_noTexas.pvalues['metric_total_matt_nppDiv10M'] = }")

fit_noTexas.summary()

# %%
fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
axs.grid(axis="y", which="both")

x = df_[x_col]
y = df_[y_col]

axs.scatter(x, y, s = 20, c="dodgerblue", marker="x");

x_line = np.arange(min(x), max(x), 0.01)
y_line = fit_noTexas.predict(pd.DataFrame(x_line).rename(columns = {0:x_col}))
axs.plot(x_line, y_line, color="r", linewidth=4)

plt.text(min(x), max(y)-.2, 'West of meridian. No texas')

axs.set_xlabel(x_col);
axs.set_ylabel(y_col);

plots_dir = data_dir_base + "00_plots/"

fig_name = plots_dir + y_col + "_" + x_col + "_WestMeridian_NoTexas.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
axs.grid(axis="y", which="both")


# Let texas be in the scatter plot and plot both lines
x = inv_prices_ndvi_npp.loc[inv_prices_ndvi_npp.EW_meridian == "W", x_col]
y = inv_prices_ndvi_npp.loc[inv_prices_ndvi_npp.EW_meridian == "W", y_col]
axs.scatter(x, y, s = 20, c="dodgerblue", marker="x");

x_line = np.arange(min(x), max(x), 0.01)
y_line = fit.predict(pd.DataFrame(x_line).rename(columns = {0:x_col}))
axs.plot(x_line, y_line, color="r", linewidth=4, label="Texas in regression")


# Line with no Texas in regression
x_line = np.arange(min(x), max(x), 0.01)
y_line = fit_noTexas.predict(pd.DataFrame(x_line).rename(columns = {0:x_col}))
axs.plot(x_line, y_line, color="k", linewidth=4, label="Texas removed")


plt.text(min(x), max(y)-.2, 'West of meridian')

axs.set_xlabel(x_col);
axs.set_ylabel(y_col);
axs.legend(loc="lower right")

plots_dir = data_dir_base + "00_plots/"

fig_name = plots_dir + y_col + "_" + x_col + "_WestMeridian_NoTexasinModel_TexasinScatter.pdf"
fig_name = plots_dir + y_col + "_" + x_col + "_WestMeridian.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
inv_prices_ndvi_npp.columns

# %%

# %%
df = inv_prices_ndvi_npp.copy()
y_col = "metric_total_matt_npp"
df = df[["year", y_col, "state_fips"]]

df = df[df.state_fips == "48"]
df.dropna(subset=["year"], inplace=True)
df.dropna(subset=[y_col], inplace=True)


fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
# axs.grid(axis="y", which="both")
axs.grid(which="both")

axs.plot(df.year, df[y_col], color="dodgerblue", linewidth=4);
axs.set_xticks(np.arange(2001, 2021, 2))
axs.set_xlabel("year");
axs.set_ylabel(y_col);

axs.title.set_text(y_col.replace("_", " ") + " in Texas (kg)")


plots_dir = data_dir_base + "00_plots/"
fig_name = plots_dir + "Texas_" + y_col + "_WestMeridian.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%

# %%
### Remove that outlier in (20, 13)
# for ```log_inventory``` against ```log_metric_total_matt_npp```


### west of Meridian
# + C(state_dummy_int)
# + C(EW_meridian)
# [inv_prices_ndvi_npp.EW_meridian == "W"]
# + beef_price_at_1982 + hay_price_at_1982
df_ = inv_prices_ndvi_npp.copy()
df_ = df_[df_.EW_meridian == "W"]
df_ = df_[["log_inventory", "log_metric_total_matt_npp"]]

m_ = df_["log_metric_total_matt_npp"].min()

print (df_.shape)
df_ = df_[df_["log_metric_total_matt_npp"] != m_]
print (df_.shape)

fit = ols('log_inventory ~ log_metric_total_matt_npp', data = df_).fit() 
print (f"{fit.pvalues['log_metric_total_matt_npp'] = }")

fit.summary()

# %%
fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
axs.grid(axis="y", which="both")

x_col = "log_metric_total_matt_npp"
y_col = "log_inventory"
x = df_[x_col]
y = df_[y_col]

axs.scatter(x, y, s = 20, c="dodgerblue", marker="x");

x_line = np.arange(min(x), max(x), 0.01)
y_line = fit.predict(pd.DataFrame(x_line).rename(columns = {0:x_col}))
axs.plot(x_line, y_line, color="r", linewidth=4)

plt.text(min(x), max(y)-.2, 'West of meridian.')

axs.set_xlabel(x_col);
axs.set_ylabel(y_col);

# plots_dir = data_dir_base + "00_plots/"
# fig_name = plots_dir + "log_log_metric_westMeridian.pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%

# %%

# %%

# %%

# %%
26786792214/755465

# %%
### Washington
inv_prices_ndvi_npp.total_matt_npp = np.log(inv_prices_ndvi_npp.total_matt_npp)

fit = ols('inventory ~ total_matt_npp + beef_price_at_1982 + hay_price_at_1982 + C(EW_meridian)', 
          data = inv_prices_ndvi_npp[inv_prices_ndvi_npp.state_fips=="53"]).fit() 

inv_prices_ndvi_npp.total_matt_npp = np.exp(inv_prices_ndvi_npp.total_matt_npp)
fit.summary()

# %%
print (all_df[all_df.state_fips=="53"].total_matt_npp.min())
print (all_df[all_df.state_fips=="53"].total_matt_npp.max())

# %%
6,451,447,773
15,454,643,636

# %%

# %%
