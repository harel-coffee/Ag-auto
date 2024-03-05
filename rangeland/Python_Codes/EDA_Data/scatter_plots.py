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
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

plots_dir = data_dir_base + "plots/scatterPlots/"
os.makedirs(plots_dir, exist_ok=True)

# %%
abb_dict = pd.read_pickle(param_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
df_OuterJoined = pd.read_pickle(reOrganized_dir + "county_data_OuterJoined.sav")
df_OuterJoined = df_OuterJoined["all_df"]
df_OuterJoined.head(2)

# %% [markdown]
# ## Pallavi Condition

# %%
df_OuterJoined = df_OuterJoined[df_OuterJoined.Pallavi == "Y"]
df_OuterJoined.reset_index(drop=True, inplace=True)

# %%
df_OuterJoined["dangerEncy"] = df_OuterJoined["danger"] + df_OuterJoined["emergency"]

df_OuterJoined["s1_dangerEncy"] = (
    df_OuterJoined["s1_danger"] + df_OuterJoined["s1_emergency"]
)
df_OuterJoined["s2_dangerEncy"] = (
    df_OuterJoined["s2_danger"] + df_OuterJoined["s2_emergency"]
)
df_OuterJoined["s3_dangerEncy"] = (
    df_OuterJoined["s3_danger"] + df_OuterJoined["s3_emergency"]
)
df_OuterJoined["s4_dangerEncy"] = (
    df_OuterJoined["s4_danger"] + df_OuterJoined["s4_emergency"]
)

# %%
df_OuterJoined.drop(
    labels=[
        "EW",
        "Pallavi",
        "normal",
        "alert",
        "danger",
        "emergency",
        "s1_normal",
        "s1_alert",
        "s1_danger",
        "s1_emergency",
        "s2_normal",
        "s2_alert",
        "s2_danger",
        "s2_emergency",
        "s3_normal",
        "s3_alert",
        "s3_danger",
        "s3_emergency",
        "s4_normal",
        "s4_alert",
        "s4_danger",
        "s4_emergency",
        "s1_avhrr_ndvi",
        "s2_avhrr_ndvi",
        "s3_avhrr_ndvi",
        "s4_avhrr_ndvi",
        "s1_gimms_ndvi",
        "s2_gimms_ndvi",
        "s3_gimms_ndvi",
        "s4_gimms_ndvi",
        "state",
    ],
    axis=1,
    inplace=True,
)

# %%
sorted(df_OuterJoined.columns)

# %%
# sns.set()
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

# %%
X = df_OuterJoined[
    [
        "inventory",
        "rangeland_acre",
        "herb_avg",
        "herb_area_acr",
        "county_total_npp",
        "irr_hay_area",
    ]
]

# %%
tick_legend_FontSize = 10

params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.2,
    "axes.titlesize": tick_legend_FontSize * 2,
    "xtick.labelsize": tick_legend_FontSize * 1.2,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1.2,  #  * 0.75
    "axes.titlepad": 10,
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%

# %%
my_scatter = pd.plotting.scatter_matrix(X, alpha=0.2, diagonal=None, figsize=(7.5, 7.5))
counter = 0
for ax in my_scatter.flatten():
    counter += 1
    # ax.xaxis.label.set_rotation(90)
    # ax.yaxis.label.set_rotation(90)
    # ax.yaxis.label.set_ha('right')
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation(0)


# fig_name = plots_dir + "pandasScatter.pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")
fig_name = plots_dir + "pandasScatter.png"
plt.savefig(fname=fig_name, dpi=200, bbox_inches="tight")

# %%
tick_legend_FontSize = 10

params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.4,
    "axes.titlesize": tick_legend_FontSize * 2,
    "xtick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "axes.titlepad": 10,
}
plt.rcParams.update(params)

# diag_kind{‘auto’, ‘hist’, ‘kde’, None}
my_scatter = sns.pairplot(X, size=2, diag_kind="None", plot_kws={"s": 4})

fig_name = plots_dir + "snsScatter.png"
plt.savefig(fname=fig_name, dpi=200, bbox_inches="tight")

# %%
X = df_OuterJoined[
    ["inventory", "rangeland_acre", "herb_area_acr", "county_total_npp", "irr_hay_area"]
]

tick_legend_FontSize = 10
params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.4,
    "axes.titlesize": tick_legend_FontSize * 2,
    "xtick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "axes.titlepad": 10,
}
plt.rcParams.update(params)

my_scatter = sns.pairplot(X, size=2, diag_kind="None", plot_kws={"s": 4})

# fig_name = plots_dir + "snsScatter.pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

fig_name = plots_dir + "snsScatter_herbArea.png"
plt.savefig(fname=fig_name, dpi=200, bbox_inches="tight")

# %%

# %%
X = df_OuterJoined[["inventory", "rangeland_acre", "herb_area_acr", "herb_avg"]]


tick_legend_FontSize = 10
params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.4,
    "axes.titlesize": tick_legend_FontSize * 2,
    "xtick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "axes.titlepad": 10,
}
plt.rcParams.update(params)


my_scatter = sns.pairplot(X, size=2, diag_kind="None", plot_kws={"s": 4})

fig_name = plots_dir + "snsScatter_herbBattle.png"
plt.savefig(fname=fig_name, dpi=200, bbox_inches="tight")

# %%
df_OuterJoined.rename(
    columns={"yr_countyMean_total_precip": "yr_cntymean_precip"}, inplace=True
)

# %%
X = df_OuterJoined[
    ["county_total_npp", "dangerEncy", "yr_cntymean_precip", "annual_avg_Tavg"]
]

tick_legend_FontSize = 10
params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.4,
    "axes.titlesize": tick_legend_FontSize * 2,
    "xtick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "axes.titlepad": 10,
}
plt.rcParams.update(params)


my_scatter = sns.pairplot(X, size=2, diag_kind="None", plot_kws={"s": 4})

fig_name = plots_dir + "snsScatter_heatPrecipTemp.png"
plt.savefig(fname=fig_name, dpi=200, bbox_inches="tight")

# %%
sorted(df_OuterJoined.columns)

# %%
cols_ = [
    "S1_countyMean_avg_Tavg",
    "S2_countyMean_avg_Tavg",
    "S3_countyMean_avg_Tavg",
    "S4_countyMean_avg_Tavg",
]

X = df_OuterJoined[["county_total_npp"] + cols_]

tick_legend_FontSize = 10
params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1,
    "axes.titlesize": tick_legend_FontSize * 2,
    "xtick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "axes.titlepad": 10,
}
plt.rcParams.update(params)


my_scatter = sns.pairplot(X, size=2, diag_kind="None", plot_kws={"s": 4})

fig_name = plots_dir + "snsScatter_SWNPP.png"
plt.savefig(fname=fig_name, dpi=200, bbox_inches="tight")

# %%

# %%

# %%

# %%

# %%
from scipy.stats import wasserstein_distance
import numpy as np

# %%
wasserstein_distance(np.arange(5), np.arange(5))

# %%

# %%

# %%

# %%
