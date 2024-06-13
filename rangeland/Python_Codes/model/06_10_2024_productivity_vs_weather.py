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
# In the meeting we haad on June 10, 2024 there was a request to model NPP/NDVI as a function of weather variables.

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

from sklearn.metrics import r2_score

current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)

# %%
from statsmodels.formula.api import ols

# %%
font = {'size' : 14}
matplotlib.rc('font', **font)

tick_legend_FontSize = 12

params = {"legend.fontsize": tick_legend_FontSize * 1.5,  # medium, large
          # 'figure.figsize': (6, 4),
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.2,
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
plots_dir =  data_dir_base + "00_plots/NDVI_NPP_Vs_Weather_Jun102024/"
os.makedirs(plots_dir, exist_ok=True)

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_fips = abb_dict["state_fips"]
state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)

# remove Kentucky
state_fips_SoI = state_fips_SoI[state_fips_SoI.state_full != "Kentucky"]
print(len(state_fips_SoI))

state_fips_SoI.head(2)

# %%
filename = reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
all_data_dict = pd.read_pickle(filename)
print(all_data_dict["Date"])
list(all_data_dict.keys())

# %%
all_df = all_data_dict["all_df_outerjoined"].copy()
all_df = all_df[all_df.state_fips.isin(list(state_fips_SoI.state_fips))].copy()
all_df.reset_index(drop=True, inplace=True)

# %%
all_df = rc.convert_lb_2_kg(df=all_df, 
                            matt_total_npp_col="total_matt_npp", 
                            new_col_name="metric_total_matt_npp")

all_df = rc.convert_lbperAcr_2_kg_in_sqM(df=all_df, 
                                         matt_unit_npp_col="unit_matt_npp", 
                                         new_col_name="metric_unit_matt_npp")

# %%
all_df.head(2)

# %%
list(all_df.columns)

# %%
keep_cols = ['year', 'state_fips', 'unit_matt_npp', 'total_matt_npp', 'unit_matt_npp_std',
             'total_matt_npp_std', 'annual_statemean_total_precip', 'annual_avg_tavg',
             'rangeland_acre', 'state_area_acre', 'rangeland_fraction', 'herb_avg',
             'herb_std', 'herb_area_acr', 'irr_hay_area', 'total_area_irrHayRelated',
             'irr_hay_as_perc', 'max_ndvi_month_modis', 'max_ndvi_in_year_modis',
             'ndvi_std_modis', 'max_ndvi_currYrseason_modis',
             # 's1_bucket_modis', 's2_bucket_modis', 's3_bucket_modis', 's4_bucket_modis',
             's1_max_modis_ndvi', 's2_max_modis_ndvi', 's3_max_modis_ndvi', 's4_max_modis_ndvi', 
             'state_dummy_int', 'metric_total_matt_npp', 'metric_unit_matt_npp',
             
             's1_statemean_total_precip', 's2_statemean_total_precip', 
             's3_statemean_total_precip', 's4_statemean_total_precip',
             
             's1_statemean_avg_tavg', 's2_statemean_avg_tavg',
             's3_statemean_avg_tavg', 's4_statemean_avg_tavg' ]

all_df = all_df[keep_cols]

print (all_df.shape)
all_df.dropna(subset=["max_ndvi_in_year_modis"], inplace=True)
all_df.reset_index(drop=True, inplace=True)
print (all_df.shape)

all_df.head(2)

# %%

# %%
# just to sort
keep_cols = ['year', 'state_fips', "state", "state_full", 
             'unit_matt_npp', 'total_matt_npp', 'unit_matt_npp_std',
             'total_matt_npp_std', 'annual_statemean_total_precip', 'annual_avg_tavg',
             'rangeland_acre', 'state_area_acre', 'rangeland_fraction', 'herb_avg',
             'herb_std', 'herb_area_acr', 'irr_hay_area', 'total_area_irrHayRelated',
             'irr_hay_as_perc', 'max_ndvi_month_modis', 'max_ndvi_in_year_modis',
             'ndvi_std_modis', 'max_ndvi_currYrseason_modis',
             's1_max_modis_ndvi', 's2_max_modis_ndvi', 's3_max_modis_ndvi', 's4_max_modis_ndvi', 
             'state_dummy_int', 'metric_total_matt_npp', 'metric_unit_matt_npp', "EW_meridian",

             's1_statemean_total_precip', 's2_statemean_total_precip', 
             's3_statemean_total_precip', 's4_statemean_total_precip',
             
             's1_statemean_avg_tavg', 's2_statemean_avg_tavg',
             's3_statemean_avg_tavg', 's4_statemean_avg_tavg']

all_df = pd.merge(all_df, state_fips_SoI, on="state_fips", how="left")

all_df = all_df[keep_cols]
all_df.head(2)

# %%
# There is no NPP in 2022 and also in 2021 there are 8 states with no NPP.
all_df.dropna(subset=["unit_matt_npp"], inplace=True)

all_df.reset_index(drop=True, inplace=True)

# %%
all_df.head(3)

# %%
all_df.columns

# %%
data_dir_base

# %%

# %%
tick_legend_FontSize = 10
params = {"legend.fontsize": tick_legend_FontSize,  # medium, large
          # 'figure.figsize': (6, 4),
          "axes.labelsize": tick_legend_FontSize * 1,
          "axes.titlesize": tick_legend_FontSize * 2,
          "xtick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
          "ytick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
          "axes.titlepad": 10}
plt.rcParams.update(params)

cols_ = ["metric_unit_matt_npp", "annual_statemean_total_precip",
         "annual_avg_tavg", "max_ndvi_in_year_modis"]

X = all_df[cols_]

my_scatter = sns.pairplot(X, size=2, diag_kind="None", plot_kws={"s": 4})

# %%
from mpl_toolkits import mplot3d
 
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection ='3d') # syntax for 3-D projection
 
# defining all 3 axis
x = all_df["annual_avg_tavg"]
y = all_df["annual_statemean_total_precip"]
z = all_df["metric_unit_matt_npp"]

# ax.view_init(20, 5)
# plotting
ax.scatter(x, y, z, c = z) # ax.plot3D(x, y, z, 'green')
ax.set_title('Metric NPP v. annual precip and temp')
ax.set_xlabel('annual_avg_tavg', fontsize=12)
ax.set_ylabel('annual_statemean_total_precip', fontsize=12)
ax.set_zlabel('metric_unit_matt_npp', fontsize=12)

fig_name = plots_dir + "unitMetricNPP_weather_3D.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

plt.show()

# %%
from mpl_toolkits import mplot3d
 
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection ='3d') # syntax for 3-D projection
 
# defining all 3 axis
x = all_df["annual_avg_tavg"]
y = all_df["annual_statemean_total_precip"]
z = all_df["max_ndvi_in_year_modis"]

# ax.view_init(20, 5)
# plotting
ax.scatter(x, y, z, c = z) # ax.plot3D(x, y, z, 'green')
ax.set_title('max NDVI v. annual precip and temp')
ax.set_xlabel('annual_avg_tavg', fontsize=12)
ax.set_ylabel('annual_statemean_total_precip', fontsize=12)
ax.set_zlabel('max_ndvi_in_year_modis', fontsize=12)

fig_name = plots_dir + "NDVI_weather_3D.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

plt.show()

# %%
all_df.columns

# %%
tick_legend_FontSize = 10
params = {"legend.fontsize": tick_legend_FontSize,  # medium, large
          # 'figure.figsize': (6, 4),
          "axes.labelsize": tick_legend_FontSize * 1,
          "axes.titlesize": tick_legend_FontSize * 2,
          "xtick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
          "ytick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
          "axes.titlepad": 10}
plt.rcParams.update(params)

# %%
cols_ = ["metric_unit_matt_npp", 
         "s1_statemean_total_precip", "s2_statemean_total_precip",
         "s3_statemean_total_precip", "s4_statemean_total_precip"]
X = all_df[cols_]
fig, axs = plt.subplots(2, 2, figsize=(10, 4), sharex=True, sharey=True,
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05})

# This makes the ylabel become common label but messed up the ticks!
# fig.add_subplot(111, frameon=False)
# ax1, ax2, ax3, ax4 = axs[0,1], axs[0,1], axs[1,0], axs[1,1]
(ax1, ax2), (ax3, ax4) = axs;
ax1.grid(axis="y", which="both"); ax2.grid(axis="y", which="both")
ax3.grid(axis="y", which="both"); ax4.grid(axis="y", which="both")

ax1.scatter(X[cols_[1]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[1].split("_")[0]);
ax2.scatter(X[cols_[2]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[2].split("_")[0]);
ax3.scatter(X[cols_[3]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[3].split("_")[0]);
ax4.scatter(X[cols_[4]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[4].split("_")[0]);

ax1.legend(loc="best"); ax2.legend(loc="best")
ax3.legend(loc="best"); ax4.legend(loc="best")

ax3.set_xlabel("seasonal precip");
ax4.set_xlabel("seasonal precip");
ax1.set_ylabel(cols_[0]);
ax3.set_ylabel(cols_[0]);

fig_name = plots_dir + "unitMetricNPP_Sprecip_scatter.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%

# %%
cols_ = ["max_ndvi_in_year_modis", 
         "s1_statemean_total_precip", "s2_statemean_total_precip",
         "s3_statemean_total_precip", "s4_statemean_total_precip"]
X = all_df[cols_]
fig, axs = plt.subplots(2, 2, figsize=(10, 4), sharex=True, sharey=True,
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05})

# This makes the ylabel become common label but messed up the ticks!
# fig.add_subplot(111, frameon=False)
# ax1, ax2, ax3, ax4 = axs[0,1], axs[0,1], axs[1,0], axs[1,1]
(ax1, ax2), (ax3, ax4) = axs;
ax1.grid(axis="y", which="both"); ax2.grid(axis="y", which="both")
ax3.grid(axis="y", which="both"); ax4.grid(axis="y", which="both")

ax1.scatter(X[cols_[1]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[1].split("_")[0]);
ax2.scatter(X[cols_[2]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[2].split("_")[0]);
ax3.scatter(X[cols_[3]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[3].split("_")[0]);
ax4.scatter(X[cols_[4]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[4].split("_")[0]);

ax1.legend(loc="best"); ax2.legend(loc="best")
ax3.legend(loc="best"); ax4.legend(loc="best")

ax3.set_xlabel("seasonal precip");
ax4.set_xlabel("seasonal precip");
ax1.set_ylabel(cols_[0]);
ax3.set_ylabel(cols_[0]);

fig_name = plots_dir + "NDVI_Sprecip_scatter.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%

# %%
cols_ = ["max_ndvi_in_year_modis", 
         "s1_statemean_avg_tavg", "s2_statemean_avg_tavg",
         "s3_statemean_avg_tavg", "s4_statemean_avg_tavg"]
X = all_df[cols_]

fig, axs = plt.subplots(2, 2, figsize=(10, 4), sharex=True, sharey=True,
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05})

# This makes the ylabel become common label but messed up the ticks!
# fig.add_subplot(111, frameon=False)
# ax1, ax2, ax3, ax4 = axs[0,1], axs[0,1], axs[1,0], axs[1,1]
(ax1, ax2), (ax3, ax4) = axs;
ax1.grid(axis="y", which="both"); ax2.grid(axis="y", which="both")
ax3.grid(axis="y", which="both"); ax4.grid(axis="y", which="both")

ax1.scatter(X[cols_[1]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[1].split("_")[0]);
ax2.scatter(X[cols_[2]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[2].split("_")[0]);
ax3.scatter(X[cols_[3]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[3].split("_")[0]);
ax4.scatter(X[cols_[4]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[4].split("_")[0]);

ax1.legend(loc="best"); ax2.legend(loc="best")
ax3.legend(loc="best"); ax4.legend(loc="best")

ax3.set_xlabel("seasonal temp");
ax4.set_xlabel("seasonal temp");
ax1.set_ylabel(cols_[0]);
ax3.set_ylabel(cols_[0]);

fig_name = plots_dir + "NDVI_STemp_scatter.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%

# %%
cols_ = ["metric_unit_matt_npp", 
         "s1_statemean_avg_tavg", "s2_statemean_avg_tavg",
         "s3_statemean_avg_tavg", "s4_statemean_avg_tavg"]
X = all_df[cols_]

fig, axs = plt.subplots(2, 2, figsize=(10, 4), sharex=True, sharey=True,
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05})

(ax1, ax2), (ax3, ax4) = axs;
ax1.grid(axis="y", which="both"); ax2.grid(axis="y", which="both")
ax3.grid(axis="y", which="both"); ax4.grid(axis="y", which="both")

ax1.scatter(X[cols_[1]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[1].split("_")[0]);
ax2.scatter(X[cols_[2]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[2].split("_")[0]);
ax3.scatter(X[cols_[3]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[3].split("_")[0]);
ax4.scatter(X[cols_[4]], X[cols_[0]], s=20, c="dodgerblue", marker="x", label=cols_[4].split("_")[0]);

ax1.legend(loc="best"); ax2.legend(loc="best")
ax3.legend(loc="best"); ax4.legend(loc="best")

ax3.set_xlabel("seasonal temp");
ax4.set_xlabel("seasonal temp");
ax1.set_ylabel(cols_[0]);
ax3.set_ylabel(cols_[0]);

fig_name = plots_dir + "unitMetricNPP_STemp_scatter.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %% [markdown]
# # Model NPP/NDVI = f(weather)

# %%
print (all_df.shape)
all_df.head(2)

# %%
list(all_df.columns)

# %%
"Kentucky" in all_df.state_full.unique()

# %% [markdown]
# # Annual weather

# %%
indp_vars = ['annual_statemean_total_precip', 'annual_avg_tavg',]
y_var = "metric_unit_matt_npp"

#################################################################
df = all_df[[y_var] + indp_vars].copy()
df.dropna(how="any", inplace=True)
X = df[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = df[y_var].astype(float)
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%

# %%
indp_vars = ['annual_statemean_total_precip', 'annual_avg_tavg',]
y_var = "max_ndvi_in_year_modis"

#################################################################
df = all_df[[y_var] + indp_vars].copy()
df.dropna(how="any", inplace=True)
X = df[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = df[y_var].astype(float)
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %% [markdown]
# # Seasonal weather

# %%
indp_vars = ['s1_statemean_total_precip', 's2_statemean_total_precip',
             's3_statemean_total_precip', 's4_statemean_total_precip',
             
             's1_statemean_avg_tavg', 's2_statemean_avg_tavg',
             's3_statemean_avg_tavg', 's4_statemean_avg_tavg']

y_var = "metric_unit_matt_npp"

#################################################################
df = all_df[[y_var] + indp_vars].copy()
df.dropna(how="any", inplace=True)
X = df[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = df[y_var].astype(float)
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%

# %%
indp_vars = ['s1_statemean_total_precip', 's2_statemean_total_precip',
             's3_statemean_total_precip', 's4_statemean_total_precip',
             
             's1_statemean_avg_tavg', 's2_statemean_avg_tavg',
             's3_statemean_avg_tavg', 's4_statemean_avg_tavg']

y_var = "max_ndvi_in_year_modis"

#################################################################
df = all_df[[y_var] + indp_vars].copy()
df.dropna(how="any", inplace=True)
X = df[indp_vars]
for col_ in indp_vars:
    X[col_] = X[col_].astype("float")

X = sm.add_constant(X)
Y = df[y_var].astype(float)
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%
