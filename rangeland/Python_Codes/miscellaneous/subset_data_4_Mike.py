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
# # May 2
#
# Mike wants to run stuff with some software.
#
#
# Mail said:
#
# inventories by state by year and annual state rangeland productivity (NDVI I guess) in a csv file

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
state_name_fips = pd.DataFrame({"state_full" : list(abb_dict["full_2_abb"].keys()),
                                "state" : list(abb_dict["full_2_abb"].values())})
state_name_fips = pd.merge(state_name_fips, abb_dict["state_fips"], on=["state"], how="left")
state_name_fips.head(2)

# %%
state_fips_SoI = state_name_fips[state_name_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
state_fips_SoI.head(2)

# %%

# %%
filename = reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
all_data_dict = pd.read_pickle(filename)
print (all_data_dict["Date"])
list(all_data_dict.keys())

# %%
all_df = all_data_dict["all_df_outerjoined"]
all_df.head(2)

# %%
test_inventory_yr = all_df[["year", "unit_matt_npp"]]
test_inventory_yr.dropna(how="any", inplace=True)
print (test_inventory_yr.year.min())
print (test_inventory_yr.year.max())

# %%
productivity = pd.read_csv(Min_data_dir_base + "statefips_annual_productivity.csv")
productivity.head(2)

productivity = pd.read_csv(Min_data_dir_base + "county_annual_productivity.csv")
productivity.head(2)

# %%
MODIS_NPP = pd.read_csv("/Users/hn/Documents/01_research_data/RangeLand/Data_large_notUsedYet/" + \
                        "Min_data/county_annual_MODIS_NPP.csv")
MODIS_NPP.head(2)

# %%
test_inventory_yr = all_df[["year", "inventory"]]
test_inventory_yr.dropna(how="any", inplace=True)
print (test_inventory_yr.year.min())
print (test_inventory_yr.year.max())

# %%
test_inventory_yr = all_df[["year", "max_ndvi_in_year_modis"]]
test_inventory_yr.dropna(how="any", inplace=True)
print (test_inventory_yr.year.min())
print (test_inventory_yr.year.max())

# %%
dummy_cols = [x for x in all_df.columns if "dumm" in x]
all_df.drop(columns = dummy_cols, inplace=True)

# %%
keep_cols = ['year', 'inventory', 'state_fips',
             'unit_matt_npp', 'total_matt_npp',
             'unit_matt_npp_std', 'total_matt_npp_std',
             'hay_price_at_1982', 'beef_price_at_1982',
             'rangeland_acre', 'max_ndvi_in_year_modis',
             'EW_meridian', 'herb_avg', 'herb_std']

print (all_df.year.min())
print (all_df.year.max())

all_df = all_df[keep_cols]
# all_df.dropna(subset = ["total_matt_npp"], inplace=True)

print (all_df.year.min())
print (all_df.year.max())

all_df = all_df[all_df.state_fips.isin(list(state_fips_SoI.state_fips))]
all_df.reset_index(drop=True, inplace=True)
all_df.head(2)

# %%
print (all_df.year.min())
print (all_df.year.max())

# %%
test_inventory_yr = all_df[["year", "inventory"]]
test_inventory_yr.dropna(how="any", inplace=True)
print (test_inventory_yr.year.min())
print (test_inventory_yr.year.max())

# %%

# %%
all_df = rc.convert_lb_2_kg(df=all_df, 
                            matt_total_npp_col="total_matt_npp", 
                            new_col_name="metric_total_matt_npp")

# %%
all_df = pd.merge(all_df, state_name_fips, on=["state_fips"], how="left")
all_df.head(2)

# %%
all_df[["year", "unit_matt_npp"]].dropna(how="any", inplace=False).year.max()

# %%
all_df[["year", "max_ndvi_in_year_modis"]].dropna(how="any", inplace=False).head(2)

# %%
15979574020

# %%
all_df.tail(2)

# %%

# %%
# converting to CSV file
all_df.to_csv(reOrganized_dir + "NPP_NDVI_Invent_Mike_2May2024.csv")

# %%
reOrganized_dir

# %%
all_df.year.max()

# %%
all_df.head(2)

# %%
Mike_Dell_df = all_df[["year", "inventory", "rangeland_acre", "unit_matt_npp", "state"]].copy()
Mike_Dell_df.dropna(subset=["inventory", "unit_matt_npp"], inplace=True)
Mike_Dell_df.reset_index(drop=True, inplace=True)

Mike_Dell_df["inventory_div_RA"] = Mike_Dell_df['inventory'] / Mike_Dell_df['rangeland_acre']
Mike_Dell_df.head(2)

# %%
Mike_Dell_df_min = Mike_Dell_df.groupby(["state"])["unit_matt_npp"].min().reset_index().round(4)
Mike_Dell_df_min.rename(columns={"unit_matt_npp": "min_unit_matt_npp"}, inplace=True)

Mike_Dell_df_mean = Mike_Dell_df.groupby(["state"])["unit_matt_npp"].mean().reset_index().round(4)
Mike_Dell_df_mean.rename(columns={"unit_matt_npp": "mean_unit_matt_npp"}, inplace=True)

Mike_Dell_df_max = Mike_Dell_df.groupby(["state"])["unit_matt_npp"].max().reset_index().round(4)
Mike_Dell_df_max.rename(columns={"unit_matt_npp": "max_unit_matt_npp"}, inplace=True)

Mike_Dell_df_range = pd.merge(Mike_Dell_df_min, Mike_Dell_df_mean, on=["state"], how="left")
Mike_Dell_df_range = pd.merge(Mike_Dell_df_range, Mike_Dell_df_max, on=["state"], how="left")
Mike_Dell_df_range.head(2)

# %%
Mike_Dell_df_min = Mike_Dell_df.groupby(["state"])["inventory_div_RA"].min().reset_index().round(4)
Mike_Dell_df_min.rename(columns={"inventory_div_RA": "min_inventory_div_RA"}, inplace=True)
Mike_Dell_df_min

Mike_Dell_df_mean = Mike_Dell_df.groupby(["state"])["inventory_div_RA"].mean().reset_index().round(4)
Mike_Dell_df_mean.rename(columns={"inventory_div_RA": "mean_inventory_div_RA"}, inplace=True)
Mike_Dell_df_mean

Mike_Dell_df_max = Mike_Dell_df.groupby(["state"])["inventory_div_RA"].max().reset_index().round(4)
Mike_Dell_df_max.rename(columns={"inventory_div_RA": "max_inventory_div_RA"}, inplace=True)

Mike_Dell_df_range = pd.merge(Mike_Dell_df_min, Mike_Dell_df_mean, on=["state"], how="left")
Mike_Dell_df_range = pd.merge(Mike_Dell_df_range, Mike_Dell_df_max, on=["state"], how="left")
Mike_Dell_df_range.head(2)

# %%
tick_legend_FontSize = 10

params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.2,
    "axes.titlesize": tick_legend_FontSize * 1.3,
    "xtick.labelsize": tick_legend_FontSize,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize,  #  * 0.75
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

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, gridspec_kw={"hspace": 0.35, "wspace": 0.05})
axs.grid(axis="y", which="both")

X_axis = np.arange(len(df.county))

bar_width_ = 1
step_size_ = 5 * bar_width_
X_axis = np.array(range(0, step_size_ * len(df.county), step_size_))

axs.bar(X_axis - bar_width_ * 2, df["double-cropped"], color=color_dict["double-cropped"],
        width=bar_width_, label="double-cropped",)

axs.bar(X_axis - bar_width_, df["single-cropped"], color=color_dict["single-cropped"],
        width=bar_width_, label="single-cropped")

axs.tick_params(axis="x", labelrotation=90)
axs.set_xticks(X_axis, df.county)

axs.set_ylabel("acreage")
axs.legend(loc="best")
axs.xaxis.set_ticks_position("none")

# send the guidelines back
ymin, ymax = axs.get_ylim()
axs.set(ylim=(ymin - 1, ymax + 25), axisbelow=True)


# %%
