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

import time
from datetime import date

import matplotlib
import matplotlib.pyplot as plt

import sys, os, os.path
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc

# %%
dir_base = "/Users/hn/Documents/01_research_data/NASA/"
in_dir = dir_base + "/VI_TS/05_SG_TS/"
SF_data_dir   = dir_base + "/data_part_of_shapefile/"

plot_dir = dir_base + "/beet_seed_investigation_plots/"
os.makedirs(plot_dir, exist_ok=True)

# %%

# %%
file_names = ["SG_AdamBenton2016_NDVI_JFD.csv", "SG_Walla2015_NDVI_JFD.csv",
              "SG_Grant2017_NDVI_JFD.csv",      "SG_FranklinYakima2018_NDVI_JFD.csv"]

# %%
SF_f_names=["AdamBenton2016.csv", "FranklinYakima2018.csv", "Grant2017.csv", "Walla2015.csv"]

SF_data = pd.DataFrame()
for file in SF_f_names:
    curr_file = pd.read_csv(SF_data_dir + file)
    SF_data = pd.concat([SF_data, curr_file])
    
beet_seed_SF = SF_data[SF_data.CropTyp == "beet seed"].copy()
beet_seed_SF.reset_index(drop=True, inplace=True)

beet_seed_SF.shape

# %%
beet_seed = pd.DataFrame()
for a_file in file_names:
    df = pd.read_csv(in_dir + a_file)
    df = df[df.ID.isin(list(beet_seed_SF.ID.unique()))]
    
    if len(df) > 0:
        beet_seed = pd.concat([beet_seed, df])
beet_seed.reset_index(drop=True, inplace=True)

beet_seed["human_system_start_time"] = pd.to_datetime(beet_seed["human_system_start_time"])
beet_seed = pd.merge(beet_seed, SF_data, on=(["ID"]), how='left')

# %%

# %%
county_year_dict = {"Adams":2016, 
                    "Benton":2016,
                    "Franklin":2018,
                    "Grant": 2017, 
                    "Walla Walla":2015,
                    "Yakima":2018}

beet_seed_properYear = pd.DataFrame()
for a_county in county_year_dict.keys():
    curr_county_DF = beet_seed[beet_seed.county==a_county].copy()
    # print ("------------------------------------------------------------------------------")
    # print (curr_county_DF.shape)
    curr_county_DF = nc.filter_by_lastSurvey(curr_county_DF, county_year_dict[a_county])
    curr_county_DF = curr_county_DF[curr_county_DF.human_system_start_time.dt.year == county_year_dict[a_county]]
    # print (curr_county_DF.shape)
    beet_seed_properYear = pd.concat([beet_seed_properYear, curr_county_DF])

beet_seed_properYear.reset_index(drop=True, inplace=True)
print (beet_seed_properYear.shape)

# %%
tick_legend_FontSize = 10

params = {'legend.fontsize': tick_legend_FontSize, # medium, large
          # 'figure.figsize': (6, 4),
          'axes.labelsize': tick_legend_FontSize*1.2,
          'axes.titlesize': tick_legend_FontSize*1.3,
          'xtick.labelsize': tick_legend_FontSize, #  * 0.75
          'ytick.labelsize': tick_legend_FontSize, #  * 0.75
          'axes.titlepad': 10}

plt.rc('font', family = 'Palatino')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update(params)

# %%
for an_ID in beet_seed_properYear.ID.unique():
    df = beet_seed_properYear[beet_seed_properYear.ID == an_ID]
    fig, axs = plt.subplots(1, 1, figsize=(12, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
    axs.grid(axis='y', which='both')

    axs.plot(df["human_system_start_time"], df["NDVI"],
             c="dodgerblue", linewidth=2, label="SG NDVI");

    axs.set_ylim([-0.3, 1.15])

    axs.set_title(df["CropTyp"].unique()[0] + ", " + \
                  str(df["Acres"].unique()[0]) + " acres, " + \
                  df["ID"].unique()[0]);
    axs.legend(loc="best")
    file_name = plot_dir + df["CropTyp"].unique()[0].replace(" ", "_") + "_" + df["ID"].unique()[0] + ".pdf"
    plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);
    plt.close()

# %%
fig, axs = plt.subplots(1, 1, figsize=(12, 3), sharex=False, # sharey='col', # sharex=True, sharey=True,
                   gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

axs.plot(df["human_system_start_time"], df["NDVI"],
         c="dodgerblue", linewidth=2, label="SG NDVI");

axs.set_ylim([-0.3, 1.15])

axs.set_title(df["CropTyp"].unique()[0] + ", " + \
              str(df["Acres"].unique()[0]) + " acres, " + \
              df["ID"].unique()[0]);
axs.legend(loc="best")

# %%

# %%

# %%
