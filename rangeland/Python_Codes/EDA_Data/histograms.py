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

plots_dir = data_dir_base + "plots/histograms/"
os.makedirs(plots_dir, exist_ok=True)

# %%
SoI = ["Alabama", "Arizona", "Arkansas", "California", "Colorado",
       "Florida", "Georgia", "Idaho", "Illinois", "Iowa", "Kansas",
       "Kentucky", "Louisiana", "Mississippi",
       "Missouri", "Montana", "Nebraska",
       "Nevada", "New Mexico", "North Dakota",
       "Oklahoma", "Oregon", "South Dakota",
       "Tennessee", "Texas", "Utah",
       "Virginia", "Washington", "Wyoming"]

abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI_abb = []
for x in SoI:
    SoI_abb = SoI_abb + [abb_dict["full_2_abb"][x]]

# %%
df_OuterJoined = pd.read_pickle(reOrganized_dir + "all_data_OuterJoined.sav")
df_OuterJoined = df_OuterJoined["all_df"]
df_OuterJoined.head(2)

# %% [markdown]
# ### Pallavi condition

# %%
df_OuterJoined.reset_index(drop=True, inplace=True)

# %%
df_OuterJoined['s1_dangerEncy'] = df_OuterJoined["s1_danger"] + df_OuterJoined["s1_emergency"]
df_OuterJoined['s2_dangerEncy'] = df_OuterJoined["s2_danger"] + df_OuterJoined["s2_emergency"]
df_OuterJoined['s3_dangerEncy'] = df_OuterJoined["s3_danger"] + df_OuterJoined["s3_emergency"]
df_OuterJoined['s4_dangerEncy'] = df_OuterJoined["s4_danger"] + df_OuterJoined["s4_emergency"]


# %%
df_OuterJoined.drop(labels=['EW', 'Pallavi',
                            'normal', 'alert', 'danger', 'emergency',
                            's1_normal', 's1_alert', 's1_danger', 's1_emergency',
                            's2_normal', 's2_alert', 's2_danger', 's2_emergency',
                            's3_normal', 's3_alert', 's3_danger', 's3_emergency',
                            's4_normal', 's4_alert', 's4_danger', 's4_emergency',
                            's1_avhrr_ndvi', 's2_avhrr_ndvi', 's3_avhrr_ndvi', 's4_avhrr_ndvi',
                            's1_gimms_ndvi', 's2_gimms_ndvi', 's3_gimms_ndvi', 's4_gimms_ndvi',
                            'state'],
                    axis=1, inplace=True)

# %%
sorted(df_OuterJoined.columns)

# %%
tick_legend_FontSize = 10

params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.8,
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


# %% [markdown]
# ## Fixed variables
#
#  - rangeland acre
#  - herb ratio
#  - irrigated hay %

# %%
fig, axes = plt.subplots(1, 1, figsize=(12, 5), sharey=False)
axes.grid(axis="y", which="both")
axes.set_axisbelow(True) # sends the grids underneath the plot

plt.hist(df_OuterJoined['rangeland_fraction']*100, bins=100);
plt.xticks(np.arange(0, 102, 2), rotation ='vertical');

axes.title.set_text("rangeland fraction")
axes.set_xlabel("rangeland fraction")
axes.set_ylabel("count")
fig.tight_layout()

fig_name = plots_dir + "rangeland_fraction_frequency.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%

# %%
fig, axes = plt.subplots(1, 1, figsize=(12, 5), sharey=False)
axes.grid(axis="y", which="both")
axes.set_axisbelow(True) # sends the grids underneath the plot

plt.hist(df_OuterJoined['herb_avg'], bins=100);
plt.xticks(np.arange(0, 102, 2), rotation ='vertical');

axes.title.set_text("herb average (taken over rangeland pixels(?)")
axes.set_xlabel("herb average")
axes.set_ylabel("count")
fig.tight_layout()

fig_name = plots_dir + "herb_average_frequency.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%

# %%
fig, axes = plt.subplots(1, 1, figsize=(12, 5), sharey=False)
axes.grid(axis="y", which="both")
axes.set_axisbelow(True) # sends the grids underneath the plot

plt.hist(df_OuterJoined['irr_hay_as_perc'], bins=100);
plt.xticks(np.arange(0, 102, 2), rotation ='vertical');

axes.title.set_text("irrigated hay as %")
axes.set_xlabel("irr hay as perc")
axes.set_ylabel("count")
fig.tight_layout()

fig_name = plots_dir + "irr_hay_as_perc_frequency.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharey=False)
axes[0].grid(axis="y", which="both")
axes[1].grid(axis="y", which="both")
axes[2].grid(axis="y", which="both")
axes[0].set_axisbelow(True) # sends the grids underneath the plot
axes[1].set_axisbelow(True) 
axes[2].set_axisbelow(True)

axes[0].hist(df_OuterJoined['irr_hay_as_perc'], bins=100);
# axes[0].set(xticks=np.arange(0, 102, 2));
x_ticks = np.arange(0, 102, 2)
axes[0].set_xticks(x_ticks)
axes[0].set_xticklabels(x_ticks, rotation="vertical")

axes[0].set_xlabel("irr hay as perc")
axes[0].set_ylabel("count")
###################################
axes[1].hist(df_OuterJoined['rangeland_fraction']*100, bins=100);
axes[1].set_xticks(x_ticks)
axes[1].set_xticklabels(x_ticks, rotation="vertical")

axes[1].set_xlabel("rangeland fraction")
axes[1].set_ylabel("count")
###################################
axes[2].hist(df_OuterJoined['herb_avg'], bins=100);
axes[2].set_xticks(x_ticks)
axes[2].set_xticklabels(x_ticks, rotation="vertical")

axes[2].set_xlabel("herb avg")
axes[2].set_ylabel("count")
###################################


fig.tight_layout()

fig_name = plots_dir + "irrHayAsPerc_RAfraction_herbAvg.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%

# %%
