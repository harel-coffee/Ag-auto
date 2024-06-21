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
# When we met on June 10, 2024 Mike mentioned it would be good to look into how much and what kind of interaction/dynamic exist between inventory and slaughter. Lets see what we can do!
#
# I think here we need Jan 1 inventory. and then see how many were slaughtered thereafter during the same year. dammit.

# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# %%
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"

reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
plot_dir = data_dir_base + "00_plots/slaughter_inventory_exploration/"

os.makedirs(plot_dir, exist_ok=True)

# %%

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_fips = abb_dict["state_fips"]
# state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
# state_fips_SoI.reset_index(drop=True, inplace=True)
# print (len(state_fips_SoI))
print (state_fips.shape)
state_fips.head(2)

# %%
state_fips = abb_dict["state_fips"]
# state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
# state_fips_SoI.reset_index(drop=True, inplace=True)
# print(len(state_fips_SoI))
# state_fips_SoI.head(2)

# %%
filename = reOrganized_dir + "monthly_NDVI_beef_slaughter.sav"

monthly_NDVI_beef_slaughter = pd.read_pickle(filename)
print (monthly_NDVI_beef_slaughter["Date"])
monthly_NDVI_beef_slaughter = monthly_NDVI_beef_slaughter["monthly_NDVI_beef_slaughter"]
monthly_NDVI_beef_slaughter.head(2)

# %%
regions = monthly_NDVI_beef_slaughter["region"].unique()

# %%
monthly_NDVI_beef_slaughter["region"].unique()

# %% [markdown]
# ### Compute annual slaughter

# %%
monthly_slaughter = monthly_NDVI_beef_slaughter[["year", "region", "slaughter_count"]].copy()
monthly_slaughter.dropna(subset=["slaughter_count"], inplace=True)
monthly_slaughter.reset_index(drop=True, inplace=True)
monthly_slaughter.head(2)

# %%
annual_slaughter = monthly_slaughter.groupby(["region", "year"])["slaughter_count"].sum().reset_index()
annual_slaughter.head(2)

# %%
df = annual_slaughter[annual_slaughter["region"] == "region_8"].copy()
print (df.shape)

df = annual_slaughter[annual_slaughter["region"] == "region_9"].copy()
print (df.shape)

# %% [markdown]
# ### Compute regional inventory

# %%
filename = reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
all_data_dict = pd.read_pickle(filename)
print (all_data_dict["Date"])
list(all_data_dict.keys())

# %%
all_data = all_data_dict["all_df_outerjoined"].copy()
inventory_df = all_data[["year", "inventory", "state_fips"]].copy()

inventory_df.dropna(subset=["inventory"], inplace=True)
inventory_df["inventory"] = inventory_df["inventory"].astype(int)
# do NOT subset to states of interest as slaughter data ignores that
# inventory_df = inventory_df[inventory_df["state_fips"].isin(list(state_fips_SoI["state_fips"].unique()))]
inventory_df.reset_index(drop=True, inplace=True)

inventory_df.head(2)

# %%

# %%
inventory_df = pd.merge(inventory_df, state_fips[['state', 'state_fips', 'state_full']], 
                        on=["state_fips"], how="left")

inventory_df.head(2)

# %%
shannon_regions_dict_abbr = {"region_1_region_2" : ['CT', 'ME', 'NH', 'VT', 'MA', 'RI', 'NY', 'NJ'], 
                             "region_3" : ['DE', 'MD', 'PA', 'WV', 'VA'],
                             "region_4" : ['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN'],
                             "region_5" : ['IL', 'IN', 'MI', 'MN', 'OH', 'WI'],
                             "region_6" : ['AR', 'LA', 'NM', 'OK', 'TX'],
                             "region_7" : ['IA', 'KS', 'MO', 'NE'],
                             "region_8" : ['CO', 'MT', 'ND', 'SD', 'UT', 'WY'],
                             "region_9" : ['AZ', 'CA', 'HI', 'NV'],
                             "region_10": ['AK', 'ID', 'OR', 'WA']}

# %%
inventory_df["region"] = "NA"
for a_key in shannon_regions_dict_abbr.keys():
    inventory_df.loc[inventory_df["state"].isin(shannon_regions_dict_abbr[a_key]), 'region'] = a_key
inventory_df.head(2)

# %% [markdown]
# ### compute inventory in each region

# %%
region_inventory = inventory_df.copy()
region_inventory = region_inventory[["year", "region", "inventory"]]

region_inventory = region_inventory.groupby(["region", "year"])["inventory"].sum().reset_index()
region_inventory.head(2)

# %%
annual_slaughter.head(2)

# %% [markdown]
# ## Add one year to each inventory data so we have Jan 1st inventory.

# %%
region_inventory["year"] = region_inventory["year"] + 1

# %%
region_slaughter_inventory = pd.merge(region_inventory, annual_slaughter, 
                                      on=["region", "year"], how="outer")

print (f"{region_inventory.shape = }")
print (f"{annual_slaughter.shape = }")
print (f"{region_slaughter_inventory.shape = }")
region_slaughter_inventory.head(2)

# %%
annual_slaughter.head(10)

# %%
region_inventory.shape

# %%
#### it seems in some years some of the data are not available
# do we want to drop NAs? Region 8 has a lot of missing slaughter
# but its inventory might be important!

# region_slaughter_inventory.dropna(how="any", inplace=True)
region_slaughter_inventory.reset_index(drop=True, inplace=True)
print (region_slaughter_inventory.shape)

# %%
region_slaughter_inventory.head(2)

# %%
NotInteresting_regions_L = ["region_1_region_2", "region_3", "region_5"]
high_inv_regions = ["region_" + str(x) for x in [4, 6, 7, 8]]
low_inv_regions = ["region_" + str(x) for x in [9, 10]]

# %%
font = {'size' : 14}
matplotlib.rc('font', **font)

tick_legend_FontSize = 10

params = {"legend.fontsize": tick_legend_FontSize * 1.2,  # medium, large
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

# %%
# These colors are from US_map_study_area.py to be consistent with regions.

col_dict = {"region_1_region_2": "cyan",
            "region_3": "black", 
            "region_4": "green",
            "region_5": "tomato",
            "region_6": "red",
            "region_7": "dodgerblue",
            "region_8": "dimgray", # gray: "#C0C0C0"
            "region_9": "#ffd343", # mild yellow
            "region_10": "steelblue"}

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
(ax1, ax2) = axs;
ax1.grid(axis="both", which="both"); ax2.grid(axis="both", which="both")
y_var = "inventory"
for a_region in high_inv_regions:
    df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    ax1.plot(df.year, df[y_var], 
             color = col_dict[a_region], linewidth=3, 
             label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); #
    ax1.legend(loc="best");
    

for a_region in low_inv_regions:
    df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    ax2.plot(df.year, df[y_var], 
             color = col_dict[a_region], linewidth=3, 
             label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); #
    ax2.legend(loc="best");
    
space = 5
ax1.xaxis.set_major_locator(ticker.MultipleLocator(space)) 

fig_name = plot_dir + "inventory_TS_" + datetime.now().strftime('%Y-%m-%d time-%H.%M') + ".pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
region_slaughter_inventory.head(2)

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
(ax1, ax2) = axs;
ax1.grid(axis="both", which="both"); ax2.grid(axis="both", which="both")
y_var = "slaughter_count"
for a_region in high_inv_regions:
    df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    ax1.plot(df.year, df[y_var], 
             color = col_dict[a_region], linewidth=3, 
             label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); #
    ax1.legend(loc="best");
    

for a_region in low_inv_regions:
    df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    ax2.plot(df.year, df[y_var], 
             color = col_dict[a_region], linewidth=3, 
             label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); #
    ax2.legend(loc="best");
    
fig_name = plot_dir + "slaughter_TS_" + datetime.now().strftime('%Y-%m-%d time-%H.%M') + ".pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
(ax1, ax2) = axs;
ax1.grid(axis="both", which="both"); ax2.grid(axis="both", which="both")

# fig.suptitle("slaughter: dashed lines");
ax1.title.set_text('slaughter: dashed lines')

y_var = "inventory"
for a_region in high_inv_regions:
    df = region_slaughter_inventory.copy()
    df = df[df["region"] == a_region].copy()
    ax1.plot(df.year, df[y_var], color = col_dict[a_region], linewidth=3,
            label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); # 

for a_region in low_inv_regions:
    df = region_slaughter_inventory.copy()
    df = df[df["region"] == a_region].copy()
    ax2.plot(df.year, df[y_var], color = col_dict[a_region], linewidth=3,
             label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); #

y_var = "slaughter_count"
for a_region in high_inv_regions:
    df = region_slaughter_inventory.copy()
    df = df[df["region"] == a_region].copy()
    ax1.plot(df.year, df[y_var], linestyle="dashed",
             color = col_dict[a_region], linewidth=3)

for a_region in low_inv_regions:
    df = region_slaughter_inventory.copy()
    df = df[df["region"] == a_region].copy()
    df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    ax2.plot(df.year, df[y_var], linestyle="dashed",
             color = col_dict[a_region], linewidth=3)

ax1.legend(loc="best"); ax2.legend(loc="best");

space = 5
ax1.xaxis.set_major_locator(ticker.MultipleLocator(space)) 

fig_name = plot_dir + "inv_slau_TS_" + datetime.now().strftime('%Y-%m-%d time-%H.%M') + ".pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
annual_slaughter.head(2)

# %%
region_inventory.head(2)

# %%
min_year = annual_slaughter.year.unique().min()
min_year = max([region_inventory.year.unique().min(), min_year])

# %%
print (region_slaughter_inventory.shape)
region_slaughter_inventory = region_slaughter_inventory[region_slaughter_inventory.year >= 1984].copy()
print (region_slaughter_inventory.shape)

# %%
inventory_annal_diff = pd.DataFrame()
for a_region in regions:
    curr_df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    curr_df = curr_df[['region', 'year', 'inventory']].copy()
    curr_df.sort_values("year", inplace=True)
    curr_diff = pd.DataFrame()
    curr_diff["inventory_delta"] = curr_df.iloc[1:]["inventory"].values - curr_df.iloc[:-1]["inventory"].values
    curr_diff["region"] = a_region
    curr_diff["year"] = curr_df.iloc[1:]["year"].astype("str").values + "_" + \
                                curr_df.iloc[:-1]["year"].astype("str").values
    
    inventory_annal_diff = pd.concat([inventory_annal_diff, curr_diff])

inventory_annal_diff = inventory_annal_diff[["region", "year", "inventory_delta"]]

inventory_annal_diff["inventory_delta"] = inventory_annal_diff["inventory_delta"].astype(int)
inventory_annal_diff.reset_index(drop=True, inplace=True)
inventory_annal_diff.head(2)

# %%
inventory_annal_diff.shape

# %%
inventory_annal_diff[inventory_annal_diff["inventory_delta"] < 0].shape

# %%
inventory_annal_diff[(inventory_annal_diff["region"]== "region_6") & 
                     (inventory_annal_diff["inventory_delta"] < 0)].shape

# %%
fig, axs = plt.subplots(1, 1, figsize=(12, 3), sharex=True, gridspec_kw={"hspace":0.15, "wspace":0.05})
axs.grid(axis="y", which="both")
axs.grid(axis="x", which="major")

region = "region_6"
df = inventory_annal_diff.copy()
df = df[df["region"] == region].copy()
axs.plot(df.year, df.inventory_delta, linewidth=1,color="dodgerblue")
axs.scatter(df.year, df.inventory_delta, label="inventory delta " +  region)

plt.xticks(rotation=90);
# axs.plot(df.year, df.slaughter_count, linewidth=3, label="slaughter "+ region, 
#          color="dodgerblue", linestyle="dashed");
space = 3
axs.xaxis.set_major_locator(ticker.MultipleLocator(space)) 

plt.title("inventory deltas")
plt.legend(loc = "best");

# %%
df.year.unique()

# %%
inventory_annal_diff[(inventory_annal_diff.region == "region_6") &
                     (inventory_annal_diff.year == "2012_2011")]

# %%
region_slaughter_inventory[(region_slaughter_inventory.region == "region_6") &
                           (region_slaughter_inventory.year == 2012)]

# %%
aa = inventory_annal_diff.loc[inventory_annal_diff.region == "region_6", "inventory_delta"].idxmin()
aa

# %%
inventory_annal_diff.loc[aa]

# %%
region_slaughter_inventory[(region_slaughter_inventory["region"] == "region_6") &
                           (region_slaughter_inventory["year"] == 2011)]

# %%
Rs = ["region_4", "region_7", "region_8", "region_9"]

region_slaughter_inventory[(region_slaughter_inventory["region"].isin(Rs)) &
                           (region_slaughter_inventory["year"].isin([2011])) ]

# %%
Rs = ["region_4", "region_7", "region_8", "region_9"]

inventory_annal_diff[(inventory_annal_diff["region"].isin(Rs)) &
                     (inventory_annal_diff["year"].isin(["2012_2011"])) ]

# %%

# %%
annual_slaughter.head(2)

# %%
region_slaughter_inventory.head(2)

# %%
# ## inventory of region 8

# fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
# axs.grid(axis="y", which="both");
# y_var = "inventory"

# for a_region in ["region_8"]:
#     df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
#     axs.scatter(df["year"], df["inventory"], label="inventory " +  a_region)
#     axs.legend(loc="best");

# %%
graph_dict = {"region_10" : ["region_8", "region_9"],
              "region_9" : ["region_6", "region_8", "region_10"],
              "region_8" : ["region_10", "region_9", "region_7", "region_5"],
              "region_7" : ["region_8", "region_6", "region_5", "region_4"],
              "region_6" : ["region_9", "region_7", "region_4"],
              "region_5" : ["region_8", "region_7", "region_4", "region_3"],
              "region_4" : ["region_7", "region_6", "region_5", "region_3"],
              "region_3" : ["region_5", "region_4", "region_1_region_2"],
              "region_1_region_2" : ["region_3"]
               }

# %%
# we need to look at the years of overlap
# between inventory and slaughter!

slr_more_than_inv_change = {}

for a_region in regions:
    curr_region_df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    for an_inv_yr in region_slaughter_inventory.year.unique():
        curr_region_df_yr = curr_region_df[curr_region_df.year == an_inv_yr].copy()
        

# %%

# %%
region_slaughter_inventory

# %%

# %%

# %%

# %%

# %%

# %%

# %%
s_8 = s[s.region == "region_8"]
s_8 = s_8[["region", "slaughter_count"]].copy()
s_8.dropna(how="any", inplace=True)

s_old_8 = s_old[s_old.region == "region_8"]
s_old_8 = s_old_8[["region", "slaughter_count"]].copy()
s_old_8.dropna(how="any", inplace=True)

s_9 = s[s.region == "region_9"]
s_9 = s_9[["region", "slaughter_count"]].copy()
s_9.dropna(how="any", inplace=True)


s_old_9 = s_old[s_old.region == "region_9"]
s_old_9 = s_old_9[["region", "slaughter_count"]].copy()
s_old_9.dropna(how="any", inplace=True)

print (s_8.shape)
print (s_old_8.shape)

print (s_9.shape)
print (s_old_9.shape)

# %%

# %%

# %%
