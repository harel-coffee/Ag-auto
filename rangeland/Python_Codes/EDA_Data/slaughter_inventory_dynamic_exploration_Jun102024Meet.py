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
filename = reOrganized_dir + "shannon_slaughter_data.sav"

slaughter = pd.read_pickle(filename)
print (slaughter["Date"])
(list(slaughter.keys()))

# %%
beef_slaught_complete_yrs = slaughter["beef_slaught_complete_yrs_tall"]
beef_slaught_complete_yrs.head(2)

# %%
regions = beef_slaught_complete_yrs["region"].unique()
regions

# %% [markdown]
# ### Compute annual slaughter

# %%
monthly_slaughter = beef_slaught_complete_yrs[["year", "region", "slaughter_count"]].copy()
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

# %%
all_data = all_data_dict["all_df_outerjoined"].copy()
inventory_df = all_data[["year", "inventory", "state_fips"]].copy()

inventory_df.dropna(subset=["inventory"], inplace=True)
inventory_df["inventory"] = inventory_df["inventory"].astype(int)
# do NOT subset to states of interest as slaughter data ignores that
# inventory_df = inventory_df[inventory_df["state_fips"].isin(list(state_fips_SoI["state_fips"].unique()))]
inventory_df.reset_index(drop=True, inplace=True)

print (f"{inventory_df.year.max() = }")
inventory_df.head(2)

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

# slaughter goes only up to 2022. So, lets do that to inventory
region_inventory = region_inventory[region_inventory["year"] < 2023].copy()

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
print (region_inventory.shape)
region_inventory[region_inventory.year == 2022]

# %%

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

# %%
print (region_slaughter_inventory.shape)
region_slaughter_inventory = region_slaughter_inventory[region_slaughter_inventory.year >= 1984].copy()
print (region_slaughter_inventory.shape)

# %% [markdown]
# # compute inventory deltas carefully
#
# There might be missing years!

# %%
inventory_annal_diff = pd.DataFrame()
for a_region in regions:
    curr_df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    curr_df = curr_df[['region', 'year', 'inventory']].copy()
    curr_df.sort_values("year", inplace=True)
    curr_region_diff = pd.DataFrame(columns=["region", "year", "inventory_delta"])
    for a_year in curr_df.year.unique():
        curr_df_yr = curr_df[curr_df.year.isin([a_year, a_year-1])].copy()
        if len(curr_df_yr) == 2:
            curr_diff = curr_df_yr.iloc[1]["inventory"] - \
                                              curr_df_yr.iloc[0]["inventory"]
            
            d = pd.DataFrame.from_dict({'region': [a_region], 
                                        'year': [str(a_year) + "_" + str(a_year-1)], 
                                        'inventory_delta': [curr_diff]})
            
            curr_region_diff = pd.concat([curr_region_diff, d])
    
    inventory_annal_diff = pd.concat([inventory_annal_diff, curr_region_diff])

inventory_annal_diff = inventory_annal_diff[["region", "year", "inventory_delta"]]

inventory_annal_diff.reset_index(drop=True, inplace=True)
inventory_annal_diff["inventory_delta"] = inventory_annal_diff["inventory_delta"].astype(int)
inventory_annal_diff.reset_index(drop=True, inplace=True)
inventory_annal_diff.head(2)

# %%
inventory_annal_diff.rename(columns={"year": "diff_years"}, inplace=True)
inventory_annal_diff.head(2)

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
axs.plot(df["diff_years"], df.inventory_delta, linewidth=1,color="dodgerblue")
axs.scatter(df["diff_years"], df.inventory_delta, label="inventory delta " +  region)

plt.xticks(rotation=90);
# axs.plot(df.year, df.slaughter_count, linewidth=3, label="slaughter "+ region, 
#          color="dodgerblue", linestyle="dashed");
space = 1
axs.xaxis.set_major_locator(ticker.MultipleLocator(space)) 

plt.title("inventory deltas")
plt.legend(loc = "best");

# %%

# %% [markdown]
# ### plot inventory diff against slaughter

# %%
inventory_annal_diff.head(2)

# %%
inventory_annal_diff["year"] = inventory_annal_diff["diff_years"].str.split("_", expand=True)[1]
inventory_annal_diff["year"] = inventory_annal_diff["year"].astype(int)
inventory_annal_diff.head(2)

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
(ax1, ax2) = axs;
ax1.grid(axis="both", which="both"); ax2.grid(axis="both", which="both")

# fig.suptitle("slaughter: dashed lines");
ax1.title.set_text('slaughter: dashed lines: multiplied by negative')

y_var = "inventory_delta"
for a_region in high_inv_regions:
    df = inventory_annal_diff.copy()
    df = df[df["region"] == a_region].copy()
    ax1.plot(df["year"], df[y_var], color = col_dict[a_region], linewidth=3,
            label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); # 

for a_region in low_inv_regions:
    df = inventory_annal_diff.copy()
    df = df[df["region"] == a_region].copy()
    ax2.plot(df["year"], df[y_var], color = col_dict[a_region], linewidth=3,
             label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); #

y_var = "slaughter_count"

for a_region in high_inv_regions:
    df = region_slaughter_inventory.copy()
    df = df[df["region"] == a_region].copy()
    ax1.plot(df.year, -df[y_var], linestyle="dashed", color = col_dict[a_region], linewidth=3)

for a_region in low_inv_regions:
    df = region_slaughter_inventory.copy()
    df = df[df["region"] == a_region].copy()
    df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    ax2.plot(df.year, -df[y_var], linestyle="dashed", color = col_dict[a_region], linewidth=3)

ax1.legend(loc="best"); ax2.legend(loc="best");

space = 1
ax1.xaxis.set_major_locator(ticker.MultipleLocator(space))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(space))
# plt.xticks(rotation=90);

# ax1.xaxis.set_ticks(list(region_slaughter_inventory.year))
# ax2.xaxis.set_ticks(list(region_slaughter_inventory.year))

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90);
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90);


fig_name = plot_dir + "invDiff_slau_TS_" + datetime.now().strftime('%Y-%m-%d time-%H.%M') + ".pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
inventory_annal_diff[(inventory_annal_diff.region == "region_6") &
                     (inventory_annal_diff.diff_years == "2012_2011")]

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
                     (inventory_annal_diff["diff_years"].isin(["2012_2011"])) ]

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
              "region_8" : ["region_10", "region_9", "region_7", "region_6", "region_5"],
              "region_7" : ["region_8", "region_6", "region_5", "region_4"],
              "region_6" : ["region_9", "region_8", "region_7", "region_4"],
              "region_5" : ["region_8", "region_7", "region_4", "region_3"],
              "region_4" : ["region_7", "region_6", "region_5", "region_3"],
              "region_3" : ["region_5", "region_4", "region_1_region_2"],
              "region_1_region_2" : ["region_3"]
               }

# %%
region_Adj = np.array([
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 1, 1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 1, 1, 1, 0, 0, 0],
                       [0, 1, 1, 0, 0, 1, 1, 0, 0],
                       [0, 0, 1, 0, 0, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 1, 0, 0],
                       [0, 0, 0, 1, 1, 1, 0, 1, 1],
                       [0, 0, 0, 0, 1, 0, 1, 0, 1],
                       [0, 0, 0, 0, 0, 0, 1, 1, 0],
                      ])
region_Adj

# %%
inventory_annal_diff.sort_values(["region", "year"], inplace=True)
inventory_annal_diff.reset_index(drop=True, inplace=True)
inventory_annal_diff.head(4)

# %%
region_slaughter_inventory.sort_values(["region", "year"], inplace=True)
region_slaughter_inventory.reset_index(drop=True, inplace=True)
region_slaughter_inventory.head(4)

# %%
inventory_annal_diff = pd.merge(inventory_annal_diff, 
                                region_slaughter_inventory[['region', 'year', 'slaughter_count']], 
                                on=["region", "year"], how="left")

inventory_annal_diff.head(2)

# %% [markdown]
# # Inventory vs. Slaughter
#
# There are 3 cases:
#
#   - Inventory decline is less than slaughter. Which means some of slaughters came into the region to be slaughtered.
#   
#     e.g. inventory goes from 100 to 80 and slaughter is 50. So, 30 cows were impoprted to be killed.
#   
# --------
#   - Inventory decline is more than slaughter, which means some of inventory moved out of state?
#   
#     e.g. inventory goes from 100 to 20 and slaughter is 50. So, 30 cows are lost (moved out of state?)
# --------
#   - Inventory increases even tho there are slaughters. So, some cows are imported and added to the inventory.
#   
#     e.g. inventory goes from 100 to 120 and slaughter is 50. So, 70 cows are imported to the region.
#     either for slaughter or all added to the inventory.

# %%
inventory_annal_diff.head(2)

# %%
incresed_inv_case = inventory_annal_diff[inventory_annal_diff.inventory_delta > 0].copy()
incresed_inv_case.reset_index(drop=True, inplace=True)

print (f"{incresed_inv_case.shape = }")
incresed_inv_case.head(2)

# %%
inventory_annal_diff.index

# %%
# we need to look at the years of overlap
# between inventory and slaughter!

A = inventory_annal_diff.copy()
A = A[A.inventory_delta < 0].copy()
A = A[abs(A["inventory_delta"]) < A["slaughter_count"]].copy()
A.reset_index(drop=True, inplace=True)
slr_more_than_inv_decline = A.copy()
del(A)

print (slr_more_than_inv_decline.shape)
slr_more_than_inv_decline.head(2)

# %%
A = inventory_annal_diff.copy()
A = A[A.inventory_delta < 0].copy()
A = A[abs(A["inventory_delta"]) > A["slaughter_count"]].copy()
A.reset_index(drop=True, inplace=True)
slr_less_than_inv_decline = A.copy()
del(A)

print (slr_less_than_inv_decline.shape)
slr_less_than_inv_decline.head(2)

# %%
A = inventory_annal_diff.copy()
A = A[A.inventory_delta < 0].copy()
A = A[abs(A["inventory_delta"]) == A["slaughter_count"]].copy()
A.reset_index(drop=True, inplace=True)
slr_equal_inv_decline = A.copy()
del(A)

print (slr_equal_inv_decline.shape)
slr_equal_inv_decline.head(2)

# %% [markdown]
# # Region_1_2 is not intersting
#
# and there is only 1 case of ```region_10```. Let us look at ```region_8```.

# %%
slr_less_than_inv_decline.groupby(["region"]).count()

# %%
slr_less_than_inv_decline[slr_less_than_inv_decline.region == "region_8"]

# %%
aa_ = slr_less_than_inv_decline[slr_less_than_inv_decline.region == "region_8"].year.values
aa_

# %% [markdown]
# #### Look at neighbors and those years
#
# to see if we can find anything. 
#
# Slaughter has been less than inventory decline. So, if we assume inventory/slaughter data are correctly recorded.:
#
# - either some of inventory has gone to neighbors.
# - or some cows from other states have been imported here solely for slaughtering.
# - or both?
#
# So, we need to find a neghbor(s) whose
# - This is not doable. What if a neighboring state's slaughter is also contamintated by importing cows to it for only slaughtering? Lets say that has not happened. If they had capacity to slaughter, they would have slaughtered their own cows.
#
# So, we need to find a neghbor(s) whose
# - relation between ```inventory_decline``` and ```slaughter``` is opposited of what we have above; ```abs(inventory_decline) < slaughter```. BUT, those neighbors have neighbors of their own. They might be interacting with them too. it is a hard thing to attack in a brute force fashion!

# %%
# we are looking for this discrepancy
target_year = aa_[0]
df_ = slr_less_than_inv_decline[(slr_less_than_inv_decline.region == "region_8") &
                                (slr_less_than_inv_decline.year == target_year)]

abs(df_["inventory_delta"].item()) - df_["slaughter_count"].item()

# %%
explore_df = inventory_annal_diff[(inventory_annal_diff["region"].isin(graph_dict["region_8"])) &
                                  (inventory_annal_diff["year"] == target_year)].copy()

explore_df

# %%
explore_df["slaughter_minus_abs_inv_decline"] = explore_df["slaughter_count"] - abs(explore_df["inventory_delta"])
explore_df

# %%

# %%
