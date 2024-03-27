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
# March 01, 2024
#
# Mike and HN had a meeting. It was mentioned that we want state-level modeling. Modeling the changes/deltas. If something significant is not discovered we will look into county-level and maybe slaughter stuff.
#
# NPP/NDVI, feed price, and beef price should be independent variables and other stuff such as RA or dummy variables are secondary.
#
# Mike emailed me a file ```LivestockPrices.xlsx``` and set me a link for [hay prices](https://www.ers.usda.gov/data-products/feed-grains-database/feed-grains-yearbook-tables/).

# %%
import shutup
shutup.please()

import pandas as pd
import numpy as np
import os

from datetime import datetime, date

import os, os.path, pickle, sys

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
mike_dir = data_dir_base + "Mike/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%

# %%
xl = pd.ExcelFile(mike_dir + "LivestockPrices.xlsx")
EX_sheet_names = xl.sheet_names  # see all sheet names
EX_sheet_names = EX_sheet_names[1:]
EX_sheet_names

# %%
ii = 0
sheet_name_ = EX_sheet_names[ii]

curr_sheet = pd.read_excel(io=mike_dir + "LivestockPrices.xlsx",
                           sheet_name=sheet_name_, header=0, skiprows=0)
# curr_sheet_columns = list(curr_sheet.columns)
# named_columns = curr_sheet_columns[0] # [x for x in curr_sheet_columns if not("Unnamed" in x)]


# %%
list(curr_sheet.columns)

# %%
curr_sheet.head(5)

# %%
beef_price = curr_sheet[["Unnamed: 0", "Unnamed: 4"]].copy()
beef_price.columns = ["date", "steers_medium_large_600_650lbs_$_cwt"]
beef_price = beef_price.loc[3:291]
beef_price.reset_index(drop=True, inplace=True)
beef_price

# %%
beef_price['date'] = pd.to_datetime(beef_price['date'], errors='coerce')

# %%
beef_price["year"] = beef_price['date'].dt.year
beef_price["month"] = beef_price['date'].dt.month
beef_price.head(3)

# %%
beef_price_annual_avg = beef_price.copy()

# %%
beef_price = beef_price[beef_price.month == 7]
beef_price.reset_index(drop=True, inplace=True)
beef_price.head(3)

# %%
# print (f"{grain_feed_cost.iloc[278, 2:14].mean().round(2) = }")
# print (f"{grain_feed_cost.iloc[279, 2:14].mean().round(2) = }")
# print (f"{grain_feed_cost.iloc[280, 2:14].mean().round(2) = }")
# print (f"{grain_feed_cost.iloc[281, 2:14].mean().round(2) = }")

# %%
beef_price.head(2)

# %%
beef_price.rename({"steers_medium_large_600_650lbs_$_cwt": "steers_medium_large_600_650lbs_$_cwt_julyPrice",
                   "nan": "date", 
                   "Wt\navg 2/": "wt_avg_2_$perTon"}, 
                   axis=1, inplace=True)

beef_price = beef_price[["year", "steers_medium_large_600_650lbs_$_cwt_julyPrice"]]
beef_price.head(2)

# %%
beef_price_annual_avg.head(2)

# %%
col_ = "steers_medium_large_600_650lbs_$_cwt"
beef_price_annual_avg = beef_price_annual_avg.groupby("year")[col_].mean().to_frame()
beef_price_annual_avg.reset_index(drop=False, inplace=True)
beef_price_annual_avg.head(2)

# %%
beef_price_annual_avg[col_] = beef_price_annual_avg[col_].astype(float)
beef_price_annual_avg[col_] = beef_price_annual_avg[col_].round(2)
beef_price_annual_avg.head(2)

# %%
col_ = "steers_medium_large_600_650lbs_$_cwt"
beef_price_annual_avg.rename({col_: "steers_medium_large_600_650lbs_$_cwt_yrAvg"}, axis=1, inplace=True)
beef_price_annual_avg.head(2)

# %%
beef_price = pd.merge(beef_price_annual_avg, beef_price, on=["year"], how="outer")
beef_price.head(2)

# %% [markdown]
# # Change commodity year to regular year

# %%
xl = pd.ExcelFile(mike_dir + "Feed Grains Yearbook Tables-All Years.xlsx")
EX_sheet_names = xl.sheet_names  # see all sheet names
EX_sheet_names = EX_sheet_names[1:]
table_11_sheet_name = [x for x in EX_sheet_names if "Table11" in x]
table_11_sheet_name


grain_feed_cost = pd.read_excel(io=mike_dir + "Feed Grains Yearbook Tables-All Years.xlsx",
                                sheet_name=table_11_sheet_name)
grain_feed_cost = grain_feed_cost["FGYearbookTable11-Full"]
grain_feed_cost.head(3)

# %%
grain_feed_cost.columns = grain_feed_cost.loc[0]
grain_feed_cost.columns = grain_feed_cost.columns.astype(str)
grain_feed_cost.head(3)

grain_feed_cost = grain_feed_cost.loc[1:]
grain_feed_cost.reset_index(drop=True, inplace=True)
grain_feed_cost.head(4)

# %%
grain_feed_cost.rename({"Commodity and marketing\nyear 1/": "commodity",
                        "nan": "date", 
                        "Wt\navg 2/": "wt_avg_2_$perTon"}, 
                        axis=1, inplace=True)

grain_feed_cost.dropna(subset=['wt_avg_2_$perTon'], inplace=True)
grain_feed_cost.reset_index(drop=True, inplace=True)
grain_feed_cost.head(2)

# %%
grain_feed_cost["commodity"] = grain_feed_cost["commodity"].astype(str)
for idx_ in grain_feed_cost.index:
    if grain_feed_cost.loc[idx_, "commodity"] == "nan":
        grain_feed_cost.loc[idx_, "commodity"] = grain_feed_cost.loc[idx_-1, "commodity"]
        
grain_feed_cost["prev_year"] = grain_feed_cost.date.str.slice(start=0, stop=4).astype(int)
grain_feed_cost["year"] = grain_feed_cost["prev_year"] + 1
grain_feed_cost.head(3)

# %% [markdown]
# # Not Monthly
#
# Before 1949, there was no monthly data. So, we cannot get calendar-year prices. So, we need
# to drop those rows/years.

# %%
grain_feed_cost.dropna(subset=['May'], inplace=True)
grain_feed_cost.reset_index(drop=True, inplace=True)
grain_feed_cost.head(2)

# %%
all_hay = grain_feed_cost[grain_feed_cost.commodity == "All hay"].copy()
alfalfa = grain_feed_cost[grain_feed_cost.commodity == "Alfalfa hay"].copy()

all_hay.reset_index(drop=True, inplace=True)
alfalfa.reset_index(drop=True, inplace=True)

# %%
all_hay["wt_avg_2_$perTon_calendar_yr"] = 0

idxs = list(all_hay.index)
idxs.pop()
first_months_ = ["Jan", "Feb", "Mar", "Apr"]
second_months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for idx_ in idxs:
    yr_avg = (all_hay.loc[idx_, first_months_].sum() + all_hay.loc[idx_+1, second_months].sum()) / 12
    all_hay.loc[idx_, "wt_avg_2_$perTon_calendar_yr"] = round(yr_avg, 2)

all_hay.head(2)

# %%
all_hay.tail(2)

# %%
alfalfa["wt_avg_2_$perTon_calendar_yr"] = 0

idxs = list(alfalfa.index)
idxs.pop()
first_months_ = ["Jan", "Feb", "Mar", "Apr"]
second_months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for idx_ in idxs:
    yr_avg = (alfalfa.loc[idx_, first_months_].sum() + alfalfa.loc[idx_+1, second_months].sum()) / 12
    alfalfa.loc[idx_, "wt_avg_2_$perTon_calendar_yr"] = round(yr_avg, 2)

alfalfa.head(2)

# %%
alfalfa.tail(2)

# %%
col_ = "wt_avg_2_$perTon_calendar_yr"
all_hay = all_hay[all_hay[col_]!=0]

alfalfa = alfalfa[alfalfa[col_]!=0]

# %%
all_hay.rename({"wt_avg_2_$perTon": "allHay_wtAvg_2_$perTon_marketPrevYearYear",
                "wt_avg_2_$perTon_calendar_yr" : "allHay_wtAvg_2_$perTon_calendar_yr"}, axis=1, inplace=True)

all_hay = all_hay[["year", 
                   "allHay_wtAvg_2_$perTon_marketPrevYearYear", 
                   "allHay_wtAvg_2_$perTon_calendar_yr"]]
all_hay.head(2)

# %%
alfalfa.rename({"wt_avg_2_$perTon": "alfalfaHay_wtAvg_2_$perTon_marketPrevYearYear",
                "wt_avg_2_$perTon_calendar_yr" : "alfalfaHay_wtAvg_2_$perTon_calendar_yr"}, axis=1, inplace=True)

alfalfa = alfalfa[["year", 
                   "alfalfaHay_wtAvg_2_$perTon_marketPrevYearYear", 
                   "alfalfaHay_wtAvg_2_$perTon_calendar_yr"]]
alfalfa.head(2)

# %%
grain_feed_cost_short = pd.merge(all_hay, alfalfa, on=["year"], how="outer")
grain_feed_cost_short.head(2)

# %%
beef_price.head(2)

# %%
beefPrice_hayCost = pd.merge(grain_feed_cost_short, beef_price, on=["year"], how="outer")
print (f"{beefPrice_hayCost.shape = }")
beefPrice_hayCost.head(2)

# %%
beefPrice_hayCost.sort_values(by=["year"], inplace=True)
beefPrice_hayCost.reset_index(drop=True, inplace=True)
beefPrice_hayCost.head(2)

# %%
beefPrice_hayCost

# %%

# %%
delta_cols = list(beefPrice_hayCost.columns[1:])
delta_cols

# %%
delta_df = pd.DataFrame(beefPrice_hayCost.loc[1:, delta_cols].values - \
                        beefPrice_hayCost.loc[0:len(beefPrice_hayCost)-2, delta_cols].values)
delta_df.columns = delta_cols
delta_df.head(2)

# %%
delta_df["year"] = list(beefPrice_hayCost.year[1:].astype(str).values + \
                              "_" + \
                              beefPrice_hayCost.year[:-1].astype(str).values)

# %%
delta_df.head(3)

# %%

# %%
import matplotlib
import matplotlib.pyplot as plt

# %%

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharey=False)
axes.grid(axis="y", which="both")
axes.set_axisbelow(True) # sends the grids underneath the plot

plt.hist(delta_df['allHay_wtAvg_2_$perTon_calendar_yr'], bins=200);
# plt.xticks(np.arange(0, 102, 2), rotation ='vertical');

axes.title.set_text("deltas of allHay_wtAvg_2_$perTon_calendar_yr")
axes.set_xlabel("deltas of allHay_wtAvg_2_$perTon_calendar_yr")
axes.set_ylabel("count")
fig.tight_layout()

# fig_name = plots_dir + "xxxxx.pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharey=False)
axes.grid(axis="y", which="both")
axes.set_axisbelow(True) # sends the grids underneath the plot

plt.hist(beefPrice_hayCost['allHay_wtAvg_2_$perTon_calendar_yr'], bins=200);
# plt.xticks(np.arange(0, 102, 2), rotation ='vertical');

axes.title.set_text("allHay_wtAvg_2_$perTon_calendar_yr")
axes.set_xlabel("allHay_wtAvg_2_$perTon_calendar_yr")
axes.set_ylabel("count")
fig.tight_layout()

# fig_name = plots_dir + "xxxxx.pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
beefPrice_hayCost.head(2)

# %%
ind_vars = list(beefPrice_hayCost.columns)
ind_vars.remove("year")
############################################################
normal_df = (beefPrice_hayCost[ind_vars] - beefPrice_hayCost[ind_vars].mean()) / \
                        beefPrice_hayCost[ind_vars].std(ddof=1)
normal_df.head(2)

normal_cols = [i + j for i, j in zip(ind_vars, ["_normal"] * len(ind_vars))]
normal_cols

beefPrice_hayCost[normal_cols] = normal_df
beefPrice_hayCost.head(2)

# %%
ind_vars = list(delta_df.columns)
ind_vars.remove("year")
ind_vars
############################################################
normal_df = (delta_df[ind_vars] - delta_df[ind_vars].mean()) / delta_df[ind_vars].std(ddof=1)
normal_df.head(2)

normal_cols = [i + j for i, j in zip(ind_vars, ["_normal"] * len(ind_vars))]
normal_cols

delta_df[normal_cols] = normal_df
delta_df.head(2)

# %% [markdown]
# ## Reorder columns

# %%
delta_df_cols_ = list(delta_df.columns)
delta_df_cols_.remove("year")
delta_df_cols_ = ["year"] + delta_df_cols_
delta_df = delta_df[delta_df_cols_]
delta_df.head(2)

# %%
beefPrice_hayCost.head(2)

# %%
filename = reOrganized_dir + "beef_hay_cost_fromMikeLinkandFile.sav"

export_ = {"beef_hay_cost_MikeLinkandFile" : beefPrice_hayCost,
           "beef_hay_costDeltas_MikeLinkandFile" : delta_df,
           "source_code": "03_01_2024_BeefPriceFeedPriceClean.ipynb",
           "Author": "HN",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

pickle.dump(export_, open(filename, "wb"))

# %%
beefPrice_hayCost.head(2)

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharey=False)
axes.grid(axis="y", which="both")
axes.set_axisbelow(True) # sends the grids underneath the plot

plt.hist((beefPrice_hayCost['allHay_wtAvg_2_$perTon_calendar_yr_normal']), bins=200);
# plt.xticks(np.arange(0, 102, 2), rotation ='vertical');

axes.title.set_text("allHay_wtAvg_2_$perTon_calendar_yr")
axes.set_xlabel("allHay_wtAvg_2_$perTon_calendar_yr")
axes.set_ylabel("count")
fig.tight_layout()

# fig_name = plots_dir + "xxxxx.pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharey=False)
axes.grid(axis="y", which="both")
axes.set_axisbelow(True) # sends the grids underneath the plot

plt.hist((beefPrice_hayCost['allHay_wtAvg_2_$perTon_calendar_yr']), bins=200);
# plt.xticks(np.arange(0, 102, 2), rotation ='vertical');

axes.title.set_text("allHay_wtAvg_2_$perTon_calendar_yr")
axes.set_xlabel("allHay_wtAvg_2_$perTon_calendar_yr")
axes.set_ylabel("count")
fig.tight_layout()

# fig_name = plots_dir + "xxxxx.pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
