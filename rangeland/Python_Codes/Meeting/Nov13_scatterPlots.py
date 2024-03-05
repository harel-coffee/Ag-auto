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

# %% [markdown]
# ## Nov. 13 Scatter Plots

# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("/Users/hn/Documents/00_GitHub/Rangeland/Python_Codes/")
import rangeland_core as rc

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"
seasonal_dir = reOrganized_dir + "seasonal_variables/02_merged_mean_over_county/"

# %%

# %%
Bhupi = pd.read_csv(param_dir + "Bhupi_25states_clean.csv")
Bhupi["SC"] = Bhupi.state + "-" + Bhupi.county

print(f"{len(Bhupi.state.unique()) = }")
print(f"{len(Bhupi.county_fips.unique()) = }")
Bhupi.head(2)

# %%

# %%
abb_dict = pd.read_pickle(param_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
USDA_data = pickle.load(open(reOrganized_dir + "USDA_data.sav", "rb"))

cattle_inventory = USDA_data["cattle_inventory"]

# pick only 25 states we want
cattle_inventory = cattle_inventory[cattle_inventory.state.isin(SoI)].copy()

print(f"{cattle_inventory.data_item.unique() = }")
print(f"{cattle_inventory.commodity.unique() = }")
print(f"{cattle_inventory.year.unique() = }")

census_years = list(cattle_inventory.year.unique())
# pick only useful columns
cattle_inventory = cattle_inventory[
    ["year", "county_fips", "cattle_cow_beef_inventory"]
]

print(f"{len(cattle_inventory.county_fips.unique()) = }")
cattle_inventory.head(2)

# %%
print(cattle_inventory.shape)
cattle_inventory = rc.clean_census(
    df=cattle_inventory, col_="cattle_cow_beef_inventory"
)
print(cattle_inventory.shape)

# %% [markdown]
# ### Min has an extra "1" as leading digit in FIPS!!

# %%
# county_annual_GPP_NPP_prod = pd.read_csv(reOrganized_dir + "county_annual_GPP_NPP_productivity.csv")
# county_annual_GPP_NPP_prod.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)

# county_annual_GPP_NPP_prod = county_annual_GPP_NPP_prod[["year", "county", "modis_npp"]].copy()
# county_annual_GPP_NPP_prod.dropna(how='any', inplace=True)
# county_annual_GPP_NPP_prod.sort_values(by=["year", "county"], inplace=True)
# county_annual_GPP_NPP_prod.reset_index(drop=True, inplace=True)
# county_annual_GPP_NPP_prod.head(2)

# NPP = pd.read_csv(reOrganized_dir + "county_annual_GPP_NPP_productivity.csv")
NPP = pd.read_csv(Min_data_base + "county_annual_MODIS_NPP.csv")
NPP.rename(columns={"NPP": "modis_npp"}, inplace=True)

NPP = rc.correct_Mins_FIPS(df=NPP, col_="county")
NPP.rename(columns={"county": "county_fips"}, inplace=True)

NPP.head(2)

# %%

# %%
filename = reOrganized_dir + "county_seasonal_temp_ppt_weighted.sav"
seasonal_weather = pd.read_pickle(filename)
print(f"{seasonal_weather.keys() = }")
seasonal_weather = seasonal_weather["seasonal"]
seasonal_weather.head(2)

# %%
seasonal_var_cols = seasonal_weather.columns[2:10]
for a_col in seasonal_var_cols:
    seasonal_weather[a_col] = seasonal_weather[a_col].astype(float)

# %%
# pick only census years
NPP = NPP[NPP.year.isin(census_years)]
NPP.reset_index(drop=True, inplace=True)
NPP.head(2)

# pick only census years
seasonal_weather = seasonal_weather[seasonal_weather.year.isin(census_years)]
seasonal_weather.reset_index(drop=True, inplace=True)
seasonal_weather.head(2)

# %%
county_id_name_fips = pd.read_csv(Min_data_base + "county_id_name_fips.csv")
county_id_name_fips = county_id_name_fips[
    county_id_name_fips.STATE.isin(SoI_abb)
].copy()

county_id_name_fips.sort_values(by=["STATE", "county"], inplace=True)

county_id_name_fips = rc.correct_Mins_FIPS(df=county_id_name_fips, col_="county")
county_id_name_fips.rename(columns={"county": "county_fips"}, inplace=True)

county_id_name_fips.reset_index(drop=True, inplace=True)
county_id_name_fips.head(2)

# %%
print(f"{len(NPP.county_fips.unique()) = }")
print(f"{len(seasonal_weather.county_fips.unique()) = }")

# %%
print(f"{len(NPP.county_fips.unique()) = }")
NPP = NPP[NPP.county_fips.isin(list(county_id_name_fips.county_fips.unique()))].copy()
print(f"{len(NPP.county_fips.unique()) = }")
NPP.head(2)

# %%
print(f"{seasonal_weather.shape = }")
LL = list(county_id_name_fips.county_fips.unique())
seasonal_weather = seasonal_weather[seasonal_weather.county_fips.isin(LL)].copy()
print(f"{seasonal_weather.shape = }")

# %%
county_id_name_fips.head(2)

# %%
print(f"{(NPP.year.unique()) = }")
print(f"{len(NPP.county_fips.unique()) = }")
print(f"{len(cattle_inventory.county_fips.unique()) = }")

# %%
for a_year in NPP.year.unique():
    df = NPP[NPP.year == a_year]
    print(f"{len(df.county_fips.unique()) = }")

NPP.head(2)

# %%
# Rangeland area and Total area:
county_RA_and_TA_fraction = pd.read_csv(
    reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv"
)
print(county_RA_and_TA_fraction.shape)
county_RA_and_TA_fraction.head(5)

# %%
county_RA_and_TA_fraction.rename(columns={"fips_id": "county_fips"}, inplace=True)
county_RA_and_TA_fraction = rc.correct_Mins_FIPS(
    df=county_RA_and_TA_fraction, col_="county_fips"
)
county_RA_and_TA_fraction.head(2)

# %%
print(f"{len(county_RA_and_TA_fraction.county_fips.unique()) = }")
LL = list(county_id_name_fips.county_fips.unique())
county_RA_and_TA_fraction = county_RA_and_TA_fraction[
    county_RA_and_TA_fraction.county_fips.isin(LL)
].copy()
print(f"{len(county_RA_and_TA_fraction.county_fips.unique()) = }")

# %%

# %%
county_annual_NPP_Ra = pd.merge(
    NPP, county_RA_and_TA_fraction, on=["county_fips"], how="left"
)
county_annual_NPP_Ra.head(2)

# %%
county_annual_SW_Ra = pd.merge(
    seasonal_weather, county_RA_and_TA_fraction, on=["county_fips"], how="left"
)
county_annual_SW_Ra.head(2)

# %%
print(f"{sorted(cattle_inventory.year.unique())     = }")
print(f"{sorted(county_annual_NPP_Ra.year.unique()) = }")
print(f"{sorted(county_annual_SW_Ra.year.unique()) = }")

# %%
cattle_inventory = cattle_inventory[
    cattle_inventory.year.isin(list(county_annual_NPP_Ra.year.unique()))
]
county_annual_SW_Ra = county_annual_SW_Ra[
    county_annual_SW_Ra.year.isin(list(county_annual_NPP_Ra.year.unique()))
]
print(sorted(cattle_inventory.year.unique()))
print(sorted(county_annual_SW_Ra.year.unique()))
print(sorted(county_annual_NPP_Ra.year.unique()))

# %%
print(len(cattle_inventory.county_fips.unique()))
print(len(county_annual_NPP_Ra.county_fips.unique()))
print(len(county_annual_SW_Ra.county_fips.unique()))

# %%
cattle_inventory_cnty_missing_from_NPP = [
    x
    for x in cattle_inventory.county_fips.unique()
    if not (x in county_annual_NPP_Ra.county_fips.unique())
]
len(cattle_inventory_cnty_missing_from_NPP)

# %%
NPP_cnty_missing_from_cattle = [
    x
    for x in county_annual_NPP_Ra.county_fips.unique()
    if not (x in cattle_inventory.county_fips.unique())
]
len(NPP_cnty_missing_from_cattle)

# %%
print("01001" in list(county_annual_NPP_Ra.county_fips.unique()))
print("01001" in list(cattle_inventory.county_fips.unique()))

# %% [markdown]
# ## NPP has a lot of missing counties
#
#  - Min says he had a threshld about rangeland/pasture.
#  - subset the NPP and Cattle to the intersection of counties present.
#  - It seems there are different number of counties in each year in cattle inventory. Find intersection of those as well.

# %%
all_cattle_counties = set(cattle_inventory.county_fips.unique())
print(f"{len(all_cattle_counties) = }")

for a_year in sorted(cattle_inventory.year.unique()):
    curr_cow = cattle_inventory[cattle_inventory.year == a_year].copy()
    curr_cow_counties = set(curr_cow.county_fips.unique())
    all_cattle_counties = all_cattle_counties.intersection(curr_cow_counties)
    print(a_year)
    print(f"{len(all_cattle_counties) = }")
    print("====================================================================")


# %%
all_county_annual_NPP_Ra = set(county_annual_NPP_Ra.county_fips.unique())
print(f"{len(all_county_annual_NPP_Ra) = }")

for a_year in sorted(county_annual_NPP_Ra.year.unique()):
    curr = county_annual_NPP_Ra[county_annual_NPP_Ra.year == a_year].copy()
    curr_counties = set(curr.county_fips.unique())
    all_county_annual_NPP_Ra = all_county_annual_NPP_Ra.intersection(curr_counties)
    print(a_year)
    print(f"{len(all_county_annual_NPP_Ra) = }")
    print("====================================================================")


# %%
all_county_annual_SW_Ra = set(county_annual_SW_Ra.county_fips.unique())
print(f"{len(all_county_annual_SW_Ra) = }")

for a_year in sorted(county_annual_SW_Ra.year.unique()):
    curr = county_annual_SW_Ra[county_annual_SW_Ra.year == a_year].copy()
    curr_counties = set(curr.county_fips.unique())
    all_county_annual_SW_Ra = all_county_annual_SW_Ra.intersection(curr_counties)
    print(a_year)
    print(f"{len(all_county_annual_SW_Ra) = }")
    print("====================================================================")

# %%
# choose only the counties that are present in all years:
cattle_inventory = cattle_inventory[
    cattle_inventory.county_fips.isin(list(all_cattle_counties))
]

# %%
SW_counties = set(county_annual_SW_Ra.county_fips.unique())
NPP_counties = set(county_annual_NPP_Ra.county_fips.unique())
cow_counties = set(cattle_inventory.county_fips.unique())

county_intersection = NPP_counties.intersection(cow_counties)
county_intersection = county_intersection.intersection(SW_counties)

# %%
county_annual_SW_Ra = county_annual_SW_Ra[
    county_annual_SW_Ra.county_fips.isin(list(county_intersection))
]
county_annual_NPP_Ra = county_annual_NPP_Ra[
    county_annual_NPP_Ra.county_fips.isin(list(county_intersection))
]
cattle_inventory = cattle_inventory[
    cattle_inventory.county_fips.isin(list(county_intersection))
]

print(f"{county_annual_SW_Ra.shape = }")
print(f"{county_annual_NPP_Ra.shape = }")
print(f"{cattle_inventory.shape     = }")
print()
print(f"{len(county_annual_SW_Ra.county_fips.unique())  = }")
print(f"{len(county_annual_NPP_Ra.county_fips.unique()) = }")
print(f"{len(cattle_inventory.county_fips.unique())     = }")
print()
print(f"{sorted(county_annual_SW_Ra.year.unique())  = }")
print(f"{sorted(county_annual_NPP_Ra.year.unique()) = }")
print(f"{sorted(cattle_inventory.year.unique())     = }")

# %%
print(
    sorted(cattle_inventory.county_fips.unique())
    == sorted(county_annual_NPP_Ra.county_fips.unique())
)
print(
    sorted(cattle_inventory.county_fips.unique())
    == sorted(county_annual_SW_Ra.county_fips.unique())
)

# %%
len(cattle_inventory.county_fips.unique())

# %%
county_annual_NPP_Ra_cattleInv = pd.merge(
    county_annual_NPP_Ra, cattle_inventory, on=["county_fips", "year"], how="left"
)

print(f"{cattle_inventory.shape = }")
print(f"{county_annual_NPP_Ra.shape = }")
print(f"{county_annual_NPP_Ra_cattleInv.shape = }")
county_annual_NPP_Ra_cattleInv.head(2)

# %%
county_annual_SW_Ra_cattleInv = pd.merge(
    county_annual_SW_Ra, cattle_inventory, on=["county_fips", "year"], how="left"
)

print(f"{cattle_inventory.shape = }")
print(f"{county_annual_SW_Ra.shape = }")
print(f"{county_annual_SW_Ra_cattleInv.shape = }")
county_annual_SW_Ra_cattleInv.head(2)

# %%

# %%
county_annual_NPP_Ra_cattleInv.sort_values(by=["year", "county_fips"], inplace=True)
county_annual_NPP_Ra_cattleInv.reset_index(drop=True, inplace=True)
county_annual_NPP_Ra_cattleInv.head(2)

# %%
county_annual_SW_Ra_cattleInv.sort_values(by=["year", "county_fips"], inplace=True)
county_annual_SW_Ra_cattleInv.reset_index(drop=True, inplace=True)
county_annual_SW_Ra_cattleInv.head(2)

# %%
NPP_needed_cols = [
    "year",
    "county_fips",
    "modis_npp",
    "rangeland_acre",
    "county_area_acre",
    "rangeland_fraction",
    "cattle_cow_beef_inventory",
]

SW_needed_cols = [
    "year",
    "county_fips",
    "S1_countyMean_total_precip",
    "S2_countyMean_total_precip",
    "S3_countyMean_total_precip",
    "S4_countyMean_total_precip",
    "S1_countyMean_avg_Tavg",
    "S2_countyMean_avg_Tavg",
    "S3_countyMean_avg_Tavg",
    "S4_countyMean_avg_Tavg",
]

# %%
cnty_ann_SW_NPP_Ra = pd.merge(
    county_annual_NPP_Ra_cattleInv[NPP_needed_cols],
    county_annual_SW_Ra_cattleInv[SW_needed_cols],
    on=["year", "county_fips"],
    how="left",
)

cnty_ann_SW_NPP_Ra.head(2)

# %%
sns.displot(
    data=cnty_ann_SW_NPP_Ra,
    x="modis_npp",
    y="cattle_cow_beef_inventory",
    kind="kde",
    height=5,
)

# %%
tick_legend_FontSize = 10

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

# %%
cnty_ann_SW_NPP_Ra[cnty_ann_SW_NPP_Ra.cattle_cow_beef_inventory > 400000]

# %%
county_id_name_fips[county_id_name_fips.county_fips == "06107"]

# %%
L = list(
    cnty_ann_SW_NPP_Ra[
        cnty_ann_SW_NPP_Ra.cattle_cow_beef_inventory > 200000
    ].county_fips
)
county_id_name_fips[county_id_name_fips.county_fips.isin(L)]

# %%
NPP = pd.read_csv(Min_data_base + "county_annual_MODIS_NPP.csv")
RA = pd.read_csv(reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv")

RA.rename(columns={"fips_id": "county_fips"}, inplace=True)
NPP.rename(columns={"county": "county_fips"}, inplace=True)

NPP_RA = pd.merge(NPP, RA, on=["county_fips"], how="left")
NPP_RA.head(2)

# %%
cnty_ann_SW_NPP_Ra.head(2)

# %%

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
(ax1, ax2) = axes
# ax1.grid(True); ax2.grid(True)
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")
##################################################
ax1.scatter(
    cnty_ann_SW_NPP_Ra.rangeland_acre, cnty_ann_SW_NPP_Ra.cattle_cow_beef_inventory, s=5
)

ax1.set_xlabel("rangeland acre")
ax1.set_ylabel("inventory")
##################################################

ax2.scatter(
    cnty_ann_SW_NPP_Ra.modis_npp, cnty_ann_SW_NPP_Ra.cattle_cow_beef_inventory, s=5
)

ax2.set_xlabel("NPP")
# ax2.set_ylabel("inventory");
plt.show()

# %%
cnty_ann_SW_NPP_Ra

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharey=True, sharex=False)
(ax1, ax2, ax3) = axes
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")
ax3.grid(axis="y", which="both")
# ax1.grid(True); ax2.grid(True)
sns.histplot(data=cnty_ann_SW_NPP_Ra.modis_npp, kde=True, color="darkblue", ax=ax1)

sns.histplot(
    data=cnty_ann_SW_NPP_Ra.cattle_cow_beef_inventory,
    kde=True,
    color="darkblue",
    ax=ax2,
)
small_inven = cnty_ann_SW_NPP_Ra.loc[
    cnty_ann_SW_NPP_Ra.cattle_cow_beef_inventory < 75000, "cattle_cow_beef_inventory"
]
sns.histplot(data=small_inven, kde=True, color="darkblue", ax=ax3)
plt.xlabel("")

# %%

# %%
fig, axes = plt.subplots(1, 1, figsize=(5, 5), sharey=True)
axes.scatter(cnty_ann_SW_NPP_Ra.rangeland_acre, cnty_ann_SW_NPP_Ra.modis_npp, s=5)
axes.set_xlabel("rangeland_acre")
axes.set_ylabel("modis_npp")

plt.show()

# %%

# %%
