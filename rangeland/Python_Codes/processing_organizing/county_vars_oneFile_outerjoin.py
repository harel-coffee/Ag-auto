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
# **<span style='color:red'>Feb 29, 2024 update</span>:** Use Min's state level data (not conversion of county to state-level)
#
# ------------------------------------------------------
# **Jan 26, 2024**
#
# The Jan 24, 2024 models (long run avg of NPP for 2017 inventory modeling) was a lesson to put all variables in one file and just take outer join of everything.
#
# **Forgotten lesson** Keep everything: ***all states, not just 25***
#
# <font color='red'>red</font>, <span style='color:green'>green</span>, $\color{blue}{\text{blue}}$

# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %%
# for bold print
start_b = "\033[1m"
end_b = "\033[0;0m"
print("This is " + start_b + "a_bold_text" + end_b + "!")

# %%
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %% [markdown]
# ## County Fips

# %%
county_fips = pd.read_pickle(reOrganized_dir + "county_fips.sav")
county_fips = county_fips["county_fips"]

print(f"{len(county_fips.state.unique()) = }")
county_fips = county_fips[county_fips.state.isin(SoI_abb)].copy()
county_fips.drop_duplicates(inplace=True)
county_fips.reset_index(drop=True, inplace=True)
county_fips = county_fips[["county_fips", "county_name", "state", "EW"]]
print(f"{len(county_fips.state.unique()) = }")

county_fips.head(2)

# %% [markdown]
# ## Inventory

# %%
USDA_data = pd.read_pickle(reOrganized_dir + "county_USDA_data.sav")
inventory = USDA_data["cattle_inventory"]
inventory.rename(columns={"cattle_cow_beef_inventory": "inventory"}, inplace=True)

print(f"{inventory.data_item.unique() = }")
print(f"{inventory.commodity.unique() = }")
print()
print(f"{len(inventory.state.unique())= }")

inventory = inventory[["year", "county_fips", "inventory"]]
inventory.head(2)

# %% [markdown]
# ### RA
#
# We need RA to convert unit NPP to total NPP.

# %%
RA = pd.read_csv(reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv")
RA.rename(columns={"fips_id": "county_fips"}, inplace=True)
RA = rc.correct_Mins_county_6digitFIPS(df=RA, col_="county_fips")
print(f"{len(RA.county_fips.unique()) = }")
RA.reset_index(drop=True, inplace=True)
RA.head(2)

# %%
RA_Pallavi = pd.read_pickle(param_dir + "filtered_counties_RAsizePallavi.sav")
RA_Pallavi = RA_Pallavi["filtered_counties_29States"]
print(f"{len(RA_Pallavi.county_fips.unique()) = }")
print(f"{len(RA_Pallavi.state.unique()) = }")

Pallavi_counties = list(RA_Pallavi.county_fips.unique())
RA_Pallavi.head(2)

# %%
cty_yr_npp = pd.read_csv(reOrganized_dir + "county_annual_GPP_NPP_productivity.csv")

cty_yr_npp.rename(columns={"county": "county_fips", "MODIS_NPP": "unit_npp"}, inplace=True)
cty_yr_npp = rc.correct_Mins_county_6digitFIPS(df=cty_yr_npp, col_="county_fips")
cty_yr_npp = cty_yr_npp[["year", "county_fips", "unit_npp"]]

# Some counties do not have unit NPPs
cty_yr_npp.dropna(subset=["unit_npp"], inplace=True)
cty_yr_npp.reset_index(drop=True, inplace=True)

cty_yr_npp.head(2)

# %%
cty_yr_npp = pd.merge(cty_yr_npp, RA[["county_fips", "rangeland_acre"]], on=["county_fips"], how="left")

cty_yr_npp = rc.covert_unitNPP_2_total(NPP_df=cty_yr_npp, npp_unit_col_="unit_npp",
                                       acr_area_col_="rangeland_acre", npp_area_col_="county_total_npp")

cty_yr_npp.head(2)

# %% [markdown]
# ## add NPP std

# %%
npp_unit_std = cty_yr_npp.groupby(["county_fips"])["unit_npp"].std(ddof=1).reset_index()
cty_yr_npp.head(2)


# %%
npp_total_std = cty_yr_npp.groupby(["county_fips"])["county_total_npp"].std(ddof=1).reset_index()
npp_unit_std.rename(columns={"unit_npp": "unit_npp_std"}, inplace=True)
npp_total_std.rename(columns={"county_total_npp": "county_total_npp_std"}, inplace=True)
npp_std = pd.merge(npp_unit_std, npp_total_std, on=["county_fips"], how="left")
npp_std.head(2)

# %%
cty_yr_npp = pd.merge(cty_yr_npp, npp_std, on=["county_fips"], how="left")
cty_yr_npp.head(2)

# %% [markdown]
# ### Weather

# %%
filename = reOrganized_dir + "county_annual_avg_Tavg.sav"
cnty_yr_avg_Tavg = pd.read_pickle(filename)
cnty_yr_avg_Tavg = cnty_yr_avg_Tavg["annual_temp"]

cnty_yr_avg_Tavg.reset_index(drop=True, inplace=True)
cnty_yr_avg_Tavg.head(2)

# %%
filename = reOrganized_dir + "county_seasonal_temp_ppt_weighted.sav"
SW = pd.read_pickle(filename)
SW = SW["seasonal"]
SW = pd.merge(SW, county_fips, on=["county_fips"], how="left")

print(f"{len(SW.county_fips.unique())=}")
SW.head(2)

# %%
seasonal_precip_vars = ["s1_countymean_total_precip", "s2_countymean_total_precip",
                        "s3_countymean_total_precip", "s4_countymean_total_precip"]

seasonal_temp_vars = ["s1_countymean_avg_tavg", "s2_countymean_avg_tavg",
                      "s3_countymean_avg_tavg", "s4_countymean_avg_tavg"]

SW_vars = seasonal_precip_vars + seasonal_temp_vars
for a_col in SW_vars:
    SW[a_col] = SW[a_col].astype(float)


SW["annual_countymean_total_precip"] = SW[seasonal_precip_vars].sum(axis=1)
# SW["annual_countymean_avg_tavg"]   = SW[seasonal_temp_vars].sum(axis=1)
# SW["annual_countymean_avg_tavg"]   = SW["annual_countymean_avg_Tavg"]/4
SW = pd.merge(SW, cnty_yr_avg_Tavg, on=["county_fips", "year"], how="outer")
del cnty_yr_avg_Tavg
SW = SW.round(3)

SW.drop(labels=["county_name", "state"], axis=1)
SW.head(2)

# %%
cnty_grid_mean_idx = pd.read_csv(Min_data_base + "county_gridmet_mean_indices.csv")
cnty_grid_mean_idx.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
cnty_grid_mean_idx.rename(columns={"county": "county_fips"}, inplace=True)
cnty_grid_mean_idx = rc.correct_Mins_county_6digitFIPS(df=cnty_grid_mean_idx, col_="county_fips")
cnty_grid_mean_idx = cnty_grid_mean_idx[["year", "month", "county_fips", "normal", "alert", "danger", "emergency"]]

for a_col in ["normal", "alert", "danger", "emergency"]:
    cnty_grid_mean_idx[a_col] = cnty_grid_mean_idx[a_col].astype(int)
cnty_grid_mean_idx.reset_index(drop=True, inplace=True)
cnty_grid_mean_idx.head(2)

# %%
S1_heat = cnty_grid_mean_idx[cnty_grid_mean_idx.month.isin([1, 2, 3])]
S2_heat = cnty_grid_mean_idx[cnty_grid_mean_idx.month.isin([4, 5, 6, 7])]
S3_heat = cnty_grid_mean_idx[cnty_grid_mean_idx.month.isin([8, 9])]
S4_heat = cnty_grid_mean_idx[cnty_grid_mean_idx.month.isin([10, 11, 12])]

S1_heat = S1_heat[["year", "county_fips", "normal", "alert", "danger", "emergency"]]
S2_heat = S2_heat[["year", "county_fips", "normal", "alert", "danger", "emergency"]]
S3_heat = S3_heat[["year", "county_fips", "normal", "alert", "danger", "emergency"]]
S4_heat = S4_heat[["year", "county_fips", "normal", "alert", "danger", "emergency"]]

S1_heat = S1_heat.groupby(["year", "county_fips"]).sum().reset_index()
S2_heat = S2_heat.groupby(["year", "county_fips"]).sum().reset_index()
S3_heat = S3_heat.groupby(["year", "county_fips"]).sum().reset_index()
S4_heat = S4_heat.groupby(["year", "county_fips"]).sum().reset_index()

S4_heat.head(2)

# %%
S1_heat.rename(columns={"normal": "s1_normal", "alert": "s1_alert", 
                        "danger": "s1_danger", "emergency": "s1_emergency"}, inplace=True)

S2_heat.rename(columns={"normal": "s2_normal", "alert": "s2_alert",
                        "danger": "s2_danger", "emergency": "s2_emergency"}, inplace=True)

S3_heat.rename(columns={"normal": "s3_normal", "alert": "s3_alert",
                        "danger": "s3_danger", "emergency": "s3_emergency"}, inplace=True)

S4_heat.rename(columns={"normal": "s4_normal", "alert": "s4_alert", 
                        "danger": "s4_danger", "emergency": "s4_emergency"}, inplace=True)

seasonal_heat = pd.merge(S1_heat, S2_heat, on=["county_fips", "year"], how="outer")
seasonal_heat = pd.merge(seasonal_heat, S3_heat, on=["county_fips", "year"], how="outer")
seasonal_heat = pd.merge(seasonal_heat, S4_heat, on=["county_fips", "year"], how="outer")

seasonal_heat.reset_index(drop=True, inplace=True)
seasonal_heat.head(2)


# %%
annual_heat = cnty_grid_mean_idx.groupby(["year", "county_fips"]).sum().reset_index()
annual_heat = annual_heat[["year", "county_fips", "normal", "alert", "danger", "emergency"]]
annual_heat.reset_index(drop=True, inplace=True)
annual_heat.head(2)

# %%

# %% [markdown]
# # Others (Controls)
#
#  - herb ratio
#  - irrigated hay
#  - feed expense
#  - population
#  - slaughter
#
#  ### Herb

# %%
#####
##### The following 2 lines are old. Supriya, on Feb 8, created a new
##### file in which we have both herb and max NDVI DoY.
#####
# herb = pd.read_pickle(data_dir_base + "Supriya/Nov30_HerbRatio/county_herb_ratio.sav")
# herb = herb["county_herb_ratio"]

herb = pd.read_pickle(reOrganized_dir + "county_state_NDVIDOY_Herb.sav")
herb = herb["County_NDVIDOY_Herb"]

herb.dropna(how="any", inplace=True)

## Compute total herb area.
herb = rc.compute_herbRatio_totalArea(herb)
herb.reset_index(drop=True, inplace=True)
herb = herb.round(3)

# herb = herb[["county_fips", "herb_avg", "herb_area_acr"]]
herb.drop(columns=["pixel_count"], inplace=True)
herb.head(2)

# %%

# %% [markdown]
# ### irrigated hay
#
# **Need to find 2012 and fill in some of those (D)/missing stuff**

# %%
irr_hay = pd.read_pickle(reOrganized_dir + "irr_hay.sav")
irr_hay = irr_hay["irr_hay_perc"]

irr_hay.rename(columns={"value_irr": "irr_hay_area"}, inplace=True)

irr_hay = irr_hay[["county_fips", "irr_hay_area", "irr_hay_as_perc"]]

irr_hay.head(2)
# irr_hay = irr_hay[["county_fips", "irr_hay_as_perc"]]

# %%
irr_hay[irr_hay.county_fips == "04005"]

# %%
irr_hay[irr_hay.county_fips == "01001"]

# %%
feed_expense = USDA_data["feed_expense"]
feed_expense = feed_expense[["year", "county_fips", "feed_expense"]]

human_population = pd.read_pickle(reOrganized_dir + "human_population.sav")
human_population = human_population["human_population"]

slaughter_Q1 = pd.read_pickle(reOrganized_dir + "slaughter_Q1.sav")
slaughter_Q1 = slaughter_Q1["slaughter_Q1"]
slaughter_Q1.rename(columns={"cattle_on_feed_sale_4_slaughter": "slaughter"}, inplace=True)
slaughter_Q1 = slaughter_Q1[["year", "county_fips", "slaughter"]]
print("max slaughter sale is [{}]".format(slaughter_Q1.slaughter.max()))

# %%
AgLand = USDA_data["AgLand"]
wetLand_area = USDA_data["wetLand_area"]
FarmOperation = USDA_data["FarmOperation"]

# %%
RA["Pallavi"] = "N"
RA.loc[RA.county_fips.isin(RA_Pallavi.county_fips), "Pallavi"] = "Y"
RA.head(2)

# %%
filename = reOrganized_dir + "county_seasonal_ndvi.sav"
seasonal_ndvi = pd.read_pickle(filename)
seasonal_ndvi = seasonal_ndvi["seasonal_ndvi"]
seasonal_ndvi.head(2)

# %%
FarmOperation.head(2)

# %%
filename = reOrganized_dir + "county_data_forOuterJoin.sav"

export_ = {
    "AgLand": AgLand,
    "FarmOperation": FarmOperation,
    "Pallavi_counties": Pallavi_counties,
    "RA": RA,
    "RA_Pallavi": RA_Pallavi,
    "SW": SW,
    "SoI": SoI,
    "SoI_abb": SoI_abb,
    "abb_dict": abb_dict,
    "heat" : cnty_grid_mean_idx,
    "annual_heat": annual_heat,
    "seasonal_heat" : seasonal_heat,
    "county_fips" : county_fips,
    "npp": cty_yr_npp,
    "feed_expense": feed_expense,
    "herb": herb,
    "irr_hay": irr_hay,
    "human_population": human_population,
    "slaughter_Q1": slaughter_Q1,
    "wetLand_area": wetLand_area,
    "cattle_inventory": inventory,
    "seasonal_ndvi": seasonal_ndvi,
    "source_code": "county_vars_oneFile_outerjoin",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %% [markdown]
# ## Do the outer join and normalize and save in another file
#
# #### First do variables that change over time then constant variables such as rangeland area.

# %%
AgLand.head(2)

# %%
print(AgLand.commodity.unique())
print(AgLand.data_item.unique())

# %%
AgLand.rename(columns={"value": "AgLand"}, inplace=True)
AgLand = AgLand[["county_fips", "year", "data_item", "AgLand"]]
AgLand.head(2)

# %% [markdown]
# ### Constant Variables

# %%
RA.head(2)

# %%
herb.head(2)

# %%
irr_hay.head(2)

# %%
constants = pd.merge(RA, herb, on=["county_fips"], how="outer")
constants = pd.merge(constants, irr_hay, on=["county_fips"], how="outer")
constants.head(2)

# %% [markdown]
# ### Annual variables

# %%
AgLand.head(2)

# %%
## Leavre Agland out for now. Since we do not know how we want to use it.
print(len(AgLand.data_item.unique()))

# %%
FarmOperation.head(2)

# %%
print(len(FarmOperation.data_item.unique()))
print(len(FarmOperation.commodity.unique()))

# %%
FarmOperation.rename(columns={"value": "number_of_farm_operation"}, inplace=True)
FarmOperation = FarmOperation[["county_fips", "year", "number_of_farm_operation"]]
FarmOperation = rc.clean_census(df=FarmOperation, col_="number_of_farm_operation", col_to_lower=False)
FarmOperation.head(2)

# %%
annual_heat.head(2)

# %%
cty_yr_npp.head(2)

# %%
cty_yr_npp = cty_yr_npp[["county_fips", "year", "unit_npp", "county_total_npp", 
                         "unit_npp_std", "county_total_npp_std"]]
cty_yr_npp.head(2)

# %%
feed_expense.head(2)

# %%
human_population.head(2)

# %%
slaughter_Q1.head(2)

# %%
inventory.head(2)

# %%
wetLand_area.head(2)

# %%
print(len(wetLand_area.data_item.unique()))
print(len(wetLand_area.commodity.unique()))

# %%
wetLand_area = wetLand_area[["county_fips", "year", "crp_wetland_acr"]]
wetLand_area.head(2)

# %%
annual_outer = pd.merge(inventory, cty_yr_npp, on=["county_fips", "year"], how="outer")
annual_outer = pd.merge(annual_outer, slaughter_Q1, on=["county_fips", "year"], how="outer")
annual_outer = pd.merge(annual_outer, human_population, on=["county_fips", "year"], how="outer")
annual_outer = pd.merge(annual_outer, feed_expense, on=["county_fips", "year"], how="outer")
annual_outer = pd.merge(annual_outer, annual_heat, on=["county_fips", "year"], how="outer")
annual_outer = pd.merge(annual_outer, FarmOperation, on=["county_fips", "year"], how="outer")
annual_outer = pd.merge(annual_outer, wetLand_area, on=["county_fips", "year"], how="outer")
annual_outer.head(2)

# %% [markdown]
# ### Seasonal variables
# #### Seasonal ```NDVI``` to be added.

# %%
SW.head(2)

# %%
seasonal_heat.head(2)

# %%
seasonal_outer = pd.merge(SW, seasonal_heat, on=["county_fips", "year"], how="outer")
seasonal_outer.head(2)

# %%
all_df = pd.merge(annual_outer, seasonal_outer, on=["county_fips", "year"], how="outer")
all_df.head(2)

# %%
all_df = pd.merge(all_df, constants, on=["county_fips"], how="outer")
all_df.head(2)

# %%
all_df[all_df.county_fips == "01001"]

# %%
all_df = pd.merge(all_df, seasonal_ndvi, on=["county_fips", "year"], how="outer")
print(all_df.shape)
all_df.head(2)

# %%
all_df.sort_values(by=["county_fips", "year"], inplace=True)

# %%

# %%
all_df.sort_values(by=["county_fips", "year"], inplace=True)
all_df.head(2)

# %%
(list(all_df.columns))

# %%
cc = ["county_fips", "max_ndvi_season_avhrr_s1", "max_ndvi_season_avhrr_s2", 
      "max_ndvi_season_avhrr_s3", "max_ndvi_season_avhrr_s4"]
all_df.loc[all_df.year==2000, cc]

# %%
cc = ["county_fips", "max_ndvi_season_avhrr_s1", "max_ndvi_season_avhrr_s2", 
      "max_ndvi_season_avhrr_s3", "max_ndvi_season_avhrr_s4"]
(all_df.loc[(all_df.county_fips=="01009") & (all_df.year==2002)])

# %% [markdown]
# # Normalization
# Maybe we need to think about normalization in case of test and train.
# For now, apply to all.

# %%
all_df_normalized = all_df.copy()
all_df_normalized.head(2)

# %%
list(all_df_normalized.columns)

# %%
all_df_normalized.max_ndvi_season_avhrr.unique()

# %%
non_normal_cols = [
    'year',
    'county_fips',
    'inventory',
     'county_name',
     'state',
     'EW',
     'Pallavi',
     'max_ndvi_month_avhrr',
     'max_ndvi_season_avhrr',
     'max_ndvi_season_avhrr_s1',
     'max_ndvi_season_avhrr_s2',
     'max_ndvi_season_avhrr_s3',
     'max_ndvi_season_avhrr_s4',
     'max_ndvi_month_gimms',
     'max_ndvi_season_gimms',
     'max_ndvi_season_gimms_s1',
     'max_ndvi_season_gimms_s2',
     'max_ndvi_season_gimms_s3',
     'max_ndvi_season_gimms_s4',
     'max_ndvi_month_modis',
     'max_ndvi_season_modis',
     'max_ndvi_season_modis_s1',
     'max_ndvi_season_modis_s2',
     'max_ndvi_season_modis_s3',
     'max_ndvi_season_modis_s4']

numeric_cols = [x for x in sorted(all_df_normalized.columns) if not (x in non_normal_cols)]

# %%
numeric_cols

# %%
len(numeric_cols) + len(non_normal_cols) == len(all_df_normalized.columns)

# %%

# %%
all_df_normalized[numeric_cols] = (all_df_normalized[numeric_cols] - all_df_normalized[numeric_cols].mean()) / \
                                   all_df_normalized[numeric_cols].std(ddof=1)
all_df_normalized.head(2)

# %%
filename = reOrganized_dir + "county_data_and_normalData_OuterJoined.sav"

export_ = {"all_df": all_df,
           "all_df_normalized": all_df_normalized,
           "source_code": "county_vars_oneFile_outerjoin",
           "Author": "HN",
           "normalized_columns" : numeric_cols,
           "non_normalized_columns": non_normal_cols,
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

pickle.dump(export_, open(filename, "wb"))

# %% [markdown]
# # Compute differences/deltas and then normalize

# %%
list(all_df.columns)

# %%
all_df.head(2)
all_df[(all_df.year==2002) & (all_df.county_fips=="01001")]
all_df.loc[(all_df.county_fips=="56045"), "maxNDVI_DoY_countyMean"][0:5]
# ["maxNDVI_DoY_countyMean"]

# %%
all_df.loc[(all_df.county_fips=="01001"), "crp_wetland_acr"][0:5]

# %%
all_df.loc[(all_df.county_fips=="01001"), "number_of_farm_operation"][0:5]

# %%
all_df["max_ndvi_season_avhrr_s2"].unique()

# %%
# %%
delta_cols = ['year',
              'county_fips',
              'inventory',
              'unit_npp',
              'county_total_npp',
              'slaughter',
              'population',
              'feed_expense',
              'normal',
              'alert',
              'danger',
              'emergency',
              'number_of_farm_operation',
              'crp_wetland_acr',
              's1_countymean_total_precip',
              's2_countymean_total_precip',
              's3_countymean_total_precip',
              's4_countymean_total_precip',
              's1_countymean_avg_tavg',
              's2_countymean_avg_tavg',
              's3_countymean_avg_tavg',
              's4_countymean_avg_tavg',
              'annual_countymean_total_precip',
              'annual_avg_tavg',
              's1_normal',
              's1_alert',
              's1_danger',
              's1_emergency',
              's2_normal',
              's2_alert',
              's2_danger',
              's2_emergency',
              's3_normal',
              's3_alert',
              's3_danger',
              's3_emergency',
              's4_normal',
              's4_alert',
              's4_danger',
              's4_emergency',
              's1_mean_ndvi_avhrr',
              's2_mean_ndvi_avhrr',
              's3_mean_ndvi_avhrr',
              's4_mean_ndvi_avhrr',
              'max_ndvi_month_avhrr',
              'max_ndvi_in_year_avhrr',
              'max_ndvi_season_avhrr',
              'max_ndvi_season_avhrr_s1',
              'max_ndvi_season_avhrr_s2',
              'max_ndvi_season_avhrr_s3',
              'max_ndvi_season_avhrr_s4',
              's1_mean_ndvi_gimms',
              's2_mean_ndvi_gimms',
              's3_mean_ndvi_gimms',
              's4_mean_ndvi_gimms',
              'max_ndvi_month_gimms',
              'max_ndvi_in_year_gimms',
              'max_ndvi_season_gimms',
              'max_ndvi_season_gimms_s1',
              'max_ndvi_season_gimms_s2',
              'max_ndvi_season_gimms_s3',
              'max_ndvi_season_gimms_s4',
              's1_mean_ndvi_modis',
              's2_mean_ndvi_modis',
              's3_mean_ndvi_modis',
              's4_mean_ndvi_modis',
              'max_ndvi_month_modis',
              'max_ndvi_in_year_modis',
              'max_ndvi_season_modis',
              'max_ndvi_season_modis_s1',
              'max_ndvi_season_modis_s2',
              'max_ndvi_season_modis_s3',
              'max_ndvi_season_modis_s4',
              's1_sum_ndvi_avhrr',
              's2_sum_ndvi_avhrr',
              's3_sum_ndvi_avhrr',
              's4_sum_ndvi_avhrr',
              's1_sum_ndvi_gimms',
              's2_sum_ndvi_gimms',
              's3_sum_ndvi_gimms',
              's4_sum_ndvi_gimms',
              's1_sum_ndvi_modis',
              's2_sum_ndvi_modis',
              's3_sum_ndvi_modis',
              's4_sum_ndvi_modis',
              's1_max_ndvi_avhrr',
              's2_max_ndvi_avhrr',
              's3_max_ndvi_avhrr',
              's4_max_ndvi_avhrr',
              's1_max_ndvi_gimms',
              's2_max_ndvi_gimms',
              's3_max_ndvi_gimms',
              's4_max_ndvi_gimms',
              's1_max_ndvi_modis',
              's2_max_ndvi_modis',
              's3_max_ndvi_modis',
              's4_max_ndvi_modis']

# %%
non_delta_cols = ['year', 'county_fips'] + [x for x in all_df.columns if not (x in delta_cols)]
print (len(non_delta_cols))
non_delta_cols

# %%
# %%time
ind_deltas_df = pd.DataFrame(columns=delta_cols)
ind_deltas_df.head(2)

for a_county in all_df.county_fips.unique():
    curr_df = all_df[all_df.county_fips == a_county].copy()
    curr_df.sort_values(by=["year"], inplace=True)
    
    curr_diff = curr_df.loc[curr_df.index[1:], delta_cols[2:]].values - \
                curr_df.loc[curr_df.index[:-1], delta_cols[2:]].values
    curr_diff = pd.DataFrame(curr_diff, columns=delta_cols[2:])
    curr_diff["year"] = curr_df.year[1:].values
    curr_diff["county_fips"] = a_county
    ind_deltas_df = pd.concat([ind_deltas_df, curr_diff])
    del(curr_diff)

# %%
nonDelta_df = all_df.copy()

nonDelta_df = nonDelta_df[non_delta_cols]
nonDelta_df.head(3)

# %%
delta_data = pd.merge(ind_deltas_df, nonDelta_df, on=["year", "county_fips"], how="left")
delta_data.head(3)

# %%
print (f"{all_df.shape = }")
print (f"{ind_deltas_df.shape = }")
print (f"{delta_data.shape = }")

# %%
delta_data_normal = delta_data.copy()
delta_data_normal[numeric_cols] = (delta_data_normal[numeric_cols] - delta_data_normal[numeric_cols].mean()) / \
                                   delta_data_normal[numeric_cols].std(ddof=1)
delta_data_normal.head(2)

# %%
filename = reOrganized_dir + "county_delta_and_normalDelta_OuterJoined.sav"

export_ = {"delta_data": delta_data,
           "delta_data_normal": delta_data_normal,
           "delta_cols": delta_cols,
           "normalized_columns" : numeric_cols,
           "non_normalized_columns": non_normal_cols,
           "source_code": "county_vars_oneFile_outerjoin",
           "Author": "HN",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

pickle.dump(export_, open(filename, "wb"))

# %%
