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
# ## Nov 15.
#
# - ```SW```: Seasonal Weather: temp. and precip.
# - On Nov. 6 Mike wanted to model cattle inventory using only ```NPP```/```SW``` and rangeland area for one year.
# - On Nov. 13 we had a meeting and they wanted to model using ```NPP``` on county level total, not unit ```NPP```.
#
# **Min's data are inconsistent:** Let us subset the counties that are in common between ```NPP``` and ```SW```, and cattle inventory.
#
# #### Seasons in Tonsor are
# - S1: Jan - Mar
# - S2: Apr - Jul
# - S3: Aug - Sep
# - S4: Oct - Dec

# %%
import shutup

shutup.please()

import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

sys.path.append("/Users/hn/Documents/00_GitHub/Rangeland/Python_Codes/")
import rangeland_core as rc

# %% [markdown]
# ## Directories

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %% [markdown]
# ## Read data

# %%
# Bhupi = pd.read_csv(param_dir + "Bhupi_25states_clean.csv")
# Bhupi["SC"] = Bhupi.state + "-" + Bhupi.county

# print (f"{len(Bhupi.state.unique()) = }")
# print (f"{len(Bhupi.county_fips.unique()) = }")
# Bhupi.head(2)

# %%
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI = abb_dict['SoI']
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %% [markdown]
# #### List of county names and FIPs

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

# %%
USDA_data = pd.read_pickle(reOrganized_dir + "USDA_data.sav")
cattle_inventory = USDA_data["cattle_inventory"]
#
# pick only 25 states we want
cattle_inventory = cattle_inventory[cattle_inventory.state.isin(SoI)].copy()

print(f"{cattle_inventory.data_item.unique() = }")
print(f"{cattle_inventory.commodity.unique() = }")
print()
print(f"{len(cattle_inventory.state.unique()) = }")

census_years = sorted(list(cattle_inventory.year.unique()))
print(f"{census_years = }")

# pick only useful columns
inv_col_ = "cattle_cow_beef_inventory"
cattle_inventory = cattle_inventory[["year", "county_fips", inv_col_]]

print(f"{len(cattle_inventory.county_fips.unique()) = }")
cattle_inventory.head(2)

# %%
cattle_inventory[cattle_inventory.county_fips == "06107"]

# %%

# %%
print(cattle_inventory.shape)
cattle_inventory = rc.clean_census(df=cattle_inventory, col_=inv_col_)
print(cattle_inventory.shape)

# %% [markdown]
# ## Read ```NPP``` and ```SW```
#
# **Min has an extra "1" as leading digit in FIPS!!**

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
print(f"{len(NPP.county_fips.unique()) = }")
print(f"{len(seasonal_weather.county_fips.unique()) = }")

# %%
print(f"{NPP.shape = }")
print(f"{len(NPP.county_fips.unique()) = }")
NPP = NPP[NPP.county_fips.isin(list(county_id_name_fips.county_fips.unique()))].copy()
print(f"{NPP.shape = }")
print(f"{len(NPP.county_fips.unique()) = }")
NPP.head(2)

# %%
print(f"{seasonal_weather.shape = }")
print(f"{len(seasonal_weather.county_fips.unique()) = }")
LL = list(county_id_name_fips.county_fips.unique())
seasonal_weather = seasonal_weather[seasonal_weather.county_fips.isin(LL)].copy()
print(f"{len(seasonal_weather.county_fips.unique()) = }")
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
for a_year in seasonal_weather.year.unique():
    df = seasonal_weather[seasonal_weather.year == a_year]
    print(f"{len(df.county_fips.unique()) = }")

seasonal_weather.head(2)

# %% [markdown]
# ### Rangeland area

# %%
# Rangeland area and Total area:
county_RA_and_TA_fraction = pd.read_csv(
    reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv"
)
county_RA_and_TA_fraction.rename(columns={"fips_id": "county_fips"}, inplace=True)

county_RA_and_TA_fraction = rc.correct_Mins_FIPS(
    df=county_RA_and_TA_fraction, col_="county_fips"
)
L = len(county_RA_and_TA_fraction.county_fips.unique())
print("number of counties are {}.".format(L))
print(county_RA_and_TA_fraction.shape)
county_RA_and_TA_fraction.head(2)

# %% [markdown]
# ### Convert unit ```NPP``` to total county-level ```NPP```
#
# Units are $\frac{\text{Kg}~\times~C}{m^2}$, whatever the ```C``` is.
#
# 1 $m^2 = 0.000247105$ acres.

# %%
county_annual_NPP_Ra = pd.merge(
    NPP, county_RA_and_TA_fraction, on=["county_fips"], how="left"
)
county_annual_NPP_Ra.head(2)

# %%
county_annual_NPP_Ra = rc.covert_unitNPP_2_total(
    NPP_df=county_annual_NPP_Ra,
    npp_col_="modis_npp",
    area_col_="rangeland_acre",
    new_col_="county_rangeland_npp",
)
### Security check to not make mistake later:
county_annual_NPP_Ra.drop(columns=["modis_npp"], inplace=True)
county_annual_NPP_Ra.head(2)

# %%

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
common_years = (
    set(cattle_inventory.year.unique())
    .intersection(set(county_annual_NPP_Ra.year.unique()))
    .intersection(set(county_annual_SW_Ra.year.unique()))
)
common_years

# %%
cattle_inventory = cattle_inventory[cattle_inventory.year.isin(list(common_years))]
county_annual_SW_Ra = county_annual_SW_Ra[county_annual_SW_Ra.year.isin(common_years)]

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

# %% [markdown]
# ## NPP has a lot of missing counties
#
#  - Min says he had a threshld about rangeland/pasture.
#  - subset the ```NPP``` and ```Cattle``` to the intersection of counties present.
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

# %% [markdown]
# ## OLS 2017 ```NPP``` (Model)

# %%
NPP_Ra_cattleInv_2017 = county_annual_NPP_Ra_cattleInv[
    county_annual_NPP_Ra_cattleInv.year == 2017
].copy()

NPP_A_2017 = NPP_Ra_cattleInv_2017[["county_rangeland_npp", "rangeland_acre"]].values
NPP_A_2017 = np.hstack([NPP_A_2017, np.ones(len(NPP_A_2017)).reshape(-1, 1)])
print(NPP_A_2017.shape)

y_2017 = NPP_Ra_cattleInv_2017[[inv_col_]].values.reshape(-1)
print(f"{y_2017.shape = }")

# %%
NPP_sol_2017, NPP_RSS_2017, NPP_rank_2017, NPP_singular_vals_2017 = np.linalg.lstsq(
    NPP_A_2017, y_2017
)

# %%
county_annual_NPP_Ra_cattleInv[county_annual_NPP_Ra_cattleInv.year == 2017].head(2)

# %%
NPP_coef_2017, Ra_coef_2017, intercept_2017 = (
    NPP_sol_2017[0],
    NPP_sol_2017[1],
    NPP_sol_2017[2],
)

# %% [markdown]
# ### Apply ```NPP``` 2017 model to 2012 data

# %%
NPP_Ra_cattleInv_2012 = county_annual_NPP_Ra_cattleInv[
    county_annual_NPP_Ra_cattleInv.year == 2012
].copy()
y_2012 = NPP_Ra_cattleInv_2012[[inv_col_]].values.reshape(-1)

# NP_A_2012 = NPP_Ra_cattleInv_2012[["modis_npp", "rangeland_acre"]].values
# NP_A_2012 = np.hstack([NP_A_2012, np.ones(len(NP_A_2012)).reshape(-1, 1)])

# %%
NPP_Ra_cattleInv_2012.head(2)

# %%
NPP_yhat2012_Model2017 = (
    NPP_coef_2017 * NPP_Ra_cattleInv_2012["county_rangeland_npp"].values
    + Ra_coef_2017 * NPP_Ra_cattleInv_2012["rangeland_acre"].values
    + intercept_2017 * np.ones(len(y_2012))
)

NPP_res2012_Model2017 = y_2012 - NPP_yhat2012_Model2017
NPP_RSS2012_Model2017 = np.dot(NPP_res2012_Model2017, NPP_res2012_Model2017)
NPP_RSS2012_Model2017 / len(y_2012)

# %%
print(f"{NPP_Ra_cattleInv_2012[inv_col_].min()=}")
print(f"{NPP_Ra_cattleInv_2012[inv_col_].max()=}")

# %% [markdown]
# ## Least Squares based on 2017 ```Weather```

# %%
SW_Ra_cattleInv_2017 = county_annual_SW_Ra_cattleInv[
    county_annual_SW_Ra_cattleInv.year == 2017
].copy()

needed_cols = SW_Ra_cattleInv_2017.columns[2:11]
print(needed_cols)
SW_A_2017 = SW_Ra_cattleInv_2017[needed_cols].values
y_2017 = SW_Ra_cattleInv_2017[[inv_col_]].values.reshape(-1)
print(f"{y_2017.shape = }")

# %%
SW_A_2017 = np.hstack([SW_A_2017, np.ones(len(SW_A_2017)).reshape(-1, 1)])
print(SW_A_2017.shape)
SW_A_2017

# %%
SW_A_2017[0]

# %%
SW_sol_2017, SW_RSS_2017, SW_rank_2017, SW_singular_vals_2017 = np.linalg.lstsq(
    SW_A_2017, y_2017
)
SW_sol_2017

# %%
# SW_yhat_2017 = SW_A_2017 @ SW_sol_2017
# SW_res = y_2017 - SW_yhat_2017
# SW_RSS = np.dot(SW_res, SW_res)
# SW_RSS
SW_RSS_2017[0]

# %% [markdown]
# ### Apply 2017 model to 2012 data

# %%
SW_var_cols = [
    "S1_countyMean_total_precip",
    "S2_countyMean_total_precip",
    "S3_countyMean_total_precip",
    "S4_countyMean_total_precip",
    "S1_countyMean_avg_Tavg",
    "S2_countyMean_avg_Tavg",
    "S3_countyMean_avg_Tavg",
    "S4_countyMean_avg_Tavg",
    "rangeland_acre",
]

# %%
SW_Ra_cattleInv_2012 = county_annual_SW_Ra_cattleInv[
    county_annual_SW_Ra_cattleInv.year == 2012
].copy()

y_2012 = SW_Ra_cattleInv_2012[[inv_col_]].values.reshape(-1)

SW_A_2012 = SW_Ra_cattleInv_2012[SW_var_cols].values
SW_A_2012 = np.hstack([SW_A_2012, np.ones(len(y_2012)).reshape(-1, 1)])

# %%
SW_yhat2012_Model2017 = SW_A_2012 @ SW_sol_2017

SW_res2012_Model2017 = y_2012 - SW_yhat2012_Model2017
SW_RSS2012_Model2017 = np.dot(SW_res2012_Model2017, SW_res2012_Model2017)
SW_RSS2012_Model2017 / len(y_2012)

# %% [markdown]
# ### unit ```NPP``` results were:
#
# - $\text{RSS}_\text{NPP} = 107,997,845,546$
# - $\text{MSE}_\text{NPP} = 155,392,584$
# - $\text{RSE}_\text{NPP} = 12,466$

# %%
print("NPP residual stats are:")
print("    RSS = {0:.0f}.".format(NPP_RSS2012_Model2017))
print("    MSE = {0:.0f}.".format(NPP_RSS2012_Model2017 / len(y_2012)))
print("    RSE = {0:.0f}.".format(np.sqrt(NPP_RSS2012_Model2017 / len(y_2012))))
print()
print("Weather residual stats are:")
print("    RSS = {0:.0f}.".format(SW_RSS2012_Model2017))
print("    MSE = {0:.0f}.".format(SW_RSS2012_Model2017 / len(y_2012)))
print("    RSE =  {0:.0f}.".format(np.sqrt(SW_RSS2012_Model2017 / len(y_2012))))

# %%
# npp, area, intercept
NPP_sol_2017
print("npp coeff  = {0:.7f}".format(NPP_coef_2017))
print("area coeff = {0:.2f}".format(Ra_coef_2017))
print("intercept  = {0:.2f}".format(intercept_2017))

# %%
SW_sol_2017

# %%
SW_var_cols

# %%
county_annual_SW_Ra_cattleInv.head(2)

# %%
county_annual_NPP_Ra_cattleInv.head(2)

# %%
print(county_annual_NPP_Ra_cattleInv.shape)
print(county_annual_SW_Ra_cattleInv.shape)

# %%
county_annual_NPP_Ra_cattleInv.columns

# %%
county_annual_SW_Ra_cattleInv.columns

# %%
NPP_needed_cols = [
    "year",
    "county_fips",
    "county_rangeland_npp",
    "rangeland_acre",
    "county_area_acre",
    "rangeland_fraction",
    inv_col_,
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
county_annual_SW_Ra_cattleInv[SW_needed_cols].head(2)

# %%
county_annual_NPP_Ra_cattleInv[NPP_needed_cols].head(2)

# %%
cnty_ann_SW_NPP_Ra = pd.merge(
    county_annual_NPP_Ra_cattleInv[NPP_needed_cols],
    county_annual_SW_Ra_cattleInv[SW_needed_cols],
    on=["year", "county_fips"],
    how="left",
)

cnty_ann_SW_NPP_Ra.head(2)

# %% [markdown]
# ### we had dropped modis_npp column to avoid mistake. Let's add it back in, before saving data:

# %%
A = pd.read_csv(Min_data_base + "county_annual_MODIS_NPP.csv")
A.rename(columns={"NPP": "modis_npp"}, inplace=True)

A = rc.correct_Mins_FIPS(df=A, col_="county")
A.rename(columns={"county": "county_fips"}, inplace=True)

cnty_ann_SW_NPP_Ra = pd.merge(
    cnty_ann_SW_NPP_Ra, A, on=["county_fips", "year"], how="left"
)
cnty_ann_SW_NPP_Ra.head(2)

# %%
filename = reOrganized_dir + "cntyNPP_SW_catt_cow_beef_invt_CommonCntyYear.sav"

export_ = {
    "cnty_ann_SW_NPP_Ra": cnty_ann_SW_NPP_Ra,
    "source_code": "SW_CountyNPP_Ra_Model",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %%
# cnty_ann_SW_NPP_Ra.modis_npp.plot(kind='density');

# %%
# plt.hist(cnty_ann_SW_NPP_Ra.modis_npp, bins=200);

# %%
cnty_ann_SW_NPP_Ra.head(2)

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharey=False, sharex=False)
axes.grid(axis="y", which="both")
sns.histplot(
    data=cnty_ann_SW_NPP_Ra.rangeland_acre,
    kde=True,
    bins=200,
    color="darkblue",
    ax=axes,
)

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharey=False, sharex=False)
(ax1, ax2) = axes
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")

sns.histplot(
    data=cnty_ann_SW_NPP_Ra.county_rangeland_npp,
    kde=True,
    bins=200,
    color="darkblue",
    ax=ax1,
)

A = cnty_ann_SW_NPP_Ra[cnty_ann_SW_NPP_Ra.county_rangeland_npp < 200000]
sns.histplot(data=A.county_rangeland_npp, kde=True, bins=200, color="darkblue", ax=ax2)

ax1.set_xlabel("")

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharey=False, sharex=False)
axes.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")
sns.histplot(
    data=cnty_ann_SW_NPP_Ra[inv_col_], kde=True, bins=200, color="darkblue", ax=axes
)

# %%
tick_legend_FontSize = 8

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
fig, axes = plt.subplots(1, 1, figsize=(4, 4), sharey=True)
axes.grid(axis="y", which="both")
axes.scatter(cnty_ann_SW_NPP_Ra.county_rangeland_npp, cnty_ann_SW_NPP_Ra[inv_col_], s=5)

axes.set_xlabel("NPP")
axes.set_ylabel("cow inventory")
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
(ax1, ax2) = axes
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")
#############
ax1.scatter(
    cnty_ann_SW_NPP_Ra.rangeland_acre, cnty_ann_SW_NPP_Ra.county_rangeland_npp, s=5
)
ax1.set_xlabel("rangeland_acre")
ax1.set_ylabel("county_rangeland_npp")
#############
ax2.scatter(
    cnty_ann_SW_NPP_Ra.rangeland_acre, cnty_ann_SW_NPP_Ra.cattle_cow_beef_inventory, s=5
)
ax2.set_xlabel("rangeland acres")
ax2.set_ylabel("cow count")
#############
fig.tight_layout()
plt.show()

# %%
# NPP = pd.read_csv(Min_data_base + "county_annual_MODIS_NPP.csv")
# RA = pd.read_csv(reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv")
# RA.rename(columns={"fips_id": "county_fips"}, inplace=True)
# NPP.rename(columns={"county": "county_fips"}, inplace=True)
# NPP_RA = pd.merge(NPP, RA, on=["county_fips"], how="left")
# fig, axes = plt.subplots(1, 1, figsize=(5, 5), sharey=True)
# axes.grid(axis='y', which='both');
# axes.scatter(NPP_RA.rangeland_acre, NPP_RA.NPP, s = 5)
# axes.set_xlabel("Acres");
# axes.set_ylabel("NPP");
# plt.show()

# %%
cnty_ann_SW_NPP_Ra.head(2)

# %% [markdown]
# # Residual Plots

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
(ax1, ax2) = axes
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")
##################################################
ax1.scatter(y_2012, NPP_res2012_Model2017, s=5)
ax1.set_xlabel("y_2012")
ax1.set_ylabel("NPP_res2012_Model2017")
##################################################
ax2.scatter(y_2012, SW_res2012_Model2017, s=5)
ax2.set_xlabel("y_2012")
ax2.set_ylabel("SW_res2012_Model2017")
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
(ax1, ax2) = axes
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")
##################################################
ax1.scatter(NPP_yhat2012_Model2017, NPP_res2012_Model2017, s=5)
ax1.set_xlabel("NPP_yhat2012_Model2017")
ax1.set_ylabel("NPP_res2012_Model2017")
##################################################
ax2.scatter(SW_yhat2012_Model2017, SW_res2012_Model2017, s=5)
ax2.set_xlabel("SW_yhat2012_Model2017")
ax2.set_ylabel("SW_res2012_Model2017")
plt.show()

# %%

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
(ax1, ax2) = axes
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")
##################################################
ax1.scatter(SW_Ra_cattleInv_2012.rangeland_acre, NPP_res2012_Model2017, s=5)
ax1.set_xlabel("acre")
ax1.set_ylabel("NPP_res2012_Model2017")
##################################################
ax2.scatter(NPP_Ra_cattleInv_2012.rangeland_acre, SW_res2012_Model2017, s=5)
ax2.set_xlabel("acre")
ax2.set_ylabel("SW_res2012_Model2017")
##################################################
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
(ax1, ax2) = axes
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")
##################################################
ax1.scatter(SW_Ra_cattleInv_2012.rangeland_acre, NPP_res2012_Model2017, s=5)
ax1.set_xlabel("acre")
ax1.set_ylabel("NPP: res2012, Model2017")
##################################################
ax2.scatter(NPP_Ra_cattleInv_2012.rangeland_acre, SW_res2012_Model2017, s=5)
ax2.set_xlabel("acre")
ax2.set_ylabel("SW: res2012, Model2017")
plt.show()

# %% [markdown]
# ## Model with ```log```
#
# #### Plot logs first.

# %%
# ax1.set_title(r'$d_n$')
# ax1.set_title('2nd root');
# ax2.set_title('log10');
# ax3.set_title('log10');

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharey=False, sharex=False)
(ax1, ax2, ax3) = axes
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")
ax3.grid(axis="y", which="both")

text_size, text_color = "large", "r"

data_ = cnty_ann_SW_NPP_Ra.county_rangeland_npp
sns.histplot(data=np.sqrt(data_), kde=True, bins=200, color="darkblue", ax=ax1)
ax1.text(
    x=np.sqrt(data_).min(), y=250, s="2nd root", fontsize=text_size, color=text_color
)
###########################################
sns.histplot(data=np.log10(data_), kde=True, bins=200, color="darkblue", ax=ax2)
ax2.text(x=np.log10(data_).min(), y=40, s="log10", fontsize=text_size, color=text_color)
###########################################
sns.histplot(data=1 / data_, kde=True, bins=200, color="darkblue", ax=ax3)
ax3.text(x=(1 / data_).min(), y=2000, s="inverse", fontsize=text_size, color=text_color)
ax1.set_xlabel("")
ax2.set_xlabel("")
ax3.set_xlabel("transformed county-rangeland-level NPP")

# %%
cnty_ann_SW_NPP_Ra.head(2)

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharey=False, sharex=False)
(ax1, ax2) = axes
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")

text_size, text_color = "large", "r"

data_ = cnty_ann_SW_NPP_Ra.rangeland_acre
sns.histplot(data=np.sqrt(data_), kde=True, bins=200, color="darkblue", ax=ax1)
ax1.text(
    x=np.sqrt(data_).min(), y=250, s="2nd root", fontsize=text_size, color=text_color
)
###########################################
sns.histplot(data=np.log10(data_), kde=True, bins=200, color="darkblue", ax=ax2)
ax2.text(x=np.log10(data_).min(), y=40, s="log10", fontsize=text_size, color=text_color)
###########################################

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharey=False, sharex=False)
(ax1, ax2) = axes
ax1.grid(axis="y", which="both")
ax2.grid(axis="y", which="both")

text_size, text_color = "large", "r"

data_ = cnty_ann_SW_NPP_Ra[inv_col_]

sns.histplot(data=np.sqrt(data_), kde=True, bins=200, color="darkblue", ax=ax1)
ax1.text(
    x=np.sqrt(data_).min(), y=40, s="2nd root", fontsize=text_size, color=text_color
)
###########################################

sns.histplot(data=np.log10(data_), kde=True, bins=200, color="darkblue", ax=ax2)
ax2.text(x=np.log10(data_).min(), y=40, s="log10", fontsize=text_size, color=text_color)
###########################################

ax1.set_xlabel("")
ax2.set_xlabel("transformed inventory")

# %% [markdown]
# ## OLS w/ 2nd root of inventory and log of NPP, and log of rangeland acre

# %%
county_annual_NPP_Ra_cattleInv.head(2)

# %%
county_annual_NPP_Ra_cattleInv["sqrt_cattle_cow_beef_inventory"] = np.sqrt(
    county_annual_NPP_Ra_cattleInv["cattle_cow_beef_inventory"]
)

county_annual_NPP_Ra_cattleInv["log_county_rangeland_npp"] = np.log10(
    county_annual_NPP_Ra_cattleInv["county_rangeland_npp"]
)

county_annual_NPP_Ra_cattleInv["log_rangeland_acre"] = np.log10(
    county_annual_NPP_Ra_cattleInv["rangeland_acre"]
)

# %%
NPP_Ra_cattleInv_2017 = county_annual_NPP_Ra_cattleInv[
    county_annual_NPP_Ra_cattleInv.year == 2017
].copy()
# NPP_Ra_cattleInv_2017 = NPP_Ra_cattleInv_2017.round(decimals=2)
NPP_Ra_cattleInv_2017.head(2)

# %%
inv_col_ = "sqrt_cattle_cow_beef_inventory"

# %%
NPP_A_2017 = NPP_Ra_cattleInv_2017[
    ["county_rangeland_npp", "log_rangeland_acre"]
].values
NPP_A_2017 = np.hstack([NPP_A_2017, np.ones(len(NPP_A_2017)).reshape(-1, 1)])
print(NPP_A_2017.shape)

y_2017 = NPP_Ra_cattleInv_2017[[inv_col_]].values.reshape(-1)
print(f"{y_2017.shape = }")

# %%
NPP_sol_2017, NPP_RSS_2017, NPP_rank_2017, NPP_singular_vals_2017 = np.linalg.lstsq(
    NPP_A_2017, y_2017
)
NPP_sol_2017

# %%
NPP_Ra_cattleInv_2012 = county_annual_NPP_Ra_cattleInv[
    county_annual_NPP_Ra_cattleInv.year == 2012
].copy()

y_2012 = NPP_Ra_cattleInv_2012[[inv_col_]].values.reshape(-1)

# %%
NPP_Ra_cattleInv_2012.head(2)

# %%
NPP_A_2012 = NPP_Ra_cattleInv_2012[
    ["log_county_rangeland_npp", "log_rangeland_acre"]
].copy()
NPP_A_2012 = NPP_A_2012.values

NPP_A_2012 = np.hstack([NPP_A_2012, np.ones(len(NPP_A_2012)).reshape(-1, 1)])

NPP_logyhat2012_Model2017 = NPP_A_2012 @ NPP_sol_2017

# %%
## inverse of log10 to get actual prediction (as opposed to log of predictions)
NPP_yhat2012_Model2017 = (NPP_logyhat2012_Model2017) ** 2

# %%
NPP_res2012_transModel2017 = y_2012 - NPP_yhat2012_Model2017
NPP_RSS2012_transModel2017 = np.dot(
    NPP_res2012_transModel2017, NPP_res2012_transModel2017
)

# %%

# %% [markdown]
# ### unit ```NPP``` results were:
#
# - $\text{RSS}_\text{unit-NPP} = 107,997,845,546$
# - $\text{MSE}_\text{unit-NPP} = 155,392,584$
# - $\text{RSE}_\text{unit-NPP} = 12,466$

# %%
print("NPP residual stats:")
print("    RSS = {0:.0f}.".format(NPP_RSS2012_Model2017))
print("    MSE = {0:.0f}.".format(NPP_RSS2012_Model2017 / len(y_2012)))
print("    RSE = {0:.0f}.".format(np.sqrt(NPP_RSS2012_Model2017 / len(y_2012))))
print()
print("NPP residual stats for transformed model:")
print("    RSS = {0:.0f}.".format(NPP_RSS2012_transModel2017))
print("    MSE = {0:.0f}.".format(NPP_RSS2012_transModel2017 / len(y_2012)))
print("    RSE = {0:.0f}.".format(np.sqrt(NPP_RSS2012_transModel2017 / len(y_2012))))

# %%

# %%

# %%
