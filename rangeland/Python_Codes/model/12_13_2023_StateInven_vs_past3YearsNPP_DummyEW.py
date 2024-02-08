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
# StateInven_vs_past3YrsNPP_DummyEW_Dec11-for-Dec13
#
# cow_inventory as function of moving average of past 3 years of NPP.
#
# DummyEW: Dummy variable for east vs. western states.
#
# On Dec. 11 Mike and I had a meeting and this decision was made.

# %%
import shutup

shutup.please()

import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

from sklearn import preprocessing
import statistics
import statsmodels.api as sm

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

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

# %%
# for bold print
start_b = "\033[1m"
end_b = "\033[0;0m"
print ("This is " + start_b + "a_bold_text" + end_b + "!")

# %% [markdown]
# # Read

# %%
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI = abb_dict['SoI']
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
county_fips = pd.read_pickle(reOrganized_dir + "county_fips.sav")
county_fips = county_fips["county_fips"]
county_fips = county_fips[["state", "state_fip", "EW"]]
county_fips = county_fips[county_fips.state.isin(SoI_abb)].copy()
county_fips.drop_duplicates(inplace=True)
county_fips.reset_index(drop=True, inplace=True)

county_fips.head(2)

# %%
state_SoI_fip = county_fips.state_fip.unique()
len(state_SoI_fip)

# %%

# %%
# herb = pd.read_csv(data_dir_base + "Supriya/Nov30_herb/state_herb_ratio.csv")
# herb = rc.correct_state_int_fips_to_str(df=herb, col_="state_fip")
# herb.sort_values(by=["state_fip"], inplace=True)
herb = pd.read_pickle(data_dir_base + "Supriya/Nov30_HerbRatio/state_herb_ratio.sav")
print (herb.keys())
herb = herb["state_herb_ratio"]
herb = herb[herb.state_fip.isin(state_SoI_fip)]
# herb.dropna(how="any", inplace=True)
herb.head(3)

# %%
herb.dropna(how="any", inplace=True)
herb.reset_index(drop=True, inplace=True)
herb.head(3)

# %% [markdown]
# ### Read Rangeland area and Total area:

# %%
# Rangeland area and Total area:
state_RA = pd.read_pickle(reOrganized_dir + "state_RA_area.sav")
state_RA = state_RA["state_RA_area"]
state_RA = state_RA[state_RA.state_fip.isin(state_SoI_fip)]
state_RA.head(2)

# %% [markdown]
# ### Read NPP

# %%
NPP = pd.read_csv(Min_data_base + "statefips_annual_MODIS_NPP.csv")
NPP.rename(columns={"NPP": "modis_npp", "statefips90m": "state_fip"}, inplace=True)
NPP = rc.correct_3digitStateFips_Min(NPP, "state_fip")

NPP = NPP[NPP.state_fip.isin(state_SoI_fip)]
NPP.reset_index(drop=True, inplace=True)

NPP.head(2)

# %% [markdown]
# ### Convert Unit NPP to State Level

# %%
state_NPP_Ra = pd.merge(NPP, state_RA, on=["state_fip"], how="left")

state_NPP_Ra = rc.covert_unitNPP_2_total(NPP_df=state_NPP_Ra, 
                                         npp_unit_col_="modis_npp",
                                         acr_area_col_="rangeland_acre", 
                                         npp_area_col_="state_rangeland_npp")

### Security check to not make mistake later:
state_NPP_Ra.drop(columns=["modis_npp"], inplace=True)
state_NPP_Ra.head(2)

# %%

# %% [markdown]
# ### Read Inventory

# %%
invent_tall = pd.read_pickle(reOrganized_dir + "Shannon_Beef_Cows_fromCATINV_tall.sav")
invent_tall = invent_tall["CATINV_annual_tall"]

invent_tall = invent_tall[invent_tall.state.isin(SoI_abb)]

invent_tall = invent_tall[invent_tall.year.isin(state_NPP_Ra.year.unique())]
invent_tall.reset_index(drop=True, inplace=True)
invent_tall.head(2)

# %%
col_3NPP = ["NPP_3", "NPP_2", "NPP_1"]

invent_tall[col_3NPP] = 0
invent_tall.head(2)

# %%
for a_state in invent_tall.state_fip.unique():
    for a_year in np.arange(invent_tall.year.min()+3, invent_tall.year.max()+1):
        past_3_yrs = [a_year-3, a_year-2, a_year-1]
        curr_NPP = state_NPP_Ra[(state_NPP_Ra.year.isin(past_3_yrs)) & (state_NPP_Ra.state_fip == a_state)]

        NPP3yrs = curr_NPP.state_rangeland_npp.values # list(curr_NPP.state_rangeland_npp)
        invent_tall.loc[(invent_tall.year == a_year) & (invent_tall.state_fip == a_state), col_3NPP] = NPP3yrs

# %%
invent_tall.head(5)

# %%
invent_tall.tail(5)

# %% [markdown]
# ## First three years
# can be either dropped or drop the first year, and for second year use only 1 year of NPP, and for third year use only 2 years of NPP.
#
# Let's just drop them.

# %%
invent_tall = invent_tall[invent_tall.year>=2004].copy()

invent_tall.reset_index(drop=True, inplace=True)
invent_tall.head(2)

# %%
state_RA.head(2)

# %%
invent_tall = pd.merge(invent_tall, state_RA[["state_fip", "rangeland_acre"]], 
                       on=["state_fip"], how="left")
invent_tall.head(2)

# %%
herb.head(2)

# %%
invent_tall = pd.merge(invent_tall, herb[["state_fip", "herb_avg"]], 
                       on=["state_fip"], how="left")
invent_tall.head(2)

# %%
county_fips.head(2)

# %%
invent_tall = pd.merge(invent_tall, county_fips[["state_fip", "EW"]], 
                       on=["state_fip"], how="left")
invent_tall.head(2)

# %%
invent_tall["EW_binary"] = invent_tall["EW"].map({"E":0, "W" : 1})
invent_tall.head(2)

# %%
# Re-order the columns.
new_order = ["state_fip", "year", "NPP_3", "NPP_2", "NPP_1", 
             "rangeland_acre", "herb_avg", "EW", "EW_binary", "inventory"]
invent_tall = invent_tall[new_order]
invent_tall.head(2)

# %%
invent_tall.tail(2)

# %% [markdown]
# ## Model inventory vs. 3 yrs NPP.

# %%
indp_vars = col_3NPP
y_var = "inventory"

# %%
print(invent_tall.year.unique().min())
yr_max = invent_tall.year.unique().max()
print(yr_max)

# %%
train_df = invent_tall[invent_tall.year < yr_max].copy()
test_df  = invent_tall[invent_tall.year == yr_max].copy()

# %%
X = train_df[indp_vars]
X = sm.add_constant(X)
Y = train_df[y_var].astype(float)
ks = sm.OLS(Y, X)
ks_result =ks.fit()
ks_result.summary()

# %%
test_A = test_df[indp_vars]
test_A = sm.add_constant(test_A)
y_test = test_df[y_var].astype(float)

yhat_test = test_A @ ks_result.params

NPP_test_res = y_test - yhat_test
NPP_RSS_test = np.dot(NPP_test_res, NPP_test_res)
NPP_MSE_test = NPP_RSS_test / len(y_test)
NPP_RSE_test = np.sqrt(NPP_MSE_test)

print (start_b + "Test residuals for 3-years-NPP:\n" + end_b)
print("    RSS_test = {0:.0f}.".format(NPP_RSS_test))
print("    MSE_test = {0:.0f}.".format(NPP_MSE_test))
print("    RSE =  {0:.0f}.".format(NPP_RSE_test))

# %% [markdown]
# ## Model inventory vs. 3 yrs NPP, RA, HerbRatio.

# %%
indp_vars = col_3NPP + ["rangeland_acre", "herb_avg"]
y_var = "inventory"
indp_vars

# %%
train_df.head(2)

# %%
X = train_df[indp_vars]
X = sm.add_constant(X)
Y = train_df[y_var].astype(float)
ks = sm.OLS(Y, X)
ks_result =ks.fit()
ks_result.summary()

# %%
test_A = test_df[indp_vars]
test_A = sm.add_constant(test_A)
y_test = test_df[y_var].astype(float)

yhat_test = test_A @ ks_result.params

NPP_test_res = y_test - yhat_test
NPP_RSS_test = np.dot(NPP_test_res, NPP_test_res)
NPP_MSE_test = NPP_RSS_test / len(y_test)
NPP_RSE_test = np.sqrt(NPP_MSE_test)

print (start_b + "Test residuals for 3-years-NPP and herb ratio and rangeland area:\n" + end_b)
print("    RSS_test = {0:.0f}.".format(NPP_RSS_test))
print("    MSE_test = {0:.0f}.".format(NPP_MSE_test))
print("    RSE =  {0:.0f}.".format(NPP_RSE_test))

# %% [markdown]
# ## Model inventory vs. 3 yrs NPP and EW Dummy.

# %%
invent_tall.head(2)

# %%
indp_vars = col_3NPP + ["EW_binary"]
indp_vars

# %%
train_df.head(2)

# %%
X = train_df[indp_vars]
X = sm.add_constant(X)
Y = train_df[y_var].astype(float)
ks = sm.OLS(Y, X)
ks_result =ks.fit()
ks_result.summary()

# %%
test_A = test_df[indp_vars]
test_A = sm.add_constant(test_A)
y_test = test_df[y_var].astype(float)

yhat_test = test_A @ ks_result.params

NPP_test_res = y_test - yhat_test
NPP_RSS_test = np.dot(NPP_test_res, NPP_test_res)
NPP_MSE_test = NPP_RSS_test / len(y_test)
NPP_RSE_test = np.sqrt(NPP_MSE_test)

print (start_b + "Test residuals for 3-years-NPP and dummy east-west:\n" + end_b)
print("    RSS_test = {0:.0f}.".format(NPP_RSS_test))
print("    MSE_test = {0:.0f}.".format(NPP_MSE_test))
print("    RSE =  {0:.0f}.".format(NPP_RSE_test))

# %% [markdown]
# ## Model inventory vs. 3 yrs NPP, RA, HerbRatio, and EW Dummy.

# %%
invent_tall.head(2)

# %%
indp_vars = col_3NPP + ["rangeland_acre", "herb_avg", "EW_binary"]
indp_vars

# %%
X = sm.add_constant(train_df[indp_vars])
Y = train_df[y_var].astype(float)
ks = sm.OLS(Y, X)
ks_result =ks.fit()
ks_result.summary()

# %%
test_A = test_df[indp_vars]
test_A = sm.add_constant(test_A)
y_test = test_df[y_var].astype(float)

yhat_test = test_A @ ks_result.params

NPP_test_res = y_test - yhat_test
NPP_RSS_test = np.dot(NPP_test_res, NPP_test_res)
NPP_MSE_test = NPP_RSS_test / len(y_test)
NPP_RSE_test = np.sqrt(NPP_MSE_test)

print (start_b + "Test residuals for 3-years-NPP, RA, herb ratio, and dummy east-west:\n" + end_b)
print("    RSS_test = {0:.0f}.".format(NPP_RSS_test))
print("    MSE_test = {0:.0f}.".format(NPP_MSE_test))
print("    RSE =  {0:.0f}.".format(NPP_RSE_test))

# %% [markdown]
# ### Normalize and model with $\ln(y)$

# %%
invent_tall.head(2)

# %%
all_indp_vars = ["NPP_3", "NPP_2", "NPP_1", "rangeland_acre", "herb_avg"]
all_indp_vars = sorted(all_indp_vars)
all_indp_vars

# %%
# standard_indp = preprocessing.scale(all_df[explain_vars_herb]) # this is biased
normal_df = (invent_tall[all_indp_vars] - invent_tall[all_indp_vars].mean()) / \
                         invent_tall[all_indp_vars].std(ddof=1)
normal_df.head(2)

# %%
normal_cols = [i + j for i, j in zip(all_indp_vars, ["_normal"] * len(all_indp_vars))]
normal_cols

# %%
invent_tall[normal_cols] = normal_df
invent_tall.head(2)

# %%
X_normal = invent_tall[normal_cols]
X_normal = sm.add_constant(X_normal)
Y = np.log(invent_tall[y_var].astype(float))
ks_normal = sm.OLS(Y, X_normal)
ks_normal_result =ks_normal.fit()
ks_normal_result.summary()

# %%
X_normal = invent_tall[normal_cols]
X_normal = sm.add_constant(X_normal)
Y = (invent_tall[y_var].astype(float))
ks_normal = sm.OLS(Y, X_normal)
ks_normal_result =ks_normal.fit()
ks_normal_result.summary()

# %%
6.38e+05 - 2.14e+05

# %%
