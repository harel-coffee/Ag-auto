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
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

from sklearn import preprocessing
import statistics
import statsmodels.api as sm

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
state_NPP_Ra.head(2)

# %%
state_NPP_Ra = rc.covert_unitNPP_2_total(NPP_df=state_NPP_Ra, npp_unit_col_="modis_npp",
                                         acr_area_col_="rangeland_acre", npp_area_col_="state_rangeland_npp")

### Security check to not make mistake later:
state_NPP_Ra.drop(columns=["modis_npp"], inplace=True)
state_NPP_Ra.head(2)

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
invent_tall_copy = invent_tall.copy()
invent_tall_copy.head(4)

# %%
col_3NPP = ["NPP_avg_past_3yrs"]

# Add zero in there so that we are using avg of NPP of 2001, 2002, and 2003 for inventory of 2004:
NPP_3yr_avg = list(state_NPP_Ra.state_rangeland_npp.rolling(3).mean())
NPP_3yr_avg.pop()
invent_tall["NPP_avg_past_3yrs"] = [0] + NPP_3yr_avg
invent_tall.head(5)

# %%
print (state_NPP_Ra.state_rangeland_npp[:3].values)
print (state_NPP_Ra.state_rangeland_npp[:3].mean())

# %%
invent_tall = invent_tall[invent_tall.year>=2004]
invent_tall.reset_index(drop=True, inplace=True)
invent_tall.head(3)

# %% [markdown]
# ## First three years
# can be either dropped or drop the first year, and for second year use only 1 year of NPP, and for third year use only 2 years of NPP.
#
# Let's just drop them.

# %%
state_RA.head(2)

# %%
invent_tall = pd.merge(invent_tall, state_RA[["state_fip", "rangeland_acre"]], 
                       on=["state_fip"], how="left")

invent_tall = pd.merge(invent_tall, herb[["state_fip", "herb_avg"]], 
                       on=["state_fip"], how="left")

invent_tall = pd.merge(invent_tall, county_fips[["state_fip", "EW"]], 
                       on=["state_fip"], how="left")
invent_tall.head(2)

# %%
invent_tall["EW_binary"] = invent_tall["EW"].map({"E":0, "W" : 1})
invent_tall.head(2)

# %%
# Re-order the columns.
new_order = ["state_fip", "year", "NPP_avg_past_3yrs", 
             "rangeland_acre", "herb_avg", "EW", "EW_binary", "inventory"]
invent_tall = invent_tall[new_order]
invent_tall.head(2)

# %%
invent_tall.tail(2)

# %% [markdown]
# ## Model inventory vs. 3 yrs Avg. NPP.

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
train_A = train_df[indp_vars].values
train_A = np.hstack([train_A, np.ones(len(train_A)).reshape(-1, 1)])
print(train_A.shape)
train_y = train_df[y_var].values.reshape(-1).astype("float")

NPP_sol, NPP_RSS, _, _ = np.linalg.lstsq(train_A, train_y)
print (f"{NPP_sol = }")
print (f"{NPP_RSS = }")

######
######   Test
######
test_A = test_df[indp_vars].values
test_A = np.hstack([test_A, np.ones(len(test_A)).reshape(-1, 1)])
y_test = test_df[[y_var]].values.reshape(-1)

yhat_test = test_A @ NPP_sol

NPP_test_res = y_test - yhat_test
NPP_RSS_test = np.dot(NPP_test_res, NPP_test_res)
NPP_MSE_test = NPP_RSS_test / len(y_test)
NPP_RSE_test = np.sqrt(NPP_MSE_test)
print ("========================================================================")
print (start_b + "Test residuals for 3-years-Avg-NPP:\n" + end_b)
print("    RSS_test = {0:.0f}.".format(NPP_RSS_test))
print("    MSE_test = {0:.0f}.".format(NPP_MSE_test))
print("    RSE =  {0:.0f}.".format(NPP_RSE_test))

# %%
X = train_df[indp_vars]
X = sm.add_constant(X)
Y = train_df[y_var].astype(float)
ks = sm.OLS(Y, X)
ks_result =ks.fit()
ks_result.summary()

# %%
yhat_train = X @ ks_result.params
NPP_train_res = train_df[y_var] - yhat_train
NPP_RSS_train = np.dot(NPP_train_res, NPP_train_res)
NPP_MSE_train = NPP_RSS_train / len(train_df[y_var])
NPP_RSE_train = np.sqrt(NPP_MSE_train)

print (start_b + "train residuals for 3-years-Avg-NPP:\n" + end_b)
print("    RSS_train = {0:.0f}.".format(NPP_RSS_train))
print("    MSE_train = {0:.0f}.".format(NPP_MSE_train))
print("    RSE =  {0:.0f}.".format(NPP_RSE_train))

# %%
test_A = test_df[indp_vars]
test_A = sm.add_constant(test_A)
y_test = test_df[y_var].astype(float)

yhat_test = test_A @ ks_result.params

NPP_test_res = y_test - yhat_test
NPP_RSS_test = np.dot(NPP_test_res, NPP_test_res)
NPP_MSE_test = NPP_RSS_test / len(y_test)
NPP_RSE_test = np.sqrt(NPP_MSE_test)

print (start_b + "Test residuals for 3-years-Avg-NPP:\n" + end_b)
print("    RSS_test = {0:.0f}.".format(NPP_RSS_test))
print("    MSE_test = {0:.0f}.".format(NPP_MSE_test))
print("    RSE =  {0:.0f}.".format(NPP_RSE_test))

# %% [markdown]
# ## Model inventory vs. 3 yrs Avg. NPP, RA, HerbRatio.

# %%
indp_vars = col_3NPP + ["rangeland_acre", "herb_avg"]
y_var = "inventory"

# %%
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

print (start_b + "Test residuals for 3-years-Avg-NPP, RA, and herb:\n" + end_b)
print("    RSS_test = {0:.0f}.".format(NPP_RSS_test))
print("    MSE_test = {0:.0f}.".format(NPP_MSE_test))
print("    RSE =  {0:.0f}.".format(NPP_RSE_test))

# %% [markdown]
# # Standardize so scales are the same.

# %%
print (f"{indp_vars = }")
print ()
train_df.head(2)

# %%
standard_indp = preprocessing.scale(train_df[indp_vars])

# %%
std_cols = [i + j for i, j in zip(indp_vars, ["_standardized"] * len(indp_vars))]
std_cols

# %%
train_df[std_cols] = standard_indp
train_df.head(2)

# %%
X = train_df[std_cols]
X = sm.add_constant(X)
Y = train_df[y_var].astype(float)
ks = sm.OLS(Y, X)
ks_result =ks.fit()
ks_result.summary()

# %%
yhat_train = X @ ks_result.params
NPP_train_res = train_df[y_var] - yhat_train
NPP_RSS_train = np.dot(NPP_train_res, NPP_train_res)
NPP_MSE_train = NPP_RSS_train / len(train_df[y_var])
NPP_RSE_train = np.sqrt(NPP_MSE_train)

print (start_b + "Standardized: train residuals for 3-years-Avg-NPP, RA, herb:\n" + end_b)
print("    RSS_train = {0:.0f}.".format(NPP_RSS_train))
print("    MSE_train = {0:.0f}.".format(NPP_MSE_train))
print("    RSE =  {0:.0f}.".format(NPP_RSE_train))

# %% [markdown]
# # Standardize y and see what happens:

# %%
train_y_std = preprocessing.scale(train_y)
NPP_sol, NPP_RSS, _, _ = np.linalg.lstsq(train_A, train_y_std)
print (NPP_sol)
print (NPP_RSS)
print ("==============================================")

yhat_train_std = train_A @ NPP_sol

# kind of un-standardize
yhat_train = (yhat_train_std * np.std(train_y)) + train_y.mean()


NPP_train_res = train_y - yhat_train
NPP_RSS_train = np.dot(NPP_train_res, NPP_train_res)
NPP_MSE_train = NPP_RSS_train / len(train_y)
NPP_RSE_train = np.sqrt(NPP_MSE_train)

print (start_b + "standardized X and Y: train residuals for 3-years-Avg-NPP, RA, Herb, and dummy east-west:\n" + \
       end_b)
print("    RSS_test = {0:.0f}.".format(NPP_RSS_train))
print("    MSE_test = {0:.0f}.".format(NPP_MSE_train))
print("    RSE =  {0:.0f}.".format(NPP_RSE_train))

# %% [markdown]
# # Standard deviation
#
# pandas std(.) and statistics.stdev() both produce the same results using ```N-1``` in denominator while np.std() and preprocessing.scale() use ```N``` in denominator.

# %%
# mu = train_df["NPP_avg_past_3yrs"].mean()
# np_std = np.std(train_df["NPP_avg_past_3yrs"])
#### the preprocessing.scale uses np.std. 
#### I do not know why statistics.stdev(.) is different!!!!
# (train_df["NPP_avg_past_3yrs"] - mu) / np_std
# statistics.stdev(train_df["NPP_avg_past_3yrs"])


import statistics

v = train_df["NPP_avg_past_3yrs"]
vcentered = v - v.mean()
vcentered_norm = np.linalg.norm(vcentered)

print (f"{v.std()                          = }")
print (f"{statistics.stdev(v)              = }")
print (f"{vcentered_norm/np.sqrt(len(v)-1) = }")

print()
print (f"{v.std(ddof=0)                    = }")
print (f"{np.std(v)                        = }")
print (f"{vcentered_norm/np.sqrt(len(v))   = }")

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

print (start_b + "Test residuals for 3-years-NPP-Avg and EW_binary:\n" + end_b)
print("    RSS_test = {0:.0f}.".format(NPP_RSS_test))
print("    MSE_test = {0:.0f}.".format(NPP_MSE_test))
print("    RSE =  {0:.0f}.".format(NPP_RSE_test))

# %% [markdown]
# ## Model inventory vs. 3 yrs NPP, RA, HerbRatio, and EW Dummy.

# %%
invent_tall.head(2)

# %%
print (invent_tall.NPP_avg_past_3yrs.min())
print (invent_tall.NPP_avg_past_3yrs.max())

# %%
indp_vars = col_3NPP + ["rangeland_acre", "herb_avg", "EW_binary"]
indp_vars

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

print (start_b + "Test residuals for 3-years-NPP-Avg., RA, herb ratio, and dummy east-west:\n" + end_b)
print("    RSS_test = {0:.0f}.".format(NPP_RSS_test))
print("    MSE_test = {0:.0f}.".format(NPP_MSE_test))
print("    RSE =  {0:.0f}.".format(NPP_RSE_test))

# %% [markdown]
# # Transform Variables:

# %%
invent_tall.head(2)

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharey=False, sharex=False)
axes[0].grid(axis="y", which="both")
axes[1].grid(axis="y", which="both")
axes[2].grid(axis="y", which="both")

text_size, text_color = "large", "r"

data_ = invent_tall.rangeland_acre
sns.histplot(data=np.sqrt(data_), kde=True, bins=200, color="darkblue", ax=axes[0])
axes[0].text(x=np.sqrt(data_).min() + 2, y=20, s="2nd root", fontsize=text_size, color=text_color)

###########################################
sns.histplot(data=np.log10(data_), kde=True, bins=200, color="darkblue", ax=axes[1])
axes[1].text(x=np.log10(data_).min(), y=2, s="log10", fontsize=text_size, color=text_color)

###########################################
# sns.histplot(data = 1/data_, kde = True, bins=200, color = 'darkblue', ax=axes[2]);
# axes[2].text(x = (1/data_).min(), y=200, s='inverse', fontsize=text_size, color=text_color)

sns.histplot(data=data_, kde=True, bins=200, color="darkblue", ax=axes[2])
axes[2].text(x=data_.min() + 3000000, y=80, s="original", fontsize=text_size, color=text_color)


axes[0].set_xlabel("");
axes[1].set_xlabel("");
axes[2].set_xlabel("transformed rangeland acr");

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharey=False, sharex=False)
axes[0].grid(axis="y", which="both")
axes[1].grid(axis="y", which="both")
axes[2].grid(axis="y", which="both")

text_size, text_color = "large", "r"

data_ = invent_tall.NPP_avg_past_3yrs
sns.histplot(data=np.sqrt(data_), kde=True, bins=200, color="darkblue", ax=axes[0])
axes[0].text(x=np.sqrt(data_).min() + 2, y=8, s="2nd root", fontsize=text_size, color=text_color)

###########################################
sns.histplot(data=np.log10(data_), kde=True, bins=200, color="darkblue", ax=axes[1])
axes[1].text(x=np.log10(data_).min(), y=8, s="log10", fontsize=text_size, color=text_color)

###########################################
# sns.histplot(data = 1/data_, kde = True, bins=200, color = 'darkblue', ax=axes[2]);
# axes[2].text(x = (1/data_).min(), y=200, s='inverse', fontsize=text_size, color=text_color)

sns.histplot(data=data_, kde=True, bins=200, color="darkblue", ax=axes[2])
axes[2].text(x=data_.min() + 3000000, y=8, s="original", fontsize=text_size, color=text_color)


axes[0].set_xlabel("");
axes[1].set_xlabel("");
axes[2].set_xlabel("transformed NPP_avg_past_3yrs");

# %% [markdown]
# ### Avg. of current year and past 3 years (4 years total):
#
#
# #### Model inventory vs. 4 yrs NPP avg, RA, HerbRatio, and EW Dummy.
#
#

# %%
invent_tall_copy.head(4)

# %%
col_3NPP = ["NPP_avg_past_4yrs"]

# Add zero in there so that we are using avg of NPP of 2001, 2002, and 2003 for inventory of 2004:
NPP_4yr_avg = list(state_NPP_Ra.state_rangeland_npp.rolling(4).mean())
invent_tall_copy["NPP_avg_past_4yrs"] = NPP_4yr_avg
invent_tall_copy.head(5)

# %%
print (state_NPP_Ra.state_rangeland_npp[:4].values)
print (state_NPP_Ra.state_rangeland_npp[:4].mean())

# %%
invent_tall_copy = invent_tall_copy[invent_tall_copy.year>=2004]
invent_tall_copy.reset_index(drop=True, inplace=True)
invent_tall_copy.head(3)

# %%
invent_tall_copy = pd.merge(invent_tall_copy, state_RA[["state_fip", "rangeland_acre"]], 
                            on=["state_fip"], how="left")

invent_tall_copy = pd.merge(invent_tall_copy, herb[["state_fip", "herb_avg"]], 
                            on=["state_fip"], how="left")

invent_tall_copy = pd.merge(invent_tall_copy, county_fips[["state_fip", "EW"]], 
                             on=["state_fip"], how="left")

invent_tall_copy.head(2)

# %%
invent_tall_copy["EW_binary"] = invent_tall_copy["EW"].map({"E":0, "W" : 1})
invent_tall_copy.head(2)

# %%
# Re-order the columns.
new_order = ["state_fip", "year", "NPP_avg_past_4yrs", 
             "rangeland_acre", "herb_avg", "EW", "EW_binary", "inventory"]
invent_tall_copy = invent_tall_copy[new_order]
invent_tall_copy.head(2)

# %%
indp_vars = col_3NPP + ["rangeland_acre", "herb_avg", "EW_binary"]
print (f"{yr_max = }")
indp_vars

# %%
train_df = invent_tall_copy[invent_tall_copy.year < yr_max].copy()
test_df  = invent_tall_copy[invent_tall_copy.year == yr_max].copy()

# %%
train_A = train_df[indp_vars].values
train_A = np.hstack([train_A, np.ones(len(train_A)).reshape(-1, 1)])
print(train_A.shape)

train_y = train_df[y_var].values.reshape(-1).astype("float")
train_df.head(3)

# %%
NPP_sol, NPP_RSS, _, _ = np.linalg.lstsq(train_A, train_y)
print (indp_vars)
NPP_sol

# %%
test_A = test_df[indp_vars].values
test_A = np.hstack([test_A, np.ones(len(test_A)).reshape(-1, 1)])
y_test = test_df[[y_var]].values.reshape(-1)

yhat_test = test_A @ NPP_sol

NPP_test_res = y_test - yhat_test
NPP_RSS_test = np.dot(NPP_test_res, NPP_test_res)
NPP_MSE_test = NPP_RSS_test / len(y_test)
NPP_RSE_test = np.sqrt(NPP_MSE_test)


print (start_b + "Test residuals for 4-years-NPP-Avg., RA, herb ratio, and dummy EW:\n" + end_b)
print("    RSS_test = {0:.0f}.".format(NPP_RSS_test))
print("    MSE_test = {0:.0f}.".format(NPP_MSE_test))
print("    RSE =  {0:.0f}.".format(NPP_RSE_test))

# %%

# %%

# %%
