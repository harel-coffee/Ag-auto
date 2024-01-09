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
# Old queries they said is not right. I had compared that to state level data from Shannon.
#
#  - Total Beef Cow inventory: https://quickstats.nass.usda.gov/#ADD6AB04-62EF-3DBF-9F83-5C23977C6DC7
#  - Inventory of Beef Cows:  https://quickstats.nass.usda.gov/#B8E64DC0-E63E-31EA-8058-3E86C0DCB74E
#  
#  --- 
# **New queries**
#
# Too many records. had to break it into two queries:
#
# Q1 and Q2 are very similar. I just did not choose the last part of first section Domain: (TOTAL vs INVENTORY of Beef COWs)
#
# In Q1 and Q2 rows are divided into different categories (1-9 head, 10-20 head). Too much work to clean it up., So, I choose Total in the domain and create Q4.
# __________________
#   - Q1_P1. https://quickstats.nass.usda.gov/#EDC639B8-9B16-3BC1-ABB7-8ACC6F9D6646
#   - Q1_P2. https://quickstats.nass.usda.gov/#BBF52292-3BBA-37B6-AE9C-379832BF2418
# __________________
#    - Q2_P1. https://quickstats.nass.usda.gov/#CEEE6107-8E75-3662-B8E0-5EE3E913F59E
#    - Q2_P2. https://quickstats.nass.usda.gov/#6F35D357-C7F7-340C-B2C2-34E285EE3147
# __________________
#    - Q3_P1. https://quickstats.nass.usda.gov/#0D301A69-C0D6-39AA-A5DF-E2EE46794D13
#    - Q3_P2. https://quickstats.nass.usda.gov/#24382805-7D70-3709-9AC9-6A0689FA721F
# __________________
#    - Q4. Cattle, Cow, Beef, Inventory: https://quickstats.nass.usda.gov/#E38831C2-2885-35D4-8A65-A152C5762BB5
# __________________
#    - Q5. Cattle, INCL CALVES, Inventory: https://quickstats.nass.usda.gov/#CBC0AA15-6F87-3A01-9C36-127DE111E760
#    

# %% [markdown]
# ## We should choose "Total":
#  - When Total is not chosen rows are broken down into different categories (1-10 head, 10-20 head).
#  - Other than item above, rows include Total as well. (so, we are concerned with Total anyway).
#  - Sum of all those rows w/ different categories, do not add up to "Total" probably because of "(D)" shit.

# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

import matplotlib
import matplotlib.pyplot as plt

sys.path.append("/Users/hn/Documents/00_GitHub/Rangeland/Python_Codes/")
import rangeland_core as rc

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"

dir_ = data_dir_base + "NASS_downloads/cow_inventory_Qs/"
reOrganized_dir = data_dir_base + "reOrganized/"
param_dir = data_dir_base + "parameters/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"


plot_path = data_dir_base + "plots/census_inven_invest/"
os.makedirs(plot_path, exist_ok=True)

# %% [markdown]
# ### Read Shannon's Beef & Calves

# %%
xl = pd.ExcelFile(Shannon_data_dir + "CATINV.xls")
EX_sheet_names = xl.sheet_names  # see all sheet names
EX_sheet_names = EX_sheet_names[1:]
print (f"{EX_sheet_names = }")

CATINV_df = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)

ii = 0
sheet_name_ = EX_sheet_names[-1]

curr_sheet = pd.read_excel(io = Shannon_data_dir + "CATINV.xls", 
                           sheet_name = sheet_name_, 
                           header = 0, skiprows = 0)
curr_sheet_columns = list(curr_sheet.columns)
named_columns = curr_sheet_columns[0] # [x for x in curr_sheet_columns if not("Unnamed" in x)]
print (f"{named_columns = }")

curr_sheet.columns = list(curr_sheet.iloc[1, ].astype(str))
curr_sheet = curr_sheet[2:].copy()
curr_sheet.rename({'nan': 'state'}, axis=1, inplace=True)
curr_sheet.rename(columns={x: x.replace('.0', '') for x in curr_sheet.columns[1:]}, inplace=True)
curr_sheet.reset_index(drop=True, inplace=True)
curr_sheet.loc[:, curr_sheet.columns[1]:curr_sheet.columns[-1]] = curr_sheet.loc[:, \
                                                curr_sheet.columns[1]:curr_sheet.columns[-1]]*1000

# Drop rows that are entirely NA
curr_sheet.dropna(axis=0, how = 'all', inplace = True)

# Drop rows where state is NA
curr_sheet.dropna(subset=['state'], inplace = True)
shannon_All_Cattle_Calves_CATINV = curr_sheet.copy()
shannon_All_Cattle_Calves_CATINV.tail(4)

# %%
# Q1_P1 = pd.read_csv(dir_ + "Q1_P1.csv", low_memory=False)
# Q1_P2 = pd.read_csv(dir_ + "Q1_P2.csv", low_memory=False)

# Q2_P1 = pd.read_csv(dir_ + "Q2_P1.csv", low_memory=False)
# Q2_P2 = pd.read_csv(dir_ + "Q2_P2.csv", low_memory=False)

# Q1 = pd.concat([Q1_P1, Q1_P2])
# Q2 = pd.concat([Q2_P1, Q2_P2])

# print (Q1.shape)
# Q1 = rc.clean_census(df=Q1, col_="Value")
# Q2 = rc.clean_census(df=Q2, col_="Value")
# print (Q1.shape)
# print ()
# print (f"{Q1.Domain.unique() = }")

# Q1_inv = Q1[(Q1.Domain == "INVENTORY OF BEEF COWS") & (Q1.County == "AUTAUGA") & (Q1.Year == 2017)].copy()
# Q1_total = Q1[(Q1.Domain == "TOTAL") & (Q1.County == "AUTAUGA") & (Q1.Year == 2017)].copy()

# Q1_inv.reset_index(drop=True, inplace=True)
# Q1_total.reset_index(drop=True, inplace=True)

# %%
Q4_beef = pd.read_csv(dir_ + "Q4.csv")
Q5_calves = pd.read_csv(dir_ + "Q5.csv")

# Clean the value column: 
#      toss "(D)" and "(Z)" in column Value, and change type from string to float
#      Change column names to lower case
#      Change state and county names to Title format.
#      Create county_fips as string of 5 character long
Q4_beef = rc.clean_census(df = Q4_beef, col_="Value")
Q5_calves = rc.clean_census(df = Q5_calves, col_="Value")

Q5_calves.state = Q5_calves.state.str.title()
Q5_calves.county = Q5_calves.county.str.title()

Q4_beef.state = Q4_beef.state.str.title()
Q4_beef.county = Q4_beef.county.str.title()

Q5_calves.head(2)

# %%
long_eq = "========================================================================================"
print (f"{Q4_beef.data_item.unique() = }")
print (f"{Q5_calves.data_item.unique() = }")
print (long_eq)
print (f"{Q4_beef.domain.unique() = }")
print (f"{Q5_calves.domain.unique() = }")
print (long_eq)
print (f"{Q4_beef.domain_category.unique() = }")
print (f"{Q5_calves.domain_category.unique() = }")

# %%
Q4_beef.head(2) 

# %%
census_needed_cols = ["year", "state", "county", "county_fips", "data_item", "domain", "value"]

Q4_beef = Q4_beef[census_needed_cols]
Q5_calves = Q5_calves[census_needed_cols]

Q5_calves.head(2)

# %%
Q4_beef.rename(columns={"value": "cattle_cows_beef_invt"}, inplace=True)
Q5_calves.rename(columns={"value": "cattle_inc_calves_invt"}, inplace=True)
Q5_calves.head(2)

# %%
SoI = ["Alabama", "Arkansas", "California", 
       "Colorado", "Florida", "Georgia", "Idaho",
       "Illinois", "Iowa", "Kansas", "Kentucky",
       "Louisiana", "Mississippi", "Missouri", "Montana", 
       "Nebraska", "New Mexico", "North Dakota", 
       "Oklahoma", "Oregon", "South Dakota", "Tennessee",
       "Texas", "Virginia", "Wyoming"]

# %%
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
state_25_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
Shannon_Beef_Cows_fromCATINV = pd.read_csv(reOrganized_dir + "Shannon_Beef_Cows_fromCATINV.csv")
Shannon_Beef_Cows_fromCATINV = Shannon_Beef_Cows_fromCATINV[Shannon_Beef_Cows_fromCATINV.state.isin(state_25_abb)]
Shannon_Beef_Cows_fromCATINV.reset_index(drop=True, inplace=True)
Shannon_Beef_Cows_fromCATINV.head(2)

# %%
shannon_years = [str(x) for x in np.arange(1997, 2018)]
cols = ["state"] + shannon_years
Shannon_Beef_Cows_fromCATINV = Shannon_Beef_Cows_fromCATINV[cols]

shannon_All_Cattle_Calves_CATINV = shannon_All_Cattle_Calves_CATINV[cols]
shannon_All_Cattle_Calves_CATINV = shannon_All_Cattle_Calves_CATINV[\
                                                        shannon_All_Cattle_Calves_CATINV.state.isin(state_25_abb)]

Shannon_Beef_Cows_fromCATINV.head(2)

# %%
Q4_beef.head(2)

# %%
L = ["state", "year", "cattle_cows_beef_invt"]
Q4_beef_state = Q4_beef[L].groupby(["year", "state"]).sum().reset_index()

L = ["state", "year", "cattle_inc_calves_invt"]
Q5_calves_state = Q5_calves[L].groupby(["year", "state"]).sum().reset_index()

Q5_calves_state.head(2)

# %%
USDA_data = pd.read_pickle(reOrganized_dir + "USDA_data.sav")
print(USDA_data.keys())
# AgLand = USDA_data["AgLand"]
# feed_expense = USDA_data["feed_expense"]
# CRPwetLand_area = USDA_data["wetLand_area"]
Q0_bad_invt = USDA_data["cattle_inventory"]
Q0_bad_invt = Q0_bad_invt[Q0_bad_invt.state.isin(SoI)].copy()
Q0_bad_invt = rc.clean_census(df=Q0_bad_invt, col_="cattle_cow_inventory")

Q0_bad_invt.head(2)

# %%
Q0_bad_invt_state = Q0_bad_invt[["state", "year", "cattle_cow_inventory"]]\
                   .groupby(["year", "state"]).sum().reset_index()

Q0_bad_invt_state.year = pd.to_datetime(Q0_bad_invt_state.year, format="%Y")
Q0_bad_invt_state.set_index("year", inplace=True)
Q0_bad_invt_state.sort_index(inplace=True)

Q0_bad_invt_state.head(2)

# %%
size = 10
title_FontSize = 8
legend_FontSize = 8
tick_FontSize = 12
label_FontSize = 14

params = {'legend.fontsize': 9, # medium, large
          # 'figure.figsize': (6, 4),
          'axes.labelsize': size,
          'axes.titlesize': size*.2,
          'xtick.labelsize': size, #  * 0.75
          'ytick.labelsize': size, #  * 0.75
          'axes.titlepad': 10}

#
#  Once set, you cannot change them, unless restart the notebook
#
plt.rc('font', family = 'Palatino')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update(params)

# %%
# sharey='col', # sharex=True, sharey=True,
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=False, gridspec_kw={"hspace": 0.35, "wspace": 0.05})
axs[0].grid(axis="y", which="both")
axs[1].grid(axis="y", which="both")
axs[2].grid(axis="y", which="both")

##########################################################################################
sh_state_, state_ = "TX", "Texas"
##
B = Q5_calves_state[Q5_calves_state.state == state_].copy()
B.year = pd.to_datetime(B.year, format="%Y")
B.set_index("year", inplace=True)
B.sort_index(inplace=True)
axs[0].plot(B.index, B.cattle_inc_calves_invt.values,
            c="red", linewidth=2, label=state_ + " Cattle inc. calves");
del(B)
###
B = Q0_bad_invt_state[Q0_bad_invt_state.state == state_].copy()
B.sort_index(inplace=True)
axs[0].plot(B.index, B.cattle_cow_inventory.values,
            c="g", linewidth=2, label=state_ + " bad one");
del(B)
###
axs[0].plot(pd.to_datetime(shannon_years, format="%Y"),
            Shannon_Beef_Cows_fromCATINV.loc[Shannon_Beef_Cows_fromCATINV.state == sh_state_, 
                                             shannon_years].values[0],
            c="dodgerblue", linewidth=2, label=state_ + " Shannon");

axs[0].plot(pd.to_datetime(shannon_years, format="%Y"),
            shannon_All_Cattle_Calves_CATINV.loc[shannon_All_Cattle_Calves_CATINV.state == sh_state_, 
                                                 shannon_years].values[0],
            c="dodgerblue", linewidth=2, label=state_ + " Shannon (Calves)", linestyle='dashed');
###
B = Q4_beef_state[Q4_beef_state.state == state_].copy()
B.year = pd.to_datetime(B.year, format="%Y")
B.set_index("year", inplace=True)
B.sort_index(inplace=True)
axs[0].plot(B.index, B.cattle_cows_beef_invt.values,
            c="k", linewidth=2, label=state_ + " Cattle beef inv.");
del(B)
##########################################################################################
sh_state_, state_ = "MO", "Missouri"
### Calves
B = Q5_calves_state[Q5_calves_state.state == state_].copy()
B.year = pd.to_datetime(B.year, format="%Y")
B.set_index("year", inplace=True)
B.sort_index(inplace=True)
axs[1].plot(B.index, B.cattle_inc_calves_invt.values,
            c="red", linewidth=2, label=state_ + " Cattle inc. calves");
del(B)
### Bad one
B = Q0_bad_invt_state[Q0_bad_invt_state.state == state_].copy()
B.sort_index(inplace=True)
axs[1].plot(B.index, B.cattle_cow_inventory.values,
            c="g", linewidth=2, label=state_ + " bad one");
del(B)
### Shannon
axs[1].plot(pd.to_datetime(shannon_years, format="%Y"),
            Shannon_Beef_Cows_fromCATINV.loc[Shannon_Beef_Cows_fromCATINV.state == sh_state_, 
                                             shannon_years].values[0],
            c="dodgerblue", linewidth=2, label=state_ + " Shannon");

axs[1].plot(pd.to_datetime(shannon_years, format="%Y"),
            shannon_All_Cattle_Calves_CATINV.loc[shannon_All_Cattle_Calves_CATINV.state == sh_state_, 
                                                 shannon_years].values[0],
            c="dodgerblue", linewidth=2, label=state_ + " Shannon (Calves)", linestyle='dashed');

###
B = Q4_beef_state[Q4_beef_state.state == state_].copy()
B.year = pd.to_datetime(B.year, format="%Y")
B.set_index("year", inplace=True)
B.sort_index(inplace=True)
axs[1].plot(B.index, B.cattle_cows_beef_invt.values,
            c="k", linewidth=2, label=state_ + " Cattle beef inv.");
del(B)
##########################################################################################
sh_state_, state_ = "TN", "Tennessee"
### Calves
B = Q5_calves_state[Q5_calves_state.state == state_].copy()
B.year = pd.to_datetime(B.year, format="%Y")
B.set_index("year", inplace=True)
B.sort_index(inplace=True)
axs[2].plot(B.index, B.cattle_inc_calves_invt.values,
            c="red", linewidth=2, label=state_ + " Cattle inc. calves");
del(B)
### Bad one
B = Q0_bad_invt_state[Q0_bad_invt_state.state == state_].copy()
B.sort_index(inplace=True)
axs[2].plot(B.index, B.cattle_cow_inventory.values,
            c="g", linewidth=2, label=state_ + " bad one");
del(B)
### Shannon
axs[2].plot(pd.to_datetime(shannon_years, format="%Y"),
            Shannon_Beef_Cows_fromCATINV.loc[Shannon_Beef_Cows_fromCATINV.state == sh_state_, 
                                             shannon_years].values[0],
            c="dodgerblue", linewidth=2, label=state_ + " Shannon");

axs[2].plot(pd.to_datetime(shannon_years, format="%Y"),
            shannon_All_Cattle_Calves_CATINV.loc[shannon_All_Cattle_Calves_CATINV.state == sh_state_, 
                                                 shannon_years].values[0],
            c="dodgerblue", linewidth=2, label=state_ + " Shannon (Calves)", linestyle='dashed');
###
B = Q4_beef_state[Q4_beef_state.state == state_].copy()
B.year = pd.to_datetime(B.year, format="%Y")
B.set_index("year", inplace=True)
B.sort_index(inplace=True)
axs[2].plot(B.index, B.cattle_cows_beef_invt.values,
            c="k", linewidth=2, label=state_ + " Cattle beef inv.");
del(B)
####################################################################################
axs[0].legend(loc="best");
axs[1].legend(loc="best");
axs[2].legend(loc="best");

fig_name = plot_path + 'TX_TN_MO_inv.pdf'
plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')

# %%

# %%
# sharey='col', # sharex=True, sharey=True,
fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=False, gridspec_kw={"hspace": 0.35, "wspace": 0.05})
axs.grid(axis="y", which="both")

sh_state_, state_ = "ID", "Idaho"
### Calves
B = Q5_calves_state[Q5_calves_state.state == state_].copy()
B.year = pd.to_datetime(B.year, format="%Y")
B.set_index("year", inplace=True)
B.sort_index(inplace=True)
axs.plot(B.index, B.cattle_inc_calves_invt.values,
            c="red", linewidth=2, label=state_ + " Cattle inc. calves");
del(B)
### Bad one
B = Q0_bad_invt_state[Q0_bad_invt_state.state == state_].copy()
B.sort_index(inplace=True)
axs.plot(B.index, B.cattle_cow_inventory.values,
            c="g", linewidth=2, label=state_ + " bad one");
del(B)
###    Shannon
axs.plot(pd.to_datetime(shannon_years, format="%Y"),
         Shannon_Beef_Cows_fromCATINV.loc[Shannon_Beef_Cows_fromCATINV.state == sh_state_, 
                                          shannon_years].values[0],
            c="dodgerblue", linewidth=2, label=state_ + " Shannon");

axs.plot(pd.to_datetime(shannon_years, format="%Y"),
         shannon_All_Cattle_Calves_CATINV.loc[shannon_All_Cattle_Calves_CATINV.state == sh_state_, 
                                              shannon_years].values[0],
            c="dodgerblue", linewidth=2, label=state_ + " Shannon (Calves)", linestyle='dashed');

###
B = Q4_beef_state[Q4_beef_state.state == state_].copy()
B.year = pd.to_datetime(B.year, format="%Y")
B.set_index("year", inplace=True)
B.sort_index(inplace=True)
axs.plot(B.index, B.cattle_cows_beef_invt.values,
         c="k", linewidth=2, label=state_ + " Cattle beef inv.");
del(B)
###
axs.legend(loc="best");

fig_name = plot_path + 'Idaho_inv.pdf'
plt.savefig(fname = fig_name, dpi=400, bbox_inches='tight')

# %%

# %%

# %%
