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
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"

reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %% [markdown]
# ## We just need sheet A (beef cows) from CATINV

# %%
# CATINV = pd.read_excel(io=param_dir + "CATINV.xlsx", sheet_name=0)
xl = pd.ExcelFile(Shannon_data_dir + "CATINV.xls")
EX_sheet_names = xl.sheet_names  # see all sheet names
EX_sheet_names = EX_sheet_names[1:]
EX_sheet_names

# %%
ii = 0
sheet_name_ = EX_sheet_names[ii]

curr_sheet = pd.read_excel(io=Shannon_data_dir + "CATINV.xls", sheet_name=sheet_name_, header=0, skiprows=0)
curr_sheet_columns = list(curr_sheet.columns)
named_columns = curr_sheet_columns[0]  # [x for x in curr_sheet_columns if not("Unnamed" in x)]
named_columns

# %%
curr_sheet.head(2)

# %%
curr_sheet_columns[:10]

# %%

# %%
curr_sheet.columns = list(curr_sheet.iloc[1,].astype(str))
curr_sheet = curr_sheet[2:].copy()
curr_sheet.rename({"nan": "state"}, axis=1, inplace=True)
curr_sheet.rename(columns={x: x.replace(".0", "") for x in curr_sheet.columns[1:]}, inplace=True)
curr_sheet.reset_index(drop=True, inplace=True)
curr_sheet.loc[:, curr_sheet.columns[1] : curr_sheet.columns[-1]] = (
             curr_sheet.loc[:, curr_sheet.columns[1] : curr_sheet.columns[-1]] * 1000)

# Drop rows that are entirely NA
curr_sheet.dropna(axis=0, how="all", inplace=True)

# Drop rows where state is NA
curr_sheet.dropna(subset=["state"], inplace=True)
Beef_Cows_CATINV = curr_sheet.copy()
# Beef_Cows_CATINV.sort_values(by=["state"], inplace=True)
Beef_Cows_CATINV.tail(4)

# %%
(sorted(Beef_Cows_CATINV.state.unique()))

# %% [markdown]
# ## Drop US

# %%
Beef_Cows_CATINV = Beef_Cows_CATINV[Beef_Cows_CATINV.state != "US"].copy()
Beef_Cows_CATINV.reset_index(drop=True, inplace=True)

# %%
Beef_Cows_CATINV.shape

# %%
Beef_Cows_CATINV.head(2)

# %%
out_name = reOrganized_dir + "Shannon_Beef_Cows_fromCATINV.csv"
Beef_Cows_CATINV.to_csv(out_name, index=False)

# %%
years = list(Beef_Cows_CATINV.columns[1:])
num_years = len(years)

CATINV_df_tall = pd.DataFrame(data=None,
                              index=range(num_years * len(Beef_Cows_CATINV.state.unique())),
                              columns=["state", "year", "inventory"],
                              dtype=None, copy=False)

idx_ = 0
for a_state in Beef_Cows_CATINV.state.unique():
    curr = Beef_Cows_CATINV[Beef_Cows_CATINV.state == a_state]
    CATINV_df_tall.loc[idx_ : idx_ + num_years - 1, "inventory"] = curr[years].values[0]
    CATINV_df_tall.loc[idx_ : idx_ + num_years - 1, "state"] = a_state
    CATINV_df_tall.loc[idx_ : idx_ + num_years - 1, "year"] = years
    idx_ = idx_ + num_years


# %%
CATINV_df_tall.head(5)

# %%
CATINV_df_tall[CATINV_df_tall.state != "US"].tail(5)

# %%
Beef_Cows_CATINV[Beef_Cows_CATINV.state == "WY"]

# %%

# %%
import sys

sys.path.append("/Users/hn/Documents/00_GitHub/Rangeland/Python_Codes/")
import rangeland_core as rc

# county_id_name_fips = pd.read_csv(Min_data_dir_base + "county_id_name_fips.csv")
# county_id_name_fips.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

# county_id_name_fips.sort_values(by=["state", "county"], inplace=True)

# county_id_name_fips = rc.correct_Mins_county_6digitFIPS(df=county_id_name_fips, col_="county")
# county_id_name_fips.rename(columns={"county": "county_fips"}, inplace=True)

# county_id_name_fips["state_fips"] = county_id_name_fips.county_fips.str.slice(0, 2)

# print(len(county_id_name_fips.state.unique()))

# county_id_name_fips = county_id_name_fips.drop(columns=["county_name", "county_fips", "fips"])
# county_id_name_fips.drop_duplicates(inplace=True)
# county_id_name_fips.reset_index(drop=True, inplace=True)
# county_id_name_fips.head(2)

# %%
param_dir = data_dir_base + "parameters/"
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI = abb_dict['SoI']
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

county_id_name_fips = abb_dict ["county_fips"]

county_id_name_fips = county_id_name_fips[["state", "state_fips"]].copy()

county_id_name_fips.drop_duplicates(inplace=True)
# county_id_name_fips = county_id_name_fips[county_id_name_fips.state.isin(SoI_abb)]
county_id_name_fips.reset_index(drop=True, inplace=True)
county_id_name_fips.head(2)

# %%
CATINV_df_tall.head(2)

# %%
CATINV_df_tall = pd.merge(CATINV_df_tall, county_id_name_fips, on=["state"], how="left")
CATINV_df_tall.head(2)

# %%
CATINV_df_tall.year = CATINV_df_tall.year.astype(int)

# %%
CATINV_df_tall.head(2)

# %%
filename = reOrganized_dir + "Shannon_Beef_Cows_fromCATINV_tall.sav"

export_ = {
    "CATINV_annual_tall": CATINV_df_tall,
    "source_code": "convertShannonData",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %% [markdown]
# ### We just need sheet A (beef cows) from ```Annual Cattle Inventory by State.xlsx```

# %%
xl = pd.ExcelFile(Shannon_data_dir + "Annual Cattle Inventory by State.xls")
EX_sheet_names = xl.sheet_names  # see all sheet names
EX_sheet_names = EX_sheet_names[1:]
EX_sheet_names

# %%
ii = 0
sheet_name_ = EX_sheet_names[ii]

curr_sheet = pd.read_excel(
    io=Shannon_data_dir + "Annual Cattle Inventory by State.xls",
    sheet_name=sheet_name_,
    header=0,
    skiprows=0,
)
curr_sheet_columns = list(curr_sheet.columns)
named_columns = curr_sheet_columns[
    0
]  # [x for x in curr_sheet_columns if not("Unnamed" in x)]

print(f"{named_columns=}")
curr_sheet.head(4)

# %%
curr_sheet.columns = list(curr_sheet.iloc[1,].astype(str))
curr_sheet = curr_sheet[2:].copy()

curr_sheet.rename({"nan": "state"}, axis=1, inplace=True)
curr_sheet.rename(
    columns={x: x.replace(".0", "") for x in curr_sheet.columns[1:]}, inplace=True
)
curr_sheet.reset_index(drop=True, inplace=True)

curr_sheet.loc[:, curr_sheet.columns[1] : curr_sheet.columns[-1]] = (
    curr_sheet.loc[:, curr_sheet.columns[1] : curr_sheet.columns[-1]] * 1000
)


# Drop rows that are entirely NA
curr_sheet.dropna(axis=0, how="all", inplace=True)

# Drop rows where state is NA
curr_sheet.dropna(subset=["state"], inplace=True)
Beef_Cows_annual = curr_sheet.copy()
Beef_Cows_annual.head(2)

# %%
Beef_Cows_annual.tail(4)

# %%
out_name = reOrganized_dir + "Beef_Cows_fromAnnualCattleInventorybyState.csv"
Beef_Cows_CATINV.to_csv(out_name, index=False)

# %%
print(f"{Beef_Cows_CATINV.shape=}")
print(f"{Beef_Cows_annual.shape=}")

# %%
Beef_Cows_CATINV.head(4)

# %%
Beef_Cows_annual.head(4)

# %%
Beef_Cows_annual.loc[:, "1920":"2020"].equals(Beef_Cows_CATINV.loc[:, "1920":"2020"])

# %%
Beef_Cows_annual.loc[:, "2021"] - (Beef_Cows_CATINV.loc[:, "2021"])

# %%

# %% [markdown]
# ## We just need sheet B (beef cows) from Weekly Regional Cow Slaughter

# %%
file_ = "Weekly Regional Cow Slaughter.xls"
xl = pd.ExcelFile(Shannon_data_dir + file_)
EX_sheet_names = xl.sheet_names  # see all sheet names
EX_sheet_names = EX_sheet_names[1:]
EX_sheet_names

# %%
ii = 0
sheet_name_ = EX_sheet_names[ii]

curr_sheet = pd.read_excel(
    io=Shannon_data_dir + file_, sheet_name=sheet_name_, header=0, skiprows=4
)
curr_sheet_columns = list(curr_sheet.columns)
curr_sheet.head(7)

# %%
curr_sheet.loc[1,] = curr_sheet.loc[0,] + curr_sheet.loc[1,]
curr_sheet = curr_sheet.loc[1:,].copy()
curr_sheet.reset_index(drop=True, inplace=True)
curr_sheet.drop(axis=1, index=1, inplace=True)
curr_sheet.reset_index(drop=True, inplace=True)

for a_col in list(curr_sheet.columns):
    if not ("Unnamed" in a_col):
        curr_index = list(curr_sheet.columns).index(a_col)
        new_part = (
            a_col.replace(".1", "")
            .replace("- ", "")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        curr_sheet.iloc[0, curr_index] = (
            new_part + "_" + curr_sheet.iloc[0, curr_index].replace(" ", "")
        )
        curr_sheet.iloc[0, curr_index + 1] = (
            new_part + "_" + curr_sheet.iloc[0, curr_index + 1].replace(" ", "")
        )

curr_sheet.iloc[0, 0] = "date"
curr_sheet.iloc[0, 1] = "week"
curr_sheet.rename(columns=curr_sheet.iloc[0], inplace=True)

# Drop first row
curr_sheet.drop(axis=1, index=0, inplace=True)
curr_sheet.reset_index(drop=True, inplace=True)

curr_sheet.head(7)

# %%
curr_sheet["Region_1_&_Region_2_beef"] = (
    curr_sheet["Region_1_&_Region_2_Beef&dairy"]
    - curr_sheet["Region_1_&_Region_2_dairy"]
)

for ii in range(3, 11):
    curr_sheet["Region_" + str(ii) + "_beef"] = (
        curr_sheet["Region_" + str(ii) + "_Beef&dairy"]
        - curr_sheet["Region_" + str(ii) + "_dairy"]
    )

# %%
out_name = reOrganized_dir + "Beef_Cows_fromWeeklyRegionalCowSlaughter.csv"
curr_sheet.to_csv(out_name, index=False)

# %%

# %%

# %%

# %%

# %%

# %%
