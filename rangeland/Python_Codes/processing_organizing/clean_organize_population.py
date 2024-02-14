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

# %%
import pandas as pd
import numpy as np
import os

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
NASS_dir = data_dir_base + "NASS_downloads/"
census_dir = NASS_dir + "census/"
reOrganized_dir = data_dir_base + "reOrganized/"
param_dir = data_dir_base + "parameters/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
pop_1990_1999_file = "CO-99-10.txt"
pop_2000_2010_file = "z_2000_2010_co-est00int-tot.csv"
pop_2010_2020_file = "z_2010-2020-co-est2020.csv"
pop_2000_file = "z_2000_2009_co-est2009-alldata.csv"

# %%
# encoding can be unicode_escape too.
pop_2000 = pd.read_csv(census_dir + pop_2000_file, encoding='latin-1')
pop_2000_2010 = pd.read_csv(census_dir + pop_2000_2010_file, encoding='latin-1')
pop_2010_2020 = pd.read_csv(census_dir + pop_2010_2020_file, encoding='latin-1')

pop_2000.drop(["SUMLEV", "REGION", "DIVISION" ], axis=1, inplace=True)
pop_2000_2010.drop(["SUMLEV", "REGION", "DIVISION" ], axis=1, inplace=True)
pop_2010_2020.drop(["SUMLEV", "REGION", "DIVISION"], axis=1, inplace=True)


pop_2000.rename(
         columns={ "STATE": "state_fip", "COUNTY": "cnty_fip", "STNAME": "state", "CTYNAME": "county"},
        inplace=True)

pop_2000_2010.rename(
         columns={ "STATE": "state_fip", "COUNTY": "cnty_fip", "STNAME": "state", "CTYNAME": "county"},
        inplace=True)

pop_2010_2020.rename(
         columns={ "STATE": "state_fip", "COUNTY": "cnty_fip", "STNAME": "state", "CTYNAME": "county"},
        inplace=True)

pop_2000.head(2)

# %%
pop_2000.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
pop_2000_2010.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
pop_2010_2020.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
pop_2000.head(2)

# %%
pop_2000.rename({'stname': 'state', 'ctyname':'county'}, axis=1, inplace=True)
pop_2000_2010.rename({'stname': 'state', 'ctyname':'county'}, axis=1, inplace=True)
pop_2010_2020.rename({'stname': 'state', 'ctyname':'county'}, axis=1, inplace=True)
pop_2000_2010.head(2)

# %%
pop_2000_2010.state_fip = pop_2000_2010.state_fip.astype(str)
pop_2000_2010.cnty_fip = pop_2000_2010.cnty_fip.astype(str)

for idx in pop_2000_2010.index:
    if len(pop_2000_2010.loc[idx, "state_fip"]) == 1:
        pop_2000_2010.loc[idx, "state_fip"] = "0" + pop_2000_2010.loc[idx, "state_fip"]

    col = "cnty_fip"
    if len(pop_2000_2010.loc[idx, col]) == 1:
        pop_2000_2010.loc[idx, col] = "00" + pop_2000_2010.loc[idx, col]
    elif len(pop_2000_2010.loc[idx, col]) == 2:
        pop_2000_2010.loc[idx, col] = "0" + pop_2000_2010.loc[idx, col]

pop_2000_2010.head(2)

# %%
pop_2010_2020.state_fip = pop_2010_2020.state_fip.astype(str)
pop_2010_2020.cnty_fip = pop_2010_2020.cnty_fip.astype(str)

for idx in pop_2010_2020.index:
    if len(pop_2010_2020.loc[idx, "state_fip"]) == 1:
        pop_2010_2020.loc[idx, "state_fip"] = "0" + pop_2010_2020.loc[idx, "state_fip"]

    col = "cnty_fip"
    if len(pop_2010_2020.loc[idx, col]) == 1:
        pop_2010_2020.loc[idx, col] = "00" + pop_2010_2020.loc[idx, col]
    elif len(pop_2010_2020.loc[idx, col]) == 2:
        pop_2010_2020.loc[idx, col] = "0" + pop_2010_2020.loc[idx, col]

pop_2010_2020.head(2)

# %% [markdown]
# ### ```cnty_fip == 000```
#
# presentes all state. i.e. sum of all counties. Drop them.

# %%
pop_2000_2010.drop(pop_2000_2010[pop_2000_2010.cnty_fip == "000"].index, inplace=True)
pop_2010_2020.drop(pop_2010_2020[pop_2010_2020.cnty_fip == "000"].index, inplace=True)

# %%
pop_2000_2010["county_fips"] = pop_2000_2010["state_fip"] + pop_2000_2010["cnty_fip"]
pop_2010_2020["county_fips"] = pop_2010_2020["state_fip"] + pop_2010_2020["cnty_fip"]

pop_2000_2010.drop(["state_fip", "cnty_fip"], axis=1, inplace=True)
pop_2010_2020.drop(["state_fip", "cnty_fip"], axis=1, inplace=True)

pop_2000_2010.head(2)

# %%
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI = abb_dict['SoI']
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
pop_2000_2010 = pop_2000_2010[pop_2000_2010.state.isin(SoI)]
pop_2010_2020 = pop_2010_2020[pop_2010_2020.state.isin(SoI)]

pop_2000_2010.reset_index(drop=True, inplace=True)
pop_2010_2020.reset_index(drop=True, inplace=True)

# %%
need_cols_2000_2010 = ["county_fips", "popestimate2002", "popestimate2007"]
need_cols_2010_2020 = ["county_fips", "popestimate2012", "popestimate2017"]

pop_2000_2010 = pop_2000_2010[need_cols_2000_2010]
pop_2010_2020 = pop_2010_2020[need_cols_2010_2020]

# %%
pop_2000_2010.head(2)

# %%
print(f"{len(pop_2000_2010) = }")
print(f"{len(pop_2010_2020) = }")

# %%
A = [x for x in pop_2010_2020.county_fips if not (x in list(pop_2000_2010.county_fips))]
A

# %%
A = [x for x in pop_2000_2010.county_fips if not (x in list(pop_2010_2020.county_fips))]
A

# %%
L = len(pop_2000_2010) + len(pop_2010_2020)
pop_wide = pd.DataFrame(columns=["county_fips", "year", "population"], index=range(L * 2))

pop_wide.county_fips = "666"
pop_wide.year = 666
pop_wide.population = 666

years_2000_2010 = [2002, 2007] * len(pop_2000_2010)
years_2010_2020 = [2012, 2017] * len(pop_2010_2020)
pop_wide.year = years_2000_2010 + years_2010_2020

pop_wide.head(5)

wide_pointer = 0

for idx in pop_2000_2010.index:
    pop_wide.loc[wide_pointer, "county_fips"] = pop_2000_2010.loc[idx, "county_fips"]
    pop_wide.loc[wide_pointer, "population"] = pop_2000_2010.loc[idx, "popestimate2002"]

    pop_wide.loc[wide_pointer + 1, "county_fips"] = pop_2000_2010.loc[idx, "county_fips"]
    pop_wide.loc[wide_pointer + 1, "population"] = pop_2000_2010.loc[idx, "popestimate2007"]
    wide_pointer += 2

for idx in pop_2010_2020.index:
    pop_wide.loc[wide_pointer, "county_fips"] = pop_2010_2020.loc[idx, "county_fips"]
    pop_wide.loc[wide_pointer, "population"] = pop_2010_2020.loc[idx, "popestimate2012"]

    pop_wide.loc[wide_pointer + 1, "county_fips"] = pop_2010_2020.loc[idx, "county_fips"]
    pop_wide.loc[wide_pointer + 1, "population"] = pop_2010_2020.loc[idx, "popestimate2017"]
    wide_pointer += 2
    
pop_wide.head(2)

# %%
pop_wide.sort_values(["county_fips", "year"], inplace=True)
pop_wide.reset_index(drop=True, inplace=True)
pop_wide.head(10)

# %%

# %%
# population_1990_1999 = pd.read_csv(f'/Users/hn/Documents/01_research_data/RangeLand/Data/' + \
#                                                        'census/CO-99-10.txt')

pop_1990_1999 = pd.read_csv(census_dir + pop_1990_1999_file, 
                                   header = 12, sep = "\t", on_bad_lines = 'skip',
                                   encoding = 'unicode_escape')

print ()
print (f"{pop_1990_1999.shape=}")
print ()
pop_1990_1999.head(5)

# %%
cols = list(pop_1990_1999.loc[0])[0].split(" ")
cols = [x.lower() for x in cols if x!=""]
cols
# [x for x in list(population_1990_1999.loc[1])[0].split(" ") if x!=""]

# %%
pop_1990_1999_clean = pd.DataFrame(columns = cols, index=range(len(pop_1990_1999)))

for a_idx in pop_1990_1999.index:
    if a_idx==0:
        pass
    else:
        curr_row = [x for x in list(pop_1990_1999.loc[a_idx])[0].split(" ") if x!=""]
        pop_1990_1999_clean.loc[a_idx] = curr_row

pop_1990_1999_clean.dropna(inplace=True)
pop_1990_1999_clean.reset_index(drop=True, inplace=True)

for a_col in pop_1990_1999_clean.columns:
    pop_1990_1999_clean[a_col] = pop_1990_1999_clean[a_col].astype(int)

# %%
pop_1990_1999 = pop_1990_1999_clean.copy()
pop_1990_1999.head(2)

# %%
Supriya_Min_FIPS = pd.read_csv(param_dir + "Supriya_Min_FIPS.csv")
Supriya_Min_FIPS.sort_values("fips", inplace=True)
Supriya_Min_FIPS.rename({'county_name': 'county_state'}, axis=1, inplace=True)
Supriya_Min_FIPS.shape

# %%
print (f"{len(Supriya_Min_FIPS.fips.unique())=}")
print (f"{len(pop_1990_1999.fips.unique())=}")

# %%
# pop_1990_1999 = pd.merge(pop_1990_1999, Supriya_Min_FIPS, on=['fips'], how='left')
pop_1990_1999["population"] = pop_1990_1999.iloc[:, 2:9].sum(axis=1)
pop_1990_1999 = pop_1990_1999[["year", "fips", "population"]].copy()

pop_1990_1999.rename({'fips': 'county_fips'}, axis=1, inplace=True)

pop_wide.sort_values(["county_fips", "year"], inplace=True)

pop_1990_1999.county_fips = pop_1990_1999.county_fips.astype(str)
pop_1990_1999.head(2)

# %%
for idx in pop_1990_1999.index:
    if len(pop_1990_1999.loc[idx, "county_fips"]) == 4:
        pop_1990_1999.loc[idx, "county_fips"] = "0" + pop_1990_1999.loc[idx, "county_fips"]

pop_1990_1999.head(2)

# %%
pop_1997 = pop_1990_1999[pop_1990_1999.year==1997].copy()
pop_1997.reset_index(drop=True, inplace=True)

pop_1997.head(5)

# %%
pop_wide.head(3)

# %%
print(len(pop_wide.county_fips.unique()))

# %%
pop_wide = pd.concat([pop_wide, pop_1997])
pop_wide.sort_values(["county_fips", "year"], inplace=True)
pop_wide.reset_index(drop=True, inplace=True)

pop_wide.head(10)

# %%
print(len(pop_wide.county_fips.unique()))

# %%
import pickle
from datetime import datetime

filename = reOrganized_dir + "human_population.sav"

export_ = {"human_population": pop_wide, 
           "source_code" : "clean_organize_population",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %% [markdown]
# # Reshape other population DFs

# %%

# %%
# pop_2000.rename({'census2000pop': 'population'}, axis=1, inplace=True)
# pop_2000["year"] = 2000
# pop_2000 = pop_2000[["state", "county", "year", "population"]]
# pop_2000.head(2)

# st_year = 2001
# end_year = 2010
# cols = [i + str(j) for i, j in zip(["popestimate"]* len(range(st_year, end_year)), range(st_year, end_year))]
# cols = ["state", "county"] + cols + ["census2010pop"]
# pop_2000_2010 = pop_2000_2010[cols]

# %%

# %%

# %%

# %%
