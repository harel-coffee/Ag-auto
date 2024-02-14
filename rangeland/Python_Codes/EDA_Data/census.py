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
census_dir = data_dir_base + "census/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
# with open(NASS_dir+"2017_cdqt_data.txt") as f:
#     contents = f.readlines()

# f = open(NASS_dir+'2017_cdqt_data.txt','r')

# %%
# cdqt_data_2017 = pd.read_csv(NASS_dir + "2017_cdqt_data.txt", header=0, sep="	", on_bad_lines='skip')
# cdqt_data_2017

# cdqt_data_2017 = pd.read_csv(NASS_dir + "2017_cdqt_data.txt", header=0, sep=" ", on_bad_lines='skip')
# cdqt_data_2017

# %%

# %%
FarmOperation = pd.read_csv("/Users/hn/Documents/01_research_data/RangeLand/Data/NASS_downloads/FarmOperation.csv")


# %%

# %%
FarmOperation_BLACK_BELT = FarmOperation[FarmOperation["Ag District"]=="BLACK BELT"]
FarmOperation_BLACK_BELT.head(3)

# %%
population_1990_1999_file = "CO-99-10.txt"
population_2000_2010_file = "z_2000_2010_co-est00int-tot.csv"
population_2010_2020_file = "z_2010-2020-co-est2020.csv"
population_2000_file = "z_2000_2009_co-est2009-alldata.csv"

# %%
population_2000 = pd.read_csv(census_dir + population_2000_file, encoding='latin-1')
population_2000_2010 = pd.read_csv(census_dir + population_2000_2010_file, encoding='latin-1')
population_2010_2020 = pd.read_csv(census_dir + population_2010_2020_file, encoding='latin-1')
population_2000.head(2)

# %%
population_2000.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
population_2000_2010.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
population_2010_2020.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
population_2000.head(2)

# %%
population_2000.rename({'state': 'state_fips', 'county':'county_fips'}, axis=1, inplace=True)
population_2000_2010.rename({'state': 'state_fips', 'county':'county_fips'}, axis=1, inplace=True)
population_2010_2020.rename({'state': 'state_fips', 'county':'county_fips'}, axis=1, inplace=True)

# %%
population_2000.rename({'stname': 'state', 'ctyname':'county'}, axis=1, inplace=True)
population_2000_2010.rename({'stname': 'state', 'ctyname':'county'}, axis=1, inplace=True)
population_2010_2020.rename({'stname': 'state', 'ctyname':'county'}, axis=1, inplace=True)
population_2000_2010.head(2)

# %%

# %%
# population_1990_1999 = pd.read_csv(f'/Users/hn/Documents/01_research_data/RangeLand/Data/' + \
#                                                        'census/CO-99-10.txt')

population_1990_1999 = pd.read_csv(census_dir + population_1990_1999_file, 
                                   header = 12, sep = "\t", on_bad_lines = 'skip',
                                   encoding = 'unicode_escape')
population_1990_1999

# %%
cols = list(population_1990_1999.loc[0])[0].split(" ")
cols = [x.lower() for x in cols if x!=""]
cols
# [x for x in list(population_1990_1999.loc[1])[0].split(" ") if x!=""]

# %%
population_1990_1999_clean = pd.DataFrame(columns = cols, index=range(len(population_1990_1999)))

for a_idx in population_1990_1999.index:
    if a_idx==0:
        pass
    else:
        curr_row = [x for x in list(population_1990_1999.loc[a_idx])[0].split(" ") if x!=""]
        population_1990_1999_clean.loc[a_idx] = curr_row

population_1990_1999_clean.dropna(inplace=True)
population_1990_1999_clean.reset_index(drop=True, inplace=True)

for a_col in population_1990_1999_clean.columns:
    population_1990_1999_clean[a_col] = population_1990_1999_clean[a_col].astype(int)

# %%
population_1990_1999 = population_1990_1999_clean.copy()
population_1990_1999.head(2)

# %%
Supriya_Min_FIPS = pd.read_csv(data_dir_base + "Supriya_Min_FIPS.csv")
Supriya_Min_FIPS.sort_values("fips", inplace=True)
Supriya_Min_FIPS.rename({'county_name': 'county_state'}, axis=1, inplace=True)
Supriya_Min_FIPS.shape

# %%
print (f"{len(Supriya_Min_FIPS.fips.unique())=}")
print (f"{len(population_1990_1999.fips.unique())=}")

# %%
population_1990_1999 = pd.merge(population_1990_1999, Supriya_Min_FIPS, on=['fips'], how='left')
population_1990_1999["population"] = population_1990_1999.iloc[:, 2:9].sum(axis=1)
population_1990_1999 = population_1990_1999[["year", "fips", "state", "county", "county_state", "population"]].copy()
population_1990_1999.head(2)

# %% [markdown]
# # Reshape other population DFs

# %%
population_2000.head(2)

# %%
population_2000.rename({'census2000pop': 'population'}, axis=1, inplace=True)
population_2000["year"] = 2000
population_2000 = population_2000[["state", "county", "year", "population"]]
population_2000.head(2)

# %% [markdown]
# # 2000 - 2010

# %%
population_2000_2010.head(2)

# %%
st_year = 2001
end_year = 2010
cols = [i + str(j) for i, j in zip(["popestimate"]* len(range(st_year, end_year)), range(st_year, end_year))]
cols = ["state", "county"] + cols + ["census2010pop"]
population_2000_2010 = population_2000_2010[cols]

# %%
population_2000_2010.head(2)

# %%
population_2000_2010_V = pd.DataFrame(columns = ["state", "county", "year", "population"],
                                      index=range(len(population_2000_2010) * len(range(st_year, end_year+1)))
                                     )

no_pop_columns = len(population_2000_2010.columns) - 2
vertical_pointer = 0

for a_idx in population_2000_2010.index:
    curr_row = population_2000_2010.loc[a_idx]
    curr_state = curr_row.state
    curr_county = curr_row.county
    pops_vect = list(curr_row.values[2:])
    
    population_2000_2010_V.loc[vertical_pointer:vertical_pointer+no_pop_columns-1, 'year'] = \
                                                                   list(range(st_year, end_year+1))

    population_2000_2010_V.loc[vertical_pointer:vertical_pointer+no_pop_columns-1, 'population'] = pops_vect
    population_2000_2010_V.loc[vertical_pointer:vertical_pointer+no_pop_columns-1, 'county'] = curr_county
    population_2000_2010_V.loc[vertical_pointer:vertical_pointer+no_pop_columns-1, 'state'] = curr_state
    
    vertical_pointer = vertical_pointer + len(range(st_year, end_year+1))
    
population_2000_2010_V.head(2)

# %%
population_2000_2010.head(2)

# %% [markdown]
# # 2010 - 2020

# %%
population_2010_2020.drop(labels=["sumlev", "region", "division"], axis="columns", inplace=True)
population_2010_2020.head(2)

# %%
st_year = 2011
end_year = 2021
cols = [i + str(j) for i, j in zip(["popestimate"]* len(range(st_year, end_year)), range(st_year, end_year))]
cols = ["state", "county"] + cols
population_2010_2020 = population_2010_2020[cols]
population_2010_2020.head(2)

# %%

# %%
