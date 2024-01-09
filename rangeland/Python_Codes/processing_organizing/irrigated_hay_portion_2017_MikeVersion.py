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
# # Jan 5, 2024
#
# Mike and I had a meeting and he shared his ```R``` code to get irrigated hay areas from full dataset of CENSUS.
#
# The census link is [here](https://www.nass.usda.gov/Publications/AgCensus/2017/index.php).
#
# I am converting it to Python.

# %%
import pandas as pd
import numpy as np
import os

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_dir_base = data_dir_base + "Min_Data/"
NASS_dir = data_dir_base + "/NASS_downloads/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %%
import sys
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

# %%
# import gzip

# file_full = NASS_dir + "2017_cdqt_data.txt.gz"
# with gzip.open(file_full,'rt') as f:
#     for line in f:
#         print('got line', line)
        
        
# import gzip
# with gzip.open(file_full, 'rb') as f:
#     file_content = f.read()

# %%
file_full = NASS_dir + "2017_cdqt_data.txt.gz"

# %%
# deleting [quotechar='"'] does not make any difference
census_df = pd.read_csv(file_full, compression='gzip', header=0, sep='\t', quotechar='"', low_memory=False)
census_df.head(2)

# %%
census_df = census_df[census_df.AGG_LEVEL_DESC == "COUNTY"]
census_df.reset_index(drop=True, inplace=True)

print (f"{census_df.shape = }")
census_df.head(2)

# %%
# %%time
# d <- d %>% unite(fips, STATE_FIPS_CODE, COUNTY_CODE, sep="", na.rm=FALSE)

census_df = rc.census_stateCntyAnsi_2_countyFips(df=census_df, 
                                                 state_fip_col="STATE_FIPS_CODE", 
                                                 county_fip_col="COUNTY_CODE")

census_df.head(2)

# %%
census_df = census_df[census_df.SECTOR_DESC == "CROPS"]
census_df.dropna(subset=['VALUE'], inplace=True)
census_df.reset_index(drop=True, inplace=True)

# %%
census_df_25 = census_df[census_df.CENSUS_TABLE == 25].copy()

# field <- field[field$SHORT_DESC %like% "ACRES HARVESTED",]
census_df_25 = census_df_25[census_df_25.SHORT_DESC.str.contains(pat="ACRES HARVESTED", case=False)].copy()

census_df_25.reset_index(drop=True, inplace=True)

# %%
print (f"{census_df.shape = }")
census_df_25 = census_df_25[~(census_df_25.SHORT_DESC.str.contains(pat="SPRING", case=False))].copy()
census_df_25 = census_df_25[~(census_df_25.SHORT_DESC.str.contains(pat="WINTER", case=False))].copy()
census_df_25 = census_df_25[~(census_df_25.SHORT_DESC.str.contains(pat="PIMA", case=False))].copy()
census_df_25 = census_df_25[~(census_df_25.SHORT_DESC.str.contains(pat="UPLAND", case=False))].copy()
census_df_25 = census_df_25[~(census_df_25.SHORT_DESC.str.contains(pat="OIL TYPE", case=False))].copy()
census_df_25 = census_df_25[~(census_df_25.SHORT_DESC.str.contains(pat="NON-OIL TYPE", case=False))].copy()
print (f"{census_df.shape = }")

# %%
# field <- field %>% filter(str_detect(SHORT_DESC, "IRRIGAT", negate=TRUE))
field_25     = census_df_25[~(census_df_25.SHORT_DESC.str.contains(pat="IRRIGAT", case=False))].copy()
field_25_irr = census_df_25[  census_df_25.SHORT_DESC.str.contains(pat="IRRIGAT", case=False)].copy()

field_25.reset_index(drop=True, inplace=True)
field_25_irr.reset_index(drop=True, inplace=True)

# %%
# field <- field %>% filter(str_detect(SHORT_DESC, "IRRIGAT", negate=TRUE))

# %%
field_25.head(2)

# %%
field_25_irr.rename(columns={"VALUE" : "VALUE_irr"}, inplace=True)
field_25_irr.rename(columns={"SHORT_DESC" : "SHORT_DESC_irr"}, inplace=True)
field_25_irr.head(2)

# %%
# field <- field %>% select(CENSUS_CHAPTER, CENSUS_TABLE, STATE_ALPHA, CENSUS_ROW, fips, SHORT_DESC, VALUE)
# field_i <- field_i %>% select(CENSUS_CHAPTER, CENSUS_TABLE, CENSUS_ROW, fips, SHORT_DESC, VALUE_I)

s_col = ["CENSUS_CHAPTER", "CENSUS_TABLE", "CENSUS_ROW", "county_fips", "SHORT_DESC", "VALUE"]
field_25 = field_25[s_col]

s_col = ["CENSUS_CHAPTER", "CENSUS_TABLE", "CENSUS_ROW", "county_fips", "SHORT_DESC_irr", "VALUE_irr"]
field_25_irr = field_25_irr[s_col]
field_25_irr.head(2)

# %%
field_25.head(2)

# %%
field_25_irr_nonIrr = pd.merge(field_25, field_25_irr, 
                               on=list(field_25_irr.columns)[:-2], how="left")

field_25_irr_nonIrr.head(2)

# %%
# field_merge2 <- field_merge %>% filter(str_detect(VALUE, "(D)", negate=TRUE))
print (field_25_irr_nonIrr.shape)
field_25_irr_nonIrr = field_25_irr_nonIrr[field_25_irr_nonIrr.VALUE != "(D)"].copy()
print (field_25_irr_nonIrr.shape)

# %%
field_25_irr_nonIrr["VALUE"]     = field_25_irr_nonIrr["VALUE"].str.replace(",", "")
field_25_irr_nonIrr["VALUE_irr"] = field_25_irr_nonIrr["VALUE_irr"].str.replace(",", "")

# %%
field_25_irr_nonIrr.head(2)

# %%
field_25_irr_nonIrr["VALUE"] = field_25_irr_nonIrr["VALUE"].astype(float)
field_25_irr_nonIrr["VALUE_irr"] = field_25_irr_nonIrr["VALUE_irr"].astype(float)
field_25_irr_nonIrr.head(2)

# %%
field_25_irr_nonIrr.rename(columns={"VALUE_irr" : "harvest"}, inplace=True)
field_25_irr_nonIrr.head(2)

# %%
field_25_irr_nonIrr.harvest.unique()

# %%
arg1 = "10"
arg1 = int(arg1)
df = pd.DataFrame(index=range(5))
df["arg1"] = arg1
df["exponents"] = 0

for counter in df.index:
    if counter == 0:
        print(
            f"This line will be printed in the file {error_file_directory} which is defined in template.sh!"
        )
        print(f"{counter = :}")
    df.loc[counter, "exponents"] = arg1**counter

# %%
df

# %%

# %%
