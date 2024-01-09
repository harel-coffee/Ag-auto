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
# %%time
# deleting [quotechar='"'] does not make any difference
census_df = pd.read_csv(file_full, compression='gzip', header=0, sep='\t', quotechar='"', low_memory=False)
census_df.head(2)

# %%
census_df = census_df[census_df.AGG_LEVEL_DESC == "COUNTY"]
census_df.reset_index(drop=True, inplace=True)

print (f"{census_df.shape = }")
census_df.head(2)

# %%
# # %%time
# # d <- d %>% unite(fips, STATE_FIPS_CODE, COUNTY_CODE, sep="", na.rm=FALSE)

# census_df = rc.census_stateCntyAnsi_2_countyFips(df=census_df, 
#                                                  state_fip_col="STATE_FIPS_CODE", 
#                                                  county_fip_col="COUNTY_CODE")

# census_df.head(2) # took 5 min on iMac. 1.5 min MacBook

# import pickle
# from datetime import datetime

# filename = NASS_dir + "census_df_irr_hay_cell4.sav"

# export_ = {"census_df": census_df, 
#            "source_code" : "irrigated_hay_portion_2017",
#            "Author": "HN",
#            "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# pickle.dump(export_, open(filename, 'wb'))

# %%

# %%
census_df = pd.read_pickle(NASS_dir + "census_df_irr_hay_cell4.sav")
census_df = census_df["census_df"]

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
print (f"{census_df_25.shape = }")
census_df_25 = census_df_25[~(census_df_25.SHORT_DESC.str.contains(pat="SPRING", case=False))].copy()
census_df_25 = census_df_25[~(census_df_25.SHORT_DESC.str.contains(pat="WINTER", case=False))].copy()
census_df_25 = census_df_25[~(census_df_25.SHORT_DESC.str.contains(pat="PIMA", case=False))].copy()
census_df_25 = census_df_25[~(census_df_25.SHORT_DESC.str.contains(pat="UPLAND", case=False))].copy()
census_df_25 = census_df_25[~(census_df_25.SHORT_DESC.str.contains(pat="OIL TYPE", case=False))].copy()
census_df_25 = census_df_25[~(census_df_25.SHORT_DESC.str.contains(pat="NON-OIL TYPE", case=False))].copy()
print (f"{census_df_25.shape = }")

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
field_25_irr_nonIrr = pd.merge(field_25, field_25_irr, on=list(field_25_irr.columns)[:-2], how="outer")
field_25_irr_nonIrr.head(2)

# %%
field_25_irr_nonIrr.shape

# %%
# field_merge2 <- field_merge %>% filter(str_detect(VALUE, "(D)", negate=TRUE))
print (field_25_irr_nonIrr.shape)
# field_25_irr_nonIrr = field_25_irr_nonIrr[field_25_irr_nonIrr.VALUE != "(D)"].copy()
field_25_irr_nonIrr = field_25_irr_nonIrr[~(field_25_irr_nonIrr["VALUE"].str.contains(pat="(D)", 
                                                                                      case=False,
                                                                                      regex = False))]
print (field_25_irr_nonIrr.shape)

# %%

# %%
# Michael does not have this: 
# He has imputed the irrigated values
field_25_irr_nonIrr.dropna(subset=['VALUE_irr'], inplace=True)
field_25_irr_nonIrr = field_25_irr_nonIrr[~(field_25_irr_nonIrr["VALUE_irr"].str.contains(pat="(D)", 
                                                                                          case=False,
                                                                                          regex = False))]
print (field_25_irr_nonIrr.shape)

# %%
field_25_irr_nonIrr["VALUE"]     = field_25_irr_nonIrr["VALUE"].str.replace(",", "")
field_25_irr_nonIrr["VALUE_irr"] = field_25_irr_nonIrr["VALUE_irr"].str.replace(",", "")

field_25_irr_nonIrr["VALUE"] = field_25_irr_nonIrr["VALUE"].astype(float)
field_25_irr_nonIrr["VALUE_irr"] = field_25_irr_nonIrr["VALUE_irr"].astype(float)

field_25_irr_nonIrr.head(2)

# %%
# field_25_irr_nonIrr.rename(columns={"VALUE_irr" : "harvest"}, inplace=True)
# field_25_irr_nonIrr.harvest.unique()
# field_25_irr_nonIrr.head(2)

# %%
field_25_irr_nonIrr.SHORT_DESC.unique()

# %% [markdown]
# # We just need Hay. Forget about Field, Veg... etc.?

# %%
census_df[census_df.SHORT_DESC.str.contains(pat="hay", case=False)].SHORT_DESC.unique()

# %%
hay = census_df[census_df.SHORT_DESC.str.contains(pat="hay", case=False)].copy()
hay.shape

# %%
hay = hay[~(hay["VALUE"].str.contains(pat="(D)", case=False, regex = True))]
hay.shape

# %%
hay.CENSUS_TABLE.unique()

# %%
print (hay[hay.CENSUS_TABLE == 1].shape)
hay[hay.CENSUS_TABLE == 1].SHORT_DESC.unique()

# %%
print (hay[hay.CENSUS_TABLE == 2].shape)
hay[hay.CENSUS_TABLE == 2].SHORT_DESC.unique()

# %%
print (hay[hay.CENSUS_TABLE == 24].shape)
hay[hay.CENSUS_TABLE == 24].SHORT_DESC.unique()

# %%
print (hay[hay.CENSUS_TABLE == 26].shape)
hay[hay.CENSUS_TABLE == 26].SHORT_DESC.unique()

# %%
hay_irr = hay[hay.SHORT_DESC.str.contains(pat="IRRIG", case=False)].copy()
hay_nonIrr = hay[~(hay.SHORT_DESC.str.contains(pat="IRRIG", case=False))].copy()

# %%
hay_irr.rename(columns={"VALUE" : "VALUE_irr",
                        "SHORT_DESC": "SHORT_DESC_irr"}, inplace=True)

# %%
comm_cols = ["CENSUS_CHAPTER", "CENSUS_TABLE", "CENSUS_ROW", "county_fips"]
hay_irr = hay_irr[comm_cols + ["SHORT_DESC_irr", "VALUE_irr"]]
hay_nonIrr = hay_nonIrr[comm_cols + ["SHORT_DESC", "VALUE"]]

# %%
hay_irr_nonIrr = pd.merge(hay_irr, hay_nonIrr, on=comm_cols, how="outer")
hay_irr_nonIrr.head(5)

# %%
census_df[(census_df.CENSUS_CHAPTER == 2) & (census_df.CENSUS_TABLE == 24)& 
          (census_df.CENSUS_ROW == 63) & (census_df.county_fips == "01001")]

# %%
hay_irr_nonIrr.tail(5)

# %%
print (f"{hay_irr.shape = }")
print (f"{hay_nonIrr.shape = }")
print (f"{hay_irr_nonIrr.shape = }")

# %%
hay_irr_nonIrr["VALUE"]     = hay_irr_nonIrr["VALUE"].str.replace(",", "")
hay_irr_nonIrr["VALUE_irr"] = hay_irr_nonIrr["VALUE_irr"].str.replace(",", "")

hay_irr_nonIrr["VALUE"]     = hay_irr_nonIrr["VALUE"].astype(float)
hay_irr_nonIrr["VALUE_irr"] = hay_irr_nonIrr["VALUE_irr"].astype(float)

hay_irr_nonIrr.head(5)

# %%
hay[(hay.CENSUS_CHAPTER == 2) & (hay.CENSUS_TABLE==24) & (hay.CENSUS_ROW==63) & (hay.county_fips == "01001")]

# %%
hay_irr_nonIrr_noNA = hay_irr_nonIrr.dropna(how="any")
hay_irr_nonIrr_noNA.reset_index(drop=True, inplace=True)
print (f"{hay_irr_nonIrr_noNA.CENSUS_TABLE.unique() = }")
hay_irr_nonIrr_noNA.head(2)

# %%
hay[(hay.CENSUS_CHAPTER == 2) & (hay.CENSUS_TABLE==26) & (hay.CENSUS_ROW==82) & (hay.county_fips == "10001")]

# %%
gl_merge = pd.read_csv(NASS_dir + "gl_merge.csv")
print (f"{gl_merge.shape = }")
gl_merge.head(2)

# %%
hay_irr_nonIrr_noNA_26 = hay_irr_nonIrr_noNA[hay_irr_nonIrr_noNA.CENSUS_TABLE == 26].copy()
print (hay_irr_nonIrr_noNA_26.shape)

# %%
print (f"{hay_irr_nonIrr_noNA.CENSUS_TABLE.unique() = }")
print (f"{gl_merge.CENSUS_TABLE.unique() = }")

# %%
print (f"{len(sorted(hay_irr_nonIrr_noNA.CENSUS_ROW.unique()) )= }")
print (f"{len(sorted(gl_merge.CENSUS_ROW.unique())) = }")

# %%
print (len(hay_irr_nonIrr_noNA["SHORT_DESC"].unique()))
print (len(gl_merge["SHORT_DESC.x"].unique()))

# %%
gl = census_df[census_df.CENSUS_TABLE == 26]
print (gl.shape)
gl_i = gl[gl.SHORT_DESC.isin(["HAY & HAYLAGE, IRRIGATED - ACRES HARVESTED",
                             "GRASSES & LEGUMES TOTALS, IRRIGATED, SEED - ACRES HARVESTED",
                             "CORN, SILAGE, IRRIGATED - ACRES HARVESTED", 
                             "SORGHUM, SILAGE, IRRIGATED - ACRES HARVESTED"])].copy()

gl = gl[gl.SHORT_DESC.isin(["HAY & HAYLAGE - ACRES HARVESTED",
                                   "GRASSES & LEGUMES TOTALS, SEED - ACRES HARVESTED",
                                   "CORN, SILAGE - ACRES HARVESTED", 
                                   "SORGHUM, SILAGE - ACRES HARVESTED"])].copy()
print (gl.shape)

# %%
gl_i.rename(columns={"VALUE" : "VALUE_irr"}, inplace=True)

gl_i = gl_i[['CENSUS_CHAPTER', 'CENSUS_TABLE', 'CENSUS_ROW', 'county_fips', 'SHORT_DESC', 'VALUE_irr']] 
gl = gl[['CENSUS_CHAPTER', 'CENSUS_TABLE', 'STATE_ALPHA', 'CENSUS_ROW', 'county_fips', 'SHORT_DESC', 'VALUE']]
gl_merge_py = pd.merge(gl, gl_i, on=["county_fips", "CENSUS_CHAPTER","CENSUS_TABLE","CENSUS_ROW"], how="outer")

# %%
print (gl_merge_py.shape)
print (gl_merge.shape)

# %%
gl_merge_py.columns

# %%

# %%
gl_merge_py.shape

# %%
gl_merge_py.dropna(subset=['VALUE']).shape

# %%
110643 - 107196

# %%
