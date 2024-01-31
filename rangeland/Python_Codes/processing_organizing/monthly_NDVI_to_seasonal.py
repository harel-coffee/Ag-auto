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

# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
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

# %%
Min_files = os.listdir(Min_data_base)
Min_files = [x for x in Min_files if x.endswith(".csv")]
Min_files = [x for x in Min_files if "NDVI" in x]
Min_files = [x for x in Min_files if "county_monthly" in x]
Min_files

# %%
file_name = Min_files[0]
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_ = "county_fips")

S1 = ndvi[ndvi.month.isin([1, 2, 3])]
S2 = ndvi[ndvi.month.isin([4, 5, 6, 7])]
S3 = ndvi[ndvi.month.isin([8, 9])]
S4 = ndvi[ndvi.month.isin([10, 11, 12])]

# drop "month"
S1 = S1[["year", "county_fips", "NDVI"]]
S2 = S2[["year", "county_fips", "NDVI"]]
S3 = S3[["year", "county_fips", "NDVI"]]
S4 = S4[["year", "county_fips", "NDVI"]]

S1 = S1.groupby(["year", "county_fips"]).mean().reset_index()
S2 = S2.groupby(["year", "county_fips"]).mean().reset_index()
S3 = S3.groupby(["year", "county_fips"]).mean().reset_index()
S4 = S4.groupby(["year", "county_fips"]).mean().reset_index()

if "MODIS" in file_name:
    new_name = "modis_ndvi"
elif "AVHRR" in file_name:
    new_name = "avhrr_ndvi"
elif "GIMMS" in file_name:
    new_name = "gimms_ndvi"

S1.rename(columns={"NDVI" : "s1_" + new_name}, inplace=True)
S2.rename(columns={"NDVI" : "s2_" + new_name}, inplace=True)
S3.rename(columns={"NDVI" : "s3_" + new_name}, inplace=True)
S4.rename(columns={"NDVI" : "s4_" + new_name}, inplace=True)

seasonal_ndvi_avhrr = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
seasonal_ndvi_avhrr = pd.merge(seasonal_ndvi_avhrr, S3, on=["county_fips", "year"], how="outer")
seasonal_ndvi_avhrr = pd.merge(seasonal_ndvi_avhrr, S4, on=["county_fips", "year"], how="outer")
seasonal_ndvi_avhrr.head(2)

# %%
del(S1, S2, S3, S4, ndvi)

# %%
file_name = Min_files[1]
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_ = "county_fips")

S1 = ndvi[ndvi.month.isin([1, 2, 3])]
S2 = ndvi[ndvi.month.isin([4, 5, 6, 7])]
S3 = ndvi[ndvi.month.isin([8, 9])]
S4 = ndvi[ndvi.month.isin([10, 11, 12])]

# drop "month"
S1 = S1[["year", "county_fips", "NDVI"]]
S2 = S2[["year", "county_fips", "NDVI"]]
S3 = S3[["year", "county_fips", "NDVI"]]
S4 = S4[["year", "county_fips", "NDVI"]]

S1 = S1.groupby(["year", "county_fips"]).mean().reset_index()
S2 = S2.groupby(["year", "county_fips"]).mean().reset_index()
S3 = S3.groupby(["year", "county_fips"]).mean().reset_index()
S4 = S4.groupby(["year", "county_fips"]).mean().reset_index()

if "MODIS" in file_name:
    new_name = "modis_ndvi"
elif "AVHRR" in file_name:
    new_name = "avhrr_ndvi"
elif "GIMMS" in file_name:
    new_name = "gimms_ndvi"

S1.rename(columns={"NDVI" : "s1_" + new_name}, inplace=True)
S2.rename(columns={"NDVI" : "s2_" + new_name}, inplace=True)
S3.rename(columns={"NDVI" : "s3_" + new_name}, inplace=True)
S4.rename(columns={"NDVI" : "s4_" + new_name}, inplace=True)

seasonal_ndvi_gimms = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
seasonal_ndvi_gimms = pd.merge(seasonal_ndvi_gimms, S3, on=["county_fips", "year"], how="outer")
seasonal_ndvi_gimms = pd.merge(seasonal_ndvi_gimms, S4, on=["county_fips", "year"], how="outer")
seasonal_ndvi_gimms.head(2)

del(S1, S2, S3, S4, ndvi)

# %%
file_name = Min_files[2]
ndvi = pd.read_csv(Min_data_base + file_name)
ndvi.rename(columns={"county": "county_fips"}, inplace=True)
ndvi = rc.correct_Mins_county_6digitFIPS(df=ndvi, col_ = "county_fips")

S1 = ndvi[ndvi.month.isin([1, 2, 3])]
S2 = ndvi[ndvi.month.isin([4, 5, 6, 7])]
S3 = ndvi[ndvi.month.isin([8, 9])]
S4 = ndvi[ndvi.month.isin([10, 11, 12])]

# drop "month"
S1 = S1[["year", "county_fips", "NDVI"]]
S2 = S2[["year", "county_fips", "NDVI"]]
S3 = S3[["year", "county_fips", "NDVI"]]
S4 = S4[["year", "county_fips", "NDVI"]]

S1 = S1.groupby(["year", "county_fips"]).mean().reset_index()
S2 = S2.groupby(["year", "county_fips"]).mean().reset_index()
S3 = S3.groupby(["year", "county_fips"]).mean().reset_index()
S4 = S4.groupby(["year", "county_fips"]).mean().reset_index()

if "MODIS" in file_name:
    new_name = "modis_ndvi"
elif "AVHRR" in file_name:
    new_name = "avhrr_ndvi"
elif "GIMMS" in file_name:
    new_name = "gimms_ndvi"

S1.rename(columns={"NDVI" : "s1_" + new_name}, inplace=True)
S2.rename(columns={"NDVI" : "s2_" + new_name}, inplace=True)
S3.rename(columns={"NDVI" : "s3_" + new_name}, inplace=True)
S4.rename(columns={"NDVI" : "s4_" + new_name}, inplace=True)

seasonal_ndvi_modis = pd.merge(S1, S2, on=["county_fips", "year"], how="outer")
seasonal_ndvi_modis = pd.merge(seasonal_ndvi_modis, S3, on=["county_fips", "year"], how="outer")
seasonal_ndvi_modis = pd.merge(seasonal_ndvi_modis, S4, on=["county_fips", "year"], how="outer")
seasonal_ndvi_modis.head(2)

# %%
print (f"{seasonal_ndvi_avhrr.shape = }")
print (f"{seasonal_ndvi_gimms.shape = }")
print (f"{seasonal_ndvi_modis.shape = }")

# %%
print (len(seasonal_ndvi_avhrr.county_fips.unique()))
print (len(seasonal_ndvi_gimms.county_fips.unique()))
print (len(seasonal_ndvi_modis.county_fips.unique()))

# %%
ndvi_avhrr = pd.read_csv(Min_data_base + Min_files[0])
ndvi_gimms = pd.read_csv(Min_data_base + Min_files[1])
ndvi_modis = pd.read_csv(Min_data_base + Min_files[2])

# %%
# it seems they must have 381 rows per county
# and there are 22 years worth of data.
incomplete_counties_avhrr = []

for a_county in ndvi_avhrr.county.unique():
    df = ndvi_avhrr[ndvi_avhrr.county == a_county]
    if len(df) != 381:
        incomplete_counties_avhrr += [a_county]

len(incomplete_counties_avhrr)

# %%
# it seems they must have 381 rows per county
incomplete_counties_gimms = []

for a_county in ndvi_gimms.county.unique():
    df = ndvi_gimms[ndvi_gimms.county == a_county]
    if len(df) != 381:
        incomplete_counties_gimms += [a_county]

len(incomplete_counties_gimms)

# %%
# it seems they must have 381 rows per county
incomplete_counties_modis = []

for a_county in ndvi_modis.county.unique():
    df = ndvi_modis[ndvi_modis.county == a_county]
    if len(df) != 381:
        incomplete_counties_modis += [a_county]
        
len(incomplete_counties_modis)

# %%
ndvi_modis[ndvi_modis.county==incomplete_counties_modis[0]].year.unique()

# %%
# for a_year in range(2001, 2023):
#     for a_county in incomplete_counties_modis:
#         df = ndvi_modis[ndvi_modis.county == a_county]
#         df = df[df.year == a_year]
        

# %% [markdown]
# # Some Months are missing.

# %%
a_county = incomplete_counties_modis[0]
# for a_year in range(2001, 2023):
#     df = ndvi_modis[ndvi_modis.county == a_county]
#     df = df[df.year == a_year]
#     print (df.shape)
#     print (a_year)
#     print ()

df = ndvi_modis[ndvi_modis.county == a_county]
df = df[df.year == 2001]
df

# %%
print (f"{seasonal_ndvi_avhrr.shape = }")
print (f"{seasonal_ndvi_gimms.shape = }")
print (f"{seasonal_ndvi_modis.shape = }")

# %% [markdown]
# # Check if some counties have missing years

# %%
print (ndvi_avhrr.year.unique())
print ()
print (ndvi_gimms.year.unique())

print ()
print (ndvi_modis.year.unique())

# %%
print (sorted(ndvi_modis.county.unique()) == sorted(ndvi_avhrr.county.unique()))
print (sorted(ndvi_modis.county.unique()) == sorted(ndvi_gimms.county.unique()))

# %%
cnty_missing_years_modis = []
max_no_years = len(ndvi_modis.year.unique())

for a_county in ndvi_modis.county.unique():
    df = ndvi_modis[ndvi_modis.county == a_county]
    if len(df.year.unique()) != max_no_years:
        cnty_missing_years_modis += [a_county]

len(cnty_missing_years_modis)

# %%
cnty_missing_years_avhrr = []
max_no_years = len(ndvi_avhrr.year.unique())

for a_county in ndvi_avhrr.county.unique():
    df = ndvi_avhrr[ndvi_avhrr.county == a_county]
    if len(df.year.unique()) != max_no_years:
        cnty_missing_years_avhrr += [a_county]

len(cnty_missing_years_avhrr)

# %%
cnty_missing_years_gimms = []
max_no_years = len(ndvi_gimms.year.unique())

for a_county in ndvi_gimms.county.unique():
    df = ndvi_gimms[ndvi_gimms.county == a_county]
    if len(df.year.unique()) != max_no_years:
        cnty_missing_years_gimms += [a_county]

len(cnty_missing_years_gimms)

# %%

# %%
# for a_year in range(2001, 2023):
#     df = ndvi_modis[ndvi_modis.county == a_county]
#     df = df[df.year == a_year]
#     print (df.shape)
#     print (a_year)
#     print ()

# %%
seasonal_ndvi_gimms.head(2)

# %%
seasonal_ndvi_modis.head(2)

# %%
seasonal_ndvi = pd.merge(seasonal_ndvi_avhrr, seasonal_ndvi_gimms, on=["county_fips", "year"], how="outer")
seasonal_ndvi = pd.merge(seasonal_ndvi, seasonal_ndvi_modis, on=["county_fips", "year"], how="outer")
seasonal_ndvi.head(2)

# %%
print (seasonal_ndvi.shape)
print (seasonal_ndvi_avhrr.shape)

# %%
import pickle
from datetime import datetime

filename = reOrganized_dir + "seasonal_ndvi.sav"

export_ = {"seasonal_ndvi":  seasonal_ndvi,
           "source_code" : "monthly_NDVI_to_seasonal",
           "Author": "HN",
           "Note": "The county level NDVI files have missing months in them and are not consistent.",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
