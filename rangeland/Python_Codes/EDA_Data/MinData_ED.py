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

reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
# county_rangeland_and_totalarea_fraction =pd.read_csv(f'/Users/hn/Documents/01_research_data/RangeLand/' + \
#                                                      'Data/Min_Data/county_rangeland_and_totalarea_fraction.txt')

# county_rangeland_and_totalarea_fraction.head(2)

# print (county_rangeland_and_totalarea_fraction.shape)
# county_rangeland_and_totalarea_fraction.drop_duplicates(inplace=True)
# print (county_rangeland_and_totalarea_fraction.shape)
# county_rangeland_and_totalarea_fraction.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
# out_name = reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv"
# county_rangeland_and_totalarea_fraction.to_csv(out_name, index = False)


# %%
Min_csvs = sorted([x for x in os.listdir(Min_data_dir_base) if x.endswith(".csv")])

# %%
# ecozone_id_name = pd.read_csv(Min_data_dir_base + "ecozone_id_name.csv")
# ecozone_id_name.head(2)

# %%
county_id_name_fips = pd.read_csv(Min_data_dir_base + "county_id_name_fips.csv")
county_id_name_fips.head(2)

# %% [markdown]
# # Annual Data

# %%
county_annual_files = sorted([x for x in Min_csvs if ("county" in x) and ("annual" in x)])
county_annual = pd.DataFrame()

for a_file_name in county_annual_files:
    print (a_file_name)
    a_file = pd.read_csv(Min_data_dir_base + a_file_name)
    print (f"{a_file.year.min()=}")
    print (f"{a_file.year.max()=}")
    print ()
    if ("GPP" in a_file.columns):
        a_file.rename(columns={"GPP": "MODIS_GPP"}, inplace=True)
    elif("NPP" in a_file.columns):
        a_file.rename(columns={"NPP": "MODIS_NPP"}, inplace=True)
        

    if len(county_annual)==0:
        county_annual = a_file.copy()
    else:
        county_annual = pd.merge(county_annual, a_file, on=['year', 'county'], how='left')

# %%
county_annual.rename(columns={"productivity": "unit_matt_npp"}, inplace=True)

# %%
out_name = reOrganized_dir + "county_annual_GPP_MattNPP.csv"
county_annual.to_csv(out_name, index = False)

county_annual.head(2)

# %%
# ecozone_annual_files = sorted([x for x in Min_csvs if ("ecozone" in x) and ("annual" in x)])
# ecozone_annual = pd.DataFrame()

# for a_file_name in ecozone_annual_files:
#     print (a_file_name)
#     a_file = pd.read_csv(Min_data_dir_base + a_file_name)
#     print (f"{a_file.year.min()=}")
#     print (f"{a_file.year.max()=}")
#     print ()
#     if ("GPP" in a_file.columns):
#         a_file.rename(columns={"GPP": "MODIS_GPP"}, inplace=True)
#     elif("NPP" in a_file.columns):
#         a_file.rename(columns={"NPP": "MODIS_NPP"}, inplace=True)
        

#     if len(ecozone_annual)==0:
#         ecozone_annual = a_file.copy()
#     else:
#         ecozone_annual = pd.merge(ecozone_annual, a_file, on=['year', 'ecozone'], how='left')

# out_name = reOrganized_dir + "ecozone_annual.csv"
# ecozone_annual.to_csv(out_name, index = False)

# ecozone_annual.head(2)

# %%

# %%
# prfgrid_annual_files = sorted([x for x in Min_csvs if ("prfgrid" in x) and ("annual" in x)])
# prfgrid_annual = pd.DataFrame()

# for a_file_name in prfgrid_annual_files:
#     print (a_file_name)
#     a_file = pd.read_csv(Min_data_dir_base + a_file_name)
#     print (f"{a_file.year.min()=}")
#     print (f"{a_file.year.max()=}")
#     print ()
#     if ("GPP" in a_file.columns):
#         a_file.rename(columns={"GPP": "MODIS_GPP"}, inplace=True)
#     elif("NPP" in a_file.columns):
#         a_file.rename(columns={"NPP": "MODIS_NPP"}, inplace=True)
        

#     if len(prfgrid_annual)==0:
#         prfgrid_annual = a_file.copy()
#     else:
#         prfgrid_annual = pd.merge(prfgrid_annual, a_file, on=['year', 'prfgrid'], how='left')

# out_name = reOrganized_dir + "prfgrid_annual.csv"
# prfgrid_annual.to_csv(out_name, index = False)

# prfgrid_annual.head(2)

# %%

# %%
# econregion_annual_files = sorted([x for x in Min_csvs if ("econregion" in x) and ("annual" in x)])
# econregion_annual = pd.DataFrame()

# for a_file_name in econregion_annual_files:
#     print (a_file_name)
#     a_file = pd.read_csv(Min_data_dir_base + a_file_name)
#     print (f"{a_file.year.min()=}")
#     print (f"{a_file.year.max()=}")
#     print ()
#     if ("GPP" in a_file.columns):
#         a_file.rename(columns={"GPP": "MODIS_GPP"}, inplace=True)
#     elif("NPP" in a_file.columns):
#         a_file.rename(columns={"NPP": "MODIS_NPP"}, inplace=True)
        

#     if len(econregion_annual)==0:
#         econregion_annual = a_file.copy()
#     else:
#         econregion_annual = pd.merge(econregion_annual, a_file, on=['year', 'econregion'], how='left')

# out_name = reOrganized_dir + "econregion_annual.csv"
# econregion_annual.to_csv(out_name, index = False)

# econregion_annual.head(2)

# %%
# statefips_annual_files = sorted([x for x in Min_csvs if ("statefips" in x) and ("annual" in x)])
# statefips_annual = pd.DataFrame()

# for a_file_name in statefips_annual_files:
#     print (a_file_name)
#     a_file = pd.read_csv(Min_data_dir_base + a_file_name)
#     print (f"{a_file.year.min()=}")
#     print (f"{a_file.year.max()=}")
#     print ()
#     if ("GPP" in a_file.columns):
#         a_file.rename(columns={"GPP": "MODIS_GPP"}, inplace=True)
#     elif("NPP" in a_file.columns):
#         a_file.rename(columns={"NPP": "MODIS_NPP"}, inplace=True)
        

#     if len(statefips_annual)==0:
#         statefips_annual = a_file.copy()
#     else:
#         statefips_annual = pd.merge(statefips_annual, a_file, on=['year', 'statefips90m'], how='left')

# out_name = reOrganized_dir + "statefips_annual.csv"
# statefips_annual.to_csv(out_name, index = False)

# statefips_annual.head(2)

# %%
# subsection_annual_files = sorted([x for x in Min_csvs if ("subsection" in x) and ("annual" in x)])
# subsection_annual = pd.DataFrame()

# for a_file_name in subsection_annual_files:
#     print (a_file_name)
#     a_file = pd.read_csv(Min_data_dir_base + a_file_name)
#     print (f"{a_file.year.min()=}")
#     print (f"{a_file.year.max()=}")
#     print ()
#     if ("GPP" in a_file.columns):
#         a_file.rename(columns={"GPP": "MODIS_GPP"}, inplace=True)
#     elif("NPP" in a_file.columns):
#         a_file.rename(columns={"NPP": "MODIS_NPP"}, inplace=True)
        

#     if len(subsection_annual)==0:
#         subsection_annual = a_file.copy()
#     else:
#         subsection_annual = pd.merge(subsection_annual, a_file, on=['year', 'subsection'], how='left')

# out_name = reOrganized_dir + "subsection_annual.csv"
# subsection_annual.to_csv(out_name, index = False)

# subsection_annual.head(2)

# %% [markdown]
# # Monthly Data

# %%
county_monthly_files = sorted([x for x in Min_csvs if ("county" in x) and ("monthly" in x)])
county_monthly = pd.DataFrame()

for a_file_name in county_monthly_files:
    print (a_file_name)
    a_file = pd.read_csv(Min_data_dir_base + a_file_name)
    print (f"{a_file.year.min()=}")
    print (f"{a_file.year.max()=}")
    print ()
    
    source = a_file_name.split("_")[2]
    variable = a_file_name.split("_")[3].split(".")[0]
    a_file.rename(columns={variable: source + "_" + variable.upper()}, inplace=True)

    if len(county_monthly)==0:
        county_monthly = a_file.copy()
    else:
        county_monthly = pd.merge(county_monthly, a_file, on=['year', 'month', 'county'], how='left')

out_name = reOrganized_dir + "county_monthly.csv"
county_monthly.to_csv(out_name, index = False)

county_monthly.head(2)

# %%
econregion_monthly_files = sorted([x for x in Min_csvs if ("econregion" in x) and ("monthly" in x)])
econregion_monthly = pd.DataFrame()

for a_file_name in econregion_monthly_files:
    print (a_file_name)
    a_file = pd.read_csv(Min_data_dir_base + a_file_name)
    print (f"{a_file.year.min()=}")
    print (f"{a_file.year.max()=}")
    print ()
    
    source = a_file_name.split("_")[2]
    variable = a_file_name.split("_")[3].split(".")[0]
    a_file.rename(columns={variable: source + "_" + variable.upper()}, inplace=True)

    if len(econregion_monthly)==0:
        econregion_monthly = a_file.copy()
    else:
        econregion_monthly = pd.merge(econregion_monthly, a_file, on=['year', 'month', 'econregion'], how='left')

out_name = reOrganized_dir + "econregion_monthly.csv"
econregion_monthly.to_csv(out_name, index = False)

econregion_monthly.head(2)

# %%
# ecozone_monthly_files = sorted([x for x in Min_csvs if ("ecozone" in x) and ("monthly" in x)])
# ecozone_monthly = pd.DataFrame()

# for a_file_name in ecozone_monthly_files:
#     print (a_file_name)
#     a_file = pd.read_csv(Min_data_dir_base + a_file_name)
#     print (f"{a_file.year.min()=}")
#     print (f"{a_file.year.max()=}")
#     print ()
    
#     source = a_file_name.split("_")[2]
#     variable = a_file_name.split("_")[3].split(".")[0]
#     a_file.rename(columns={variable: source + "_" + variable.upper()}, inplace=True)

#     if len(ecozone_monthly)==0:
#         ecozone_monthly = a_file.copy()
#     else:
#         ecozone_monthly = pd.merge(ecozone_monthly, a_file, on=['year', 'month', 'ecozone'], how='left')

# out_name = reOrganized_dir + "ecozone_monthly.csv"
# ecozone_monthly.to_csv(out_name, index = False)

# ecozone_monthly.head(2)

# %%

# %%
# prfgrid_monthly_files = sorted([x for x in Min_csvs if ("prfgrid" in x) and ("monthly" in x)])
# prfgrid_monthly = pd.DataFrame()

# for a_file_name in prfgrid_monthly_files:
#     print (a_file_name)
#     a_file = pd.read_csv(Min_data_dir_base + a_file_name)
#     print (f"{a_file.year.min()=}")
#     print (f"{a_file.year.max()=}")
#     print ()
    
#     source = a_file_name.split("_")[2]
#     variable = a_file_name.split("_")[3].split(".")[0]
#     a_file.rename(columns={variable: source + "_" + variable.upper()}, inplace=True)

#     if len(prfgrid_monthly)==0:
#         prfgrid_monthly = a_file.copy()
#     else:
#         prfgrid_monthly = pd.merge(prfgrid_monthly, a_file, on=['year', 'month', 'prfgrid'], how='left')

# out_name = reOrganized_dir + "prfgrid_monthly.csv"
# prfgrid_monthly.to_csv(out_name, index = False)

# prfgrid_monthly.head(2)

# %%
statefips_monthly_files = sorted([x for x in Min_csvs if ("statefips" in x) and ("monthly" in x)])
statefips_monthly = pd.DataFrame()

for a_file_name in statefips_monthly_files:
    print (a_file_name)
    a_file = pd.read_csv(Min_data_dir_base + a_file_name)
    print (f"{a_file.year.min()=}")
    print (f"{a_file.year.max()=}")
    print ()
    
    source = a_file_name.split("_")[2]
    variable = a_file_name.split("_")[3].split(".")[0]
    a_file.rename(columns={variable: source + "_" + variable.upper()}, inplace=True)

    if len(statefips_monthly)==0:
        statefips_monthly = a_file.copy()
    else:
        statefips_monthly = pd.merge(statefips_monthly, a_file, on=['year', 'month', 'statefips90m'], how='left')

out_name = reOrganized_dir + "statefips_monthly.csv"
statefips_monthly.to_csv(out_name, index = False)

statefips_monthly.head(2)

# %%

# %%
subsection_monthly_files = sorted([x for x in Min_csvs if ("subsection" in x) and ("monthly" in x)])
subsection_monthly = pd.DataFrame()

for a_file_name in subsection_monthly_files:
    print (a_file_name)
    a_file = pd.read_csv(Min_data_dir_base + a_file_name)
    print (f"{a_file.year.min()=}")
    print (f"{a_file.year.max()=}")
    print ()
    
    source = a_file_name.split("_")[2]
    variable = a_file_name.split("_")[3].split(".")[0]
    a_file.rename(columns={variable: source + "_" + variable.upper()}, inplace=True)

    if len(subsection_monthly)==0:
        subsection_monthly = a_file.copy()
    else:
        subsection_monthly = pd.merge(subsection_monthly, a_file, on=['year', 'month', 'subsection'], how='left')

out_name = reOrganized_dir + "subsection_monthly.csv"
subsection_monthly.to_csv(out_name, index = False)

subsection_monthly.head(2)

# %%

# %%
Min_DroughtIndices_dir = Min_data_dir_base + "Min_DroughtIndices/"
Min_DroughtIndices_csvs = sorted([x for x in os.listdir(Min_DroughtIndices_dir) if x.endswith(".csv")])

for a_file_name in Min_DroughtIndices_csvs:
    a_file = pd.read_csv(Min_DroughtIndices_dir + a_file_name)
    print (f"{a_file.year.min()=}")
    print (f"{a_file.year.max()=}")
    print ()

# %%

# %%
Min_WeeklyClimateMean_dir = Min_data_dir_base + "Min_WeeklyClimateMean/"
Min_WeeklyClimateMean_csvs = sorted([x for x in os.listdir(Min_WeeklyClimateMean_dir) if x.endswith(".csv")])

for a_file_name in Min_WeeklyClimateMean_csvs:
    if "et0" in a_file_name:
        a_file = pd.read_csv(Min_WeeklyClimateMean_dir + a_file_name)
        print (a_file_name)
        print (f"{a_file.year.min()=}")
        print (f"{a_file.year.max()=}")
        print ()

# %%
a_file

# %%
Min_csvs

# %%
satellit_info = {"L4" : ['LANDSAT/LT04/C02/T1_L2', 1984, 1993],
                 "L5_early" : ['LANDSAT/LT05/C02/T1_L2', 1984, 1993],
                 "L5_late" : ['LANDSAT/LT05/C02/T1_L2', 1999, 2007],
                 "L7" : ['LANDSAT/LE07/C02/T1_L2', 1999, 2007]}

for a_sat_key in ["L5_early"]:
    a_sat_info = satellit_info[a_sat_key]
    satelliteChoice = a_sat_info[0]
    start_year = a_sat_info[1]
    end_year = a_sat_info[2]
    # for a_year in range(start_year, end_year+1):
    for a_year in reversed(range(start_year, end_year+1)):
        print (a_year)


# %%
