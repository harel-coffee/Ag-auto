import shutup

shutup.please()

import numpy as np
import pandas as pd
import sys, os, os.path, shutil
from datetime import date, datetime

####################################################################################
###
###      Parameters
###
####################################################################################
VI_idx = sys.argv[1]
smooth_type = sys.argv[2]
print(f"Passed Args. are: {VI_idx=:}, {smooth_type=:}!")

"""
   # Read Training Set Labels
"""
dir_base = "/data/project/agaid/h.noorazar/NASA/Data_Models_4_RegionalStat/"
in_dir = dir_base + "00_SmoothedData/"
out_dir = dir_base + "01_wideData/"
os.makedirs(out_dir, exist_ok=True)


print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

file_names = [
    smooth_type + "_Walla2015_" + VI_idx + "_JFD.csv",
    smooth_type + "_AdamBenton2016_" + VI_idx + "_JFD.csv",
    smooth_type + "_Grant2017_" + VI_idx + "_JFD.csv",
    smooth_type + "_FranklinYakima2018_" + VI_idx + "_JFD.csv",
]

data = pd.DataFrame()

for file in file_names:
    curr_file = pd.read_csv(in_dir + file)
    curr_file["human_system_start_time"] = pd.to_datetime(curr_file["human_system_start_time"])

    # These data are for 3 years. The middle one is the correct one
    all_years = sorted(curr_file.human_system_start_time.dt.year.unique())
    if len(all_years) == 3 or len(all_years) == 2:
        proper_year = all_years[1]
    elif len(all_years) == 1:
        proper_year = all_years[0]

    curr_file = curr_file[curr_file.human_system_start_time.dt.year == proper_year]
    data = pd.concat([data, curr_file])

data.reset_index(drop=True, inplace=True)
data.head(2)
#
#     Widen
#
VI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37)]
columnNames = ["ID"] + VI_colnames
data_wide = pd.DataFrame(columns=columnNames, index=range(len(data.ID.unique())))
data_wide["ID"] = data.ID.unique()

for an_ID in data.ID.unique():
    curr_df = data[data.ID == an_ID]

    data_wide_indx = data_wide[data_wide.ID == an_ID].index
    if VI_idx == "EVI":
        data_wide.loc[data_wide_indx, "EVI_1":"EVI_36"] = curr_df.EVI.values[:36]
    elif VI_idx == "NDVI":
        data_wide.loc[data_wide_indx, "NDVI_1":"NDVI_36"] = curr_df.NDVI.values[:36]

out_name = out_dir + VI_idx + "_" + smooth_type + "_wide.csv"
data_wide.to_csv(out_name, index=False)

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")
