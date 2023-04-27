####
#### Nov 16, 2021
####

"""
  Regularize the EVI and NDVI of fields in individual years for training set creation.
"""

import csv
import numpy as np
import pandas as pd
from math import factorial
import scipy
import scipy.signal
import os, os.path
from datetime import date
import datetime
import time
import sys

start_time = time.time()

# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
####################################################################################
###
###                      Aeolus Core path
###
####################################################################################

sys.path.append("/home/hnoorazar/NASA/")
import NASA_core as nc
import NASA_plot_core as ncp

####################################################################################
###
###      Parameters
###
####################################################################################
indeks = sys.argv[1]
batch_number = int(sys.argv[2])
print("Terminal Arguments are: ")
print(indeks)
print(batch_number)
print("__________________________________________")

if indeks == "NDVI":
    NoVI = "EVI"
else:
    NoVI = "NDVI"

IDcolName = "ID"
####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
data_base = "/data/hydro/users/Hossein/NASA/"
data_dir = data_base + "/02_outliers_removed/"
SF_data_dir = "/data/hydro/users/Hossein/NASA/000_shapefile_data_part/"
output_dir = data_base + "/03_jumps_removed/"
os.makedirs(output_dir, exist_ok=True)

print("data_dir is: " + data_dir)
print("output_dir is: " + output_dir)
########################################################################################
###
###                   process data
###
########################################################################################

SF_data_IDs = pd.read_csv(SF_data_dir + "10_intersect_East_Irr_2008_2018_2cols_data_part.csv")
SF_data_IDs.sort_values(by=["ID"], inplace=True)
SF_data_IDs.reset_index(drop=True, inplace=True)

# there are
batch_size = int(np.ceil(69271 / 40))

batch_IDs = SF_data_IDs.loc[(batch_number - 1) * batch_size : (batch_number * batch_size - 1)]

print("batch_IDs")
print(batch_IDs[:3])
print(batch_IDs[-3:])

out_name = (
    output_dir + "NoJump_intersect_" + indeks + "batch_number" + str(batch_number) + "_JFD.csv"
)

common_part = "T1C2L2_inters_2008_2018_EastIrr_2008-01-01_2022-01-01"
f_names = [
    "noOutlier_" + "L5_" + common_part + "_" + indeks + ".csv",
    "noOutlier_" + "L7_" + common_part + "_" + indeks + ".csv",
    "noOutlier_" + "L8_" + common_part + "_" + indeks + ".csv",
]

L5 = pd.read_csv(data_dir + f_names[0], low_memory=False)
L7 = pd.read_csv(data_dir + f_names[1], low_memory=False)
L8 = pd.read_csv(data_dir + f_names[2], low_memory=False)


L5.drop(["system_start_time"], axis=1, inplace=True)
L5 = L5[L5[indeks].notna()]

L7.drop(["system_start_time"], axis=1, inplace=True)
L7 = L7[L7[indeks].notna()]

L8.drop(["system_start_time"], axis=1, inplace=True)
L8 = L8[L8[indeks].notna()]

L578 = pd.concat([L5, L7, L8])
del (L5, L7, L8)

L578["human_system_start_time"] = pd.to_datetime(L578["human_system_start_time"])
L578["ID"] = L578["ID"].astype(str)

L578 = L578[L578.ID.isin(list(batch_IDs.ID))].copy()

########################################################################################
###
### List of unique polygons
###
IDs = L578[IDcolName].unique()
print(len(IDs))

########################################################################################
###
###  initialize output data.
###

output_df = pd.DataFrame(data=None, index=np.arange(L578.shape[0]), columns=L578.columns)
counter = 0
row_pointer = 0

for a_poly in IDs:
    if counter % 1000 == 0:
        print(counter)
    curr_field = L578[L578[IDcolName] == a_poly].copy()

    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=["human_system_start_time"], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)

    ################################################################
    no_Outlier_TS = nc.correct_big_jumps_1DaySeries_JFD(
        dataTMS_jumpie=curr_field, give_col=indeks, maxjump_perDay=0.018
    )

    output_df[row_pointer : row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
    counter += 1
    row_pointer += curr_field.shape[0]

####################################################################################
###
###                   Write the outputs
###
####################################################################################
output_df.drop_duplicates(inplace=True)
output_df.to_csv(out_name, index=False)

end_time = time.time()
print("it took {:.0f} minutes to run this code.".format((end_time - start_time) / 60))
