####
#### Nov 16, 2021
####

"""
  remove outliers that are beyond -1 and 1 in NDVI and EVI.
  Looking at 2017 data I did not see any NDVI beyond those boundaries. 
  EVI had outliers only.
"""

import csv
import numpy as np
import pandas as pd
import scipy
import scipy.signal
import os, os.path
import time
import datetime
from datetime import date
from patsy import cr
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
county = sys.argv[2]

print("Terminal Arguments are: ")
print(indeks)
print(county)
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
data_dir = data_base + "01_raw_GEE/"
output_dir = data_base + "/02_outliers_removed/"
os.makedirs(output_dir, exist_ok=True)

print("data_dir is: " + data_dir)
print("output_dir is: " + output_dir)

########################################################################################
###
###                   process data
###
########################################################################################
if county == "Monterey2014":
    f_names = [
        "L7_T1C2L2_Scaled_Monterey2014_2013-01-01_2016-01-01.csv",
        "L8_T1C2L2_Scaled_Monterey2014_2013-01-01_2016-01-01.csv",
    ]

elif county == "Walla2015":
    f_names = [
        "L7_T1C2L2_Scaled_Walla2015_2014-01-01_2016-12-31.csv",
        "L8_T1C2L2_Scaled_Walla2015_2014-01-01_2016-12-31.csv",
    ]

elif county == "AdamBenton2016":
    f_names = [
        "L7_T1C2L2_Scaled_AdamBenton2016_2015-01-01_2017-10-14.csv",
        "L8_T1C2L2_Scaled_AdamBenton2016_2015-01-01_2017-10-14.csv",
    ]

elif county == "Grant2017":
    f_names = [
        "L7_T1C2L2_Scaled_Grant2017_2016-01-01_2018-10-14.csv",
        "L8_T1C2L2_Scaled_Grant2017_2016-01-01_2018-10-14.csv",
    ]

elif county == "FranklinYakima2018":
    f_names = [
        "L7_T1C2L2_Scaled_FranklinYakima2018_2017-01-01_2019-10-14.csv",
        "L8_T1C2L2_Scaled_FranklinYakima2018_2017-01-01_2019-10-14.csv",
    ]

L7 = pd.read_csv(data_dir + f_names[0])
L8 = pd.read_csv(data_dir + f_names[1])


L7.drop([NoVI], axis=1, inplace=True)
L7 = L7[L7[indeks].notna()]

L8.drop([NoVI], axis=1, inplace=True)
L8 = L8[L8[indeks].notna()]
L578 = pd.concat([L7, L8])
del (L7, L8)
L578["ID"] = L578["ID"].astype(str)
IDs = np.sort(L578["ID"].unique())

L578 = nc.add_human_start_time_by_system_start_time(L578)
L578 = nc.initial_clean(df=L578, column_to_be_cleaned=indeks)
print("Number of unique fields is: ")
print(len(IDs))
print("__________________________________________")

print("Dimension of the data is: " + str(L578.shape))
print("__________________________________________")

#########################################################

output_df = pd.DataFrame(data=None, index=np.arange(L578.shape[0]), columns=L578.columns)

counter = 0
row_pointer = 0
for a_poly in IDs:
    if counter % 1000 == 0:
        print(counter)
    curr_field = L578[L578[IDcolName] == a_poly].copy()
    # small fields may have nothing in them!
    if curr_field.shape[0] > 2:
        # ************************************************
        #
        #    Set negative indeks values to zero.
        #
        # ************************************************
        """
        we are killing some of the ourliers here and put them
        in the normal range! do we want to do it here? No, lets do it later.
        """
        # curr_field.loc[curr_field[indeks] < 0 , indeks] = 0
        no_Outlier_TS = nc.interpolate_outliers_EVI_NDVI(outlier_input=curr_field, given_col=indeks)
        no_Outlier_TS.loc[no_Outlier_TS[indeks] < 0, indeks] = 0

        """
        it is possible that for a field we only have x=2 data points
        where all the EVI/NDVI is outlier. Then, there is nothing to 
        use for interpolation. So, hopefully interpolate_outliers_EVI_NDVI is returning an empty data table.
        """
        if len(no_Outlier_TS) > 0:
            output_df[row_pointer : row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
            counter += 1
            row_pointer += curr_field.shape[0]

####################################################################################
###
###                   Write the outputs
###
####################################################################################

out_name = output_dir + "noOutlier_" + county + "_" + indeks + ".csv"

output_df.drop_duplicates(inplace=True)
# output_df.dropna(inplace=True)
output_df.to_csv(out_name, index=False)

print("it took {:.0f} minutes to run this code.".format((end_time - start_time) / 60))
