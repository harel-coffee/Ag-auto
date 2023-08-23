import shutup, random, time

shutup.please()

import numpy as np
import pandas as pd

from datetime import date, datetime
from random import seed, random

import sys, os, os.path, shutil

####################################################################################
###
###                      Time It!
###
####################################################################################

start_time = time.time()
print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

####################################################################################
###
###                      Kamiak Core path
###
####################################################################################

sys.path.append("/home/h.noorazar/NASA/")
import NASA_core as nc

try:
    print("numpy.__version__=", numpy.__version__)
except:
    print("numpy.__version__ not printed")

####################################################################################
###
###      Parameters
###
####################################################################################

VI_idx = sys.argv[1]
smooth = sys.argv[2]
batch_no = str(sys.argv[3])

####################################################################################
###
###      Directories
###
####################################################################################
data_base = "/data/project/agaid/h.noorazar/NASA/"
if smooth == "regular":
    in_dir = data_base + "VI_TS/04_regularized_TS/"
else:
    in_dir = data_base + "VI_TS/05_SG_TS/"

out_dir = in_dir
os.makedirs(out_dir, exist_ok=True)

#####################################################################
######
######                           Body
######
#####################################################################
f_name = VI_idx + "_" + smooth + "_" + "intersect_batchNumber" + batch_no + "_JFD_pre2008.csv"
data = pd.read_csv(in_dir + f_name)
data["human_system_start_time"] = pd.to_datetime(data["human_system_start_time"])

out_name = (
    VI_idx + "_" + smooth + "_" + "intersect_batchNumber" + batch_no + "_wide_JFD_pre2008.csv"
)
out_name = out_dir + out_name
##############################
##
##     Widen
##
##############################
#
# Form an empty dataframe to populate
#
VI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37)]
columnNames = ["ID", "year"] + VI_colnames

years = data.human_system_start_time.dt.year.unique()
IDs = data.ID.unique()
no_rows = len(IDs) * len(years)

data_wide = pd.DataFrame(columns=columnNames, index=range(no_rows))
data_wide.ID = list(IDs) * len(years)
data_wide.sort_values(by=["ID"], inplace=True)
data_wide.reset_index(drop=True, inplace=True)
data_wide.year = list(years) * len(IDs)


for an_ID in IDs:
    curr_field = data[data.ID == an_ID]
    curr_years = curr_field.human_system_start_time.dt.year.unique()
    for a_year in curr_years:
        curr_field_year = curr_field[curr_field.human_system_start_time.dt.year == a_year]
        data_wide_indx = data_wide[(data_wide.ID == an_ID) & (data_wide.year == a_year)].index

        """ The reason we did the following is that some years in pre2008 has less than 36 points in them.
        We need to address this(?)
        e.g. NDVI_SG_9_pre2008.csv where 9 is batch number. 
             ID:   i2268
             year: 1988
        """
        if VI_idx == "EVI":
            V = curr_field_year.EVI.values
            if len(V) >= 36:
                data_wide.loc[data_wide_indx, "EVI_1":"EVI_36"] = V[:36]
            else:
                currV = list(V) + [V[-1]] * (36 - len(V))
                data_wide.loc[data_wide_indx, "EVI_1":"EVI_36"] = currV
        elif VI_idx == "NDVI":
            V = curr_field_year.NDVI.values
            if len(V) >= 36:
                data_wide.loc[data_wide_indx, "NDVI_1":"NDVI_36"] = V[:36]
            else:
                currV = list(V) + [V[-1]] * (36 - len(V))
                data_wide.loc[data_wide_indx, "NDVI_1":"NDVI_36"] = currV

data_wide.drop_duplicates(inplace=True)
data_wide.dropna(inplace=True)
data_wide.to_csv(out_name, index=False)

end_time = time.time()
print("it took {:.0f} minutes to run this code.".format((end_time - start_time) / 60))
print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")
