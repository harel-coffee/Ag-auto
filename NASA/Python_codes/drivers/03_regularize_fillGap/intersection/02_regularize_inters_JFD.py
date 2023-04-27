####
#### April 27, 2023
####

"""
  This script is modified from 00_Regularize_2Yrs.py from remote sensing project.
"""

import csv
import numpy as np
import pandas as pd
import scipy, scipy.signal
import time, datetime
from datetime import date
import sys, os, os.path

start_time = time.time()
# search path for modules# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

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
batch = str(sys.argv[2])
regular_window_size = 10
IDcolName = "ID"

# do the following since walla walla has two parts and we have to use walla_walla in terminal
print("Terminal Arguments are: ")
print("indeks= ", indeks)
print("batch= ", batch)
print("__________________________________________")

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
data_base = "/data/hydro/users/Hossein/NASA/03_jumps_removed/"
data_dir = data_base

output_dir = "/data/hydro/users/Hossein/NASA/04_regularized_TS/"
os.makedirs(output_dir, exist_ok=True)

########################################################################################
###
###                  Read and Process Data
###
########################################################################################

f_name = "NoJump_intersect_" + indeks + "_batchNumber" + str(batch) + "_JFD.csv"
out_name = output_dir + indeks + "_regular_intersect_batchNumber" + batch + "_JFD.csv"

an_EE_TS = pd.read_csv(data_dir + f_name, low_memory=False)
if "system_start_time" in an_EE_TS.columns:
    an_EE_TS.drop(["system_start_time"], axis=1, inplace=True)

an_EE_TS["human_system_start_time"] = pd.to_datetime(an_EE_TS["human_system_start_time"])
an_EE_TS["ID"] = an_EE_TS["ID"].astype(str)
print(an_EE_TS.head(2))

###
### List of unique polygons
###
ID_list = an_EE_TS[IDcolName].unique()
print(len(ID_list))

########################################################################################
###
###  initialize output data. all polygons in this case
###  will have the same length.
###

reg_cols = ["ID", "human_system_start_time", indeks]  # list(an_EE_TS.columns)
print("reg_cols= ", reg_cols)

st_yr = an_EE_TS.human_system_start_time.dt.year.min()
end_yr = an_EE_TS.human_system_start_time.dt.year.max()
no_days = (end_yr - st_yr + 1) * 366  # 14 years, each year 366 days!

no_steps = int(np.ceil(no_days / regular_window_size))  # no_days // regular_window_size
nrows = no_steps * len(ID_list)

output_df = pd.DataFrame(data=None, index=np.arange(nrows), columns=reg_cols)
print("st_yr is {}!".format(st_yr))
print("end_yr is {}!".format(end_yr))
print("nrows is {}!".format(nrows))
########################################################################################

counter = 0
row_pointer = 0
for a_poly in ID_list:
    if counter % 1000 == 0:
        print(counter)
    curr_field = an_EE_TS[an_EE_TS[IDcolName] == a_poly].copy()
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=["human_system_start_time"], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)

    ################################################################
    regularized_TS = nc.regularize_a_field(
        a_df=curr_field,
        V_idks=indeks,
        interval_size=regular_window_size,
        start_year=st_yr,
        end_year=end_yr,
    )

    regularized_TS = nc.fill_theGap_linearLine(a_regularized_TS=regularized_TS, V_idx=indeks)
    if counter == 0:
        print("output_df columns = ", list(output_df.columns))
        print("regularized_TS columns", list(regularized_TS.columns))
    counter += 1


####################################################################################
###
###                   Write the outputs
###
####################################################################################
output_df.drop_duplicates(inplace=True)
output_df.dropna(inplace=True)
output_df.to_csv(out_name, index=False)


end_time = time.time()
print("it took {:.0f} minutes to run this code.".format((end_time - start_time) / 60))
