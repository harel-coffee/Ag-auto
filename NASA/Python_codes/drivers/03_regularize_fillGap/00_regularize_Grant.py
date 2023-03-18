####
#### Nov 3, 2021
####

"""
  This script is modified from 00_Regularize_2Yrs.py from remote sensing project.
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
###                      Local
###
####################################################################################

###
### Core path
###

# sys.path.append('/Users/hn/Documents/00_GitHub/Ag/remote_sensing/python/')

###
### Directories
###

# param_dir = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"
####################################################################################
###
###                      Aeolus Core path
###
####################################################################################

sys.path.append('/home/hnoorazar/NASA/')
import NASA_core as nc
import NASA_plot_core as ncp

####################################################################################
###
###      Parameters                   
###
####################################################################################

indeks = sys.argv[1]
random_or_all = sys.argv[2]
regular_window_size = 10
randCount = 100

# do the following since walla walla has two parts and we have to use walla_walla in terminal
print ("Terminal Arguments are: ")
print (indeks)
print (random_or_all)
print ("__________________________________________")
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
data_base = "/data/hydro/users/Hossein/NASA/03_jumps_removed/"
data_dir = data_base

output_dir = "/data/hydro/users/Hossein/NASA/04_regularized_TS/"
os.makedirs(output_dir, exist_ok=True)

########################################################################################
###
###                   process data
###
########################################################################################
if random_or_all == "random":
    f_name = "NoJump_int_Grant_Irr_2008_2018_" + indeks + "_" + str(randCount) + "randomfields.csv"
    out_name = output_dir + "regular_int_Grant_Irr_2008_2018_" + indeks + "_" + str(randCount) + "randomfields.csv"
else:
    f_name = "NoJump_int_Grant_Irr_2008_2018_" + indeks + ".csv"
    out_name = output_dir + "regular_int_Grant_Irr_2008_2018_" + indeks + ".csv"


an_EE_TS = pd.read_csv(data_dir + f_name, low_memory=False)
an_EE_TS.drop(["system_start_time"], axis=1, inplace=True)
an_EE_TS['human_system_start_time'] = pd.to_datetime(an_EE_TS['human_system_start_time'])
print (an_EE_TS.head(2))

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

reg_cols = ['ID', 'human_system_start_time', indeks] # list(an_EE_TS.columns)
print (reg_cols)

# We have 51 below, because in total there are 515 days in 17 months
# we have
st_yr = 2008
end_yr = 2021
no_days = (end_yr - st_yr + 1) * 366 # 14 years, each year 366 days!
no_steps = no_days // regular_window_size

nrows = no_steps * len(ID_list)
output_df = pd.DataFrame(data = None,
                         index = np.arange(nrows), 
                         columns = reg_cols)
########################################################################################

counter = 0

for a_poly in ID_list:
    if (counter % 300 == 0):
        print (counter)
    curr_field = an_EE_TS[an_EE_TS[IDcolName]==a_poly].copy()
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)
    
    ################################################################
    regularized_TS = nc.regularize_a_field(a_df = curr_field, \
                                           V_idks = indeks, \
                                           interval_size = regular_window_size,\
                                           start_year=st_yr, \
                                           end_year=end_yr)
    
    regularized_TS = nc.fill_theGap_linearLine(a_regularized_TS = regularized_TS, V_idx = indeks)
    if (counter == 0):
        print ("output_df columns:",     output_df.columns)
        print ("regularized_TS.columns", regularized_TS.columns)
    
    ################################################################
    row_pointer = no_steps * counter
    
    """
    The reason for the following line is that we assume all years are 366 days!
    so, the actual thing might be smaller!
    """
    right_pointer = row_pointer + min(no_steps, regularized_TS.shape[0])
    output_df[row_pointer: right_pointer] = regularized_TS.values
    counter += 1


####################################################################################
###
###                   Write the outputs
###
####################################################################################
output_df.drop_duplicates(inplace=True)
output_df.dropna(inplace=True)

output_df.to_csv(out_name, index = False)
end_time = time.time()
print ("it took {:.0f} minutes to run this code.".format((end_time - start_time)/60))

