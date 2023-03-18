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
regular_window_size = 10

# do the following since walla walla has two parts and we have to use walla_walla in terminal
print ("Terminal Arguments are: ")
print (indeks)
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
            
param_dir = "/home/hnoorazar/remote_sensing_codes/parameters/"

data_base = "/data/hydro/users/Hossein/remote_sensing/02_Eastern_WA_EE_TS/2Years/"
data_dir = data_base

output_dir = "/data/hydro/users/Hossein/remote_sensing/03_Regularized_TS/"
os.makedirs(output_dir, exist_ok=True)

########################################################################################
###
###                   process data
###
########################################################################################

an_EE_TS = pd.read_csv(data_dir + f_name, low_memory=False)
an_EE_TS.head(2)

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
reg_cols = [IDcolName, indeks, "human_system_start_time"]

# We have 51 below, because in total there are 515 days in 17 months
# we have

no_days = 13 * 366 # 13 years, each year 366 days!
regular_window_size = 10
no_steps = int(no_days/regular_window_size)

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
                                           interval_size = regular_window_size)
    
    ################################################################
    row_pointer = no_steps * counter
    output_df[row_pointer: row_pointer + no_steps] = regularized_TS.values
    counter += 1


####################################################################################
###
###                   Write the outputs
###
####################################################################################

out_name = output_dir + "00_noJumpsRegularized_" + indeks + ".csv"
os.makedirs(output_dir, exist_ok=True)
output_df.to_csv(out_name, index = False)

end_time = time.time()
print(end_time - start_time)





