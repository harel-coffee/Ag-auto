####
#### Nov 3, 2021
####

"""
  This script is modified from 01_fillGaps_2Yrs.py from remote sensing project.
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
SF_year = int(sys.argv[2])
county = sys.argv[3]
cloud_type = sys.argv[4]
jumps = sys.argv[5]

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
param_dir = "/home/hnoorazar/remote_sensing_codes/parameters/"

data_dir  = "/data/hydro/users/Hossein/remote_sensing/03_Regularized_TS/" + cloud_type + "/2Yrs/"
output_dir="/data/hydro/users/Hossein/remote_sensing/03_Regularized_TS/" + cloud_type + "/2Yrs/"
os.makedirs(output_dir, exist_ok=True)

########################################################################################
###
###                   updates based on wJumps or noJumps
###
########################################################################################
if jumps == "noJumps":
  data_dir = data_dir + "noJump_Regularized/"
  f_name = "00_noJumpsRegularized_" + county + "_SF_" + str(SF_year) + "_" + indeks + ".csv"
  output_dir = output_dir + "noJump_Regularized/"
  os.makedirs(output_dir, exist_ok=True)
else:
  f_name = "00_Regularized_" + county + "_SF_" + str(SF_year) + "_" + indeks + ".csv"

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
polygon_list = an_EE_TS['ID'].unique()
print(len(polygon_list))

########################################################################################
###
###  initialize output data. all polygons in this case
###  will have the same length. 
###
###

reg_cols = an_EE_TS.columns

no_days = 13 * 366 # 13 years, each year 366 days!
regular_window_size = 10
no_steps = int(no_days/regular_window_size)

nrows = no_steps * len(polygon_list)
output_df = pd.DataFrame(data = None,
                         index = np.arange(nrows), 
                         columns = reg_cols)
########################################################################################

counter = 0

for a_poly in polygon_list:
    if (counter % 10 == 0):
        print (counter)
    curr_field = an_EE_TS[an_EE_TS['ID']==a_poly].copy()
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)
    
    ################################################################
    curr_field = nc.fill_theGap_linearLine(a_regularized_TS = curr_field, V_idx = indeks)

    ################################################################
    row_pointer = no_steps * counter
    output_df[row_pointer: row_pointer + no_steps] = curr_field.values
    counter += 1


# nc.convert_human_system_start_time_to_systemStart_time(output_df)
####################################################################################
###
###                   Write the outputs
###
####################################################################################
out_name = output_dir + "01_Regular_filledGap_" + indeks + ".csv"
os.makedirs(output_dir, exist_ok=True)
output_df.to_csv(out_name, index = False)

end_time = time.time()
print(end_time - start_time)





