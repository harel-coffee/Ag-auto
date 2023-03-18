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
randCount = 100

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
data_base = "/data/hydro/users/Hossein/NASA/"
data_dir = data_base + "04_regularized_TS/"

output_dir = data_base + "/05_SG_TS/"
os.makedirs(output_dir, exist_ok=True)

########################################################################################
###
###                   process data
###
########################################################################################
if random_or_all == "random":
    f_name = "regular_int_Grant_Irr_2008_2018_" + indeks + "_" + str(randCount) + "randomfields.csv"
    out_name = output_dir + "SG_int_Grant_Irr_2008_2018_" + indeks + "_" + str(randCount) + "randomfields.csv"
else:
    f_name = "regular_int_Grant_Irr_2008_2018_" + indeks + ".csv"
    out_name = output_dir + "SG_int_Grant_Irr_2008_2018_" + indeks + ".csv"

print (f_name)

an_EE_TS = pd.read_csv(data_dir + f_name, low_memory=False)
an_EE_TS['human_system_start_time'] = pd.to_datetime(an_EE_TS['human_system_start_time'])

print ("an_EE_TS dimension is:", str(an_EE_TS.shape))
print (an_EE_TS.head(2))
###
### List of unique polygons
###
ID_list = an_EE_TS[IDcolName].unique()
print("len(ID_list) is: " + str(len(ID_list)))

########################################################################################
###
###  initialize output data. all polygons in this case
###  will have the same length. 
###
counter = 0

an_EE_TS.sort_values(by=[IDcolName, 'human_system_start_time'], inplace=True)
an_EE_TS.reset_index(drop=True, inplace=True)

for a_poly in ID_list:
    if (counter % 300 == 0):
        print (counter)
    
    curr_field = an_EE_TS[an_EE_TS[IDcolName]==a_poly].copy()
    
    # Smoothen by Savitzky-Golay
    SG = scipy.signal.savgol_filter(curr_field[indeks].values, window_length=7, polyorder=3)
    SG[SG > 1 ] = 1 # SG might violate the boundaries. clip them:
    SG[SG < -1 ] = -1
    if counter == 0:
        print(curr_field.head(2))
        print(curr_field.index)
        print (an_EE_TS.loc[curr_field.index, ])
        print("len(SG) is " + str(len(SG)))
        print (SG[1:10])

    an_EE_TS.loc[curr_field.index, indeks] = SG
    counter += 1


####################################################################################
###
###                   Write the outputs
###
####################################################################################
an_EE_TS.drop_duplicates(inplace=True)
an_EE_TS.dropna(inplace=True)

an_EE_TS.to_csv(out_name, index = False)

end_time = time.time()

print ("it took {:.0f} minutes to run this code.".format((end_time - start_time)/60))
