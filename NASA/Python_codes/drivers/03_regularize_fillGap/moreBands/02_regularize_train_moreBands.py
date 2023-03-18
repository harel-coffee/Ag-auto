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

# search path for modules# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

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
county = sys.argv[2]
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
data_base = "/data/hydro/users/Hossein/NASA/01_raw_GEE/moreBands_train_set/"
data_dir = data_base

output_dir = "/data/hydro/users/Hossein/NASA/04_regularized_TS/"
os.makedirs(output_dir, exist_ok=True)

########################################################################################
###
###                   band names
###
########################################################################################
L7_dict = {"SR_B1":"blue",
           "SR_B2":"green",
           "SR_B3":"red",
           "SR_B4":"NIR",
           "SR_B5":"short_I_1",
           "SR_B7":"short_I_2",
           }

L8_dict = {"SR_B2":"blue",
           "SR_B3":"green",
           "SR_B4":"red",
           "SR_B5":"NIR",
           "SR_B6":"short_I_1",
           "SR_B7":"short_I_2",
           }

########################################################################################
###
###                   process data
###
########################################################################################

if county == "Monterey2014":
    f_names = ["X",
               "Y"]

elif county == "Walla2015":
    f_names = ["L7_T1C2L2_Scaled_Walla2015_2015-01-01_2016-01-01.csv",
               "L8_T1C2L2_Scaled_Walla2015_2015-01-01_2016-01-01.csv"]

elif county == "AdamBenton2016":
    f_names = ["L7_T1C2L2_Scaled_AdamBenton2016_2016-01-01_2017-01-01.csv",
               "L8_T1C2L2_Scaled_AdamBenton2016_2016-01-01_2017-01-01.csv"]

elif county == "Grant2017":
    f_names = ["L7_T1C2L2_Scaled_Grant2017_2017-01-01_2018-01-01.csv",
               "L8_T1C2L2_Scaled_Grant2017_2017-01-01_2018-01-01.csv"]

elif county == "FranklinYakima2018":
    f_names = ["L7_T1C2L2_Scaled_FranklinYakima2018_2018-01-01_2019-01-01.csv",
               "L8_T1C2L2_Scaled_FranklinYakima2018_2018-01-01_2019-01-01.csv"]

out_name = output_dir + "regular_" + county + "_" + indeks + "_moreBands_train_set.csv"

L7 = pd.read_csv(data_dir + f_names[0])
L8 = pd.read_csv(data_dir + f_names[1])
#
# Change the column names so they are consistent
#
L7.rename(columns = L7_dict, inplace = True)
L8.rename(columns = L8_dict, inplace = True)


L7 = L7[['ID', 'system_start_time', indeks]]
L8 = L8[['ID', 'system_start_time', indeks]]

L7 = L7[L7[indeks].notna()]
L8 = L8[L8[indeks].notna()]

L578 = pd.concat([L7, L8])
del(L7, L8)
L578["ID"] = L578["ID"].astype(str)
IDs = np.sort(L578["ID"].unique())

L578 = nc.add_human_start_time_by_system_start_time(L578)
L578 = nc.initial_clean(df = L578, column_to_be_cleaned = indeks)

an_EE_TS = L578.copy()
del(L578)

if ("system_start_time" in list(an_EE_TS.columns)):
    an_EE_TS.drop(["system_start_time"], axis=1, inplace=True)

an_EE_TS['human_system_start_time'] = pd.to_datetime(an_EE_TS['human_system_start_time'])
an_EE_TS["ID"] = an_EE_TS["ID"].astype(str)
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
st_yr = an_EE_TS.human_system_start_time.dt.year.min()
end_yr = an_EE_TS.human_system_start_time.dt.year.max()
no_days = (end_yr - st_yr + 1) * 366 # 14 years, each year 366 days!
no_steps = int(np.ceil(no_days / regular_window_size)) # no_days // regular_window_size

nrows = no_steps * len(ID_list)
output_df = pd.DataFrame(data = None,
                         index = np.arange(nrows), 
                         columns = reg_cols)

print('st_yr is {}!'.format(st_yr))
print('end_yr is {}!'.format(end_yr))
print('nrows is {}!'.format(nrows))
########################################################################################

counter = 0
row_pointer = 0
for a_poly in ID_list:
    if (counter % 1000 == 0):
        print (counter)
    curr_field = an_EE_TS[an_EE_TS[IDcolName]==a_poly].copy()
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)
    
    ################################################################
    regularized_TS = nc.regularize_a_field_annual_basis(a_df = curr_field, \
                                                        V_idks = indeks, \
                                                        interval_size = regular_window_size,\
                                                        start_year = st_yr, \
                                                        end_year = end_yr)
    
    regularized_TS = nc.fill_theGap_linearLine(a_regularized_TS = regularized_TS, V_idx=indeks)
    if (counter == 0):
        print ("output_df columns:",     output_df.columns)
        print ("regularized_TS.columns", regularized_TS.columns)
    
    ################################################################
    # row_pointer = no_steps * counter
    
    """
       The reason for the following line is that we assume all years are 366 days!
       so, the actual thing might be smaller!
    """
    # why this should not work?: It may leave some empty rows in output_df
    # but we drop them at the end.
    output_df[row_pointer : (row_pointer+regularized_TS.shape[0])] = regularized_TS.values
    row_pointer += regularized_TS.shape[0]

    # right_pointer = row_pointer + min(no_steps, regularized_TS.shape[0])
    # print('right_pointer - row_pointer + 1 is {}!'.format(right_pointer - row_pointer + 1))
    # print('len(regularized_TS.values) is {}!'.format(len(regularized_TS.values)))
    # try:
    #     ### I do not know why the hell the following did not work for training set!
    #     ### So, I converted this to try-except statement! hopefully, this will
    #     ### work, at least as temporary remedy! Why it worked well with 2008-2021 but not 2013-2015
    #     output_df[row_pointer: right_pointer] = regularized_TS.values
    # except:
    #     output_df[row_pointer: right_pointer+1] = regularized_TS.values
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

