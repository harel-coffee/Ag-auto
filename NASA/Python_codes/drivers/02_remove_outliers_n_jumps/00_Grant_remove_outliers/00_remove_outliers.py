####
#### Nov 2, 2021
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

# county = "Grant"
# SF_year = 2017

indeks = sys.argv[1]
random_or_all = sys.argv[2]

# do the following since walla walla has two parts and we have to use walla_walla in terminal
print ("Terminal Arguments are: ")
print (indeks)
print (random_or_all)
print ("__________________________________________")

import random
random.seed(10)
np.random.seed(seed=10)
randCount = 100

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

print ("data_dir is: " +  data_dir)
print ("output_dir is: " + output_dir)

########################################################################################
###
###                   process data
###
########################################################################################
f_names = ["L5_T1C2L2_Scaled_int_Grant_Irr_2008_2018_2cols_102008-01-01_2012-05-05.csv", 
           "L7_T1C2L2_Scaled_int_Grant_Irr_2008_2018_2cols_10_2008-01-01_2021-09-23.csv",
           "L8_T1C2L2_Scaled_int_Grant_Irr_2008_2018_2cols_10_2008-01-01_2021-10-14.csv"]

L5 = pd.read_csv(data_dir + f_names[0], low_memory=False)
L7 = pd.read_csv(data_dir + f_names[1], low_memory=False)
L8 = pd.read_csv(data_dir + f_names[2], low_memory=False)

L5.drop([NoVI], axis=1, inplace=True)
L5 = L5[L5[indeks].notna()]

L7.drop([NoVI], axis=1, inplace=True)
L7 = L7[L7[indeks].notna()]

L8.drop([NoVI], axis=1, inplace=True)
L8 = L8[L8[indeks].notna()]

IDs = np.sort(L5[IDcolName].unique())
L578 = pd.concat([L5, L7, L8])
del(L5, L7, L8)

########################################################
#######
#######   Choose X random fields
#######
if random_or_all == "random":
    IDs = random.sample(list(IDs), k=randCount)
    L578 = L578[L578.ID.isin(IDs)]
    L578.reset_index(drop=True, inplace=True)

L578 = nc.add_human_start_time_by_system_start_time(L578)

print ("Number of unique fields is: ")
print(len(IDs))
print ("__________________________________________")

#########################################################
print ("Dimension of the data is: " + str(L578.shape))
print ("__________________________________________")

#########################################################


L578 = nc.initial_clean(df = L578, column_to_be_cleaned = indeks)


#########################################################

output_df = pd.DataFrame(data = None,
                         index = np.arange(L578.shape[0]), 
                         columns = L578.columns)

counter = 0
row_pointer = 0

for a_poly in IDs:
    if (counter % 300 == 0):
        print (counter)
    curr_field = L578[L578[IDcolName]==a_poly].copy()

    # small fields may have nothing in them!
    if curr_field.shape[0] > 2:
        #********#********#********#********#********#********
        #
        #    Set negative indeks values to zero.
        #
        ##********#********#********#********#********#********
        """
         we are killing some of the ourliers here and put them
         in the normal range! do we want to do it here? No, lets do it later.
        """
        # curr_field.loc[curr_field[indeks] < 0 , indeks] = 0 
        no_Outlier_TS = nc.interpolate_outliers_EVI_NDVI(outlier_input = curr_field, given_col = indeks)
        no_Outlier_TS.loc[no_Outlier_TS[indeks] < 0 , indeks] = 0 

        """
        it is possible that for a field we only have x=2 data points
        where all the EVI/NDVI is outlier. Then, there is nothing to 
        use for interpolation. So, hopefully interpolate_outliers_EVI_NDVI is returning an empty data table.
        """
        if len(no_Outlier_TS) > 0:
            output_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
            counter += 1
            row_pointer += curr_field.shape[0]

####################################################################################
###
###                   Write the outputs
###
####################################################################################

if random_or_all == "random":
    out_name = output_dir + "noOutlier_int_Grant_Irr_2008_2018_" + indeks + "_" + str(randCount) + "randomfields.csv"
else:
    out_name = output_dir + "noOutlier_int_Grant_Irr_2008_2018_" + indeks + ".csv"

output_df.drop_duplicates(inplace=True)
# output_df.dropna(inplace=True)
output_df.to_csv(out_name, index = False)

end_time = time.time()
print ("it took {:.0f} minutes to run this code.".format((end_time - start_time)/60))

