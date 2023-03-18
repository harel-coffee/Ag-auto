####
#### July 23, 2020
####

"""
  remove outliers that are beyond -1 and 1 in NDVI and EVI.
  Looking at 2017 data I did not see any NDVI beyond those boundaries. 
  EVI had outliers only.
"""

import csv
import numpy as np
import pandas as pd
# import geopandas as gpd
from IPython.display import Image
# from shapely.geometry import Point, Polygon
from math import factorial
import scipy
import scipy.signal
import os, os.path

from datetime import date
import datetime
import time

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from patsy import cr

# from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sb

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

sys.path.append('/home/hnoorazar/remote_sensing_codes/')

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################

data_dir = "/data/hydro/users/Hossein/remote_sensing/03_Regularized_TS/2Yrs/"
param_dir = "/home/hnoorazar/remote_sensing_codes/parameters/"

####################################################################################
###
###                   Import remote cores
###
####################################################################################

import remote_sensing_core as rc
import remote_sensing_core as rcp

####################################################################################
###
###      Parameters                   
###
####################################################################################

indeks = sys.argv[1]
county = "Grant"
SF_year = 2017

########################################################################################
###
###                   process data
###
########################################################################################

f_name = "01_Regular_filledGap_" + county + "_SF_" + str(SF_year) + "_" + indeks + ".csv"
an_EE_TS = pd.read_csv(data_dir + f_name, low_memory=False)

"""
   The following three lines are written because we needed system_start_time
   for linear interpolation. But we had missed that in the process of regularization.
   So, we added them here, and we added this piece back into the regularization code as well.
   So, we do not need them here anymore.
"""
# rc.convert_human_system_start_time_to_systemStart_time(an_EE_TS)
# nn = data_dir + f_name
# an_EE_TS.to_csv(nn, index = False)


########################################################################################

# an_EE_TS = an_EE_TS[an_EE_TS['county'] == county] # Filter Grant
an_EE_TS['SF_year'] = SF_year

########################################################################################

output_dir = data_dir + "/outliers_removed/"
os.makedirs(output_dir, exist_ok=True)
########################################################################################

if (indeks == "EVI"):
    an_EE_TS = rc.initial_clean_EVI(an_EE_TS)
else:
    an_EE_TS = rc.initial_clean_NDVI(an_EE_TS)

an_EE_TS.head(2)

###
### List of unique polygons
###
polygon_list = an_EE_TS['ID'].unique()
print(len(polygon_list))

########################################################################################

output_df = pd.DataFrame(data = None,
                         index = np.arange(an_EE_TS.shape[0]), 
                         columns = an_EE_TS.columns)

counter = 0
row_pointer = 0

for a_poly in polygon_list:
    if (counter % 300 == 0):
        print (counter)
    curr_field = an_EE_TS[an_EE_TS['ID']==a_poly].copy()
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['image_year', 'doy'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)
    
    # print ("print(curr_field.shape")
    # print(curr_field.shape)
    # print ("__________________________________________")
    ################################################################
    no_Outlier_TS = rc.interpolate_outliers_EVI_NDVI(outlier_input = curr_field, given_col = indeks)

    output_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
    counter += 1
    row_pointer += curr_field.shape[0]


####################################################################################
###
###                   Write the outputs
###
####################################################################################
                                         
out_name = output_dir + "00_noOutlier_regular_" + county + "_SF_" + str(SF_year) + "_" + indeks + ".csv"

os.makedirs(output_dir, exist_ok=True)
output_df.to_csv(out_name, index = False)

end_time = time.time()
print(end_time - start_time)





