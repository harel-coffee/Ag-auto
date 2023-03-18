####
#### July 27, 2020
####

"""
  Regularize the EVI and NDVI of fields in Grant, 2017.
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
import remote_sensing_core as rc
import remote_sensing_core as rcp

####################################################################################
###
###      Parameters                   
###
####################################################################################

# indeks = sys.argv[1]
# county = "Grant"
# SF_year = 2017

indeks = sys.argv[1]
SF_year = sys.argv[2]
county = sys.argv[3]
cloud_type = sys.argv[4]

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
param_dir = "/home/hnoorazar/remote_sensing_codes/parameters/"

data_base = "/data/hydro/users/Hossein/remote_sensing/02_Eastern_WA_EE_TS/2Years/" + cloud_type + "/"
data_dir = data_base + "/00_outliers_removed/"

########################################################################################
###
###                   process data
###
########################################################################################

f_name = "00_noOutlier_" + county + "_SF_" + str(SF_year) + "_" + indeks + ".csv"
an_EE_TS = pd.read_csv(data_dir + f_name, low_memory=False)

########################################################################################

output_dir = data_base + "/01_jumps_removed/"
os.makedirs(output_dir, exist_ok=True)

########################################################################################

an_EE_TS = rc.initial_clean(df = an_EE_TS, column_to_be_cleaned = indeks)
an_EE_TS.head(2)

###
### List of unique polygons
###
polygon_list = an_EE_TS['ID'].unique()
print(len(polygon_list))

########################################################################################
###
###  initialize output data.
###

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

    no_Outlier_TS = rc.correct_big_jumps_1DaySeries(dataTMS_jumpie = curr_field, 
                                                    give_col = indeks, 
                                                    maxjump_perDay = 0.015)

    output_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
    counter += 1
    row_pointer += curr_field.shape[0]


output_df = rc.add_human_start_time_by_YearDoY(output_df)
####################################################################################
###
###                   Write the outputs
###
####################################################################################

out_name = output_dir + "01_outlier_n_jump_removed_" + county + "_SF_" + str(SF_year) + "_" + indeks + ".csv"
os.makedirs(output_dir, exist_ok=True)
output_df.to_csv(out_name, index = False)

end_time = time.time()
print(end_time - start_time)





