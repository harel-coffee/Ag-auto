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

import scipy
import scipy.signal
import os, os.path

import time
import datetime
from datetime import date
from patsy import cr

from IPython.display import Image
from sklearn.linear_model import LinearRegression
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# from pprint import pprint
import seaborn as sb
import matplotlib.pyplot as plt

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

# county = "Grant"
# SF_year = 2017

indeks = sys.argv[1]
SF_year = int(sys.argv[2])
county = sys.argv[3]
cloud_type = sys.argv[4]

# do the following since walla walla has two parts and we have to use walla_walla in terminal
county = county.replace("_", " ")
print ("Terminal Arguments are: ")
print (indeks)
print (SF_year)
print (county)
print (cloud_type)
print ("__________________________________________")

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
param_dir = "/home/hnoorazar/remote_sensing_codes/parameters/"

data_base = "/data/hydro/users/Hossein/remote_sensing/02_Eastern_WA_EE_TS/2Years/"
data_dir = data_base + cloud_type + "/"

########################################################################################
###
###                   process data
###
########################################################################################

# if "max" in cloud_type:
#     f_name = "Eastern_WA_" + str(SF_year) + "_" + cloud_type.split("_")[0] + "cloud_selectors_max.csv"
# else:
f_name = "Eastern_WA_" + str(SF_year) + "_" + cloud_type.split("_")[0] + "cloud_selectors.csv"

an_EE_TS = pd.read_csv(data_dir + f_name, low_memory=False)

print ("data_dir is ")
print (data_dir)

print ("List of unique counties is: ")
print (an_EE_TS.county.unique())

########################################################################################

an_EE_TS = an_EE_TS[an_EE_TS['county'] == county] # Filter county

if not('SF_year' in an_EE_TS.columns):
    an_EE_TS['SF_year'] = SF_year

print ("Dimension of the data is: ")
print (an_EE_TS.shape)
print ("__________________________________________")

print ("List of unique counties is: ")
print (an_EE_TS.county.unique())

########################################################################################

output_dir = data_dir + "/00_outliers_removed/"
os.makedirs(output_dir, exist_ok=True)
########################################################################################

an_EE_TS = rc.initial_clean(df = an_EE_TS, column_to_be_cleaned = indeks)
print ("After initial cleaning we have: ")
print (an_EE_TS.shape)

###
### List of unique polygons
###
polygon_list = an_EE_TS['ID'].unique()

print ("Number of unique fields is: ")
print(len(polygon_list))
print ("__________________________________________")


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

    #********#********#********#********#********#********
    #
    #    Set negative EVI values to zero.
    #
    ##********#********#********#********#********#********

    curr_field.loc[curr_field[indeks] < 0 , indeks] = 0

    no_Outlier_TS = rc.interpolate_outliers_EVI_NDVI(outlier_input = curr_field, given_col = indeks)

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
county = county.replace(" ", "_")
out_name = output_dir + "00_noOutlier_" + county + "_SF_" + str(SF_year) + "_" + indeks + ".csv"

os.makedirs(output_dir, exist_ok=True)
output_df.to_csv(out_name, index = False)

end_time = time.time()
print(end_time - start_time)





