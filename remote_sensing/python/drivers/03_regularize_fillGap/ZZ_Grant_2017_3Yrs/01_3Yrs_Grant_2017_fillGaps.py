####
#### June 24, 2019
####

"""
  Regularize the EVI and NDVI of fields in Gran, 2017.
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

data_dir = "/data/hydro/users/Hossein/remote_sensing/03_Regularized_TS/3Yrs/"
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
regular_window_size = 10
########################################################################################
###
###                   process data
###
########################################################################################

f_name = "Regularized_" + county + "_SF_" + str(SF_year) + "_" + indeks + ".csv"
an_EE_TS = pd.read_csv(data_dir + f_name, low_memory=False)

########################################################################################

output_dir = "/data/hydro/users/Hossein/remote_sensing/03_Regularized_TS/3Yrs/"
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
###
###  initialize output data. all polygons in this case
###  will have the same length. 
###  9 steps in the first three months, followed by 36 points in the full year,
###  9 months in the last year
###

reg_cols = an_EE_TS.columns

nrows = 54 * len(polygon_list)
output_df = pd.DataFrame(data = None,
                         index = np.arange(nrows), 
                         columns = reg_cols)
########################################################################################

counter = 0

for a_poly in polygon_list:
    if (counter % 300 == 0):
        print (counter)
    curr_field = an_EE_TS[an_EE_TS['ID']==a_poly].copy()
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['image_year', 'doy'], inplace=True)
    
    curr_field.reset_index(drop=True, inplace=True)
    
    print ("print(curr_field.shape)")
    print(curr_field.shape)
    print ("__________________________________________")
    ################################################################
    curr_field = rc.fill_theGap_linearLine(curr_field, indeks = indeks, SF_year = 2017)

    ################################################################
    row_pointer = 54 * counter
    output_df[row_pointer: row_pointer+54] = curr_field.values
    counter += 1


####################################################################################
###
###                   Write the outputs
###
####################################################################################

out_name = output_dir + "Regular_filledGap" + county + "_SF_" + str(SF_year) + "_" + indeks + ".csv"
output_df.to_csv(out_name, index = False)

end_time = time.time()
print(end_time - start_time)





