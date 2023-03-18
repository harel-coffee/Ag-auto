####
#### July 27, 2020
####

"""
  Regularize the EVI and NDVI of fields in Grant, 2017.
"""
import glob
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

indeks = sys.argv[1]
SF_year = sys.argv[2]
cloud_type = sys.argv[3]

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
param_dir = "/home/hnoorazar/remote_sensing_codes/parameters/"

data_base = "/data/hydro/users/Hossein/remote_sensing/02_Eastern_WA_EE_TS/2Years/" + cloud_type + "/"
data_dir = data_base + "/01_jumps_removed/"
output_dir = data_base + "/02_noOutlierNoJumpMerged/"
os.makedirs(output_dir, exist_ok=True)

########################################################################################
###
###  initialize output data.
###

# list of files
# pattern = '/**/*' + str(SF_year) + "_" + indeks
# FilenamesList = glob.glob(data_dir + pattern, recursive=True)

pattern = '*' + str(SF_year) + "_" + indeks + ".csv"
FilenamesList = glob.glob(data_dir + pattern)


print ("data_dir is:")
print (data_dir)

print ("pattern is: ")
print (pattern)

print ("argument is")
print (data_dir + pattern)

print ("FilenamesList is:")
print (FilenamesList)

output_df = pd.DataFrame(data = None)

for file in FilenamesList:
    an_EE_TS = pd.read_csv(file, low_memory=False)
    frames = [output_df, an_EE_TS]
    output_df = pd.concat(frames)


out_name = output_dir + "Eastern_WA_SF_" + str(SF_year) + "_70cloud_" + indeks + ".csv"
os.makedirs(output_dir, exist_ok=True)
output_df.to_csv(out_name, index = False)

end_time = time.time()
print(end_time - start_time)





