"""
Compute acreage per cultivar for double peak data
Do this for all cultivars labeled as double peaked, and then, also, filter
the cultivars by those that are potentially double cropped. (i.e. Filter out orchard stuff.)
"""

import csv
import numpy as np
import pandas as pd
# import geopandas as gpd
from IPython.display import Image
# from shapely.geometry import Point, Polygon
from math import factorial
import datetime
import time
import scipy
import os, os.path

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from patsy import cr

# from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sb


import sys
start_time = time.time()

####################################################################################
#
#      Local Computer directories and parameters
#
####################################################################################

sys.path.append('/Users/hn/Documents/00_GitHub/Ag/remote_sensing/python/')
data_dir_base = "/Users/hn/Documents/01_research_data/remote_sensing/02_peaks_and_plots/"
param_dir = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"

county = "Grant"
year = 2016
look_ahead = 8
freedom_dg = 9

####################################################################################
#
#      Aeolus directories and parameters
#
####################################################################################

sys.path.append('/home/hnoorazar/remote_sensing_codes/')
data_dir_base = "/data/hydro/users/Hossein/remote_sensing/01_NDVI_TS/Grant/No_EVI/Grant_10_cloud/"
param_dir = "/home/hnoorazar/remote_sensing_codes/parameters/"

county = "Grant"
year = 2016
look_ahead = 8
freedom_dg = 9

county = str(sys.argv[1])
year = int(sys.argv[2])
freedom_df = int(sys.argv[3])
look_ahead = int(sys.argv[4])

####################################################################################
#
#      Read Parameters
#
####################################################################################
double_crop_potential_plants = pd.read_csv(param_dir + "double_crop_potential_plants.csv")
double_crop_potential_plants.head(2)

####################################################################################
#
#       Read data
#
####################################################################################

data_dir = data_dir_base + county + "_" + str(year) + "/"

double_dt = pd.read_csv(data_dir + "LA_" + str(look_ahead) + "_df_" + str(freedom_dg) + "_double_polygons.csv")
double_dt.tail(2)


####################################################################################
#
#       Clean data
#
####################################################################################

#
# I saved an additonal empty row at the end
# remove it
#
last_row_id = double_dt.shape[0] - 1
double_dt = double_dt.drop([last_row_id], axis=0)
#
# convert the "year" column to integer.
#
double_dt = double_dt.astype({"year": int})
double_dt.tail(2)

####################################################################################
#
#       Compute acreage of each double peaked cultivar
#
####################################################################################
#
# all cultivars
#
acreage_per_cultivar_all = double_dt.groupby(["county", "year", "CropTyp"]).ExctAcr.sum().reset_index()


####
####    Potential Cultivars
####
#
# Filter the double-peaked cultivars
#
double_crop_poten = double_dt[double_dt.CropTyp.isin(double_crop_potential_plants['Crop_Type'])]
acr_per_potential_doubles = double_crop_poten.groupby(["county", "year", "CropTyp"]).ExctAcr.sum().reset_index()
acr_per_potential_doubles.head(2)

####################################################################################
#
#       Save the data
#
####################################################################################

#
# Saving path
#
out_dir = data_dir + "/acreage_tables/"
os.makedirs(out_dir, exist_ok=True)

all_acr_path_name = out_dir + "all_cult_acr_LA_" + str(look_ahead) + "_df_"  + str(freedom_dg) + ".csv"

potential_double_acr_path_name = out_dir + "potential_cult_acr_LA_" + \
                                 str(look_ahead) + "_df_"  + str(freedom_dg) + ".csv"

acreage_per_cultivar_all.to_csv(all_acr_path_name, index = False)
acr_per_potential_doubles.to_csv(potential_double_acr_path_name, index = False)

end_time = time.time()
print(end_time - start_time)




