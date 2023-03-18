"""
Peak and plot simultaneously
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

# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

####################################################################################
###
###                      Core path
###
####################################################################################
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/remote_sensing/python/')

####################################################################################
###
###                      Directories
###
####################################################################################
data_dir = "/Users/hn/Documents/01_research_data/remote_sensing/" + \
           "01_NDVI_TS/00_TS_Batches_70_cloud/"

param_dir = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"
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

data_dir = "/data/hydro/users/Hossein/remote_sensing/01_NDVI_TS/batches_70_cloud/"
param_dir = "/home/hnoorazar/remote_sensing_codes/parameters/"
####################################################################################
###
###                   Parameters
###
####################################################################################

# double_crop_potential_plants = pd.read_csv(param_dir + "double_crop_potential_plants.csv")
# double_crop_potential_plants.head(2)

####################################################################################
###
###                   Import remote cores
###
####################################################################################

import remote_sensing_core as rc
import remote_sensing_core as rcp

output_dir = data_dir
plot_dir_base = data_dir + "/double_potential_plots/"


####################################################################################
###
###                   Data Reading
###
####################################################################################

file_names = ["batch_A_2017_TS"]
file_N = file_names[0]
an_EE_TS = pd.read_csv(data_dir + file_N + ".csv")

################################################
###
### Just keep the potential fields
###
################################################

# an_EE_TS = an_EE_TS[an_EE_TS.CropTyp.isin(double_crop_potential_plants['Crop_Type'])]


####################################################################################
###
###                   process data
###
####################################################################################
#
#  Pick Grant
#
an_EE_TS = an_EE_TS[an_EE_TS['county']=="Grant"]
an_EE_TS.county.unique()

#
# The following columns do not exist in the old data
#
if not('DataSrc' in a_df.columns):
    print ("Data source is being set to NA")
    a_df['DataSrc'] = "NA"

if not('CovrCrp' in a_df.columns):
    print ("Data source is being set to NA")
    a_df['CovrCrp'] = "NA"

an_EE_TS_NDVI = rc.initial_clean_NDVI(an_EE_TS)
an_EE_TS_EVI = rc.initial_clean_EVI(an_EE_TS)
an_EE_TS_EVI.head(2)

### List of unique polygons
polygon_list = an_EE_TS_NDVI['geo'].unique()
print(len(polygon_list))

counter = 0
for a_poly in polygon_list:
    if (counter%1000 == 0):
        print (counter)
    counter += 1
    curr_field_NDVI = an_EE_TS_NDVI[an_EE_TS_NDVI['geo']==a_poly]
    curr_field_EVI = an_EE_TS_EVI[an_EE_TS_EVI['geo']==a_poly]

    year = int(curr_field_NDVI['year'].unique())
    plant = curr_field_NDVI['CropTyp'].unique()[0]
    
    # Take care of names, replace "/" and "," and " " by "_"
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")

    county = curr_field_NDVI['county'].unique()[0]
    TRS = curr_field_NDVI['TRS'].unique()[0]

    x_NDVI = curr_field_NDVI['doy']
    y_NDVI = curr_field_NDVI['NDVI']

    x_EVI = curr_field_EVI['doy']
    y_EVI = curr_field_EVI['EVI']

    #############################################
    ###
    ###             plot
    ###
    #############################################        
    sub_out = "/plant_based_plots/" + file_N + "/" + county + "/" +  plant + "/"
    plot_path = plot_dir_base + sub_out
    os.makedirs(plot_path, exist_ok=True)
    if (len(os.listdir(plot_path))<100):
        plot_title = county + ", " + plant + ", " + str(year) + " (" + TRS + ")"
        sb.set();
        fig, ax = plt.subplots(figsize=(8,6));
        ax.plot(x_NDVI, y_NDVI, label="NDVI");
        ax.plot(x_EVI, y_EVI, 'r', label="EVI")
        ax.set_title(plot_title);
        ax.set(xlabel='DoY', ylabel='NDVI & EVI')
        ax.legend(loc="best");

        fig_name = plot_path + county + "_" + plant + "_" + str(year) + "_" + str(counter) + '.png'
        plt.savefig(fname = fig_name, \
                    dpi=400,
                    bbox_inches='tight')
        plt.close()
        del(plot_path, sub_out, county, plant, year)

####################################################################################
###
###                   Compute double crop area
###
####################################################################################


end_time = time.time()
print(end_time - start_time)
