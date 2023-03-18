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
data_dir = "/Users/hn/Documents/01_research_data/remote_sensing/01_NDVI_TS/"
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
plot_dir_base = data_dir


####################################################################################
###
###                   Data Reading
###
####################################################################################

file_names = ["Grant_2017_4_Kirti_plot_TS_70_cloud"]
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
# The following columns do not exist in the old data
#
print (an_EE_TS.county.unique())
an_EE_TS = an_EE_TS[an_EE_TS['county']=="Grant"]
print (an_EE_TS.county.unique())

#
# The following columns do not exist in the old data
#
if not('DataSrc' in a_df.columns):
    print ("Data source is being set to NA")
    a_df['DataSrc'] = "NA"

if not('CovrCrp' in a_df.columns):
    print ("Data source is being set to NA")
    a_df['CovrCrp'] = "NA"

an_EE_TS_NDVI = rc.initial_clean(df = an_EE_TS, column_to_be_cleaned='NDVI')
an_EE_TS_EVI = rc.initial_clean(df = an_EE_TS, column_to_be_cleaned='EVI')

an_EE_TS_BSI = rc.initial_clean(df = an_EE_TS, column_to_be_cleaned='BSI')
an_EE_TS_NDWI = rc.initial_clean(df = an_EE_TS, column_to_be_cleaned='NDWI')
an_EE_TS_PSRI = rc.initial_clean(df = an_EE_TS, column_to_be_cleaned='PSRI')
an_EE_TS_LSWI = rc.initial_clean(df = an_EE_TS, column_to_be_cleaned='LSWI')

an_EE_TS_NDVI.head(2)

### List of unique polygons
polygon_list = an_EE_TS_NDVI['geo'].unique()
print(len(polygon_list))

counter = 0
for a_poly in polygon_list:
    if (counter%1000 == 0):
        print (counter)

    curr_field_NDVI = an_EE_TS_NDVI[an_EE_TS_NDVI['geo']==a_poly]
    curr_field_EVI = an_EE_TS_EVI[an_EE_TS_EVI['geo']==a_poly]
    
    curr_field_BSI = an_EE_TS_BSI[an_EE_TS_BSI['geo']==a_poly]
    curr_field_NDWI = an_EE_TS_NDWI[an_EE_TS_NDWI['geo']==a_poly]
    curr_field_PSRI = an_EE_TS_PSRI[an_EE_TS_PSRI['geo']==a_poly]
    curr_field_LSWI = an_EE_TS_LSWI[an_EE_TS_LSWI['geo']==a_poly]

    year = int(curr_field_NDVI['year'].unique())
    plant = curr_field_NDVI['CropTyp'].unique()[0]
    
    # Take care of names, replace "/" and "," and " " by "_"
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")

    county = curr_field_NDVI['county'].unique()[0]
    TRS = curr_field_NDVI['TRS'].unique()[0]
    ID = curr_field_NDVI['ID'].unique()[0]
    source = curr_field_NDVI['Source'].unique()[0]

    x_NDVI = curr_field_NDVI['doy']
    y_NDVI = curr_field_NDVI['NDVI']

    x_EVI = curr_field_EVI['doy']
    y_EVI = curr_field_EVI['EVI']

    x_BSI = curr_field_BSI['doy']
    y_BSI = curr_field_BSI['BSI']

    x_NDWI = curr_field_NDWI['doy']
    y_NDWI = curr_field_NDWI['NDWI']

    x_PSRI = curr_field_PSRI['doy']
    y_PSRI = curr_field_PSRI['PSRI']

    x_LSWI = curr_field_LSWI['doy']
    y_LSWI = curr_field_LSWI['LSWI']

    #############################################
    ###
    ###             plot
    ###
    #############################################        
    sub_out = "/plots/bare_indices/" + plant + "/"
    plot_path = plot_dir_base + sub_out
    os.makedirs(plot_path, exist_ok=True)
    if (len(os.listdir(plot_path)) < 70):
        plot_title = county + ", " + plant + ", " + str(year) + " (" + ID + ")"
        sb.set();
        fig, ax = plt.subplots(figsize=(8,6));
        ax.plot(x_NDVI, y_NDVI, label="NDVI");
        ax.plot(x_EVI, y_EVI, label="EVI")

        ax.plot(x_BSI, y_BSI, label="BSI")
        ax.plot(x_NDWI, y_NDWI, label="NWDI")

        ax.plot(x_PSRI, y_PSRI, label="PSRI")
        ax.plot(x_LSWI, y_LSWI, label="LSWI")

        ax.set_title(plot_title);
        ax.set(xlabel='DoY', ylabel='indices values')
        ax.legend(loc="best");
        
        ax.set_title(plot_title);
        ax.set(xlabel='DoY', ylabel='indices values')
        ax.legend(loc="best");

        fig_name = plot_path + county + "_" + source + "_" + str(year) + "_ID" + str(ID) + str(counter) + '.png'
        plt.savefig(fname = fig_name, \
                    dpi=400,
                    bbox_inches='tight')
        plt.close()
        del(plot_path, sub_out, county, plant, year)
    counter += 1

####################################################################################
###
###                   Compute double crop area
###
####################################################################################


end_time = time.time()
print(end_time - start_time)

