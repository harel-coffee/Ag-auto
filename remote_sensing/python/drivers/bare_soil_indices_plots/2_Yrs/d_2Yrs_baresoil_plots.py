####
#### July 4, 2020
####


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
import scipy.signal
import os, os.path
import matplotlib

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from patsy import cr

# from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sb

from pandas.plotting import register_matplotlib_converters

import sys
start_time = time.time()

# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

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
            
data_dir = "/data/hydro/users/Hossein/remote_sensing/02_Eastern_WA_EE_TS/2Years/"
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

SF_year = int(sys.argv[1])
given_county = sys.argv[2]
irrigated_only = int(sys.argv[3])

####################################################################################
###
###                   process data
###
####################################################################################
file_N = "Eastern_WA_" + str(SF_year) + "_70cloud_selectors.csv"
a_df = pd.read_csv(data_dir + file_N, low_memory=False)

##################################################################
##################################################################
####
####  plots has to be exact. So, we need 
####  to filter out NASS, and filter by last survey date
####
##################################################################
##################################################################

a_df = a_df[a_df['county']== given_county] # Filter given_county
a_df = rc.filter_out_NASS(a_df) # Toss NASS
a_df = rc.filter_by_lastSurvey(a_df, year = SF_year) # filter by last survey date
a_df['SF_year'] = SF_year

if irrigated_only == True:
    a_df = rc.filter_out_nonIrrigated(a_df)
    output_Irr = "irrigated_only"
else:
    output_Irr = "non_irrigated_only"
    a_df = rc.filter_out_Irrigated(a_df)

##################################################################

output_dir = "/data/hydro/users/Hossein/remote_sensing/05_bare_soil_plots/" + \
             given_county + "_" + str(SF_year)

plot_dir_base = output_dir
print ("plot_dir_base is " + plot_dir_base)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir_base, exist_ok=True)

######################

# The following columns do not exist in the old data
#

if not('CovrCrp' in a_df.columns):
    print ("Data source is being set to NA")
    a_df['CovrCrp'] = "NA"

####################################################################################

an_EE_TS_NDVI = rc.initial_clean(df = a_df, column_to_be_cleaned='NDVI')
an_EE_TS_EVI = rc.initial_clean(df = a_df, column_to_be_cleaned='EVI')
an_EE_TS_BSI = rc.initial_clean(df = a_df, column_to_be_cleaned='BSI')
an_EE_TS_NDWI = rc.initial_clean(df = a_df, column_to_be_cleaned='NDWI')
an_EE_TS_PSRI = rc.initial_clean(df = a_df, column_to_be_cleaned='PSRI')
an_EE_TS_LSWI = rc.initial_clean(df = a_df, column_to_be_cleaned='LSWI')

an_EE_TS_NDVI = rc.add_human_start_time(an_EE_TS_NDVI)
an_EE_TS_EVI = rc.add_human_start_time(an_EE_TS_EVI)
an_EE_TS_BSI = rc.add_human_start_time(an_EE_TS_BSI)
an_EE_TS_NDWI = rc.add_human_start_time(an_EE_TS_NDWI)
an_EE_TS_PSRI = rc.add_human_start_time(an_EE_TS_PSRI)
an_EE_TS_LSWI = rc.add_human_start_time(an_EE_TS_LSWI)

####################################################################################

### List of unique polygons
polygon_list = a_df['ID'].unique()
print(len(polygon_list))

counter = 0

for a_poly in polygon_list:
    if (counter%1000 == 0):
        print (counter)
    counter += 1

    ##################################################################################
    curr_field_NDVI = an_EE_TS_NDVI[an_EE_TS_NDVI['ID'] == a_poly]
    curr_field_EVI = an_EE_TS_EVI[an_EE_TS_EVI['ID'] == a_poly]
    curr_field_BSI = an_EE_TS_BSI[an_EE_TS_BSI['ID'] == a_poly]
    curr_field_NDWI = an_EE_TS_NDWI[an_EE_TS_NDWI['ID'] == a_poly]
    curr_field_PSRI = an_EE_TS_PSRI[an_EE_TS_PSRI['ID'] == a_poly]
    curr_field_LSWI = an_EE_TS_LSWI[an_EE_TS_LSWI['ID'] == a_poly]


    curr_field_NDVI = rc.initial_clean(df = curr_field_NDVI, column_to_be_cleaned='NDVI')
    curr_field_EVI  = rc.initial_clean(df = curr_field_EVI,  column_to_be_cleaned='EVI')
    curr_field_BSI  = rc.initial_clean(df = curr_field_BSI,  column_to_be_cleaned='BSI')
    curr_field_NDWI = rc.initial_clean(df = curr_field_NDWI, column_to_be_cleaned='NDWI')
    curr_field_PSRI = rc.initial_clean(df = curr_field_PSRI, column_to_be_cleaned='PSRI')
    curr_field_LSWI = rc.initial_clean(df = curr_field_LSWI, column_to_be_cleaned='LSWI')


    ##################################################################################
    # year = int(curr_field_NDVI['SF_year'].unique())
    plant = curr_field_NDVI['CropTyp'].unique()[0]
    # Take care of names, replace "/" and "," and " " by "_"
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")

    ID = curr_field_NDVI['ID'].unique()[0]
    source = curr_field_NDVI['DataSrc'].unique()[0]
    Irrigation = curr_field_NDVI['Irrigtn'].unique()[0]

    sub_out = "/" + output_Irr + "/" + plant + "/"
    plot_path = plot_dir_base + sub_out
    os.makedirs(plot_path, exist_ok=True)

    if (len(os.listdir(plot_path)) < 70):

        ##################################################################################

        x_NDVI = curr_field_NDVI['human_system_start_time']
        y_NDVI = curr_field_NDVI['NDVI']
        y_NDVI_smooth = scipy.signal.savgol_filter(y_NDVI, window_length= 7, polyorder=2)

        x_EVI = curr_field_EVI['human_system_start_time']
        y_EVI = curr_field_EVI['EVI']
        y_EVI_smooth = scipy.signal.savgol_filter(y_EVI, window_length= 7, polyorder=2)

        x_BSI = curr_field_BSI['human_system_start_time']
        y_BSI = curr_field_BSI['BSI']

        x_NDWI = curr_field_NDWI['human_system_start_time']
        y_NDWI = curr_field_NDWI['NDWI']

        x_PSRI = curr_field_PSRI['human_system_start_time']
        y_PSRI = curr_field_PSRI['PSRI']

        x_LSWI = curr_field_LSWI['human_system_start_time']
        y_LSWI = curr_field_LSWI['LSWI']

        ##################################################################################
       

        plot_title = given_county + ", " + plant + ", " + str(SF_year) + " (" + ID + ", " + source + ", "+ Irrigation + ")"

        fig, ax = plt.subplots(figsize=(20, 6));

        ax.plot(x_NDVI, y_NDVI_smooth, label="NDVI, SG 72", c = "blue");
        ax.scatter(x_NDVI, y_NDVI, label="Raw NDVI", s = 30, c = "blue");

        ax.plot(x_EVI, y_EVI_smooth , label="EVI, SG 72", c="red")
        ax.scatter(x_EVI, y_EVI, label="Raw EVI", s = 30, c="red");

        ax.plot(x_BSI, y_BSI, label="BSI", c = "gray")
        ax.plot(x_NDWI, y_NDWI, label="NWDI")

        ax.plot(x_PSRI, y_PSRI, label="PSRI")
        ax.plot(x_LSWI, y_LSWI, label="LSWI")

        ax.set_title(plot_title);
        plt.ylabel('indices values', fontsize=16)
        plt.yticks(size = 12)
        plt.xticks(x_EVI[::3], rotation = 90, size = 12)
        ax.legend(loc="best");
        plt.grid(True)

        fig_name = plot_path + given_county + "_" + plant + "_SF_year_" + str(SF_year) + "_" + ID + '.png'
        plt.savefig(fname = fig_name, \
                     dpi = 250,
                     bbox_inches='tight')
        plt.close()
        del(plot_path, sub_out)


end_time = time.time()
print ("Time it took:")
print(end_time - start_time)





