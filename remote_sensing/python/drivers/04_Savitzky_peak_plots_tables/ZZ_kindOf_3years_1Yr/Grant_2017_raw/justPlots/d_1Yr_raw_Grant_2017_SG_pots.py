####
#### May 26, 2019
####

"""
Just generate peak tables for Grant 2018 Irrigated fields 
for all cultivars; EVI and my peak finder

"""
import matplotlib.backends.backend_pdf
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

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from patsy import cr

# from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sb

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import sys
start_time = time.time()

# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

####################################################################################
###
###                      Local
###
####################################################################################

################
###
### Core path
###

sys.path.append('/Users/hn/Documents/00_GitHub/Ag/remote_sensing/python/')

################
###
### Directories
###

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

data_dir = "/data/hydro/users/Hossein/remote_sensing/02_Eastern_WA_EE_TS/3Years/"
param_dir = "/home/hnoorazar/remote_sensing_codes/parameters/"

####################################################################################
###
###                   Import remote cores
###
####################################################################################

import remote_sensing_core as rc
import remote_sensing_plot_core as rcp

####################################################################################
###
###      Parameters                   
###
####################################################################################
eleven_colors = ["gray", "lightcoral", "red", "peru",
                 "darkorange", "gold", "olive", "green",
                 "blue", "violet", "deepskyblue"]

# Sav_win_size = 9
# sav_order = 1
# delt = 0.1
irrigated_only = 1
SF_year = 2017
indeks = "EVI"

# we are creating panels where each panel
# consist of different parameters of Savitzky, 
# So, we do not need the following two

# Sav_win_size = int(sys.argv[1]) 
# sav_order = int(sys.argv[2])

# delt = float(sys.argv[1])

indeks = sys.argv[1]
irrigated_only = int(sys.argv[2])
SF_year = int(sys.argv[3])

# print ("delta = {fileShape}".format(fileShape=delt))

####################################################################################
###
###                   process data
###
####################################################################################

f_name = "Eastern_WA_" + str(SF_year) + "_70cloud_selectors.csv"
a_df = pd.read_csv(data_dir + f_name, low_memory=False)

##################################################################
##################################################################
####
####  plots has to be exact. So, we need 
#### to filter out NASS, and filter by last survey date
####
##################################################################
##################################################################

a_df = a_df[a_df['county']== "Grant"] # Filter Grant
a_df = rc.filter_out_NASS(a_df) # Toss NASS
a_df = rc.filter_by_lastSurvey(a_df, year = SF_year) # filter by last survey date

a_df['SF_year'] = SF_year

########################
######################## Do this for now, till you learn
######################## how to plot 2 years where x is DoY 
######################## 

a_df = a_df[a_df['image_year'] == SF_year]

########################
########################

if irrigated_only == True:
    a_df = rc.filter_out_nonIrrigated(a_df)
    output_Irr = "irrigated_only"
else:
    output_Irr = "non_irrigated_only"
    a_df = rc.filter_out_Irrigated(a_df)


##################################################################

output_dir = "/data/hydro/users/Hossein/remote_sensing/02_Eastern_WA_plots_tbls/" + \
             "1Yr_plots/Grant_" + str(SF_year) + "_raw_" + output_Irr + "_" + indeks + "/" 

plot_dir_base = output_dir
print ("plot_dir_base is " + plot_dir_base)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir_base, exist_ok=True)

######################

# The following columns do not exist in the old data
#
if not('DataSrc' in a_df.columns):
    print ("Data source is being set to NA")
    a_df['DataSrc'] = "NA"

if not('CovrCrp' in a_df.columns):
    print ("CovrCrp is being set to NA")
    a_df['CovrCrp'] = "NA"

if (indeks == "EVI"):
    a_df = rc.initial_clean_EVI(a_df)
else:
    a_df = rc.initial_clean_NDVI(a_df)

a_df.head(2)
an_EE_TS = a_df.copy()

### List of unique polygons
polygon_list = np.sort(an_EE_TS['ID'].unique())
print(len(polygon_list))

counter = 0

for a_poly in polygon_list:
    if (counter%1000 == 0):
        print (counter)

    curr_field = an_EE_TS[an_EE_TS['ID']==a_poly].copy()
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['image_year', 'doy'], inplace=True)
    ID = curr_field['ID'].unique()[0]

    ################################################################

    plant = curr_field['CropTyp'].unique()[0]
    # Take care of names, replace "/" and "," and " " by "_"
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")

    county = curr_field['county'].unique()[0]

    sub_out = plant + "/" # "/plant_based_plots/" + plant + "/"
    plot_path = plot_dir_base + sub_out
    plot_path = plot_path   # +  str(len(SG_max_DoYs_series)) + "_peaks/"
    os.makedirs(plot_path, exist_ok=True)
    # print ("plot_path is " + plot_path)
    if (len(os.listdir(plot_path)) < 70):
        S1 = rcp.subplots_savitzky(current_field = curr_field, idx = indeks, deltA = 0.1)
        S2 = rcp.subplots_savitzky(current_field = curr_field, idx = indeks, deltA = 0.2)
        S3 = rcp.subplots_savitzky(current_field = curr_field, idx = indeks, deltA = 0.3)
        S4 = rcp.subplots_savitzky(current_field = curr_field, idx = indeks, deltA = 0.4)

        fig_name = plot_path + county + "_" + plant + "_SF_year_" + str(SF_year) + "_" + ID + '.pdf'
        pdf = matplotlib.backends.backend_pdf.PdfPages(fig_name)
        pdf.savefig( S1 )
        pdf.savefig( S2 )
        pdf.savefig( S3 )
        pdf.savefig( S4 )
        pdf.close()
        counter += 1      

print ("done")
print (time.time() - start_time)



