####
#### July 2. This is a copy of the version we had from before. plotting one year.
#### Here we are extending it to 2 years. Since August of a given year to the end
#### of the next year.
####

"""
Just generate peak plots for Grant 2017 fields 
for all cultivars; EVI and my peak finder
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

start = time.time()

# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

####################################################################################
###
###                      Local
###
####################################################################################
print ("****************************************************")

print('matplotlib: {}'.format(matplotlib.__version__))
print('numpy: {}'.format(np.__version__))
print('pandas: {}'.format(pd.__version__))

print ("****************************************************")

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

data_dir_raw = "/data/hydro/users/Hossein/remote_sensing/02_Eastern_WA_EE_TS/2Years/70_cloud/"
data_dir_noJumps = data_dir_raw + "/02_noOutlierNoJumpMerged/"
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

# indeks = "EVI"
# irrigated_only = 1
# SF_year = 2017
jumps = sys.argv[1]
indeks = sys.argv[2]
irrigated_only = int(sys.argv[3])
SF_year = int(sys.argv[4])

given_county = "Grant"
####################################################################################
###
###                   Read the  data
###
####################################################################################

if jumps == "yes":
    data_dir = data_dir_raw
    f_name = "Eastern_WA_" + str(SF_year) + "_70cloud_selectors.csv"
else:
    data_dir = data_dir_noJumps
    f_name = "Eastern_WA_SF_" + str(SF_year) + "_70cloud_" + indeks + ".csv"

a_df = pd.read_csv(data_dir + f_name, low_memory=False)

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
if jumps == "yes":
    output_dir = "/data/hydro/users/Hossein/remote_sensing/02_Eastern_WA_plots_tbls/" + \
                 "2Yr_plt_70cloud_notRegular_wJump/" + given_county + "_" + str(SF_year) + "_raw_" + output_Irr + "_" + indeks + "/" 
else:
    output_dir = "/data/hydro/users/Hossein/remote_sensing/02_Eastern_WA_plots_tbls/" + \
                 "2Yr_plt_70cloud_notRegular_noJump/" + given_county + "_" + str(SF_year) + "_noJump_" + output_Irr + "_" + indeks + "/" 


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
    print ("initial_clean_EVI")
else:
    a_df = rc.initial_clean_NDVI(a_df)
    print ("initial_clean_NDVI")

a_df.head(2)
an_EE_TS = a_df.copy()
del(a_df)

### List of unique polygons
polygon_list = np.sort(an_EE_TS['ID'].unique())
print(len(polygon_list))

counter = 0

for a_poly in polygon_list:
    if (counter%1000 == 0):
        print (counter)

    curr_field = an_EE_TS[an_EE_TS['ID']==a_poly].copy()
    
    ################################################################
    #
    # Sort by DoY (sanitary check)
    # 

    curr_field.sort_values(by=['image_year', 'doy'], inplace=True)

    ID = curr_field['ID'].unique()[0]
    plant = curr_field['CropTyp'].unique()[0]
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
    if (len(os.listdir(plot_path)) < 100):

        # 
        #  Set up Canvas
        #
        fig, axs = plt.subplots(2, 2, figsize=(20,12),
                                sharex='col', sharey='row',
                                gridspec_kw={'hspace': 0.1, 'wspace': .1})

        (ax1, ax2), (ax3, ax4) = axs
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)

        rcp.savitzky_2yrs_panel(crr_fld = curr_field, idx = indeks, deltA = 0.1, SFYr = SF_year, ax = ax1)
        rcp.savitzky_2yrs_panel(crr_fld = curr_field, idx = indeks, deltA = 0.2, SFYr = SF_year, ax = ax2)
        rcp.savitzky_2yrs_panel(crr_fld = curr_field, idx = indeks, deltA = 0.3, SFYr = SF_year, ax = ax3)
        rcp.savitzky_2yrs_panel(crr_fld = curr_field, idx = indeks, deltA = 0.4, SFYr = SF_year, ax = ax4)

        fig_name = plot_path + county + "_" + plant + "_SF_year_" + str(SF_year) + "_" + ID + '.png'

        plt.savefig(fname = fig_name, dpi=250, bbox_inches='tight')
        counter += 1


print ("done")
end_time = time.time()
print(end_time - start_time)


