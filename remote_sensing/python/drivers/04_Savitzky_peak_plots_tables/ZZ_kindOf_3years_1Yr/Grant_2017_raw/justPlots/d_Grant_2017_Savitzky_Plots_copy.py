####
#### May 26, 2019
####

"""
Just generate peak tables for Grant 2018 Irrigated fields 
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

#########################################
###
###     Local Core path
###

sys.path.append('/Users/hn/Documents/00_GitHub/Ag/remote_sensing/python/')

################
###
### Directories
###
data_dir = "/Users/hn/Documents/01_research_data" + \
           "/remote_sensing/01_NDVI_TS/00_Eastern_WA_withYear/"

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

data_dir = "/data/hydro/users/Hossein/remote_sensing/01_NDVI_TS/00_Eastern_WA_EE/"
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
eleven_colors = ["gray", "lightcoral", "red", "peru",
                 "darkorange", "gold", "olive", "green",
                 "blue", "violet", "deepskyblue"]

Sav_win_size = 9
sav_order = 1
delt = 0.1
irrigated_only = 1
SF_year = 2017
indeks = "EVI"

# we are creating panels where each panel
# consist of different parameters of Savitzky, 
# So, we do not need the following two

# Sav_win_size = int(sys.argv[1]) 
# sav_order = int(sys.argv[2])

# delt = float(sys.argv[1])
# indeks = sys.argv[1]
# irrigated_only = int(sys.argv[2])
# SF_year = int(sys.argv[3])

print ("delta = {fileShape}".format(fileShape=delt))

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

output_dir = "/data/hydro/users/Hossein/remote_sensing/01_NDVI_TS/01_Eastern_WA_plots_tbls/" + \
             "plots/Grant_" + str(SF_year) + "/" + \
              output_Irr + "/savitzky_" + indeks + "/" + \
             "/delta" + str(delt) + "/"

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
    print ("Data source is being set to NA")
    a_df['CovrCrp'] = "NA"

if (indeks == "EVI"):
    a_df = rc.initial_clean_EVI(a_df)
else:
    a_df = rc.initial_clean_NDVI(a_df)

a_df.head(2)
an_EE_TS = a_df.copy()

### List of unique polygons
polygon_list = an_EE_TS['ID'].unique()
print(len(polygon_list))

counter = 0

for a_poly in polygon_list:
    if (counter%1000 == 0):
        print (counter)
    counter += 1
    curr_field = an_EE_TS[an_EE_TS['ID']==a_poly].copy()
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['image_year', 'doy'], inplace=True)

    ################################################################

    plant = curr_field['CropTyp'].unique()[0]
    # Take care of names, replace "/" and "," and " " by "_"
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")

    sub_out = plant + "/" # "/plant_based_plots/" + plant + "/"
    plot_path = plot_dir_base + sub_out
    plot_path = plot_path   # +  str(len(SG_max_DoYs_series)) + "_peaks/"
    os.makedirs(plot_path, exist_ok=True)
    # print ("plot_path is " + plot_path)
    if (len(os.listdir(plot_path))<50):

        county = curr_field['county'].unique()[0]
        ID = curr_field['ID'].unique()[0]

        X = curr_field['doy']
        y = curr_field[indeks]

        #############################################
        ###
        ###             Smoothen
        ###
        #############################################
        # differences are minor, but lets keep using Pythons function
        # my_savitzky_pred = rc.savitzky_golay(y, window_size=Sav_win_size, order=sav_order)

        SG_pred_31 = scipy.signal.savgol_filter(y, window_length= 3, polyorder=1)
        SG_pred_32 = scipy.signal.savgol_filter(y, window_length= 3, polyorder=2)
        
        SG_pred_51 = scipy.signal.savgol_filter(y, window_length= 5, polyorder=1)
        SG_pred_52 = scipy.signal.savgol_filter(y, window_length= 5, polyorder=2)
        SG_pred_53 = scipy.signal.savgol_filter(y, window_length= 5, polyorder=3)
        
        SG_pred_71 = scipy.signal.savgol_filter(y, window_length= 7, polyorder=1)
        SG_pred_72 = scipy.signal.savgol_filter(y, window_length= 7, polyorder=2)
        SG_pred_73 = scipy.signal.savgol_filter(y, window_length= 7, polyorder=3)

        SG_pred_91 = scipy.signal.savgol_filter(y, window_length= 9, polyorder=1)
        SG_pred_92 = scipy.signal.savgol_filter(y, window_length= 9, polyorder=2)
        SG_pred_93 = scipy.signal.savgol_filter(y, window_length= 9, polyorder=3)

       
        # preds_df = pd.DataFrame(data = {'SG 31':SG_pred_31, 'SG 32':SG_pred_32, 
        #                                 'SG 51':SG_pred_51, 'SG 52':SG_pred_52, 'SG 53':SG_pred_53, 
        #                                 'SG 71':SG_pred_71, 'SG 72':SG_pred_72, 'SG 73':SG_pred_73, 
        #                                 'SG 91':SG_pred_91, 'SG 92':SG_pred_92, 'SG 93':SG_pred_93}, 
        #                         index = X)
        # df2.set_index(keys = ['a'], drop=True,  inplace=True)
        #############################################
        ###
        ###             find peaks
        ###
        #############################################

        SG_max_min_31 = rc.my_peakdetect(y_axis=SG_pred_31, x_axis=X, delta=delt);
        SG_max_31 =  SG_max_min_31[0]; SG_min_31 =  SG_max_min_31[1];
        SG_max_31 = rc.separate_x_and_y(m_list = SG_max_31);
        SG_min_31 = rc.separate_x_and_y(m_list = SG_min_31);
        SG_max_DoYs_series_31 = pd.Series(SG_max_31[0]);
        SG_max_series_31 = pd.Series(SG_max_31[1]);
        SG_min_DoYs_series_31 = pd.Series(SG_min_31[0]);
        SG_min_series_31 = pd.Series(SG_min_31[1]);


        SG_max_min_32 = rc.my_peakdetect(y_axis=SG_pred_32, x_axis=X, delta=delt);
        SG_max_32 =  SG_max_min_32[0]; SG_min_32 =  SG_max_min_32[1];
        SG_max_32 = rc.separate_x_and_y(m_list = SG_max_32);
        SG_min_32 = rc.separate_x_and_y(m_list = SG_min_32);
        SG_max_DoYs_series_32 = pd.Series(SG_max_32[0]);
        SG_max_series_32 = pd.Series(SG_max_32[1]);
        SG_min_DoYs_series_32 = pd.Series(SG_min_32[0]);
        SG_min_series_32 = pd.Series(SG_min_32[1]);

        ############
        ############ window 5
        ############

        SG_max_min_51 = rc.my_peakdetect(y_axis=SG_pred_51, x_axis=X, delta=delt);
        SG_max_51 =  SG_max_min_51[0]; SG_min_51 =  SG_max_min_51[1];
        SG_max_51 = rc.separate_x_and_y(m_list = SG_max_51);
        SG_min_51 = rc.separate_x_and_y(m_list = SG_min_51);
        SG_max_DoYs_series_51 = pd.Series(SG_max_51[0]);
        SG_max_series_51 = pd.Series(SG_max_51[1]);
        SG_min_DoYs_series_51 = pd.Series(SG_min_51[0]);
        SG_min_series_51 = pd.Series(SG_min_51[1]);

        SG_max_min_52 = rc.my_peakdetect(y_axis=SG_pred_52, x_axis=X, delta=delt);
        SG_max_52 =  SG_max_min_52[0]; SG_min_52 =  SG_max_min_52[1];
        SG_max_52 = rc.separate_x_and_y(m_list = SG_max_52);
        SG_min_52 = rc.separate_x_and_y(m_list = SG_min_52);
        SG_max_DoYs_series_52 = pd.Series(SG_max_52[0]);
        SG_max_series_52 = pd.Series(SG_max_52[1]);
        SG_min_DoYs_series_52 = pd.Series(SG_min_52[0]);
        SG_min_series_52 = pd.Series(SG_min_52[1]);

        SG_max_min_53 = rc.my_peakdetect(y_axis=SG_pred_53, x_axis=X, delta=delt);
        SG_max_53 =  SG_max_min_53[0]; SG_min_53 =  SG_max_min_53[1];
        SG_max_53 = rc.separate_x_and_y(m_list = SG_max_53);
        SG_min_53 = rc.separate_x_and_y(m_list = SG_min_53);
        SG_max_DoYs_series_53 = pd.Series(SG_max_53[0]);
        SG_max_series_53 = pd.Series(SG_max_53[1]);
        SG_min_DoYs_series_53 = pd.Series(SG_min_53[0]);
        SG_min_series_53 = pd.Series(SG_min_53[1]);

        ############
        ############ window 7
        ############

        SG_max_min_71 = rc.my_peakdetect(y_axis=SG_pred_71, x_axis=X, delta=delt);
        SG_max_71 =  SG_max_min_71[0]; SG_min_71 =  SG_max_min_71[1];
        SG_max_71 = rc.separate_x_and_y(m_list = SG_max_71);
        SG_min_71 = rc.separate_x_and_y(m_list = SG_min_71);
        SG_max_DoYs_series_71 = pd.Series(SG_max_71[0]);
        SG_max_series_71 = pd.Series(SG_max_71[1]);
        SG_min_DoYs_series_71 = pd.Series(SG_min_71[0]);
        SG_min_series_71 = pd.Series(SG_min_71[1]);

        SG_max_min_72 = rc.my_peakdetect(y_axis=SG_pred_72, x_axis=X, delta=delt);
        SG_max_72 =  SG_max_min_72[0]; SG_min_72 =  SG_max_min_72[1];
        SG_max_72 = rc.separate_x_and_y(m_list = SG_max_72);
        SG_min_72 = rc.separate_x_and_y(m_list = SG_min_72);
        SG_max_DoYs_series_72 = pd.Series(SG_max_72[0]);
        SG_max_series_72 = pd.Series(SG_max_72[1]);
        SG_min_DoYs_series_72 = pd.Series(SG_min_72[0]);
        SG_min_series_72 = pd.Series(SG_min_72[1]);

        SG_max_min_73 = rc.my_peakdetect(y_axis=SG_pred_73, x_axis=X, delta=delt);
        SG_max_73 =  SG_max_min_73[0]; SG_min_73 =  SG_max_min_73[1];
        SG_max_73 = rc.separate_x_and_y(m_list = SG_max_73);
        SG_min_73 = rc.separate_x_and_y(m_list = SG_min_73);
        SG_max_DoYs_series_73 = pd.Series(SG_max_73[0]);
        SG_max_series_73 = pd.Series(SG_max_73[1]);
        SG_min_DoYs_series_73 = pd.Series(SG_min_73[0]);
        SG_min_series_73 = pd.Series(SG_min_73[1]);

        ############
        ############ window 9
        ############

        SG_max_min_91 = rc.my_peakdetect(y_axis=SG_pred_91, x_axis=X, delta=delt);
        SG_max_91 =  SG_max_min_91[0]; SG_min_91 =  SG_max_min_91[1];
        SG_max_91 = rc.separate_x_and_y(m_list = SG_max_91);
        SG_min_91 = rc.separate_x_and_y(m_list = SG_min_91);
        SG_max_DoYs_series_91 = pd.Series(SG_max_91[0]);
        SG_max_series_91 = pd.Series(SG_max_91[1]);
        SG_min_DoYs_series_91 = pd.Series(SG_min_91[0]);
        SG_min_series_91 = pd.Series(SG_min_91[1]);

        SG_max_min_92 = rc.my_peakdetect(y_axis=SG_pred_92, x_axis=X, delta=delt);
        SG_max_92 =  SG_max_min_92[0]; SG_min_92 =  SG_max_min_92[1];
        SG_max_92 = rc.separate_x_and_y(m_list = SG_max_92);
        SG_min_92 = rc.separate_x_and_y(m_list = SG_min_92);
        SG_max_DoYs_series_92 = pd.Series(SG_max_92[0]);
        SG_max_series_92 = pd.Series(SG_max_92[1]);
        SG_min_DoYs_series_92 = pd.Series(SG_min_92[0]);
        SG_min_series_92 = pd.Series(SG_min_92[1]);

        SG_max_min_93 = rc.my_peakdetect(y_axis=SG_pred_93, x_axis=X, delta=delt);
        SG_max_93 =  SG_max_min_93[0]; SG_min_93 =  SG_max_min_93[1];
        SG_max_93 = rc.separate_x_and_y(m_list = SG_max_93);
        SG_min_93 = rc.separate_x_and_y(m_list = SG_min_93);
        SG_max_DoYs_series_93 = pd.Series(SG_max_93[0]);
        SG_max_series_93 = pd.Series(SG_max_93[1]);
        SG_min_DoYs_series_93 = pd.Series(SG_min_93[0]);
        SG_min_series_93 = pd.Series(SG_min_93[1]);

        ########################################################################################################
        ########################################################################################################

        plotting_dic = { "SG_pred_31" : [SG_pred_31, SG_max_DoYs_series_31, SG_max_series_31],
                         "SG_pred_32" : [SG_pred_32, SG_max_DoYs_series_32, SG_max_series_32],

                         "SG_pred_51" : [SG_pred_51, SG_max_DoYs_series_51, SG_max_series_51],
                         "SG_pred_52" : [SG_pred_52, SG_max_DoYs_series_52, SG_max_series_52],
                         "SG_pred_53" : [SG_pred_53, SG_max_DoYs_series_53, SG_max_series_53],

                         "SG_pred_71" : [SG_pred_71, SG_max_DoYs_series_71, SG_max_series_71],
                         "SG_pred_72" : [SG_pred_72, SG_max_DoYs_series_72, SG_max_series_72],
                         "SG_pred_73" : [SG_pred_73, SG_max_DoYs_series_73, SG_max_series_73],

                         "SG_pred_91" : [SG_pred_91, SG_max_DoYs_series_91, SG_max_series_91],
                         "SG_pred_92" : [SG_pred_92, SG_max_DoYs_series_92, SG_max_series_92],
                         "SG_pred_93" : [SG_pred_93, SG_max_DoYs_series_93, SG_max_series_93]
        }

        #############################################
        ###
        ###             plot
        ###
        #############################################

        # sub_out = plant + "/" # "/plant_based_plots/" + plant + "/"
        # plot_path = plot_dir_base + sub_out
        # plot_path = plot_path   # +  str(len(SG_max_DoYs_series)) + "_peaks/"
        # os.makedirs(plot_path, exist_ok=True)
        # # print ("plot_path is " + plot_path)
        # if (len(os.listdir(plot_path))<50):
            
        plot_title = county + ", " + plant + " (" + ID + ")"
        sb.set();

        fig, ax = plt.subplots(figsize=(8,6));
        ax.scatter(X, y, label="Raw data", s=30);

        for co, ite in enumerate(plotting_dic):
            ax.plot(X, plotting_dic[ite][0], label = ite, c = eleven_colors[co])
            ax.scatter(plotting_dic[ite][1], plotting_dic[ite][2], s=100, marker='*', c = eleven_colors[co]);

        ax.set_title(plot_title);
        ax.set(xlabel='DoY', ylabel=indeks)

        ################################################
        #
        #    bare soil indices plots
        #

        # an_EE_TS_BSI = rc.initial_clean(df = curr_field, column_to_be_cleaned='BSI')
        # # an_EE_TS_NDWI = rc.initial_clean(df = curr_field, column_to_be_cleaned='NDWI')
        # an_EE_TS_PSRI = rc.initial_clean(df = curr_field, column_to_be_cleaned='PSRI')
        # an_EE_TS_LSWI = rc.initial_clean(df = curr_field, column_to_be_cleaned='LSWI')

        # ax.plot(an_EE_TS_BSI['doy'], an_EE_TS_BSI['BSI'], label="BSI")
        # # ax.plot(x_NDWI, y_NDWI, label="NWDI")

        # ax.plot(an_EE_TS_PSRI['doy'], an_EE_TS_PSRI['PSRI'], label="PSRI")
        # ax.plot(an_EE_TS_LSWI['doy'], an_EE_TS_LSWI['LSWI'], label="LSWI")

        ax.legend(loc="best");
        fig_name = plot_path + county + "_" + plant + "_SF_year_" + str(SF_year) + "_" + str(counter) + '.png'
        plt.savefig(fname = fig_name, \
                    dpi=300,
                    bbox_inches='tight')
        plt.close()
        del(plot_path, sub_out) #  county, plant, year
 