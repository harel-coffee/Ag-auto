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
data_dir = "/Users/hn/Documents/01_research_data/Ag_check_point/" + \
           "remote_sensing/01_NDVI_TS/Grant/No_EVI/Grant_10_cloud/Grant_2016/"

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

data_dir = "/data/hydro/users/Hossein/remote_sensing/" + \
           "01_NDVI_TS/Grant/No_EVI/Grant_10_cloud/Grant_2016/"
param_dir = "/home/hnoorazar/remote_sensing_codes/parameters/"
####################################################################################
###
###                   Parameters
###
####################################################################################
freedom_df = 7
look_ahead = 8
freedom_df = int(sys.argv[1])
look_ahead = int(sys.argv[2])

double_crop_potential_plants = pd.read_csv(param_dir + "double_crop_potential_plants.csv")
double_crop_potential_plants.head(2)

####################################################################################
###
###                   Import remote cores
###
####################################################################################

import remote_sensing_core as rc
import remote_sensing_core as rcp

output_dir = data_dir
plot_dir_base = data_dir + "/plots/"


####################################################################################
###
###                   Data Reading
###
####################################################################################

file_names = ["Grant_2016_TS.csv"]
file_N = file_names[0]
a_df = pd.read_csv(data_dir + file_N)

####################################################################################
###
###                   process data
###
####################################################################################
#
# The following columns do not exist in the old data
#
if not('DataSrc' in a_df.columns):
    print ("Data source is being set to NA")
    a_df['DataSrc'] = "NA"

if not('CovrCrp' in a_df.columns):
    print ("Data source is being set to NA")
    a_df['CovrCrp'] = "NA"

a_df = rc.initial_clean_NDVI(a_df)
a_df.head(2)
an_EE_TS = a_df.copy()
# an_EE_TS = rc.initial_clean_NDVI(an_EE_TS)

################
###
### Just keep the potential fields
###
################

a_df = a_df[a_df.CropTyp.isin(double_crop_potential_plants['Crop_Type'])]


### List of unique polygons
polygon_list = an_EE_TS['geo'].unique()
print(len(polygon_list))

output_columns = ['Acres', 'CovrCrp', 'CropGrp', 'CropTyp',
                  'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
                  'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'county', 'year', 'geo',
                  'peak_Doy', 'peak_value', 'peak_count']

all_polygons_and_their_peaks = pd.DataFrame(data=None, 
                                            index=np.arange(3*len(an_EE_TS)), 
                                            columns=output_columns)

double_columns = ['Acres', 'CovrCrp', 'CropGrp', 'CropTyp',
                  'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
                  'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'county', 'year', 'geo',
                  'peak_count']

double_polygons = pd.DataFrame(data=None, 
                               index=np.arange(2*len(an_EE_TS)), 
                               columns=double_columns)
pointer = 0
double_pointer = 0
counter = 0
for a_poly in polygon_list:
    if (counter%1000 == 0):
        print (counter)
    counter += 1
    curr_field = an_EE_TS[an_EE_TS['geo']==a_poly]

    year = int(curr_field['year'].unique())
    plant = curr_field['CropTyp'].unique()[0]
    
    # Take care of names, replace "/" and "," and " " by "_"
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")

    county = curr_field['county'].unique()[0]
    TRS = curr_field['TRS'].unique()[0]

    ### 
    ###  There is a chance that a polygon is repeated twice?
    ###

    X = curr_field['doy']
    y = curr_field['NDVI']
    #############################################
    ###
    ###             Smoothen
    ###
    #############################################

    # Generate spline basis with "freedom_df" degrees of freedom
    x_basis = cr(X, df=freedom_df, constraints='center')

    # Fit model to the data
    model = LinearRegression().fit(x_basis, y)

    # Get estimates
    y_hat = model.predict(x_basis)

    #############################################
    ###
    ###             find peaks
    ###
    #############################################
    # peaks_LWLS_1 = peakdetect(LWLS_1[:, 1], lookahead = 10, delta=0)
    # max_peaks = peaks_LWLS_1[0]
    # peaks_LWLS_1 = form_xs_ys_from_peakdetect(max_peak_list = max_peaks, doy_vect=X)

    peaks_spline = rc.peakdetect(y_hat, lookahead = look_ahead, delta=0)
    max_peaks =  peaks_spline[0]
    peaks_spline = rc.form_xs_ys_from_peakdetect(max_peak_list = max_peaks, doy_vect=X)
    # print(peaks_spline)
    DoYs_series = pd.Series(peaks_spline[0])
    peaks_series = pd.Series(peaks_spline[1])

    peak_df = pd.DataFrame({ 
                       'peak_Doy': DoYs_series,
                       'peak_value': peaks_series
                      })
    # add number of peaks to the data frame.
    peak_df['peak_count'] = peak_df.shape[0]

    WSDA_df = rc.keep_WSDA_columns(curr_field)
    WSDA_df = WSDA_df.drop_duplicates()
    
    if (len(peak_df)>0):
        WSDA_df = pd.concat([WSDA_df]*peak_df.shape[0]).reset_index()
        # WSDA_df = pd.concat([WSDA_df, peak_df], axis=1, ignore_index=True)
        WSDA_df = WSDA_df.join(peak_df)
        if ("index" in WSDA_df.columns):
            WSDA_df = WSDA_df.drop(columns=['index'])

        # all_polygons_and_their_peaks = all_polygons_and_their_peaks.append(WSDA_df, sort=False)

        """
        copy the .values. Otherwise the index inconsistency between
        WSDA_df and all_poly... will prevent the copying.
        """
        all_polygons_and_their_peaks.iloc[pointer:(pointer + len(WSDA_df))] = WSDA_df.values
        #
        #  if we have double peaks add them to the double_polygons
        #
        if (len(WSDA_df) == 2):
            # print(plant, county, year, counter)
            WSDA_df = WSDA_df.drop(columns=['peak_Doy', 'peak_value'])
            WSDA_df = WSDA_df.drop_duplicates()
            double_polygons.iloc[double_pointer:(double_pointer + len(WSDA_df))] = WSDA_df.values
            double_pointer += len(WSDA_df)

        pointer += len(WSDA_df)
        
        #############################################
        ###
        ###             plot
        ###
        #############################################        
        sub_out = "/plant_based_plots/LA_" + str(look_ahead) + "_df_" + str(freedom_df) + "/" + plant + "/"
        plot_path = plot_dir_base + sub_out
        plot_path = plot_path + str(peak_df.shape[0]) + "_peaks/"
        os.makedirs(plot_path, exist_ok=True)
        if (len(os.listdir(plot_path))<100):
            plot_title = county + ", " + plant + ", " + str(year) + " (" + TRS + ")"
            sb.set();
            fig, ax = plt.subplots(figsize=(8,6));
            ax.plot(X, y, label="NDVI");
            ax.plot(X, y_hat, 'r', label="smoothing spline")
            ax.scatter(DoYs_series, peaks_series, s=100, c='g', marker='*');
            ax.set_title(plot_title);
            ax.set(xlabel='DoY', ylabel='NDVI')
            ax.legend(loc="best");

            fig_name = plot_path + county + "_" + plant + "_" + str(year) + "_" + str(counter) + '.png'
            plt.savefig(fname = fig_name, \
                        dpi=500,
                        bbox_inches='tight')
            plt.close()
            del(plot_path, sub_out, county, plant, year)

        # print(plot_path)
        # print(sub_out)
        # print(county)
        # print(plant)
        # print(year)
        """
        if peak_df.shape[0]==2:
            double_plot_path = plot_dir_base + "/double_peaks/" + sub_out
            os.makedirs(double_plot_path, exist_ok=True)
            plot_title = county + ", " + plant + ", " + str(year) + " (" + TRS + ")"
            sb.set();
            fig, ax = plt.subplots(figsize=(8, 6));
            ax.plot(X, y, label="NDVI data");
            ax.plot(X, y_hat, 'r', label="smoothing spline result")
            ax.scatter(DoYs_series, peaks_series, s=100, c='g', marker='*');
            ax.set_title(plot_title);
            ax.set(xlabel='DoY', ylabel='NDVI')
            ax.legend(loc="best");

            fig_name = double_plot_path + county + "_" + plant + "_" + str(year) + "_" + str(counter) + '.png'
            plt.savefig(fname = fig_name, \
                        dpi=500,
                        bbox_inches='tight')
            
            if (len(os.listdir(plot_path))<100):
                fig_name = plot_path + county + "_" + plant + "_" + str(year) + "_" + str(counter) + '.png'
                plt.savefig(fname = fig_name, \
                            dpi=500, 
                            bbox_inches='tight')
            plt.close()
            del(plot_path, sub_out, county, plant, year)

        else:
            if (len(os.listdir(plot_path))<100):
                plot_title = county + ", " + plant + ", " + str(year) + " (" + TRS + ")"
                sb.set();
                fig, ax = plt.subplots(figsize=(8,6));
                ax.plot(X, y, label="NDVI data");
                ax.plot(X, y_hat, 'r', label="smoothing spline result")
                ax.scatter(DoYs_series, peaks_series, s=100, c='g', marker='*');
                ax.set_title(plot_title);
                ax.set(xlabel='DoY', ylabel='NDVI')
                ax.legend(loc="best");

                fig_name = plot_path + county + "_" + plant + "_" + str(year) + "_" + str(counter) + '.png'
                plt.savefig(fname = fig_name, \
                            dpi=500, 
                            bbox_inches='tight')
                plt.close()
                del(plot_path, sub_out, county, plant, year)
        """

        # to make sure the reference by address thing 
        # will not cause any problem.
    del(WSDA_df)

####################################################################################
###
###                   Compute double crop area
###
####################################################################################



####################################################################################
###
###                   Write the outputs
###
####################################################################################

all_polygons_and_their_peaks = all_polygons_and_their_peaks[0:(pointer+1)]
double_polygons = double_polygons[0:(double_pointer+1)]

out_name = output_dir + "LA_" + str(look_ahead) + "_df_"+ str(freedom_df) + "_all_polygons_and_their_peaks.csv"
all_polygons_and_their_peaks.to_csv(out_name, index = False)

out_name = output_dir + "LA_" + str(look_ahead) + "_df_"+ str(freedom_df) + "_double_polygons.csv"
double_polygons.to_csv(out_name, index = False)

end_time = time.time()
print(end_time - start_time)
