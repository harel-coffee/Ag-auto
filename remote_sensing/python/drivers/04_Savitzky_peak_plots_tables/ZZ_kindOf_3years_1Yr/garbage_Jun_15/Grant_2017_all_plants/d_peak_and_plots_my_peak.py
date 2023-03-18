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

data_dir = "/Users/hn/Documents/01_research_data/" + \
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
           "01_NDVI_TS/00_Grant/No_EVI/Grant_10_cloud/Grant_2017/"
param_dir = "/home/hnoorazar/remote_sensing_codes/parameters/"
####################################################################################
###
###                   Parameters
###
####################################################################################

double_crop_potential_plants = pd.read_csv(param_dir + "double_crop_potential_plants.csv")
double_crop_potential_plants.head(2)

####################################################################################
###
###                   Import remote cores
###
####################################################################################

import remote_sensing_core as rc
import remote_sensing_core as rcp


####################################################################################
###
###                   Data Reading
###
####################################################################################
freedom_df = 7
delt = 0.2
file_names = ["Grant_2017_TS.csv"]
file_N = file_names[0]
a_df = pd.read_csv(data_dir + file_N)

output_dir = data_dir + "/savitzky/delta_" + str(delt) + "/"
os.makedirs(output_dir, exist_ok=True)
plot_dir_base = output_dir
####################################################################################
###
###                   process data
###
####################################################################################

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

# a_df = a_df[a_df.CropTyp.isin(double_crop_potential_plants['Crop_Type'])]


### List of unique polygons
polygon_list = an_EE_TS['geo'].unique()
print(len(polygon_list))

max_output_columns = ['Acres', 'CovrCrp', 'CropGrp', 'CropTyp',
                      'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
                      'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'county', 'year', 'geo',
                      'max_Doy', 'max_value', 'max_count']

all_poly_and_maxs_spline = pd.DataFrame(data=None, 
                                        index=np.arange(3*len(an_EE_TS)), 
                                        columns=max_output_columns)

all_poly_and_maxs_savitzky = pd.DataFrame(data=None, 
                                          index=np.arange(3*len(an_EE_TS)), 
                                          columns=max_output_columns)


min_output_columns = ['Acres', 'CovrCrp', 'CropGrp', 'CropTyp',
                      'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
                      'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'county', 'year', 'geo',
                      'min_Doy', 'min_value', 'min_count']

all_poly_and_mins_spline = pd.DataFrame(data=None, 
                                                 index=np.arange(3*len(an_EE_TS)), 
                                                 columns=min_output_columns)

all_poly_and_mins_savitzky = pd.DataFrame(data=None, 
                                                   index=np.arange(3*len(an_EE_TS)), 
                                                   columns=min_output_columns)

# double_max_columns = ['Acres', 'CovrCrp', 'CropGrp', 'CropTyp',
#                       'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
#                       'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'county', 'year', 'geo',
#                       'max_count']

# double_poly_max_spline = pd.DataFrame(data=None, 
#                                       index=np.arange(2*len(an_EE_TS)), 
#                                       columns=double_max_columns)

# double_poly_max_savitzky = pd.DataFrame(data=None, 
#                                         index=np.arange(2*len(an_EE_TS)), 
#                                         columns=double_max_columns)


# double_min_columns = ['Acres', 'CovrCrp', 'CropGrp', 'CropTyp',
#                       'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
#                       'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'county', 'year', 'geo',
#                       'min_count']

# double_poly_min_spline = pd.DataFrame(data=None, 
#                                       index=np.arange(2*len(an_EE_TS)), 
#                                       columns=double_min_columns)

# double_poly_min_savitzky = pd.DataFrame(data=None, 
#                                         index=np.arange(2*len(an_EE_TS)), 
#                                         columns=double_min_columns)

pointer_max_spline = 0
pointer_min_spline = 0

pointer_max_savitzky = 0
pointer_min_savitzky = 0

counter = 0
# double_max_spline_pointer = 0
# double_min_spline_pointer = 0

# double_max_savitzky_pointer = 0
# double_min_savitzky_pointer = 0


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
    #
    #   Spline
    #

    x_basis = cr(X, df=freedom_df, constraints='center') # Generate spline basis with "freedom_df" degrees of freedom
    model = LinearRegression().fit(x_basis, y) # Fit model to the data
    spline_pred = model.predict(x_basis) # Get estimates

    #
    # savitzky
    #

    savitzky_pred = rc.savitzky_golay(y, window_size=5, order=1)

    #############################################
    ###
    ###             find peaks
    ###
    #############################################
    # 
    # Spline peaks
    # 
    spline_max_min = rc.my_peakdetect(y_axis=spline_pred, x_axis=X, delta=delt);

    spline_max =  spline_max_min[0];
    spline_min =  spline_max_min[1];

    spline_max = rc.separate_x_and_y(m_list = spline_max);
    spline_min = rc.separate_x_and_y(m_list = spline_min);

    spline_max_DoYs_series = pd.Series(spline_max[0]);
    spline_max_series = pd.Series(spline_max[1]);

    spline_min_DoYs_series = pd.Series(spline_min[0]);
    spline_min_series = pd.Series(spline_min[1]);


    spline_max_df = pd.DataFrame({ 
                           'max_Doy': spline_max_DoYs_series,
                           'max_value': spline_max_series
                          })
    # add number of max to the data frame.
    spline_max_df['max_count'] = spline_max_df.shape[0]

    spline_min_df = pd.DataFrame({ 
                           'min_Doy': spline_min_DoYs_series,
                           'min_value': spline_min_series
                          })
    # add number of max to the data frame.
    spline_min_df['max_count'] = spline_min_df.shape[0]

    #################################################################################
    #
    #    savitzky
    #

    savitzky_max_min = rc.my_peakdetect(y_axis=savitzky_pred, x_axis=X, delta=delt);

    savitzky_max =  savitzky_max_min[0];
    savitzky_min =  savitzky_max_min[1];

    savitzky_max = rc.separate_x_and_y(m_list = savitzky_max);
    savitzky_min = rc.separate_x_and_y(m_list = savitzky_min);

    savitzky_max_DoYs_series = pd.Series(savitzky_max[0]);
    savitzky_max_series = pd.Series(savitzky_max[1]);

    savitzky_min_DoYs_series = pd.Series(savitzky_min[0]);
    savitzky_min_series = pd.Series(savitzky_min[1]);


    savitzky_max_df = pd.DataFrame({ 
                           'max_Doy': savitzky_max_DoYs_series,
                           'max_value': savitzky_max_series
                          })
    # add number of max to the data frame.
    savitzky_max_df['max_count'] = savitzky_max_df.shape[0]

    savitzky_min_df = pd.DataFrame({ 
                           'min_Doy': savitzky_min_DoYs_series,
                           'min_value': savitzky_min_series
                          })
    # add number of max to the data frame.
    savitzky_min_df['max_count'] = savitzky_min_df.shape[0]
    ########################################################################################################
    ########################################################################################################

    #############################################
    ###
    ###             plot
    ###
    #############################################        
    sub_out = "/plant_based_plots/" + plant + "/"
    plot_path = plot_dir_base + sub_out
    plot_path = plot_path + str(savitzky_max_df.shape[0]) + "_peaks/"
    os.makedirs(plot_path, exist_ok=True)
    if (len(os.listdir(plot_path)) < 70):
        
        plot_title = county + ", " + plant + ", " + str(year) + " (" + TRS + ")"
        sb.set();

        fig, ax = plt.subplots(figsize=(8,6));
        ax.scatter(X, y, label="Data", s=30);

        ax.plot(X, savitzky_pred, 'k--', label="savitzky")
        ax.scatter(savitzky_max_DoYs_series, savitzky_max_series, s=200, c='k', marker='*');

        ax.plot(X, spline_pred, 'r--', label="Spline")
        ax.scatter(spline_max_DoYs_series, spline_max_series, s=100, c='r', marker='*');
        ax.legend(loc="best");

        ax.set_title(plot_title);
        ax.set(xlabel='DoY', ylabel='NDVI')
        ax.legend(loc="best");

        fig_name = plot_path + county + "_" + plant + "_" + str(year) + "_" + str(counter) + '.png'
        os.makedirs(plot_path, exist_ok=True)
        
        plt.savefig(fname = fig_name, \
                     dpi=300,
                     bbox_inches='tight')
        plt.close()
        del(plot_path, sub_out) #  county, plant, year

    WSDA_df = rc.keep_WSDA_columns(curr_field)
    WSDA_df = WSDA_df.drop_duplicates()
    
    if (len(spline_max_df)>0):
        WSDA_max_df_spline = pd.concat([WSDA_df]*spline_max_df.shape[0]).reset_index()
        # WSDA_max_df_spline = pd.concat([WSDA_max_df_spline, spline_max_df], axis=1, ignore_index=True)
        WSDA_max_df_spline = WSDA_max_df_spline.join(spline_max_df)
        if ("index" in WSDA_max_df_spline.columns):
            WSDA_max_df_spline = WSDA_max_df_spline.drop(columns=['index'])
        """
        copy the .values. Otherwise the index inconsistency between
        WSDA_max_df_spline and all_poly... will prevent the copying.
        """
        if (pointer_max_spline > all_poly_and_maxs_spline.shape[0]):
            empty = pd.DataFrame(data=None, index=np.arange(500), columns=max_output_columns)
            all_poly_and_maxs_spline = pd.concat([all_poly_and_maxs_spline, empty]).reset_index()

        all_poly_and_maxs_spline.iloc[pointer_max_spline:(pointer_max_spline + \
                                                len(WSDA_max_df_spline))] = WSDA_max_df_spline.values
        pointer_max_spline += len(WSDA_max_df_spline)

    if (len(spline_min_df)>0):
        WSDA_min_df_spline = pd.concat([WSDA_df]*spline_min_df.shape[0]).reset_index()
        # WSDA_min_df_spline = pd.concat([WSDA_min_df_spline, spline_min_df], axis=1, ignore_index=True)
        WSDA_min_df_spline = WSDA_min_df_spline.join(spline_min_df)
        if ("index" in WSDA_min_df_spline.columns):
            WSDA_min_df_spline = WSDA_min_df_spline.drop(columns=['index'])
        """
        copy the .values. Otherwise the index inconsistency between
        WSDA_min_df_spline and all_poly... will prevent the copying.
        """
        if (pointer_min_spline > all_poly_and_mins_spline.shape[0]):
            empty = pd.DataFrame(data=None, index=np.arange(500), columns=min_output_columns)
            all_poly_and_mins_spline = pd.concat([all_poly_and_mins_spline, empty]).reset_index()

        all_poly_and_mins_spline.iloc[pointer_min_spline:(pointer_min_spline + \
                                                            len(WSDA_min_df_spline))] = WSDA_min_df_spline.values
        pointer_min_spline += len(WSDA_min_df_spline)

    if (len(savitzky_max_df)>0):
        WSDA_max_df_savitzky = pd.concat([WSDA_df]*savitzky_max_df.shape[0]).reset_index()
        # WSDA_max_df_savitzky = pd.concat([WSDA_max_df_savitzky, savitzky_max_df], axis=1, ignore_index=True)
        WSDA_max_df_savitzky = WSDA_max_df_savitzky.join(savitzky_max_df)
        if ("index" in WSDA_max_df_savitzky.columns):
            WSDA_max_df_savitzky = WSDA_max_df_savitzky.drop(columns=['index'])
        """
        copy the .values. Otherwise the index inconsistency between
        WSDA_max_df_savitzky and all_poly... will prevent the copying.
        """
        if (pointer_max_savitzky > all_poly_and_maxs_savitzky.shape[0]):
            empty = pd.DataFrame(data=None, index=np.arange(500), columns=min_output_columns)
            all_poly_and_maxs_savitzky = pd.concat([all_poly_and_maxs_savitzky, empty]).reset_index()

        all_poly_and_maxs_savitzky.iloc[pointer_max_savitzky:(pointer_max_savitzky + \
                                                                    len(WSDA_max_df_savitzky))] = WSDA_max_df_savitzky.values
        pointer_max_savitzky += len(WSDA_max_df_savitzky)

    if (len(savitzky_min_df)>0):
        WSDA_min_df_savitzky = pd.concat([WSDA_df]*savitzky_min_df.shape[0]).reset_index()
        # WSDA_min_df_savitzky = pd.concat([WSDA_min_df_savitzky, savitzky_min_df], axis=1, ignore_index=True)
        WSDA_min_df_savitzky = WSDA_min_df_savitzky.join(savitzky_min_df)
        if ("index" in WSDA_min_df_savitzky.columns):
            WSDA_min_df_savitzky = WSDA_min_df_savitzky.drop(columns=['index'])
        """
        copy the .values. Otherwise the index inconsistency between
        WSDA_min_df_savitzky and all_poly... will prevent the copying.
        """
        if (pointer_min_savitzky > all_poly_and_mins_savitzky.shape[0]):
            empty = pd.DataFrame(data=None, index=np.arange(500), columns=min_output_columns)
            all_poly_and_mins_savitzky = pd.concat([all_poly_and_mins_savitzky, empty]).reset_index()

        all_poly_and_mins_savitzky.iloc[pointer_min_savitzky:(pointer_min_savitzky + \
                                                                  len(WSDA_min_df_savitzky))] = WSDA_min_df_savitzky.values
        pointer_min_savitzky += len(WSDA_min_df_savitzky)
    
    del(WSDA_df)


####################################################################################
###
###                   Write the outputs
###
####################################################################################
###########
########### max
###########

all_poly_and_maxs_spline = all_poly_and_maxs_spline[0:(pointer_max_spline+1)]
out_name = output_dir + "/df_"+ str(freedom_df) + "_all_poly_and_maxs_spline.csv"
all_poly_and_maxs_spline.to_csv(out_name, index = False)

all_poly_and_maxs_savitzky = all_poly_and_maxs_savitzky[0:(pointer_max_savitzky+1)]
out_name = output_dir + "/all_poly_and_maxs_savitzky.csv"
all_poly_and_maxs_savitzky.to_csv(out_name, index = False)

###########
########### min
###########

all_poly_and_mins_spline = all_poly_and_mins_spline[0:(pointer_min_spline+1)]
out_name = output_dir + "/df_"+ str(freedom_df) + "_all_poly_and_mins_spline.csv"
all_poly_and_mins_spline.to_csv(out_name, index = False)

all_poly_and_mins_savitzky = all_poly_and_mins_savitzky[0:(pointer_min_savitzky+1)]
out_name = output_dir + "/all_poly_and_mins_savitzky.csv"
all_poly_and_mins_savitzky.to_csv(out_name, index = False)


# out_name = output_dir + "_df_"+ str(freedom_df) + "_double_polygons_spline.csv"
# double_poly_max_spline = double_poly_max_spline[0:(double_max_pointer+1)]
# double_poly_max_spline.to_csv(out_name, index = False)

end_time = time.time()
print(end_time - start_time)


