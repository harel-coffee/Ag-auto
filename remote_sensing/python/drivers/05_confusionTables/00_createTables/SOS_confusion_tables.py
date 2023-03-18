  
####
#### Sept. 1
####

####
#### Sept. 9, edit to include more threshold directories.
####

"""
Jupyter Notebook is called on iMac:
Acreages_ConfusionStyle_raw_n_regular_SOS
"""

import csv
import time
import scipy
import datetime
import itertools
import numpy as np
import os, os.path
import scipy.signal
import pandas as pd
from patsy import cr
from math import factorial
from IPython.display import Image
from sklearn.linear_model import LinearRegression
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import matplotlib.pyplot as plt
import seaborn as sb

# import geopandas as gpd
# from shapely.geometry import Point, Polygon
# from pprint import pprint

import sys
start_time = time.time()

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

raw_or_regular = "regular"

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
param_dir = "/home/hnoorazar/remote_sensing_codes/parameters/"
SF_data_dir = "/data/hydro/users/Hossein/remote_sensing/01_shapefiles_data_part_not_filtered/"

# the following data came from 70% cloud.
regular_in_base = "/data/hydro/users/Hossein/remote_sensing/04_noJump_Regularized_plt_tbl_SOSEOS/"

# regular_coarse_in_dir = regular_in_base + "/2Yrs_tables_regular_coarse_SEOS5/"
regular_output_dir = "/data/hydro/users/Hossein/remote_sensing/05_Regular_SOS_confusions/"

###################################
#
#  First I had written for loop for years and counties!
#  on Sept. 9 changed it to terminal arguments
#
SF_years = [2016, 2017, 2018]
counties = ["Grant", "Whitman", "Asotin",
            "Garfield", "Ferry", "Franklin",
            "Columbia", "Adams","Benton",
            "Chelan", "Douglas", "Kittitas", "Klickitat",
            "Lincoln", "Okanogan", "Spokane", "Stevens",
            "Yakima",'Pend_Oreille', 'Walla_Walla']


counties = sys.argv[1]
SF_years = int(sys.argv[2])
indekses = sys.argv[3]
SEOS_cut = int(sys.argv[4])

sos_thresh = int(SEOS_cut / 10)/10 # grab the first digit as SOS cut
eos_thresh = (SEOS_cut % 10) / 10  # grab the second digit as EOS cut

data_dir = regular_in_base + "2Yrs_tbl_reg_fineGranular_SOS" + str(int(sos_thresh*10)) + "_" + "EOS" + str(int(eos_thresh*10)) + "/"
output_dir = regular_output_dir + "fine_SOS" + str(int(sos_thresh*10)) + "_" + "EOS" + str(int(eos_thresh*10))  + "/00_allYCtables_separate/"
os.makedirs(output_dir, exist_ok=True)

print ("_________________________________________________________")
print ("data dir is: " + data_dir)
print ("_________________________________________________________")
print ("output_dir is: " +  output_dir)
print ("_________________________________________________________")

#############################################################################################

deltas = [0.1, 0.2, 0.3, 0.4]
delta_windows_degrees = [[5, 1], [5, 3], [7, 3], [9, 3]]

output_columns = ['parameters', 'actual_2_pred_2', 'actual_2_pred_Not2',
                  'actual_Not2_pred_2', 'actual_Not2_pred_Not2']

for given_county in [counties]:
    for SF_year in [SF_years]:
        print ("given_county is " + given_county.replace("_", " "))

        ####################################################################################
        ###
        ###                   Read parameters and data
        ###
        ####################################################################################
        double_crop_potens = pd.read_csv(param_dir + "double_crop_potential_plants.csv")

        WSDA_DataTable = pd.read_csv(SF_data_dir + "WSDA_DataTable_" + str(SF_year) + ".csv")
        WSDA_DataTable = WSDA_DataTable[WSDA_DataTable.county == given_county.replace("_", " ")]
        WSDA_DataTable["DataSrc"] = WSDA_DataTable["DataSrc"].str.lower()
        WSDA_DataTable["CropTyp"] = WSDA_DataTable["CropTyp"].str.lower()

        print (double_crop_potens.shape)
        print (WSDA_DataTable.shape)
        
        for indeks in [indekses]:
            for exactly_2_seasons in [False, True]:
                for double_by_Note in [False]:
                    for NASS_out in [False]:
                        # we have dropped out non-irrigated fields in the peak_finding step
                        for non_Irr_out in [True]: 
                            for perennials_out in [True]: # 
                                #### 
                                #### build output dataframe
                                #### 
                                output = pd.DataFrame(data=None, 
                                                      #delta_windows_degrees
                                                      index = np.arange(len(delta_windows_degrees)),
                                                      columns = output_columns)
                                output['parameters'] = delta_windows_degrees

                                #### 
                                #### Build shapeFile info accordingly
                                #### 
                                curr_SF = WSDA_DataTable.copy()

                                if double_by_Note == False:
                                    dbl_name = "_dblNotFiltered_"
                                else:
                                    curr_SF = rc.filter_double_by_Notes(curr_SF)
                                    dbl_name = "_onlyDblByNotes_"

                                if NASS_out == True:
                                    curr_SF = rc.filter_out_NASS(curr_SF)
                                    NASS_name = "NASSOut_"
                                else:
                                    NASS_name = "NASSin_"

                                if non_Irr_out == True:
                                    curr_SF = rc.filter_out_nonIrrigated(curr_SF)
                                    non_Irr_name = "JustIrr"
                                else:
                                    non_Irr_name = "BothIrr"

                                if perennials_out == True:
                                    print ("line 165")
                                    print (curr_SF.shape)
                                    curr_SF = curr_SF[curr_SF.CropTyp.isin(double_crop_potens['Crop_Type'])]
                                    print ("line 167")
                                    print (curr_SF.shape)
                                    Pere_name = "_PereOut_"
                                else:
                                    Pere_name = "_PereIn_"

                                print ("NASS_out: " + str(NASS_out) + ", non_Irr_out: " + str(non_Irr_out) + \
                                       ", perennials_out: " + str(perennials_out))

                                for location, params in enumerate(output['parameters']):
                                    window = params[0]
                                    degree = params[1]

                                    f_name = given_county  + "_" + str(SF_year) + "_regular_" + indeks + \
                                             "_SG_win" + str(window) + "_Order" + str(degree) + ".csv"

                                    doubl_pk_file = data_dir + f_name
                                    doubl_season_table = pd.read_csv(doubl_pk_file, low_memory=False)
                                    print ("line 187")
                                    print (doubl_season_table.shape)

                                    doubl_season_table["CropTyp"] = doubl_season_table["CropTyp"].str.lower()

                                    if double_by_Note == True:
                                        doubl_season_table = rc.filter_double_by_Notes(doubl_season_table)

                                    if NASS_out == True:
                                        doubl_season_table = rc.filter_out_NASS(doubl_season_table)

                                    if non_Irr_out == True:
                                        doubl_season_table = rc.filter_out_nonIrrigated(doubl_season_table)
                                    
                                    if perennials_out == True:
                                        doubl_season_table = doubl_season_table[\
                                                 doubl_season_table.CropTyp.isin(double_crop_potens['Crop_Type'])]
                                        print ("line 214")
                                        print (doubl_season_table.shape)
                                
                                    doubl_season_table.drop(['doy', 'EVI', 'Date', 
                                                           'human_system_start_time', 
                                                           'EVI_ratio','SOS', 'EOS'], axis=1, inplace=True)

                                    doubl_season_table.drop_duplicates(inplace=True)
                                    print ("line 229")
                                    print (doubl_season_table.shape)
                                    
                                    #### 
                                    #### Populate output dataframe
                                    #### 
                                    actual_double_cropped = rc.filter_double_by_Notes(curr_SF)
                                    actual_Notdouble_cropped = rc.filter_Notdouble_by_Notes(curr_SF)
                                    print ("line 238")
                                    print (actual_double_cropped.shape)
                                    print (actual_Notdouble_cropped.shape)

                                    if exactly_2_seasons == False:
                                        predicted_double_seasons = doubl_season_table[\
                                                                 doubl_season_table.season_count >= 2].copy()

                                        predicted_Notdouble_seasons = doubl_season_table[\
                                                                    doubl_season_table.season_count < 2].copy()
                                        exactly_2_seasons_name = "morethan2seasons"
                                    else:
                                        predicted_double_seasons = doubl_season_table[\
                                                        doubl_season_table.season_count == 2].copy()


                                        predicted_Notdouble_seasons = doubl_season_table[\
                                                                    doubl_season_table.season_count != 2].copy()

                                        exactly_2_seasons_name = "exactly2seasons"

                                    # print ("There are [%(nrow)d] IDs in curr_SF." % \
                                    #       {"nrow":len(curr_SF['ID'])})

                                    # print ("of which [%(nrow)d] are unique." % \
                                    #        {"nrow":len(curr_SF['ID'].unique())})

                                    actual_2_pred_2 = actual_double_cropped[\
                                                    actual_double_cropped['ID'].isin(\
                                                                            predicted_double_seasons['ID'])]

                                    actual_2_pred_2 = actual_2_pred_2['ExctAcr'].sum()

                                    actual_Not2_pred_2 = actual_Notdouble_cropped[\
                                                           actual_Notdouble_cropped['ID'].isin(\
                                                                          predicted_double_seasons['ID'])]
                                    actual_Not2_pred_2 = actual_Not2_pred_2['ExctAcr'].sum()

                                    actual_2_pred_Not2 = actual_double_cropped['ExctAcr'].sum() - actual_2_pred_2

                                    actual_Not2_pred_Not2 = actual_Notdouble_cropped['ExctAcr'].sum() - \
                                                              actual_Not2_pred_2

                                    fillin_col = ["actual_2_pred_2", "actual_2_pred_Not2", \
                                                  "actual_Not2_pred_2", "actual_Not2_pred_Not2"]

                                    fillin_vals = [actual_2_pred_2, actual_2_pred_Not2, \
                                                   actual_Not2_pred_2, actual_Not2_pred_Not2]

                                    output.loc[location, fillin_col] = fillin_vals


                                ###########
                                output['parameters'] = output['parameters'].astype("str")

                                filename = output_dir + given_county + "_" + str(SF_year) + \
                                           "_" + indeks + \
                                           Pere_name + NASS_name + non_Irr_name + dbl_name + \
                                           "confusion_Acr_" + exactly_2_seasons_name + "_" + \
                                            raw_or_regular + ".csv"


                                output['actual_2_pred_2'] = output['actual_2_pred_2'].astype(float)
                                output['actual_2_pred_Not2'] = output['actual_2_pred_Not2'].astype(float)
                                output['actual_Not2_pred_2'] = output['actual_Not2_pred_2'].astype(float)
                                output['actual_Not2_pred_Not2'] = output['actual_Not2_pred_Not2'].astype(float)
                                output = output.round(decimals=2)
                                print (output)

                                output.to_csv(filename, index = False)



print ("done")
print (time.time() - start_time)