####
#### Oct. 30, Produce a list of crops that are double cropped.
####


import csv
import time
import scipy
import datetime
import itertools
import numpy as np
import os, os.path, fnmatch
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
import remote_sensing_plot_core as rcp


####################################################################################
###
###                   Aeolus Directories
###
####################################################################################

data_dir_base = "/data/hydro/users/Hossein/remote_sensing/04_noJump_Regularized_plt_tbl_SOSEOS/"

output_dir = "/data/hydro/users/Hossein/remote_sensing/06_list_of_double_cropped_plants/"
os.makedirs(output_dir, exist_ok=True)

print ("_________________________________________________________")
print ("data dir is: " + data_dir_base)
print ("_________________________________________________________")
print ("output_dir is: " +  output_dir)
print ("_________________________________________________________")
###################################

sos_list = [3, 4, 5]
years = [2016, 2017, 2018]
SGparams = ["51", "53", "73", "93"]

needed_columns = ["ID", "Acres", "county", "CropTyp", "DataSrc", "Irrigtn", "LstSrvD", "Notes", 
                  "RtCrpTy", "image_year", "SF_year", "season_count"]

output_columns = needed_columns.copy()
output_columns.append("SG_params")

limited_columns = ["CropTyp", "SG_params", "image_year"]

for seos in sos_list:
    output_dataframe = pd.DataFrame(data = None, 
                                    # index = np.arange(len(delta_windows_degrees)),
                                    columns = output_columns)

    data_dir = data_dir_base + "2Yrs_tbl_reg_fineGranular_SOS" + str(seos) + "_EOS" + str(seos) + "/"

    for yr in years:
        patt = "*" + str(yr) + "*"
        list_files = fnmatch.filter(os.listdir(data_dir), patt)

        for SG in SGparams:
            SG_patt = "win" + SG[0] + "_Order" + SG[1]
            list_files_SG = [k for k in list_files if SG_patt in k]

            for a_file in list_files_SG:
                curr_file = pd.read_csv(data_dir + a_file)
                ####
                #### Drop NASS and filter by last survey date
                ####
                curr_file = curr_file[curr_file.DataSrc != "nass"]
                curr_file = curr_file[curr_file['LstSrvD'].str.contains(str(yr))]

                #
                # Include only irrigated fileds
                #
                # curr_file = rc.filter_out_nonIrrigated(curr_file)

                ####
                curr_file = curr_file[needed_columns]
                curr_file = curr_file[curr_file.season_count >= 2]
                curr_file.drop_duplicates(inplace=True)
                curr_file["SG_params"] = SG
                output_dataframe = pd.concat([output_dataframe, curr_file])

    filename = output_dir + "extended_double_cropped_plants_SEOS" + str(seos) + ".csv"
    output_dataframe.to_csv(filename, index = False)

    #
    # Drop columns such as RtCrpTy so that we can drop duplicate rows
    # to get unique crop types.
    #
    output_dataframe = output_dataframe[limited_columns]
    output_dataframe.drop_duplicates(inplace=True)

    output_dataframe.sort_values(by=['SG_params', 'image_year', 'CropTyp'], inplace=True)

    filename = output_dir + "double_cropped_plants_SEOS" + str(seos) + ".csv"
    output_dataframe.to_csv(filename, index = False)
    



print ("done")
print (time.time() - start_time)



