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

param_dir = "/home/hnoorazar/remote_sensing_codes/parameters/"
data_dir = "/data/hydro/users/Hossein/remote_sensing/04_noJump_Regularized_plt_tbl_SOSEOS/2Yrs_tbl_reg_fineGranular_SOS3_EOS3/"

output_dir = "/data/hydro/users/Hossein/remote_sensing/06_list_of_2cropped_IDs_4_website/"
os.makedirs(output_dir, exist_ok=True)

annuals_list = pd.read_csv(param_dir + "double_crop_potential_plants.csv", low_memory=False)

####################################################################################
###
###                   Body
###
####################################################################################

counties = ["Grant", "Franklin", "Yakima",
            "Walla_Walla", "Adams", "Benton",
            "Whitman", "Asotin", "Garfield", 
            "Ferry", "Columbia", "Chelan", "Douglas", 
            "Kittitas", "Klickitat", "Lincoln", 
            "Okanogan", "Spokane", "Stevens",
            "Pend_Oreille"]

SF_years = ["2016", "2017", "2018"]
postfix_1 = "_regular_EVI_"
postfix_2 = "SG_win7_Order3.csv"


NASS_out_options = [False]
last_survey_date_options = [True, False]
double_potential_options = [True] # This must be the only option
irrigated_options = [True]


for NASS_out in NASS_out_options:
    for last_survey_date in last_survey_date_options:
        for double_potential in double_potential_options:
            for irrigated in irrigated_options:
                for SF_year in SF_years:
                    all_data = pd.DataFrame()
                    for county in counties:
                        file_name = county + "_" + SF_year + postfix_1 + postfix_2
                        a_df = pd.read_csv(data_dir + file_name, low_memory=False)

                        print (file_name)
                        print (a_df.SF_year.unique())

                        if NASS_out == True:
                            a_df = rc.filter_out_NASS(a_df)
                            out_name_NASS = "NASSOut"
                        else:
                            out_name_NASS = "NASSIn"

                        if last_survey_date == True:
                            a_df = a_df[a_df['LstSrvD'].str.contains(SF_year)]
                            out_name_survey = "LastSrvCorrect"
                        else:
                            out_name_survey = "LastSrvFalse"

                        if irrigated == True:
                            a_df = rc.filter_out_nonIrrigated(a_df)
                            out_name_irrigated = "JustIrr"
                        else:
                            out_name_irrigated = "IrrAndNonIrr"

                        if double_potential == True:
                            """
                               We have to do the following, since we missed Caneberry and a couple (1 or 2?) of
                               other perennials that came up in the meeting with Perry.
                            """
                            a_df = a_df[a_df.CropTyp.isin(annuals_list['Crop_Type'])]
                            double_potential_name = "onlyAnnuals"

                        # Filter season count more than 2
                        a_df = a_df[a_df.season_count >= 2]
                        all_data = pd.concat([all_data, a_df])

                    output_name = "doubleCroppedFields_" + \
                                  SF_year                + "_" + \
                                  double_potential_name  + "_" + \
                                  out_name_irrigated     + "_" + \
                                  out_name_survey        + "_" + \
                                  out_name_NASS          + "_" + \
                                  postfix_2

                    all_data.to_csv(output_dir + output_name, index = False)
















