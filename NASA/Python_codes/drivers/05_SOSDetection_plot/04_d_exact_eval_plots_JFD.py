####
#### Nov 18, 2021
####

import csv
import numpy as np
import pandas as pd

import datetime
from datetime import date
import time

import scipy
import scipy.signal
import os, os.path

from patsy import cr

# from pprint import pprint
import matplotlib
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
###                      Aeolus Core path
###
####################################################################################

sys.path.append('/home/hnoorazar/NASA/')
import NASA_core as nc
import NASA_plot_core as ncp

####################################################################################
###
###      Parameters                   
###
####################################################################################
county = sys.argv[1]
print (county)
####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
param_dir = "/data/hydro/users/Hossein/NASA/000_shapefile_data_part/"
raw_dir = "/data/hydro/users/Hossein/NASA/01_raw_GEE/"
data_dir = "/data/hydro/users/Hossein/NASA/05_SG_TS/"
SOS_plot_dir = "/data/hydro/users/Hossein/NASA/06_SOS_plots_exactEval/"
eval_dir = "/data/hydro/users/Hossein/NASA/0000_parameters/"
print ("_________________________________________________________")
print ("data dir is: " + data_dir)
print ("_________________________________________________________")

####################################################################################
###
###                   Read data
###
####################################################################################

if county == "Monterey2014":
    raw_names = ["L7_T1C2L2_Scaled_Monterey2014_2013-01-01_2016-01-01.csv",
                 "L8_T1C2L2_Scaled_Monterey2014_2013-01-01_2016-01-01.csv"]

elif county == "AdamBenton2016":
    raw_names = ["L7_T1C2L2_Scaled_AdamBenton2016_2015-01-01_2017-10-14.csv",
                 "L8_T1C2L2_Scaled_AdamBenton2016_2015-01-01_2017-10-14.csv"]

elif county == "FranklinYakima2018":
    raw_names = ["L7_T1C2L2_Scaled_FranklinYakima2018_2017-01-01_2019-10-14.csv",
                 "L8_T1C2L2_Scaled_FranklinYakima2018_2017-01-01_2019-10-14.csv"]

elif county == "Grant2017":
    raw_names = ["L7_T1C2L2_Scaled_Grant2017_2016-01-01_2018-10-14.csv",
                 "L8_T1C2L2_Scaled_Grant2017_2016-01-01_2018-10-14.csv"]

elif county == "Walla2015":
    raw_names = ["L7_T1C2L2_Scaled_Walla2015_2014-01-01_2016-12-31.csv",
                 "L8_T1C2L2_Scaled_Walla2015_2014-01-01_2016-12-31.csv"]


SF_data_name = county + ".csv"

SG_df_NDVI = pd.read_csv(data_dir + "SG_" + county + "_NDVI_JFD.csv")
SG_df_EVI  = pd.read_csv(data_dir + "SG_" + county + "_EVI_JFD.csv")
eval_tb = pd.read_csv(eval_dir + "evaluation_set.csv")

# convert the strings to datetime format
SG_df_NDVI['human_system_start_time'] = pd.to_datetime(SG_df_NDVI['human_system_start_time'])
SG_df_EVI['human_system_start_time'] = pd.to_datetime(SG_df_EVI['human_system_start_time'])

# Monterays ID will be read as integer, convert to string
SG_df_EVI["ID"] = SG_df_EVI["ID"].astype(str)
SG_df_NDVI["ID"] = SG_df_NDVI["ID"].astype(str)

"""
    Read and Clean the raw data
"""
L7 = pd.read_csv(raw_dir + raw_names[0], low_memory=False)
L8 = pd.read_csv(raw_dir + raw_names[1], low_memory=False)
raw_df = pd.concat([L7, L8])
raw_df["ID"] = raw_df["ID"].astype(str)
del (L7, L8)
"""
  Plots should be exact. Therefore, we need to filter by
  last survey year, toss out NASS, and we are sticking to irrigated
  fields for now.
"""
SF_data = pd.read_csv(param_dir + SF_data_name)
SF_data["ID"] = SF_data["ID"].astype(str)

print ("line 116")
print (raw_df.shape)
print (SG_df_NDVI.shape)
print (SG_df_EVI.shape)

print (SF_data.head(1))
print (raw_df.head(1))
print (SG_df_NDVI.head(1))
print (SG_df_EVI.head(1))

if county != "Monterey2014":
    # filter by last survey date. Last 4 digits of county name!
    SF_data = nc.filter_by_lastSurvey(SF_data, year = county[-4:]) 
    SF_data = nc.filter_out_NASS(SF_data)         # Toss NASS
    SF_data = nc.filter_out_nonIrrigated(SF_data) # keep only irrigated lands
    print ("line 130")
    print (SF_data.shape)
    print (SF_data.head(2))
    
    fk = list(SF_data.ID)
    raw_df    = raw_df[raw_df.ID.isin(fk)]
    SG_df_EVI = SG_df_EVI[SG_df_EVI.ID.isin(fk)]
    SG_df_NDVI= SG_df_NDVI[SG_df_NDVI.ID.isin(fk)]

    print ("line 138")
    print (raw_df.shape)
    print (SG_df_NDVI.shape)
    print (SG_df_EVI.shape)

    print (raw_df.head(1))
    print (SG_df_NDVI.head(1))
    print (SG_df_EVI.head(1))

raw_df_EVI = raw_df.copy()
raw_df_NDVI = raw_df.copy()
del(raw_df)

raw_df_EVI.drop(["NDVI"], axis=1, inplace=True)
raw_df_NDVI.drop(["EVI"], axis=1, inplace=True)

raw_df_EVI = raw_df_EVI[raw_df_EVI["EVI"].notna()]
raw_df_NDVI = raw_df_NDVI[raw_df_NDVI["NDVI"].notna()]

raw_df_EVI = nc.add_human_start_time_by_system_start_time(raw_df_EVI)
raw_df_NDVI= nc.add_human_start_time_by_system_start_time(raw_df_NDVI)

########################################

SG_df_NDVI = nc.initial_clean(df = SG_df_NDVI, column_to_be_cleaned = "NDVI")
SG_df_EVI = nc.initial_clean(df = SG_df_EVI, column_to_be_cleaned = "EVI")

raw_df_NDVI = nc.initial_clean(df = raw_df_NDVI, column_to_be_cleaned = "NDVI")
raw_df_EVI = nc.initial_clean(df = raw_df_EVI, column_to_be_cleaned = "EVI")

counter = 0

### List of unique fields
IDs = np.sort(eval_tb["ID"].unique())
print ("_____________________________________")
print('len(IDs) is {}!'.format(len(IDs)))
print ("_____________________________________")


given_year = int(county[-4:])
min_year = pd.to_datetime(datetime.datetime(given_year-1, 1, 1))
max_year = pd.to_datetime(datetime.datetime(given_year+1, 12, 31))
SG_df_NDVI = SG_df_NDVI[SG_df_NDVI.human_system_start_time >= min_year]
SG_df_NDVI = SG_df_NDVI[SG_df_NDVI.human_system_start_time <= max_year]

SG_df_EVI = SG_df_EVI[SG_df_EVI.human_system_start_time >= min_year]
SG_df_EVI = SG_df_EVI[SG_df_EVI.human_system_start_time <= max_year]

print ("line 172")
print (SG_df_NDVI.shape)
print (SG_df_EVI.shape)


for ID in IDs:
    if (counter%1000 == 0):
        print ("_____________________________________")
        print ("counter: " + str(counter))
        print (ID)

    curr_SF_data = SF_data[SF_data['ID'] == ID].copy()
    curr_SG_NDVI = SG_df_NDVI[SG_df_NDVI['ID'] == ID].copy()
    curr_SG_NDVI.sort_values(by=['human_system_start_time'], inplace=True)
    curr_SG_NDVI.reset_index(drop=True, inplace=True)

    curr_raw_NDVI = raw_df_NDVI[raw_df_NDVI['ID'] == ID].copy()
    curr_raw_NDVI.sort_values(by=['human_system_start_time'], inplace=True)
    curr_raw_NDVI.reset_index(drop=True, inplace=True)


    curr_SG_EVI = SG_df_EVI[SG_df_EVI['ID'] == ID].copy()
    curr_SG_EVI.sort_values(by=['human_system_start_time'], inplace=True)
    curr_SG_EVI.reset_index(drop=True, inplace=True)

    curr_raw_EVI = raw_df_EVI[raw_df_EVI['ID'] == ID].copy()
    curr_raw_EVI.sort_values(by=['human_system_start_time'], inplace=True)
    curr_raw_EVI.reset_index(drop=True, inplace=True)
    ################################################################
    
    fig, axs = plt.subplots(2, 1, figsize=(18, 6),
                            sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.1, 'wspace': .1});

    (ax1, ax2) = axs;
    ax1.grid(True); ax2.grid(True); 
    # ax3.grid(True); ax4.grid(True); ax5.grid(True); ax6.grid(True);

    # Plot NDVIs
    ncp.SG_clean_SOS_orchardinPlot(raw_dt = curr_raw_NDVI,
                                   SG_dt = curr_SG_NDVI,
                                   idx = "NDVI",
                                   ax = ax1,
                                   onset_cut = 0.3, 
                                   offset_cut = 0.3);

    # Plot EVIs
    ncp.SG_clean_SOS_orchardinPlot(raw_dt = curr_raw_EVI,
                                   SG_dt = curr_SG_EVI,
                                   idx = "EVI",
                                   ax = ax2,
                                   onset_cut = 0.3, 
                                   offset_cut = 0.3);

    """
       Title is already set in the function above. 
       We can replace/overwrite it here:
    """
    if county == "Monterey2014":
        plant = curr_SF_data['Crop2014'].unique()[0].lower().replace(" ", "_").replace(",", "").replace("/", "_")
        data_source = "Land IQ"
        survey_date = curr_SF_data['LstModDat'].unique()[0]
        plot_title = county + ", " + plant + " (" + ID + ", " + data_source + ", " + survey_date +")"
    else:
        plant = curr_SF_data['CropTyp'].unique()[0].lower().replace(" ", "_").replace(",", "").replace("/", "_")
        data_source = curr_SF_data['DataSrc'].unique()[0]
        irrig_type = curr_SF_data['Irrigtn'].unique()[0]
        survey_date = curr_SF_data['LstSrvD'].unique()[0]
        plot_title = county + ", " + plant + " (" + ID + ", " + data_source + ", " + irrig_type + ", " + survey_date +")"

    ax1.set_title(plot_title);
    ax2.set_title("");

    plot_path = SOS_plot_dir + "/train_plots_JFD/" + plant + "/"
    os.makedirs(plot_path, exist_ok=True)

    fig_name = plot_path + county + "_" + ID +'.pdf'
    plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight')
    plt.close('all')
    counter += 1


print ("done")

end_time = time.time()
print ("it took {:.0f} minutes to run this code.".format((end_time - start_time)/60))


