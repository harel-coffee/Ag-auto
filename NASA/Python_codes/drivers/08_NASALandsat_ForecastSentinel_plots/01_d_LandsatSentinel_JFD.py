####
#### Feb 22, 2022
####
"""    
     After creating the first 250 eval/train set
     there are inconsistencies between NASA/Landsat
     labels and Forecast/Sentinel labels from experts.
     Here we are.
"""
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
from matplotlib.dates import ConciseDateFormatter
import matplotlib.dates as mdates
from datetime import datetime
register_matplotlib_converters()

import sys
start_time = time.time()

####################################################################################
###
###                      Aeolus Core path
###
####################################################################################

sys.path.append('/home/hnoorazar/NASA/')
import NASA_core as nc
import NASA_plot_core as ncp

sys.path.append('/home/hnoorazar/remote_sensing_codes/')
import remote_sensing_core as rc
import remote_sensing_plot_core as rcp


####################################################################################
###
###      Parameters                   
###
####################################################################################
county = sys.argv[1]
NDVI_ratio_cut = 0.3
print (county)

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
SF_dataPart_dir = "/data/hydro/users/Hossein/NASA/000_shapefile_data_part/"
param_dir = "/data/hydro/users/Hossein/NASA/0000_parameters/"
NASA_raw_dir = "/data/hydro/users/Hossein/NASA/01_raw_GEE/"
NASA_data_dir = "/data/hydro/users/Hossein/NASA/05_SG_TS/"
SOS_plot_dir = "/data/hydro/users/Hossein/NASA/08_Sentinel_landsat_plots/"
print ("_________________________________________________________")
print ("data dir is: " + NASA_data_dir)
print ("_________________________________________________________")

####################################################################################
###
###                   Read data
###
####################################################################################

if county == "Monterey2014":
    raw_names = ["L7_T1C2L2_Scaled_Monterey2014_2013-01-01_2016-01-01.csv",
                 "L8_T1C2L2_Scaled_Monterey2014_2013-01-01_2016-01-01.csv"]

if county == "AdamBenton2016":
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

# print ("line 101")
SF_data_name = county + ".csv"

SG_df_NDVI = pd.read_csv(NASA_data_dir + "SG_" + county + "_NDVI_JFD.csv")
SG_df_EVI  = pd.read_csv(NASA_data_dir + "SG_" + county + "_EVI_JFD.csv")
eval_tb = pd.read_csv(param_dir + "evaluation_set.csv")

if county == "AdamBenton2016":
    eval_tb = eval_tb[eval_tb.county.isin(["Adams", "Benton"])]
elif county == "FranklinYakima2018":
    eval_tb = eval_tb[eval_tb.county.isin(["Franklin", "Yakima"])]
elif county == "Grant2017":
    eval_tb = eval_tb[eval_tb.county == "Grant"]
elif county == "Walla2015":
    eval_tb = eval_tb[eval_tb.county == "Walla Walla"]


# convert the strings to datetime format
SG_df_NDVI['human_system_start_time'] = pd.to_datetime(SG_df_NDVI['human_system_start_time'])
SG_df_EVI['human_system_start_time'] = pd.to_datetime(SG_df_EVI['human_system_start_time'])

# Monterays ID will be read as integer, convert to string
SG_df_EVI["ID"] = SG_df_EVI["ID"].astype(str)
SG_df_NDVI["ID"] = SG_df_NDVI["ID"].astype(str)

"""
    Read and Clean the raw data
"""
L7 = pd.read_csv(NASA_raw_dir + raw_names[0], low_memory=False)
L8 = pd.read_csv(NASA_raw_dir + raw_names[1], low_memory=False)
NASA_raw_df = pd.concat([L7, L8])
NASA_raw_df["ID"] = NASA_raw_df["ID"].astype(str)
del (L7, L8)
"""
  Plots should be exact. Therefore, we need to filter by
  last survey year, toss out NASS, and we are sticking to irrigated
  fields for now.
"""
SF_data = pd.read_csv(SF_dataPart_dir + SF_data_name)
SF_data["ID"] = SF_data["ID"].astype(str)


if county != "Monterey2014":
    # filter by last survey date. Last 4 digits of county name!
    SF_data = nc.filter_by_lastSurvey(SF_data, year=county[-4:]) 
    SF_data = nc.filter_out_NASS(SF_data)         # Toss NASS
    SF_data = nc.filter_out_nonIrrigated(SF_data) # keep only irrigated lands
    
    fk = list(SF_data.ID)
    NASA_raw_df    = NASA_raw_df[NASA_raw_df.ID.isin(fk)]
    SG_df_EVI = SG_df_EVI[SG_df_EVI.ID.isin(fk)]
    SG_df_NDVI= SG_df_NDVI[SG_df_NDVI.ID.isin(fk)]


NASA_raw_df_EVI = NASA_raw_df.copy()
NASA_raw_df_NDVI = NASA_raw_df.copy()
del(NASA_raw_df)

NASA_raw_df_EVI.drop(["NDVI"], axis=1, inplace=True)
NASA_raw_df_NDVI.drop(["EVI"], axis=1, inplace=True)

NASA_raw_df_EVI = NASA_raw_df_EVI[NASA_raw_df_EVI["EVI"].notna()]
NASA_raw_df_NDVI = NASA_raw_df_NDVI[NASA_raw_df_NDVI["NDVI"].notna()]

NASA_raw_df_EVI = nc.add_human_start_time_by_system_start_time(NASA_raw_df_EVI)
NASA_raw_df_NDVI= nc.add_human_start_time_by_system_start_time(NASA_raw_df_NDVI)
# print ("line 167")
########################################

SG_df_NDVI = nc.initial_clean(df = SG_df_NDVI, column_to_be_cleaned = "NDVI")
SG_df_EVI = nc.initial_clean(df = SG_df_EVI, column_to_be_cleaned = "EVI")

NASA_raw_df_NDVI= nc.initial_clean(df=NASA_raw_df_NDVI, column_to_be_cleaned = "NDVI")
NASA_raw_df_EVI = nc.initial_clean(df=NASA_raw_df_EVI, column_to_be_cleaned = "EVI")

counter = 0

### List of unique fields
IDs = np.sort(eval_tb["ID"].unique())
print ("_____________________________________")
print('len(IDs) is {}!'.format(len(IDs)))
print ("_____________________________________")

# print ("line 184")
given_year = int(county[-4:])
# min_year = pd.to_datetime(datetime.datetime(given_year-1, 1, 1))
# max_year = pd.to_datetime(datetime.datetime(given_year+1, 12, 31))

min_year = pd.to_datetime(datetime(given_year-1, 1, 1))
max_year = pd.to_datetime(datetime(given_year+1, 12, 31))

SG_df_NDVI = SG_df_NDVI[SG_df_NDVI.human_system_start_time >= min_year]
SG_df_NDVI = SG_df_NDVI[SG_df_NDVI.human_system_start_time <= max_year]

SG_df_EVI = SG_df_EVI[SG_df_EVI.human_system_start_time >= min_year]
SG_df_EVI = SG_df_EVI[SG_df_EVI.human_system_start_time <= max_year]

# print ("line 198")
######################################################################
######################################################################
######################################################################
############
############       Read and Clean Sentinel EVI data
############
######################################################################
######################################################################
######################################################################

regularized = True
indeks = "EVI"
irrigated_only = True
given_county = sys.argv[1]

if irrigated_only == True:
    output_Irr = "irrigated_only"
else:
    output_Irr = "non_irrigated_only"

regular_data_dir = "/data/hydro/users/Hossein/remote_sensing/03_Regularized_TS/70_cloud/2Yrs/noJump_Regularized/"
sentinel_raw_dir = "/data/hydro/users/Hossein/remote_sensing/02_Eastern_WA_EE_TS/2Years/70_cloud/"
# print ("line 221")

if county == "AdamBenton2016":
    f_names = ["01_Regular_filledGap_Adams_SF_2016_EVI.csv",
               "01_Regular_filledGap_Benton_SF_2016_EVI.csv"]

    a_sentinel_df = pd.read_csv(regular_data_dir + f_names[0], low_memory=False)
    a_sentinel_df_2 = pd.read_csv(regular_data_dir + f_names[1], low_memory=False)
    # print ("line 229: ", a_sentinel_df.shape)
    # print ("line 230: ", a_sentinel_df_2.shape)
    a_sentinel_df = pd.concat([a_sentinel_df, a_sentinel_df_2])

elif county == "FranklinYakima2018":
    f_names = ["01_Regular_filledGap_Franklin_SF_2018_EVI.csv",
               "01_Regular_filledGap_Yakima_SF_2018_EVI.csv"]
    a_sentinel_df = pd.read_csv(regular_data_dir + f_names[0], low_memory=False)
    a_sentinel_df_2 = pd.read_csv(regular_data_dir + f_names[1], low_memory=False)
    a_sentinel_df = pd.concat([a_sentinel_df, a_sentinel_df_2])

elif county == "Grant2017":
    f_names = ["01_Regular_filledGap_Grant_SF_2017_EVI.csv"]
    a_sentinel_df = pd.read_csv(regular_data_dir + f_names[0], low_memory=False)

elif county == "Walla2015":
    f_names = ["01_Regular_filledGap_Walla_Walla_SF_2015_EVI.csv"]
    a_sentinel_df = pd.read_csv(regular_data_dir + f_names[0], low_memory=False)

# print ("line 246")
SF_year = int(county[-4:])
raw_f_name = "Eastern_WA_" + str(SF_year) + "_70cloud_selectors.csv"
sentinel_raw_df = pd.read_csv(sentinel_raw_dir + raw_f_name, low_memory=False)
# print ("line 252: ", sentinel_raw_df.shape)
# print ("line 253: ", sentinel_raw_df.county.unique())

# print ("line 255")
if 'Date' in a_sentinel_df.columns:
    if type(a_sentinel_df.Date.iloc[0]) == str:
        a_sentinel_df['Date'] = pd.to_datetime(a_sentinel_df.Date.values).values


a_sentinel_df['SF_year'] = SF_year
a_sentinel_df = rc.initial_clean(df=a_sentinel_df, column_to_be_cleaned=indeks)
sentinel_raw_df = rc.initial_clean(df=sentinel_raw_df, column_to_be_cleaned=indeks)
# print ("line 264")
if not("human_system_start_time" in sentinel_raw_df.columns):
    sentinel_raw_df = rc.add_human_start_time_by_YearDoY(sentinel_raw_df)

if 'Date' in sentinel_raw_df.columns:
    if type(sentinel_raw_df.Date.iloc[0]) == str:
        sentinel_raw_df['Date'] = pd.to_datetime(sentinel_raw_df.Date.values).values
else: 
    sentinel_raw_df['Date'] = pd.to_datetime(sentinel_raw_df.human_system_start_time.values).values

# print ("line 274")
an_EE_TS = a_sentinel_df.copy()
del(a_sentinel_df)

######################################################################
######################################################################
######################################################################
############
############       Plot Side by sude.
############
######################################################################
######################################################################
######################################################################
for ID in IDs:
    if (counter%1000 == 0):
        print ("_____________________________________")
        print ("counter: " + str(counter))
        print (ID)

    curr_SF_data = SF_data[SF_data['ID'] == ID].copy()
    curr_SG_NDVI = SG_df_NDVI[SG_df_NDVI['ID'] == ID].copy()
    curr_SG_NDVI.sort_values(by=['human_system_start_time'], inplace=True)
    curr_SG_NDVI.reset_index(drop=True, inplace=True)

    curr_raw_NDVI = NASA_raw_df_NDVI[NASA_raw_df_NDVI['ID'] == ID].copy()
    curr_raw_NDVI.sort_values(by=['human_system_start_time'], inplace=True)
    curr_raw_NDVI.reset_index(drop=True, inplace=True)


    curr_SG_EVI = SG_df_EVI[SG_df_EVI['ID'] == ID].copy()
    curr_SG_EVI.sort_values(by=['human_system_start_time'], inplace=True)
    curr_SG_EVI.reset_index(drop=True, inplace=True)

    curr_raw_EVI = NASA_raw_df_EVI[NASA_raw_df_EVI['ID'] == ID].copy()
    curr_raw_EVI.sort_values(by=['human_system_start_time'], inplace=True)
    curr_raw_EVI.reset_index(drop=True, inplace=True)
    ################################################################
    ##
    ##   Sentinel Part of the loop
    ##
    # print ("line 314")
    curr_field_two_years = an_EE_TS[an_EE_TS['ID'] == ID].copy()
    sentinel_curr_raw_EVI = sentinel_raw_df[sentinel_raw_df['ID'] == ID].copy()
    # print ("line 317: " , curr_field_two_years.shape)
    # print ("line 318: " , sentinel_curr_raw_EVI.shape)
    curr_field_two_years.sort_values(by=['image_year', 'doy'], inplace=True)
    sentinel_curr_raw_EVI.sort_values(by=['image_year', 'doy'], inplace=True)
    # print ("line 321")
    # print ("line 322", curr_field_two_years.columns)
    # print ("line 323", sentinel_curr_raw_EVI.columns)
    # print ("line 324", list(curr_field_two_years.image_year.unique())[0])
    # print ("line 325", list(sentinel_curr_raw_EVI.image_year.unique())[0])
    # print ("line 326", curr_field_two_years.shape)
    # print ("line 327", sentinel_curr_raw_EVI.shape)
    ################################################################
    
    fig, axs = plt.subplots(3, 1, figsize=(18, 9.5),
                        sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.1, 'wspace': .1});

    (ax1, ax2, ax3) = axs;
    ax1.grid(True); ax2.grid(True); ax3.grid(True); 
    # ax3.grid(True); ax4.grid(True); ax5.grid(True); ax6.grid(True);

    # Plot NDVIs
    ncp.SG_clean_SOS_orchardinPlot_VerticalLine(raw_dt=curr_raw_NDVI,
                                                SG_dt=curr_SG_NDVI,
                                                idx="NDVI",
                                                ax=ax1,
                                                onset_cut=NDVI_ratio_cut, 
                                                offset_cut=NDVI_ratio_cut);
    # print ("line 343")
    # Plot EVIs
    ncp.SG_clean_SOS_orchardinPlot_VerticalLine(raw_dt=curr_raw_EVI,
                                                SG_dt=curr_SG_EVI,
                                                idx="EVI",
                                                ax=ax2,
                                                onset_cut=NDVI_ratio_cut, 
                                                offset_cut=NDVI_ratio_cut);

    ax1.axvspan(datetime(given_year+1, 1, 1), datetime(given_year+2, 1, 1), facecolor='.01', alpha=0.3)
    ax1.axvspan(datetime(given_year-1, 1, 1), datetime(given_year, 1, 1), facecolor='.01', alpha=0.3)
    ax2.axvspan(datetime(given_year+1, 1, 1), datetime(given_year+2, 1, 1), facecolor='.01', alpha=0.3)
    ax2.axvspan(datetime(given_year-1, 1, 1), datetime(given_year, 1, 1), facecolor='.01', alpha=0.3)

    # Plot Sentinel EVI. That is just what we have right now. (Feb. 22, 2022)
    # print ("line 360")
    rcp.panel_SOS_Sentinel4Landsat(twoYears_raw = sentinel_curr_raw_EVI,
                                   twoYears_regular = curr_field_two_years,
                                   idx = indeks, SG_params=[7, 3],
                                   SFYr = SF_year, ax=ax3,
                                   onset_cut = NDVI_ratio_cut, 
                                   offset_cut = NDVI_ratio_cut);
    # print ("line 367")


    start = str(given_year) + '-01-01'
    end = str(given_year) + '-12-01'
    vertical_dates = pd.date_range(start, end, freq='MS')
    vertical_dates = vertical_dates[::3]
    # ax1.vlines(x=vertical_dates, lw=1,
    #            ymin=-1, ymax=1, color='r', 
    #            label='test lines')

    # ax2.vlines(x=vertical_dates, lw=1,
    #            ymin=-1, ymax=1, color='r', 
    #            label='test lines')

    vertical_annotations = vertical_dates[::3]
    # style = dict(size=10, color='g', rotation='vertical')
    # for ii in np.arange(0, len(vertical_annotations)):
    #     ax1.text(x = vertical_annotations[ii],
    #              y = 0, 
    #              s= str(vertical_dates[ii].month) + "-" + str(vertical_dates[ii].day),
    #              **style
    #             )
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
        cc =  curr_SF_data.county.unique()[0]
        area = curr_SF_data.Acres.unique()[0]
        
        plot_title = cc + ", " + plant + " (" + ID + ", " + data_source + ", " + \
                       irrig_type + ", " + survey_date + ", Area: " + str(area) + " acres)"

    ax1.set_title(plot_title);
    ax2.set_title("");

    # import matplotlib.dates as mdates
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    # Rotates and right-aligns the x labels so they don't crowd each other.
    # for label in ax1.get_xticklabels(which='major'):
    #     label.set(rotation=90, horizontalalignment='right')

    # ax1.xaxis.set_major_formatter(
    #     mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))
    
    ax2.xaxis.set_major_formatter(ConciseDateFormatter(ax2.xaxis.get_major_locator()))
    ax1.xaxis.set_major_formatter(ConciseDateFormatter(ax1.xaxis.get_major_locator()))
    # ax3.xaxis.set_major_formatter(ConciseDateFormatter(ax3.xaxis.get_major_locator()))
    # ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # locator = mdates.AutoDateLocator(minticks=36, maxticks=36)
    # formatter = mdates.ConciseDateFormatter(locator)
    # ax3.xaxis.set_major_locator(locator)
    # ax3.xaxis.set_major_formatter(formatter)
    
    # ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax3.xaxis.set_major_formatter(ConciseDateFormatter(ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))))


    # matplotlib.dates.ConciseDateFormatter(ax3.xaxis.get_major_locator())

    plot_path = SOS_plot_dir + "/exactEval6000_landSentinel/" + plant + "/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + county + "_" + ID +'.png'
    plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight', facecolor="w")

    # save flat
    plot_path = SOS_plot_dir + "/exactEval6000_landSentinel_flat/"
    os.makedirs(plot_path, exist_ok=True)
    fig_name = plot_path + county + "_" + ID +'.png'
    plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight', facecolor="w")
    
    plt.close('all')
    counter += 1


print ("done")

end_time = time.time()
print ("it took {:.0f} minutes to run this code.".format((end_time - start_time)/60))


