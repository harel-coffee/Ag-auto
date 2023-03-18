# import libraries

import numpy as np
import pandas as pd
# import geopandas as gpd
from IPython.display import Image
# from shapely.geometry import Point, Polygon
from math import factorial
import datetime
import time
import scipy
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from patsy import cr

from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sb
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import ConciseDateFormatter
import matplotlib.dates as mdates
import os, os.path

from datetime import datetime
register_matplotlib_converters()

import sys
# search path for modules
# look @ https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/remote_sensing/python/')
import remote_sensing_core as rc

################################################################
#####
#####                   Function definitions
#####
################################################################
def panel_SOS_Sentinel4Landsat(twoYears_raw, twoYears_regular, idx, SG_params, 
                               SFYr, ax, onset_cut=0.5, offset_cut=0.5):

    """  Feb 2022. Sentinel Vs Landsat
    This is clean version of SG_1yr_panels_clean_sciPy_My_Peaks_SOS_fineGranularity(.)
    HERE, I have removed the minimums and maximums from the plot along with the "processed" data points
    that plotted some data till before SG smoothing. 
    I will use this to plot Sentinel and landsat side by side.


    The function SG_1yr_panels_clean_sciPy_My_Peaks_SOS_fineGranularity(.) itself
    has the following comment in it:
    This function has additional part to plot SOS and EOS.
    and it is updated version of the function savitzky_1yr_panels_clean_sciPy_and_My_PeakFinder(.)
    """
    ## crr_fld = dataAB.copy() #***
    crr_fld_twoYrs_regular = twoYears_regular.copy() #***
    # print ("RS core - line 50, ", crr_fld_twoYrs_regular.shape)
    # print ("RS core - line 51, ", crr_fld_twoYrs_regular.columns)
    # print ("RS core - line 52, ", crr_fld_twoYrs_regular.image_year.unique())
    if (not("human_system_start_time" in list(crr_fld_twoYrs_regular.columns))):
        crr_fld_twoYrs_regular = rc.add_human_start_time(crr_fld_twoYrs_regular)

    plant = crr_fld_twoYrs_regular['CropTyp'].unique()[0]
    #
    # Take care of names, replace "/" and "," and " " by "_"
    #
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")

    county = crr_fld_twoYrs_regular['county'].unique()[0]
    ID = crr_fld_twoYrs_regular['ID'].unique()[0]
    data_source = crr_fld_twoYrs_regular['DataSrc'].unique()[0]
    irrig_type = crr_fld_twoYrs_regular['Irrigtn'].unique()[0]
    survey_date = crr_fld_twoYrs_regular['LstSrvD'].unique()[0]

    ### Copy the regularized values that are processed up to SG smoothing.
    twoYrs_regular_Yvalues = crr_fld_twoYrs_regular[idx].copy()
    twoYrs_regular_Xvalues = crr_fld_twoYrs_regular['Date'].copy()

    #############################################
    ###
    ###             Smoothen
    ###
    #############################################
    window_len = SG_params[0]
    poly_order = SG_params[1]
    SG_pred_twoYears = scipy.signal.savgol_filter(twoYrs_regular_Yvalues, window_length=window_len, polyorder=poly_order)

    # SG might violate the boundaries. clip them:
    SG_pred_twoYears[SG_pred_twoYears > 1 ] = 1
    SG_pred_twoYears[SG_pred_twoYears < -1 ] = -1
    
    crr_fld_twoYrs_regular[idx] = SG_pred_twoYears
    #############################################
    # crr_fld will be one year from now on
    crr_fld = crr_fld_twoYrs_regular.copy()
    # print ("line 92 RS core, ", crr_fld.shape, " ", crr_fld.image_year.unique())
    
    # print ("line 94: ",  type(crr_fld.image_year.unique()[0]))
    # print ("line 95: ", type(crr_fld.image_year.unique()[1]))
    # print ("line 96 RS core, ", SFYr, ", ", type(SFYr))
    crr_fld = crr_fld[crr_fld.image_year == SFYr]
    # print ("line 98 RS core, ", crr_fld.image_year.unique())
    #############################################
    ###
    ###             fine granularity table
    ###
    #############################################
    # create the full calendar to make better estimation of SOS and EOS.
    fine_granular_table = rc.create_calendar_table(SF_year = SFYr)
    fine_granular_table = pd.merge(fine_granular_table, crr_fld, on=['Date', 'SF_year', 'doy'], how='left')
    # print ("line 107 RS core, ", fine_granular_table.shape)

    ###### We need to fill the NAs that are created because they were not created in fine_granular_table
    fine_granular_table["image_year"] = crr_fld["image_year"].unique()[0]
    # print ("line 109 RS core, ", fine_granular_table.shape)
    fine_granular_table["ID"]     = crr_fld["ID"].unique()[0]
    fine_granular_table["Acres"]  = crr_fld["Acres"].unique()[0]
    fine_granular_table["county"] = crr_fld["county"].unique()[0]
    fine_granular_table["CropTyp"] = crr_fld["CropTyp"].unique()[0]
    fine_granular_table["DataSrc"] = crr_fld["DataSrc"].unique()[0]
    fine_granular_table["Irrigtn"] = crr_fld["Irrigtn"].unique()[0]
    fine_granular_table["LstSrvD"] = crr_fld["LstSrvD"].unique()[0]

    # fine_granular_table["CropGrp"] = crr_fld["CropGrp"].unique()[0] # commented out on Feb 2022
    # fine_granular_table["ExctAcr"] = crr_fld["ExctAcr"].unique()[0] # commented out on Feb 2022
    # fine_granular_table["IntlSrD"] = crr_fld["IntlSrD"].unique()[0] # commented out on Feb 2022
    # fine_granular_table["Notes"]   = crr_fld["Notes"].unique()[0]   # commented out on Feb 2022
    # fine_granular_table["RtCrpTy"] = crr_fld["RtCrpTy"].unique()[0] # commented out on Feb 2022
    # fine_granular_table["Shap_Ar"] = crr_fld["Shap_Ar"].unique()[0] # commented out on Feb 2022
    # fine_granular_table["Shp_Lng"] = crr_fld["Shp_Lng"].unique()[0] # commented out on Feb 2022
    # fine_granular_table["TRS"] = crr_fld["TRS"].unique()[0] # commented out on Feb 2022

    fine_granular_table = rc.add_human_start_time_by_YearDoY(fine_granular_table)

    # replace NAs with -1.5. Because, that is what the function fill_theGap_linearLine()
    # uses as indicator for missing values
    fine_granular_table.fillna(value={idx:-1.5}, inplace=True)
    
    fine_granular_table = rc.fill_theGap_linearLine(regular_TS = fine_granular_table, 
                                                    V_idx=idx, 
                                                    SF_year=SFYr)
    
    # update SG_pred so that we do not have to update too many other stuff.
    SG_pred = fine_granular_table[idx].values.copy()
    crr_fld = fine_granular_table
    y = fine_granular_table[idx].copy()
    #############################################
    ###
    ###   Form a data table of X and Y values
    ###
    #############################################

    if len(fine_granular_table['image_year'].unique()) == 2:
        X = rc.extract_XValues_of_2Yrs_TS(fine_granular_table, SF_yr = SFYr)
    elif len(fine_granular_table['image_year'].unique()) == 1:
        X = fine_granular_table['doy']

    d = {'DoY': X, 'Date': pd.to_datetime(fine_granular_table.human_system_start_time.values).values}
    date_df = pd.DataFrame(data=d)

    #############################################
    ###
    ###      Form a dictionary for plotting
    ###
    #############################################

    keyy = "SG: [" + str(window_len) + ", " + str(poly_order) + "]"
    plotting_dic = { keyy : [SG_pred]}

    #############################################
    ###
    ###             plot
    ###
    #############################################
    plot_title = county + ", " + plant + " (" + ID + ", " + data_source + ", " + irrig_type + ", " + survey_date +")" 
    ax.set_ylim([-0.3, 1.15])
    
    # plot regularized version of two years
    # ax.scatter(twoYrs_regular_Xvalues.values, twoYrs_regular_Yvalues, 
    #            label="2YRs process. data", s=7, c='#E4D00A'); # commented out on Feb 2022

    # plot SG smoothed version of two years
    ax.plot(crr_fld_twoYrs_regular['Date'].values, crr_fld_twoYrs_regular[idx], label= keyy, c='k');
    ax.scatter(date_df.Date.values, y.values, label="1-Day Time Series", s=7, c='b');

    ###
    ###   plot SOS and EOS
    ###
    # Update the EVI/NDVI values to the smoothed version.
    crr_fld [idx] = SG_pred
    crr_fld = rc.addToDF_SOS_EOS_White(pd_TS=crr_fld, VegIdx=idx, 
                                       onset_thresh=onset_cut, 
                                       offset_thresh=offset_cut)

    ##
    ##  Kill bad detected seasons 
    ##
    crr_fld = rc.Null_SOS_EOS_by_DoYDiff(pd_TS = crr_fld, min_season_length=40)

    #
    #  Start of the season
    #
    # SOS = crr_fld[crr_fld['SOS'] != 0]
    # ax.scatter(SOS['Date'], SOS['SOS'], marker='+', s=155, c='g')

    # annotate SOS
    # for ii in np.arange(0, len(SOS)):
    #     style = dict(size=10, color='g', rotation='vertical')
    #     ax.text(x = SOS.iloc[ii]['Date'].date(), 
    #             y = -0.7, 
    #             s = 'DoY=' + str(SOS.iloc[ii]['doy']), 
    #             **style)
    #
    #  End of the season
    #
    # EOS = crr_fld[crr_fld['EOS'] != 0]
    # ax.scatter(EOS['Date'], EOS['EOS'], marker='+', s=155, c='r')

    # annotate EOS
    # for ii in np.arange(0, len(EOS)):
    #     style = dict(size=10, color='r', rotation='vertical')
    #     ax.text(x = EOS.iloc[ii]['Date'].date(), 
    #             y = -0.7, 
    #             s = 'DoY=' + str(EOS.iloc[ii]['doy']), 
    #             **style)

    # Plot ratios:
    # ax.plot(crr_fld['Date'], crr_fld['EVI_ratio'], c='gray', label="EVI Ratio")

    # ax.axhline(0 , color = 'r', linewidth=.5)
    # ax.axhline(1 , color = 'r', linewidth=.5)
    # ax.axhline(-1, color = 'r', linewidth=.5)

    # plot raw data of two years
    ax.scatter(twoYears_raw['Date'].values, twoYears_raw[idx], label="raw", s=7, c='dodgerblue');

    ax.set_title(plot_title);
    ax.set(ylabel=idx)
    ax.legend(loc="upper left");

    ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # locator = mdates.AutoDateLocator(minticks=36, maxticks=36)
    # formatter = mdates.ConciseDateFormatter(locator)
    # ax.xaxis.set_major_locator(locator)
    # ax.xaxis.set_major_formatter(formatter)
    
    # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))))


def plot_for_landsat_proposal(twoYears_EVI8Day, SFYr, ax, onset_cut=0.3, offset_cut=0.3):
    """
    This function has additional part to plot SOS and EOS.
    and it is updated version of the function savitzky_1yr_panels_clean_sciPy_and_My_PeakFinder(.)
    """
    crr_fld_twoYrs_regular = twoYears_EVI8Day.copy() #***
    if (not("human_system_start_time" in list(crr_fld_twoYrs_regular.columns))):
        crr_fld_twoYrs_regular = rc.add_human_start_time(crr_fld_twoYrs_regular)

    eleven_colors = ["gray", "lightcoral", "red", "peru",
                     "darkorange", "gold", "olive", "green",
                     "blue", "violet", "deepskyblue"]

    plant = crr_fld_twoYrs_regular['CropTyp'].unique()[0]
    #
    # Take care of names, replace "/" and "," and " " by "_"
    #
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")

    county = crr_fld_twoYrs_regular['county'].unique()[0]
    ID = crr_fld_twoYrs_regular['ID'].unique()[0]
    data_source = crr_fld_twoYrs_regular['DataSrc'].unique()[0]
    irrig_type = crr_fld_twoYrs_regular['Irrigtn'].unique()[0]
    survey_date = crr_fld_twoYrs_regular['LstSrvD'].unique()[0]

    ###
    ###             Smoothen
    ###

    #############################################
    #############################################
    # crr_fld will be one year from now on
    crr_fld = crr_fld_twoYrs_regular.copy()
    crr_fld = crr_fld[crr_fld.image_year == SFYr]

    #############################################
    ###
    ###             fine granularity table
    ###
    #############################################
    # create the full calendar to make better estimation of SOS and EOS.
    fine_granular_table = rc.create_calendar_table(SF_year = SFYr)
    fine_granular_table.rename(columns={"Date": "human_system_start_time"}, inplace=True)
    fine_granular_table = pd.merge(fine_granular_table, crr_fld, on=['human_system_start_time', 'SF_year', 'doy'], how='left')

    ###### We need to fill the NAs that are created because they were not created in fine_granular_table
    fine_granular_table["image_year"] = crr_fld["image_year"].unique()[0]
    fine_granular_table["ID"]     = crr_fld["ID"].unique()[0]
    fine_granular_table["county"] = crr_fld["county"].unique()[0]
    fine_granular_table["CropTyp"] = crr_fld["CropTyp"].unique()[0]
    fine_granular_table["DataSrc"] = crr_fld["DataSrc"].unique()[0]

    fine_granular_table["Irrigtn"] = crr_fld["Irrigtn"].unique()[0]

    fine_granular_table["LstSrvD"] = crr_fld["LstSrvD"].unique()[0]
    fine_granular_table["Notes"]   = crr_fld["Notes"].unique()[0]
    fine_granular_table["RtCrpTy"] = crr_fld["RtCrpTy"].unique()[0]

    # fine_granular_table = rc.add_human_start_time_by_YearDoY(fine_granular_table)

    # replace NAs with -1.5. Because, that is what the function fill_theGap_linearLine()
    # uses as indicator for missing values
    fine_granular_table.fillna(value={"SG_EVI":-1.5}, inplace=True)
    
    fine_granular_table = rc.fill_theGap_linearLine(regular_TS = fine_granular_table, 
                                                    V_idx = "SG_EVI", 
                                                    SF_year=SFYr)
    
    # update SG_pred so that we do not have to update too many other stuff.
    SG_pred = fine_granular_table["SG_EVI"].values.copy()
    crr_fld = fine_granular_table
    y = fine_granular_table["SG_EVI"].copy()
    #############################################
    ###
    ###   Form a data table of X and Y values
    ###
    #############################################

    if len(fine_granular_table['image_year'].unique()) == 2:
        X = rc.extract_XValues_of_2Yrs_TS(fine_granular_table, SF_yr = SFYr)
    elif len(fine_granular_table['image_year'].unique()) == 1:
        X = fine_granular_table['doy']

    d = {'DoY': X, 'human_system_start_time': pd.to_datetime(fine_granular_table.human_system_start_time.values).values}
    date_df = pd.DataFrame(data=d)

    #############################################
    ###
    ###             plot
    ###
    #############################################
    plot_title = county + ", " + plant + " (" + ID + ", " + data_source + ", " + irrig_type + ", " + survey_date +")" 
    ax.set_ylim([-1.15, 1.15])

    # plot SG smoothed version of two years

    # ax.plot(crr_fld_twoYrs_regular['human_system_start_time'].values, 
    #         crr_fld_twoYrs_regular["SG_EVI"], 
    #         label= "Smoothed EVI", c='k');

    # plot raw data of two years
    ax.scatter(twoYears_EVI8Day['human_system_start_time'].values, 
               twoYears_EVI8Day["EVI"], 
               label = "Landsat EVI 8-day", 
               s=7, c='k'); # dodgerblue #E4D00A

    ax.scatter(date_df.human_system_start_time.values, y.values, label="Processed EVI", s=7, c='dodgerblue');

    ###
    ###   plot SOS and EOS
    ###
    # Update the EVI/NDVI values to the smoothed version.
    crr_fld ["SG_EVI"] = SG_pred
    crr_fld = rc.addToDF_SOS_EOS_White(pd_TS = crr_fld, VegIdx = "SG_EVI", 
                                       onset_thresh = onset_cut, offset_thresh = offset_cut)

    ##
    ##  Kill bad detected seasons 
    ##
    crr_fld['Date'] = crr_fld.human_system_start_time
    crr_fld = rc.Null_SOS_EOS_by_DoYDiff(pd_TS = crr_fld, min_season_length=40)

    #
    #  Start of the season
    #
    SOS = crr_fld[crr_fld['SOS'] != 0]
    ax.scatter(SOS['human_system_start_time'], SOS['SOS'], marker='+', s=155, c = 'g', label="SOS")

    # annotate SOS
    for ii in np.arange(0, len(SOS)):
        style = dict(size=10, color='g', rotation='vertical')
        ax.text(x = SOS.iloc[ii]['human_system_start_time'].date(), 
                y = -0.95, 
                s = 'DoY=' + str(SOS.iloc[ii]['doy']), 
                **style)

    #
    #  End of the season
    #

    EOS = crr_fld[crr_fld['EOS'] != 0]
    ax.scatter(EOS['human_system_start_time'], EOS['EOS'], marker='+', s=155, c='r', label="EOS")

    # annotate EOS
    for ii in np.arange(0, len(EOS)):
        style = dict(size=10, color='r', rotation='vertical')
        ax.text(x = EOS.iloc[ii]['human_system_start_time'].date(), 
                y = -0.95, 
                s = 'DoY=' + str(EOS.iloc[ii]['doy']), 
                **style)

    ax.axhline(0 , color = 'r', linewidth=.5)
    ax.axhline(1 , color = 'r', linewidth=.5)
    ax.axhline(-1, color = 'r', linewidth=.5)

    # # plot raw data of two years
    # ax.scatter(twoYears_EVI8Day['human_system_start_time'].values, 
    #            twoYears_EVI8Day["EVI"], 
    #            label = "Landsat EVI 8-day", 
    #            s=7, c='#E4D00A'); # dodgerblue

    ax.set_title(plot_title);
    ax.set(ylabel = "EVI")
    ax.legend(loc="upper left");

############################################################################################

def SG_1yr_panels_noPeak_SOS_fineGranularity(twoYears_raw, 
                                             twoYears_regular, 
                                             idx, SG_params, SFYr, ax, deltA = 0.4, 
                                             onset_cut=0.5, offset_cut=0.5):
    """
    This function has additional part to plot SOS and EOS.
    and it is updated version of the function savitzky_1yr_panels_clean_sciPy_and_My_PeakFinder(.)
    """
    ## crr_fld = dataAB.copy() #***
    crr_fld_twoYrs_regular = twoYears_regular.copy() #***
    if (not("human_system_start_time" in list(crr_fld_twoYrs_regular.columns))):
        crr_fld_twoYrs_regular = rc.add_human_start_time(crr_fld_twoYrs_regular)

    eleven_colors = ["gray", "lightcoral", "red", "peru",
                     "darkorange", "gold", "olive", "green",
                     "blue", "violet", "deepskyblue"]

    plant = crr_fld_twoYrs_regular['CropTyp'].unique()[0]
    #
    # Take care of names, replace "/" and "," and " " by "_"
    #
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")

    county = crr_fld_twoYrs_regular['county'].unique()[0]
    ID = crr_fld_twoYrs_regular['ID'].unique()[0]
    ###
    ### Copy the regularized values that are processed up to SG smoothing.
    ###
    twoYrs_regular_Yvalues = crr_fld_twoYrs_regular[idx].copy()
    twoYrs_regular_Xvalues = crr_fld_twoYrs_regular['Date'].copy()

    crr_fld_twoYrs_regular_1Year = crr_fld_twoYrs_regular.copy()
    crr_fld_twoYrs_regular_1Year = crr_fld_twoYrs_regular_1Year[crr_fld_twoYrs_regular_1Year.image_year >= SFYr]
    twoYrs_regular_Yvalues_1Year = crr_fld_twoYrs_regular_1Year[idx].copy()
    twoYrs_regular_Xvalues_1Year = crr_fld_twoYrs_regular_1Year['Date'].copy()

    #############################################
    ###
    ###             Smoothen
    ###
    #############################################
    # differences are minor, but lets keep using Pythons function
    # my_savitzky_pred = rc.savitzky_golay(y, window_size=Sav_win_size, order=sav_order)
    window_len = SG_params[0]
    poly_order = SG_params[1]

    ## SG_pred = scipy.signal.savgol_filter(y, window_length= window_len, polyorder=poly_order) #***

    SG_pred_twoYears = scipy.signal.savgol_filter(twoYrs_regular_Yvalues, 
                                                  window_length= window_len, 
                                                  polyorder=poly_order)

    # SG might violate the boundaries. clip them:
    SG_pred_twoYears[SG_pred_twoYears > 1 ] = 1
    SG_pred_twoYears[SG_pred_twoYears < -1 ] = -1
    
    crr_fld_twoYrs_regular[idx] = SG_pred_twoYears
    
    # crr_fld will be one year from now on
    crr_fld = crr_fld_twoYrs_regular.copy()
    crr_fld = crr_fld[crr_fld.image_year == SFYr]
    #############################################
    ###
    ###             fine granularity table
    ###
    #############################################
    # create the full calendar to make better estimation of SOS and EOS.
    fine_granular_table = rc.create_calendar_table(SF_year = SFYr)
    fine_granular_table = pd.merge(fine_granular_table, crr_fld, on=['Date', 'SF_year', 'doy'], how='left')

    ###### We need to fill the NAs that are created because they were not created in fine_granular_table
    fine_granular_table["image_year"] = crr_fld["image_year"].unique()[0]
    fine_granular_table["ID"]     = crr_fld["ID"].unique()[0]

    fine_granular_table = rc.add_human_start_time_by_YearDoY(fine_granular_table)

    # replace NAs with -1.5. Because, that is what the function fill_theGap_linearLine()
    # uses as indicator for missing values
    fine_granular_table.fillna(value={idx:-1.5}, inplace=True)
    
    fine_granular_table = rc.fill_theGap_linearLine(regular_TS = fine_granular_table, 
                                                    V_idx=idx, 
                                                    SF_year=SFYr)
    
    # update SG_pred so that we do not have to update too many other stuff.
    SG_pred = fine_granular_table[idx].values.copy()
    crr_fld = fine_granular_table
    y = fine_granular_table[idx].copy()
    #############################################
    ###
    ###   Form a data table of X and Y values
    ###
    #############################################

    if len(fine_granular_table['image_year'].unique()) == 2:
        X = rc.extract_XValues_of_2Yrs_TS(fine_granular_table, SF_yr = SFYr)
    elif len(fine_granular_table['image_year'].unique()) == 1:
        X = fine_granular_table['doy']

    d = {'DoY': X, 'Date': pd.to_datetime(fine_granular_table.human_system_start_time.values).values}
    date_df = pd.DataFrame(data=d)

    #############################################
    ###
    ###             plot
    ###
    #############################################
    plant.replace("_", " ")
    ax.set_ylim([-1.15, 1.15])

    ###
    ###   plot SOS and EOS
    ###
    # Update the EVI/NDVI values to the smoothed version.
    crr_fld [idx] = SG_pred
    crr_fld = rc.addToDF_SOS_EOS_White(pd_TS = crr_fld, VegIdx = idx, 
                                       onset_thresh = onset_cut, 
                                       offset_thresh = offset_cut)
   
    crr_fld = rc.Null_SOS_EOS_by_DoYDiff(pd_TS = crr_fld, min_season_length=40)
    
    # Plot ratios:
    crr_fld = crr_fld[crr_fld.Date >= pd.Timestamp(year=SFYr, month=1, day=1)]
    ax.plot(crr_fld['Date'], crr_fld['EVI_ratio'], c='gray', label="EVI Ratio")

    ax.scatter(date_df.Date.values, y.values, label="1-Day Time Series", s=7, c='b');
    # plot SG smoothed version of two years
    keyy = "SG: [" + str(window_len) + ", " + str(poly_order) + "]"
    crr_fld_twoYrs_regular = crr_fld_twoYrs_regular[crr_fld_twoYrs_regular.image_year >= SFYr]
    ax.plot(crr_fld_twoYrs_regular['Date'].values, crr_fld_twoYrs_regular[idx], label= keyy, c='k');

    SOS = crr_fld[crr_fld['SOS'] != 0]
    ax.scatter(SOS['Date'], SOS['SOS'], marker='+', s=155, c='g')

    EOS = crr_fld[crr_fld['EOS'] != 0]
    ax.scatter(EOS['Date'], EOS['EOS'], marker='+', s=155, c='r')
    
    # plot regularized version of two years
    ax.scatter(twoYrs_regular_Xvalues_1Year.values, twoYrs_regular_Yvalues_1Year, 
               label="process. data", s=7, c='#E4D00A');

    # plot raw data of two years
    twoYears_raw = twoYears_raw[twoYears_raw.image_year >= SFYr]
    ax.scatter(twoYears_raw['Date'].values, twoYears_raw[idx], label="raw data", s=7, c='dodgerblue');

    plot_title = plant.replace("_", " ")
    ax.set_title(plot_title);
    ax.set(ylabel=idx)
    ax.legend(loc="best");



def SG_1yr_panels_clean_sciPy_My_Peaks_SOS_fineGranularity(twoYears_raw, twoYears_regular, idx, SG_params, 
                                                           SFYr, ax, deltA = 0.4, onset_cut=0.5, offset_cut=0.5):
    """
    This function has additional part to plot SOS and EOS.
    and it is updated version of the function savitzky_1yr_panels_clean_sciPy_and_My_PeakFinder(.)
    """
    ## crr_fld = dataAB.copy() #***
    crr_fld_twoYrs_regular = twoYears_regular.copy() #***
    if (not("human_system_start_time" in list(crr_fld_twoYrs_regular.columns))):
        crr_fld_twoYrs_regular = rc.add_human_start_time(crr_fld_twoYrs_regular)

    eleven_colors = ["gray", "lightcoral", "red", "peru",
                     "darkorange", "gold", "olive", "green",
                     "blue", "violet", "deepskyblue"]

    plant = crr_fld_twoYrs_regular['CropTyp'].unique()[0]
    #
    # Take care of names, replace "/" and "," and " " by "_"
    #
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")

    county = crr_fld_twoYrs_regular['county'].unique()[0]
    ID = crr_fld_twoYrs_regular['ID'].unique()[0]
    data_source = crr_fld_twoYrs_regular['DataSrc'].unique()[0]
    irrig_type = crr_fld_twoYrs_regular['Irrigtn'].unique()[0]
    survey_date = crr_fld_twoYrs_regular['LstSrvD'].unique()[0]

    ## y = crr_fld[idx].copy() #***
    ###
    ### Copy the regularized values that are processed up to SG smoothing.
    ###
    twoYrs_regular_Yvalues = crr_fld_twoYrs_regular[idx].copy()
    twoYrs_regular_Xvalues = crr_fld_twoYrs_regular['Date'].copy()

    #############################################
    ###
    ###             Smoothen
    ###
    #############################################
    # differences are minor, but lets keep using Pythons function
    # my_savitzky_pred = rc.savitzky_golay(y, window_size=Sav_win_size, order=sav_order)
    window_len = SG_params[0]
    poly_order = SG_params[1]

    ## SG_pred = scipy.signal.savgol_filter(y, window_length= window_len, polyorder=poly_order) #***

    SG_pred_twoYears = scipy.signal.savgol_filter(twoYrs_regular_Yvalues, window_length= window_len, polyorder=poly_order)

    # SG might violate the boundaries. clip them:
    SG_pred_twoYears[SG_pred_twoYears > 1 ] = 1
    SG_pred_twoYears[SG_pred_twoYears < -1 ] = -1
    
    crr_fld_twoYrs_regular[idx] = SG_pred_twoYears
    
    #############################################
    #############################################

    # crr_fld will be one year from now on
    crr_fld = crr_fld_twoYrs_regular.copy()
    crr_fld = crr_fld[crr_fld.image_year == SFYr]
    #############################################
    ###
    ###             fine granularity table
    ###
    #############################################
    # create the full calendar to make better estimation of SOS and EOS.
    fine_granular_table = rc.create_calendar_table(SF_year = SFYr)
    fine_granular_table = pd.merge(fine_granular_table, crr_fld, on=['Date', 'SF_year', 'doy'], how='left')

    ###### We need to fill the NAs that are created because they were not created in fine_granular_table
    fine_granular_table["image_year"] = crr_fld["image_year"].unique()[0]
    fine_granular_table["ID"]     = crr_fld["ID"].unique()[0]
    fine_granular_table["Acres"]  = crr_fld["Acres"].unique()[0]
    fine_granular_table["county"] = crr_fld["county"].unique()[0]

    fine_granular_table["CropGrp"] = crr_fld["CropGrp"].unique()[0]
    fine_granular_table["CropTyp"] = crr_fld["CropTyp"].unique()[0]
    fine_granular_table["DataSrc"] = crr_fld["DataSrc"].unique()[0]
    fine_granular_table["ExctAcr"] = crr_fld["ExctAcr"].unique()[0]

    fine_granular_table["IntlSrD"] = crr_fld["IntlSrD"].unique()[0]
    fine_granular_table["Irrigtn"] = crr_fld["Irrigtn"].unique()[0]

    fine_granular_table["LstSrvD"] = crr_fld["LstSrvD"].unique()[0]
    fine_granular_table["Notes"]   = crr_fld["Notes"].unique()[0]
    fine_granular_table["RtCrpTy"] = crr_fld["RtCrpTy"].unique()[0]
    fine_granular_table["Shap_Ar"] = crr_fld["Shap_Ar"].unique()[0]
    fine_granular_table["Shp_Lng"] = crr_fld["Shp_Lng"].unique()[0]
    fine_granular_table["TRS"] = crr_fld["TRS"].unique()[0]

    fine_granular_table = rc.add_human_start_time_by_YearDoY(fine_granular_table)

    # replace NAs with -1.5. Because, that is what the function fill_theGap_linearLine()
    # uses as indicator for missing values
    fine_granular_table.fillna(value={idx:-1.5}, inplace=True)
    
    fine_granular_table = rc.fill_theGap_linearLine(regular_TS = fine_granular_table, 
                                                    V_idx=idx, 
                                                    SF_year=SFYr)
    
    # update SG_pred so that we do not have to update too many other stuff.
    SG_pred = fine_granular_table[idx].values.copy()
    crr_fld = fine_granular_table
    y = fine_granular_table[idx].copy()
    #############################################
    ###
    ###   Form a data table of X and Y values
    ###
    #############################################

    if len(fine_granular_table['image_year'].unique()) == 2:
        X = rc.extract_XValues_of_2Yrs_TS(fine_granular_table, SF_yr = SFYr)
    elif len(fine_granular_table['image_year'].unique()) == 1:
        X = fine_granular_table['doy']

    d = {'DoY': X, 'Date': pd.to_datetime(fine_granular_table.human_system_start_time.values).values}
    date_df = pd.DataFrame(data=d)

    min_val_for_being_peak = 0.5

    #############################################
    ###
    ###             find peaks scipy
    ###
    #############################################
    """
    distance : Required minimal horizontal distance (>= 1) in samples 
               between neighbouring peaks. Smaller peaks are removed first 
               until the condition is fulfilled for all remaining peaks.

    This is gonna prevent detecting false peaks.
    """

    # scipy.signal.argrelextrema(SG_pred, np.greater)
    peaks_indxs, all_properties = scipy.signal.find_peaks(x = SG_pred, 
                                                          height = min_val_for_being_peak, 
                                                          threshold = None, 
                                                          distance = 3, #  
                                                          prominence = 0.2, 
                                                          width = 3, 
                                                          wlen = None, 
                                                          rel_height = 0.5, 
                                                          plateau_size=None)

    scipy_SG_max_DoYs_series = X.iloc[peaks_indxs]
    scipy_SG_max_series = pd.Series(all_properties['peak_heights'])

    # keyy = "SG: [" + str(window_len) + ", " + str(poly_order) + "]"
    # plotting_dic = { keyy : [SG_pred, SG_max_DoYs_series, SG_max_series]}

    #############################################
    ###
    ###             find troughs scipy
    ###
    #############################################

    scipy_miminum_indexes = scipy.signal.argrelextrema(SG_pred, np.less)[0]
    scipy_SG_min_DoYs_series = X.iloc[scipy_miminum_indexes]
    scipy_SG_min_series = pd.Series(SG_pred[scipy_miminum_indexes])

    #############################################
    ###
    ###  find peaks and troughs of MATLAB
    ###
    #############################################
    my_SG_max_min = rc.my_peakdetect(y_axis = SG_pred, x_axis = X, delta=deltA);
    my_SG_max =  my_SG_max_min[0]; my_SG_min =  my_SG_max_min[1];
    my_SG_max = rc.separate_x_and_y(m_list = my_SG_max);
    my_SG_min = rc.separate_x_and_y(m_list = my_SG_min);
    my_SG_max_DoYs_series = pd.Series(my_SG_max[0]);
    my_SG_max_series = pd.Series(my_SG_max[1]);
    my_SG_min_DoYs_series = pd.Series(my_SG_min[0]);
    my_SG_min_series = pd.Series(my_SG_min[1]);

    #############################################
    ###
    ###      Form a dictionary for plotting
    ###
    #############################################

    keyy = "SG: [" + str(window_len) + ", " + str(poly_order) + "]"
    plotting_dic = { keyy : [SG_pred, 
                             scipy_SG_max_DoYs_series, scipy_SG_max_series, # Scipy peak and troughs
                             scipy_SG_min_DoYs_series, scipy_SG_min_series,

                             my_SG_max_DoYs_series, my_SG_max_series, # my peak and troughs
                             my_SG_min_DoYs_series, my_SG_min_series,

                            ]}

    #############################################
    ###
    ###             plot
    ###
    #############################################
    # + ", delta = " + str(deltA)
    plot_title = county + ", " + plant + " (" + ID + ", " + data_source + ", " + irrig_type + ", " + survey_date +")" 
    ax.set_ylim([-1.15, 1.15])
    # sb.set();
    
    # plot regularized version of two years
    ax.scatter(twoYrs_regular_Xvalues.values, twoYrs_regular_Yvalues, label="2YRs process. data", s=7, c='#E4D00A');

    # plot SG smoothed version of two years
    ax.plot(crr_fld_twoYrs_regular['Date'].values, crr_fld_twoYrs_regular[idx], label= keyy, c='k');


    ax.scatter(date_df.Date.values, y.values, label="1-Day Time Series", s=7, c='b');


    for co, ite in enumerate(plotting_dic):
        lbl = ite # + ", Peaks: " + str(len(plotting_dic[ite][2]))
        """
        The following is now plotted above. So, we comment it out here.
        """
        # ax.plot(date_df.Date.values, plotting_dic[ite][0], label = lbl, c = 'k')

        ############################################
        #
        # plot the SciPy outputs
        #
        ############################################
        # plot the SciPy peaks
        Scipy_date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][1])]
        ax.scatter(Scipy_date_df_specific.Date.values, 
                   plotting_dic[ite][2], s=150, marker='*', c = '#00CC99');

    
        # plot My peaks
        my_date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][5])]
        ax.scatter(my_date_df_specific.Date.values, plotting_dic[ite][6], s=100, marker=4, c = "r");

        # plot My Troughs
        My_date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][7])]
        ax.scatter(My_date_df_specific.Date.values, plotting_dic[ite][8], s=100, marker=4, c = 'r');

        # annotate SciPy peaks
        for max_count in np.arange(0, len(Scipy_date_df_specific)):
            style = dict(size=10, color='grey', rotation='vertical')
            ax.text(x = Scipy_date_df_specific.iloc[max_count]['Date'].date(), 
                    y = -0.7, 
                    s = 'DoY=' + str(Scipy_date_df_specific.iloc[max_count]['DoY']), 
                    **style)

        # annotate My troughs
        for min_count in np.arange(0, len(My_date_df_specific)):
            style = dict(size=10, color='grey', rotation='vertical')
            ax.text(x = My_date_df_specific.iloc[min_count]['Date'].date(), 
                    y = -0.7, 
                    s = 'DoY=' + str(My_date_df_specific.iloc[min_count]['DoY']), 
                    **style)

    ###
    ###   plot SOS and EOS
    ###
    # Update the EVI/NDVI values to the smoothed version.
    crr_fld [idx] = SG_pred
    crr_fld = rc.addToDF_SOS_EOS_White(pd_TS = crr_fld, 
                                       VegIdx = idx, 
                                       onset_thresh = onset_cut, 
                                       offset_thresh = offset_cut)

    ##
    ##  Kill bad detected seasons 
    ##
    crr_fld = rc.Null_SOS_EOS_by_DoYDiff(pd_TS = crr_fld, min_season_length=40)

    #
    #  Start of the season
    #
    SOS = crr_fld[crr_fld['SOS'] != 0]
    ax.scatter(SOS['Date'], SOS['SOS'], marker='+', s=155, c='g')

    # annotate SOS
    for ii in np.arange(0, len(SOS)):
        style = dict(size=10, color='g', rotation='vertical')
        ax.text(x = SOS.iloc[ii]['Date'].date(), 
                y = -0.7, 
                s = 'DoY=' + str(SOS.iloc[ii]['doy']), 
                **style)

    #
    #  End of the season
    #

    EOS = crr_fld[crr_fld['EOS'] != 0]
    ax.scatter(EOS['Date'], EOS['EOS'], marker='+', s=155, c='r')

    # annotate EOS
    for ii in np.arange(0, len(EOS)):
        style = dict(size=10, color='r', rotation='vertical')
        ax.text(x = EOS.iloc[ii]['Date'].date(), 
                y = -0.7, 
                s = 'DoY=' + str(EOS.iloc[ii]['doy']), 
                **style)

    # Plot ratios:
    ax.plot(crr_fld['Date'], crr_fld['EVI_ratio'], c='gray', label="EVI Ratio")

    ax.axhline(0 , color = 'r', linewidth=.5)
    ax.axhline(1 , color = 'r', linewidth=.5)
    ax.axhline(-1, color = 'r', linewidth=.5)

    # plot raw data of two years
    ax.scatter(twoYears_raw['Date'].values, twoYears_raw[idx], label="raw data", s=7, c='dodgerblue');

    ax.set_title(plot_title);
    ax.set(ylabel=idx)
    ax.legend(loc="best");

#####################################################################################################################

def SG_1yr_panels_clean_sciPy_My_Peaks_SOS_fineGranularity_1Year(dataAB, idx, SG_params, SFYr, ax, deltA = 0.4, onset_cut=0.5, offset_cut=0.5):
    """
    This function has additional part to plot SOS and EOS.
    and it is updated version of the function savitzky_1yr_panels_clean_sciPy_and_My_PeakFinder(.)
    """
    crr_fld = dataAB.copy()
    if (not("human_system_start_time" in list(crr_fld.columns))):
        crr_fld = rc.add_human_start_time(crr_fld)

    eleven_colors = ["gray", "lightcoral", "red", "peru",
                     "darkorange", "gold", "olive", "green",
                     "blue", "violet", "deepskyblue"]

    plant = crr_fld['CropTyp'].unique()[0]
    # Take care of names, replace "/" and "," and " " by "_"
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")

    county = crr_fld['county'].unique()[0]
    ID = crr_fld['ID'].unique()[0]

    y = crr_fld[idx].copy()
    processed_Idx = crr_fld[idx].copy()
    processed_Dates = crr_fld['Date'].copy()

    #############################################
    ###
    ###             Smoothen
    ###
    #############################################
    # differences are minor, but lets keep using Pythons function
    # my_savitzky_pred = rc.savitzky_golay(y, window_size=Sav_win_size, order=sav_order)
    window_len = SG_params[0]
    poly_order = SG_params[1]

    SG_pred = scipy.signal.savgol_filter(y, window_length= window_len, polyorder=poly_order)

    # SG might violate the boundaries. clip them:
    SG_pred[SG_pred > 1 ] = 1
    SG_pred[SG_pred < -1 ] = -1
    
    crr_fld[idx] = SG_pred
    
    #############################################
    ###
    ###             fine granularity table
    ###
    #############################################
    # create the full calendar to make better estimation of SOS and EOS.
    fine_granular_table = rc.create_calendar_table(SF_year = SFYr)
    fine_granular_table = pd.merge(fine_granular_table, crr_fld, on=['Date', 'SF_year', 'doy'], how='left')

    ###### We need to fill the NAs that are created because they were not created in fine_granular_table
    fine_granular_table["image_year"] = crr_fld["image_year"].unique()[0]
    fine_granular_table["ID"]     = crr_fld["ID"].unique()[0]
    fine_granular_table["Acres"]  = crr_fld["Acres"].unique()[0]
    fine_granular_table["county"] = crr_fld["county"].unique()[0]

    fine_granular_table["CropGrp"] = crr_fld["CropGrp"].unique()[0]
    fine_granular_table["CropTyp"] = crr_fld["CropTyp"].unique()[0]
    fine_granular_table["DataSrc"] = crr_fld["DataSrc"].unique()[0]
    fine_granular_table["ExctAcr"] = crr_fld["ExctAcr"].unique()[0]

    fine_granular_table["IntlSrD"] = crr_fld["IntlSrD"].unique()[0]
    fine_granular_table["Irrigtn"] = crr_fld["Irrigtn"].unique()[0]

    fine_granular_table["LstSrvD"] = crr_fld["LstSrvD"].unique()[0]
    fine_granular_table["Notes"]   = crr_fld["Notes"].unique()[0]
    fine_granular_table["RtCrpTy"] = crr_fld["RtCrpTy"].unique()[0]
    fine_granular_table["Shap_Ar"] = crr_fld["Shap_Ar"].unique()[0]
    fine_granular_table["Shp_Lng"] = crr_fld["Shp_Lng"].unique()[0]
    fine_granular_table["TRS"] = crr_fld["TRS"].unique()[0]

    fine_granular_table = rc.add_human_start_time_by_YearDoY(fine_granular_table)

    # replace NAs with -1.5. Because, that is what the function fill_theGap_linearLine()
    # uses as indicator for missing values
    fine_granular_table.fillna(value={idx:-1.5}, inplace=True)
    
    fine_granular_table = rc.fill_theGap_linearLine(regular_TS = fine_granular_table, 
                                                    V_idx=idx, 
                                                    SF_year=SFYr)
    
    # update SG_pred so that we do not have to update too many other stuff.
    SG_pred = fine_granular_table[idx].values.copy()
    crr_fld = fine_granular_table
    y = fine_granular_table[idx].copy()
    #############################################
    ###
    ###   Form a data table of X and Y values
    ###
    #############################################

    if len(fine_granular_table['image_year'].unique()) == 2:
        X = rc.extract_XValues_of_2Yrs_TS(fine_granular_table, SF_yr = SFYr)
    elif len(fine_granular_table['image_year'].unique()) == 1:
        X = fine_granular_table['doy']

    d = {'DoY': X, 'Date': pd.to_datetime(fine_granular_table.human_system_start_time.values).values}
    date_df = pd.DataFrame(data=d)

    min_val_for_being_peak = 0.5

    #############################################
    ###
    ###             find peaks scipy
    ###
    #############################################
    """
    distance : Required minimal horizontal distance (>= 1) in samples 
               between neighbouring peaks. Smaller peaks are removed first 
               until the condition is fulfilled for all remaining peaks.
    This is gonna prevent detecting false peaks.
    """

    # scipy.signal.argrelextrema(SG_pred, np.greater)
    peaks_indxs, all_properties = scipy.signal.find_peaks(x = SG_pred, 
                                                          height = min_val_for_being_peak, 
                                                          threshold = None, 
                                                          distance = 3, #  
                                                          prominence = 0.2, 
                                                          width = 3, 
                                                          wlen = None, 
                                                          rel_height = 0.5, 
                                                          plateau_size=None)

    scipy_SG_max_DoYs_series = X.iloc[peaks_indxs]
    scipy_SG_max_series = pd.Series(all_properties['peak_heights'])

    # keyy = "SG: [" + str(window_len) + ", " + str(poly_order) + "]"
    # plotting_dic = { keyy : [SG_pred, SG_max_DoYs_series, SG_max_series]}

    #############################################
    ###
    ###             find troughs scipy
    ###
    #############################################

    scipy_miminum_indexes = scipy.signal.argrelextrema(SG_pred, np.less)[0]
    scipy_SG_min_DoYs_series = X.iloc[scipy_miminum_indexes]
    scipy_SG_min_series = pd.Series(SG_pred[scipy_miminum_indexes])

    #############################################
    ###
    ###  find peaks and troughs of MATLAB
    ###
    #############################################
    my_SG_max_min = rc.my_peakdetect(y_axis = SG_pred, x_axis = X, delta=deltA);
    my_SG_max =  my_SG_max_min[0]; my_SG_min =  my_SG_max_min[1];
    my_SG_max = rc.separate_x_and_y(m_list = my_SG_max);
    my_SG_min = rc.separate_x_and_y(m_list = my_SG_min);
    my_SG_max_DoYs_series = pd.Series(my_SG_max[0]);
    my_SG_max_series = pd.Series(my_SG_max[1]);
    my_SG_min_DoYs_series = pd.Series(my_SG_min[0]);
    my_SG_min_series = pd.Series(my_SG_min[1]);

    #############################################
    ###
    ###      Form a dictionary for plotting
    ###
    #############################################

    keyy = "SG: [" + str(window_len) + ", " + str(poly_order) + "]"
    plotting_dic = { keyy : [SG_pred, 
                             scipy_SG_max_DoYs_series, scipy_SG_max_series, # Scipy peak and troughs
                             scipy_SG_min_DoYs_series, scipy_SG_min_series,

                             my_SG_max_DoYs_series, my_SG_max_series, # my peak and troughs
                             my_SG_min_DoYs_series, my_SG_min_series,

                            ]}

    #############################################
    ###
    ###             plot
    ###
    #############################################

    plot_title = county + ", " + plant + " (" + ID + "), delta = " + str(deltA)
    ax.set_ylim([-1.15, 1.15])
    # sb.set();

    ax.scatter(date_df.Date.values, y.values, label="1-Day Time Series", s=7, c='b');
    ax.scatter(processed_Dates.values, processed_Idx, label="processed data", s=7, c='#E4D00A');

    for co, ite in enumerate(plotting_dic):
        lbl = ite # + ", Peaks: " + str(len(plotting_dic[ite][2]))
        ax.plot(date_df.Date.values, plotting_dic[ite][0], label = lbl, c = 'k')

        ############################################
        #
        # plot the SciPy outputs
        #
        ############################################
        # plot the SciPy peaks
        Scipy_date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][1])]
        ax.scatter(Scipy_date_df_specific.Date.values, 
                   plotting_dic[ite][2], s=150, marker='*', c = '#00CC99');

    
        # plot My peaks
        my_date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][5])]
        ax.scatter(my_date_df_specific.Date.values, plotting_dic[ite][6], s=100, marker=4, c = "r");

        # plot My Troughs
        My_date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][7])]
        ax.scatter(My_date_df_specific.Date.values, plotting_dic[ite][8], s=100, marker=4, c = 'r');

        # annotate SciPy peaks
        for max_count in np.arange(0, len(Scipy_date_df_specific)):
            style = dict(size=10, color='grey', rotation='vertical')
            ax.text(x = Scipy_date_df_specific.iloc[max_count]['Date'].date(), 
                    y = -0.7, 
                    s = 'DoY=' + str(Scipy_date_df_specific.iloc[max_count]['DoY']), 
                    **style)

        # annotate My troughs
        for min_count in np.arange(0, len(My_date_df_specific)):
            style = dict(size=10, color='grey', rotation='vertical')
            ax.text(x = My_date_df_specific.iloc[min_count]['Date'].date(), 
                    y = -0.7, 
                    s = 'DoY=' + str(My_date_df_specific.iloc[min_count]['DoY']), 
                    **style)

    ###
    ###   plot SOS and EOS
    ###
    # Update the EVI/NDVI values to the smoothed version.
    crr_fld[idx] = SG_pred
    #
    #  EVI Ratio is added in this step
    #
    crr_fld = rc.addToDF_SOS_EOS_White(pd_TS = crr_fld, 
                                       VegIdx = idx, 
                                       onset_thresh = onset_cut, 
                                       offset_thresh = offset_cut)

    ##
    ##  Kill bad detected seasons 
    ##
    crr_fld = rc.Null_SOS_EOS_by_DoYDiff(pd_TS = crr_fld, min_season_length=40)

    #
    #  Start of the season
    #
    SOS = crr_fld[crr_fld['SOS'] != 0]
    ax.scatter(SOS['Date'], SOS['SOS'], marker='+', s=155, c='g')

    # annotate  EOS
    for ii in np.arange(0, len(SOS)):
        style = dict(size=10, color='green', rotation='vertical')
        ax.text(x = SOS.iloc[ii]['Date'].date(), 
                y = -0.7, 
                s = 'DoY=' + str(SOS.iloc[ii]['doy']), 
                **style)

    #
    #  End of the season
    #

    EOS = crr_fld[crr_fld['EOS'] != 0]
    ax.scatter(EOS['Date'], EOS['EOS'], marker='+', s=155, c='r')

    # annotate EOS
    for ii in np.arange(0, len(EOS)):
        style = dict(size=10, color='red', rotation='vertical')
        ax.text(x = EOS.iloc[ii]['Date'].date(), 
                y = -0.7, 
                s = 'DoY=' + str(EOS.iloc[ii]['doy']), 
                **style)

    # Plot ratios:
    ax.plot(crr_fld['Date'], crr_fld['EVI_ratio'], c='gray', label="EVI Ratio")

    ax.axhline(0 , color = 'r', linewidth=.5)
    ax.axhline(1 , color = 'r', linewidth=.5)
    ax.axhline(-1, color = 'r', linewidth=.5)

    ax.set_title(plot_title);
    ax.set(ylabel=idx)
    ax.legend(loc="best");

#####################################################################################################################

def SG_1yr_panels_clean_sciPy_My_Peaks_SOS(dataAB, idx, SG_params, SFYr, ax, deltA=0.4, onset_cut=0.5, offset_cut=0.5):
    """
    This function has additional part to plot SOS and EOS.
    and it is updated version of the function savitzky_1yr_panels_clean_sciPy_and_My_PeakFinder(.)
    """
    crr_fld = dataAB.copy()
    if (not("human_system_start_time" in list(crr_fld.columns))):
        crr_fld = rc.add_human_start_time(crr_fld)

    eleven_colors = ["gray", "lightcoral", "red", "peru",
                     "darkorange", "gold", "olive", "green",
                     "blue", "violet", "deepskyblue"]

    plant = crr_fld['CropTyp'].unique()[0]
    # Take care of names, replace "/" and "," and " " by "_"
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")

    county = crr_fld['county'].unique()[0]
    ID = crr_fld['ID'].unique()[0]

    y = crr_fld[idx].copy()

    #############################################
    ###
    ###             Smoothen
    ###
    #############################################
    # differences are minor, but lets keep using Pythons function
    # my_savitzky_pred = rc.savitzky_golay(y, window_size=Sav_win_size, order=sav_order)
    window_len = SG_params[0]
    poly_order = SG_params[1]

    SG_pred = scipy.signal.savgol_filter(y, window_length = window_len, polyorder = poly_order)

    # SG might violate the boundaries. clip them:
    SG_pred[SG_pred > 1 ] = 1
    SG_pred[SG_pred < -1 ] = -1

    #############################################
    ###
    ###   Form a data table of X and Y values
    ###
    #############################################

    if len(crr_fld['image_year'].unique()) == 2:
        X = rc.extract_XValues_of_2Yrs_TS(crr_fld, SF_yr = SFYr)
    elif len(crr_fld['image_year'].unique()) == 1:
        X = crr_fld['doy']

    d = {'DoY': X, 'Date': pd.to_datetime(crr_fld.human_system_start_time.values).values}
    date_df = pd.DataFrame(data=d)

    min_val_for_being_peak = 0.5

    #############################################
    ###
    ###             find peaks scipy
    ###
    #############################################
    """
    distance : Required minimal horizontal distance (>= 1) in samples 
               between neighbouring peaks. Smaller peaks are removed first 
               until the condition is fulfilled for all remaining peaks.

    This is gonna prevent detecting false peaks.
    """

    # scipy.signal.argrelextrema(SG_pred, np.greater)
    peaks_indxs, all_properties = scipy.signal.find_peaks(x = SG_pred, 
                                                          height = min_val_for_being_peak, 
                                                          threshold = None, 
                                                          distance = 3, #  
                                                          prominence = 0.3, 
                                                          width = 3, 
                                                          wlen = None, 
                                                          rel_height = 0.5, 
                                                          plateau_size=None)

    scipy_SG_max_DoYs_series = X.iloc[peaks_indxs]
    scipy_SG_max_series = pd.Series(all_properties['peak_heights'])

    # keyy = "SG: [" + str(window_len) + ", " + str(poly_order) + "]"
    # plotting_dic = { keyy : [SG_pred, SG_max_DoYs_series, SG_max_series]}

    #############################################
    ###
    ###             find troughs scipy
    ###
    #############################################

    scipy_miminum_indexes = scipy.signal.argrelextrema(SG_pred, np.less)[0]
    scipy_SG_min_DoYs_series = X.iloc[scipy_miminum_indexes]
    scipy_SG_min_series = pd.Series(SG_pred[scipy_miminum_indexes])

    #############################################
    ###
    ###  find peaks and troughs of MATLAB
    ###
    #############################################
    my_SG_max_min = rc.my_peakdetect(y_axis = SG_pred, x_axis = X, delta=deltA);
    my_SG_max =  my_SG_max_min[0]; my_SG_min =  my_SG_max_min[1];
    my_SG_max = rc.separate_x_and_y(m_list = my_SG_max);
    my_SG_min = rc.separate_x_and_y(m_list = my_SG_min);
    my_SG_max_DoYs_series = pd.Series(my_SG_max[0]);
    my_SG_max_series = pd.Series(my_SG_max[1]);
    my_SG_min_DoYs_series = pd.Series(my_SG_min[0]);
    my_SG_min_series = pd.Series(my_SG_min[1]);

    #############################################
    ###
    ###      Form a dictionary for plotting
    ###
    #############################################

    keyy = "SG: [" + str(window_len) + ", " + str(poly_order) + "]"
    plotting_dic = { keyy : [SG_pred, 
                             scipy_SG_max_DoYs_series, scipy_SG_max_series, # Scipy peak and troughs
                             scipy_SG_min_DoYs_series, scipy_SG_min_series,

                             my_SG_max_DoYs_series, my_SG_max_series, # my peak and troughs
                             my_SG_min_DoYs_series, my_SG_min_series,

                            ]}

    #############################################
    ###
    ###             plot
    ###
    #############################################

    plot_title = county + ", " + plant + " (" + ID + ")" # + ", delta =" + str(deltA)
    ax.set_ylim([-1.15, 1.15])
    # sb.set();

    ax.scatter(date_df.Date.values, y.values, label="processed data", s=10, c='#E4D00A');

    for co, ite in enumerate(plotting_dic):
        lbl = ite # + ", Peaks: " + str(len(plotting_dic[ite][2]))
        ax.plot(date_df.Date.values, plotting_dic[ite][0], label = lbl, c = 'k')

        ############################################
        #
        # plot the SciPy outputs
        #
        ############################################
        # plot the SciPy peaks
        Scipy_date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][1])]
        ax.scatter(Scipy_date_df_specific.Date.values, 
                   plotting_dic[ite][2], s=150, marker='*', c = '#00CC99');

        # plot the SciPy troughs
        # Scipy_date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][3])]
        # ax.scatter(Scipy_date_df_specific.Date.values, plotting_dic[ite][4], s=150, marker='*', c = '#00CC99');

        # annotate SciPy troughs
        # for min_count in np.arange(0, len(Scipy_date_df_specific)):
        #     style = dict(size=10, color='grey', rotation='vertical')
        #     ax.text(x = Scipy_date_df_specific.iloc[min_count]['Date'].date(), 
        #             y = -1, 
        #             s = 'DoY=' + str(Scipy_date_df_specific.iloc[min_count]['DoY']), 
        #             **style)

        # plot My peaks
        my_date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][5])]
        ax.scatter(my_date_df_specific.Date.values, plotting_dic[ite][6], s=100, marker=4, c = "r");

        # plot My Troughs
        My_date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][7])]
        ax.scatter(My_date_df_specific.Date.values, plotting_dic[ite][8], s=100, marker=4, c = 'r');

        # annotate My troughs
        for min_count in np.arange(0, len(My_date_df_specific)):
            style = dict(size=10, color='grey', rotation='vertical')
            ax.text(x = My_date_df_specific.iloc[min_count]['Date'].date(), 
                    y = -0.7, 
                    s = 'DoY=' + str(My_date_df_specific.iloc[min_count]['DoY']), 
                    **style)

    ###
    ###   plot SOS and EOS
    ###
    # Update the EVI/NDVI values to the smoothed version.
    crr_fld [idx] = SG_pred
    crr_fld = rc.addToDF_SOS_EOS_White(pd_TS = crr_fld, 
                                       VegIdx = idx, 
                                       onset_thresh = onset_cut, 
                                       offset_thresh = offset_cut)

    ##
    ##  Kill bad detected seasons 
    ##
    crr_fld = rc.Null_SOS_EOS_by_DoYDiff(pd_TS = crr_fld, min_season_length=40)

    #
    #  Start of the season
    #
    SOS = crr_fld[crr_fld['SOS'] != 0]
    ax.scatter(SOS['Date'], SOS['SOS'], marker='+', s=155, c='g')

    # annotate EOS
    for ii in np.arange(0, len(SOS)):
        style = dict(size=10, color='grey', rotation='vertical')
        ax.text(x = SOS.iloc[ii]['Date'].date(), 
                y = -0.7, 
                s = 'DoY=' + str(SOS.iloc[ii]['doy']), 
                **style)

    #
    #  End of the season
    #

    EOS = crr_fld[crr_fld['EOS'] != 0]
    ax.scatter(EOS['Date'], EOS['EOS'], marker='+', s=155, c='r')

    # annotate EOS
    for ii in np.arange(0, len(EOS)):
        style = dict(size=10, color='grey', rotation='vertical')
        ax.text(x = EOS.iloc[ii]['Date'].date(), 
                y = -0.7, 
                s = 'DoY=' + str(EOS.iloc[ii]['doy']), 
                **style)

    # Plot ratios:
    # ax.plot(crr_fld['Date'], crr_fld['EVI_ratio'], c='r', label="EVI Ratio")

    ax.axhline(0 , color = 'r', linewidth=.5)
    ax.axhline(1 , color = 'r', linewidth=.5)
    ax.axhline(-1, color = 'r', linewidth=.5)

    ax.set_title(plot_title);
    ax.set(ylabel=idx)
    ax.legend(loc="best");

#####################################################################################################################

def savitzky_1yr_panels_clean_sciPy_and_My_PeakFinder(A_Data, idx, SG_params, SFYr, ax, deltA = 0.4, min_val_for_being_peak = 0.5):
    
    crr_fld = A_Data.copy()
    if (not("human_system_start_time" in list(crr_fld.columns))):
        crr_fld = rc.add_human_start_time(crr_fld)

    eleven_colors = ["gray", "lightcoral", "red", "peru",
                     "darkorange", "gold", "olive", "green",
                     "blue", "violet", "deepskyblue"]

    plant = crr_fld['CropTyp'].unique()[0]
    # Take care of names, replace "/" and "," and " " by "_"
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")

    county = crr_fld['county'].unique()[0]
    ID = crr_fld['ID'].unique()[0]

    y = crr_fld[idx]

    #############################################
    ###
    ###             Smoothen
    ###
    #############################################
    # differences are minor, but lets keep using Pythons function
    # my_savitzky_pred = rc.savitzky_golay(y, window_size=Sav_win_size, order=sav_order)
    window_len = SG_params[0]
    poly_order = SG_params[1]

    SG_pred = scipy.signal.savgol_filter(y, window_length = window_len, polyorder = poly_order)

    # we do not need the following since we update EVI by replacing it with smoothed version.
    # smooth_col_name = "smooth_" + idx
    # crr_fld[smooth_col_name] = SG_pred

    #############################################
    ###
    ###   Form a data table of X and Y values
    ###
    #############################################

    if len(crr_fld['image_year'].unique()) == 2:
        X = rc.extract_XValues_of_2Yrs_TS(crr_fld, SF_yr = SFYr)
    elif len(crr_fld['image_year'].unique()) == 1:
        X = crr_fld['doy']

    d = {'DoY': X, 'Date': pd.to_datetime(crr_fld.human_system_start_time.values).values}
    date_df = pd.DataFrame(data=d)

    #############################################
    ###
    ###             find peaks scipy
    ###
    #############################################
    """
    distance : Required minimal horizontal distance (>= 1) in samples 
               between neighbouring peaks. Smaller peaks are removed first 
               until the condition is fulfilled for all remaining peaks.

    This is gonna prevent detecting false peaks.
    """

    # scipy.signal.argrelextrema(SG_pred, np.greater)
    peaks_indxs, all_properties = scipy.signal.find_peaks(x = SG_pred, 
                                                          height = min_val_for_being_peak, 
                                                          threshold = None, 
                                                          distance = None, #  
                                                          prominence = None, 
                                                          width = None, 
                                                          wlen = None, 
                                                          rel_height = 0.5, 
                                                          plateau_size=None)

    scipy_SG_max_DoYs_series = X.iloc[peaks_indxs]
    scipy_SG_max_series = pd.Series(all_properties['peak_heights'])

    # keyy = "SG: [" + str(window_len) + ", " + str(poly_order) + "]"
    # plotting_dic = { keyy : [SG_pred, SG_max_DoYs_series, SG_max_series]}

    #############################################
    ###
    ###             find troughs scipy
    ###
    #############################################

    scipy_miminum_indexes = scipy.signal.argrelextrema(SG_pred, np.less)[0]
    scipy_SG_min_DoYs_series = X.iloc[scipy_miminum_indexes]
    scipy_SG_min_series = pd.Series(SG_pred[scipy_miminum_indexes])

    #############################################
    ###
    ###  find peaks and troughs of MATLAB
    ###
    #############################################
    my_SG_max_min = rc.my_peakdetect(y_axis = SG_pred, x_axis = X, delta=deltA);
    my_SG_max =  my_SG_max_min[0]; my_SG_min =  my_SG_max_min[1];
    my_SG_max = rc.separate_x_and_y(m_list = my_SG_max);
    my_SG_min = rc.separate_x_and_y(m_list = my_SG_min);
    my_SG_max_DoYs_series = pd.Series(my_SG_max[0]);
    my_SG_max_series = pd.Series(my_SG_max[1]);
    my_SG_min_DoYs_series = pd.Series(my_SG_min[0]);
    my_SG_min_series = pd.Series(my_SG_min[1]);

    #############################################
    ###
    ###      Form a dictionary for plotting
    ###
    #############################################

    keyy = "SG: [" + str(window_len) + ", " + str(poly_order) + "]"
    plotting_dic = { keyy : [SG_pred, 

                             scipy_SG_max_DoYs_series, scipy_SG_max_series, # Scipy peak and troughs
                             scipy_SG_min_DoYs_series, scipy_SG_min_series,

                             my_SG_max_DoYs_series, my_SG_max_series, # my peak and troughs
                             my_SG_min_DoYs_series, my_SG_min_series,

                            ]}

    #############################################
    ###
    ###             plot
    ###
    #############################################
    
    plot_title = county + ", " + plant + " (" + ID + "), delta = " + str(deltA)
    ax.set_ylim([-1.15, 1.15])
    # sb.set();

    ax.scatter(date_df.Date.values, y.values, label="data", s=20, c='#E4D00A');

    for co, ite in enumerate(plotting_dic):
        lbl = ite + ", Peaks: " + str(len(plotting_dic[ite][2]))
        ax.plot(date_df.Date.values, plotting_dic[ite][0], label = lbl, c = 'k')

        ############################################
        #
        # plot the SciPy outputs
        #
        ############################################
        # plot the SciPy peaks
        Scipy_date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][1])]
        ax.scatter(Scipy_date_df_specific.Date.values, plotting_dic[ite][2], s=150, marker='*', c = '#00CC99');

        # annotate SciPy peaks
        # for max_count in np.arange(0, len(Scipy_date_df_specific)):
        #     style = dict(size=10, color='grey', rotation='vertical')
        #     ax.text(x = Scipy_date_df_specific.iloc[max_count]['Date'].date(), 
        #             y = -1, 
        #             s = 'DoY=' + str(Scipy_date_df_specific.iloc[max_count]['DoY']), 
        #             **style)


        # plot the SciPy troughs
        Scipy_date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][3])]
        ax.scatter(Scipy_date_df_specific.Date.values, plotting_dic[ite][4], s=150, marker='*', c = '#00CC99');

        # annotate SciPy troughs
        for min_count in np.arange(0, len(Scipy_date_df_specific)):
            style = dict(size=10, color='grey', rotation='vertical')
            ax.text(x = Scipy_date_df_specific.iloc[min_count]['Date'].date(), 
                    y = -1, 
                    s = 'DoY=' + str(Scipy_date_df_specific.iloc[min_count]['DoY']), 
                    **style)

        # plot My peaks
        my_date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][5])]
        ax.scatter(my_date_df_specific.Date.values, plotting_dic[ite][6], s=100, marker=4, c = "r");


        # plot My Troughs
        My_date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][7])]
        ax.scatter(My_date_df_specific.Date.values, plotting_dic[ite][8], s=100, marker=4, c = 'r');

        # annotate My troughs
        for min_count in np.arange(0, len(My_date_df_specific)):
            style = dict(size=10, color='grey', rotation='vertical')
            ax.text(x = My_date_df_specific.iloc[min_count]['Date'].date(), 
                    y = -1, 
                    s = 'DoY=' + str(My_date_df_specific.iloc[min_count]['DoY']), 
                    **style)

    ax.set_title(plot_title);
    ax.set(ylabel=idx)
    ax.legend(loc="best");

#####################################################################################################################

def savitzky_1yr_panels_clean_myPeak(crr_fld, idx, SG_params, SFYr, ax, deltA = 0.4):
    """
    Aug. 3
    _clean_ means one smoothing method per subplot
    In this plots we want to have figures that have 4 subplots in them,
    each subplot corresponds to 1 set of parameters of SG. Delta will be set
    to a given constant.
    """

    if (not("human_system_start_time" in list(crr_fld.columns))):
        crr_fld = rc.add_human_start_time(crr_fld)

    eleven_colors = ["gray", "lightcoral", "red", "peru",
                     "darkorange", "gold", "olive", "green",
                     "blue", "violet", "deepskyblue"]

    plant = crr_fld['CropTyp'].unique()[0]
    # Take care of names, replace "/" and "," and " " by "_"
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")
    
    county = crr_fld['county'].unique()[0]
    ID = crr_fld['ID'].unique()[0]

    y = crr_fld[idx]

    #############################################
    ###
    ###             Smoothen
    ###
    #############################################
    # differences are minor, but lets keep using Pythons function
    # my_savitzky_pred = rc.savitzky_golay(y, window_size=Sav_win_size, order=sav_order)
    window_len = SG_params[0]
    poly_order = SG_params[1]

    SG_pred = scipy.signal.savgol_filter(y, window_length = window_len, polyorder = poly_order)
    
    #############################################
    ###
    ###             find peaks
    ###
    #############################################
    if len(crr_fld['image_year'].unique()) == 2:
        X = rc.extract_XValues_of_2Yrs_TS(crr_fld, SF_yr = SFYr)
    elif len(crr_fld['image_year'].unique()) == 1:
        X = crr_fld['doy']
    

    d = {'DoY': X, 'Date': pd.to_datetime(crr_fld.human_system_start_time.values).values}
    date_df = pd.DataFrame(data=d)
    
    SG_max_min = rc.my_peakdetect(y_axis = SG_pred, x_axis = X, delta=deltA);
    SG_max =  SG_max_min[0]; SG_min =  SG_max_min[1];
    SG_max = rc.separate_x_and_y(m_list = SG_max);
    SG_min = rc.separate_x_and_y(m_list = SG_min);
    SG_max_DoYs_series = pd.Series(SG_max[0]);
    SG_max_series = pd.Series(SG_max[1]);
    SG_min_DoYs_series = pd.Series(SG_min[0]);
    SG_min_series = pd.Series(SG_min[1]);

    ########################################################################################################

    keyy = "SG: [" + str(window_len) + ", " + str(poly_order) + "]"
    plotting_dic = { keyy : [SG_pred, SG_max_DoYs_series, SG_max_series, SG_min_DoYs_series, SG_min_series]}

    #############################################
    ###
    ###             plot
    ###
    #############################################       
    plot_title = county + ", " + plant + " (" + ID + "), delta = " + str(deltA)
    ax.set_ylim([-1.15, 1.15])
    # sb.set();
    
    ax.scatter(date_df.Date.values, y.values, label="data", s = 30);

    for co, ite in enumerate(plotting_dic):
        lbl = ite + ", Peaks: " + str(len(plotting_dic[ite][2]))
        # ax.plot(X, plotting_dic[ite][0], label = lbl, c = eleven_colors[co])
        
        ax.plot(date_df.Date.values, plotting_dic[ite][0], label = lbl, c = 'k')

        # plot the peaks
        date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][1])]
        ax.scatter(date_df_specific.Date.values, plotting_dic[ite][2], s=150, marker=4, c = 'r');
        #
        # plot the troughs
        #
        date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][3])]
        ax.scatter(date_df_specific.Date.values, plotting_dic[ite][4], s=150, marker=4, c = 'r');

        ################################################
        #
        # annotate troughs
        #
        ################################################

        # Setting up the parameters 
        for min_count in np.arange(0, len(date_df_specific)):
        # for currIDX in date_df_specific.index:
            # xdata = date_df_specific.loc[currIDX, 'DoY']
            # ydata = date_df_specific.loc[currIDX, 'Date']
            
            style = dict(size=10, color='grey', rotation='vertical')
            ax.text(x = date_df_specific.iloc[min_count]['Date'].date(), 
                    y = -1, 
                    s = 'DoY=' + str(date_df_specific.iloc[min_count]['DoY']), 
                    **style)

    ax.set_title(plot_title);
    ax.set(ylabel=idx)
    ax.legend(loc="best");

#####################################################################################################################

def savitzky_2yrs_panel(crr_fld, idx, deltA, SFYr, ax):

    if (not("human_system_start_time" in list(crr_fld.columns))):
        crr_fld = rc.add_human_start_time(crr_fld)

    eleven_colors = ["gray", "lightcoral", "red", "peru",
                     "darkorange", "gold", "olive", "green",
                     "blue", "violet", "deepskyblue"]

    plant = crr_fld['CropTyp'].unique()[0]
    # Take care of names, replace "/" and "," and " " by "_"
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")
    
    county = crr_fld['county'].unique()[0]
    ID = crr_fld['ID'].unique()[0]

    y = crr_fld[idx]

    #############################################
    ###
    ###             Smoothen
    ###
    #############################################
    # differences are minor, but lets keep using Pythons function
    # my_savitzky_pred = rc.savitzky_golay(y, window_size=Sav_win_size, order=sav_order)

    SG_pred_31 = scipy.signal.savgol_filter(y, window_length = 3, polyorder = 1)
    SG_pred_32 = scipy.signal.savgol_filter(y, window_length = 3, polyorder = 2)
    
    SG_pred_51 = scipy.signal.savgol_filter(y, window_length = 5, polyorder = 1)
    SG_pred_52 = scipy.signal.savgol_filter(y, window_length = 5, polyorder = 2)
    SG_pred_53 = scipy.signal.savgol_filter(y, window_length = 5, polyorder = 3)
    
    SG_pred_71 = scipy.signal.savgol_filter(y, window_length = 7, polyorder = 1)
    SG_pred_72 = scipy.signal.savgol_filter(y, window_length = 7, polyorder = 2)
    SG_pred_73 = scipy.signal.savgol_filter(y, window_length = 7, polyorder = 3)

    SG_pred_91 = scipy.signal.savgol_filter(y, window_length = 9, polyorder = 1)
    SG_pred_92 = scipy.signal.savgol_filter(y, window_length = 9, polyorder = 2)
    SG_pred_93 = scipy.signal.savgol_filter(y, window_length = 9, polyorder = 3)
    #############################################
    ###
    ###             find peaks
    ###
    #############################################
    # X = crr_fld['doy']
    X = rc.extract_XValues_of_2Yrs_TS(crr_fld, SF_yr = SFYr)

    d = {'DoY': X, 'Date': pd.to_datetime(crr_fld.human_system_start_time.values).values}
    date_df = pd.DataFrame(data=d)
    """
    Jul 1.
    This function is written since Kirti said
    we do not need to have parts of the next year. i.e. 
    if we are looking at what is going on in a field in 2017,
    we only need data since Aug. 2016 till the end of 2017.
    We do not need anything in 2018.
    """

    SG_max_min_31 = rc.my_peakdetect(y_axis=SG_pred_31, x_axis=X, delta=deltA);
    SG_max_31 =  SG_max_min_31[0]; SG_min_31 =  SG_max_min_31[1];
    SG_max_31 = rc.separate_x_and_y(m_list = SG_max_31);
    SG_min_31 = rc.separate_x_and_y(m_list = SG_min_31);
    SG_max_DoYs_series_31 = pd.Series(SG_max_31[0]);
    SG_max_series_31 = pd.Series(SG_max_31[1]);
    SG_min_DoYs_series_31 = pd.Series(SG_min_31[0]);
    SG_min_series_31 = pd.Series(SG_min_31[1]);


    SG_max_min_32 = rc.my_peakdetect(y_axis=SG_pred_32, x_axis=X, delta=deltA);
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

    SG_max_min_51 = rc.my_peakdetect(y_axis=SG_pred_51, x_axis=X, delta=deltA);
    SG_max_51 =  SG_max_min_51[0]; SG_min_51 =  SG_max_min_51[1];
    SG_max_51 = rc.separate_x_and_y(m_list = SG_max_51);
    SG_min_51 = rc.separate_x_and_y(m_list = SG_min_51);
    SG_max_DoYs_series_51 = pd.Series(SG_max_51[0]);
    SG_max_series_51 = pd.Series(SG_max_51[1]);
    SG_min_DoYs_series_51 = pd.Series(SG_min_51[0]);
    SG_min_series_51 = pd.Series(SG_min_51[1]);

    SG_max_min_52 = rc.my_peakdetect(y_axis=SG_pred_52, x_axis=X, delta=deltA);
    SG_max_52 =  SG_max_min_52[0]; SG_min_52 =  SG_max_min_52[1];
    SG_max_52 = rc.separate_x_and_y(m_list = SG_max_52);
    SG_min_52 = rc.separate_x_and_y(m_list = SG_min_52);
    SG_max_DoYs_series_52 = pd.Series(SG_max_52[0]);
    SG_max_series_52 = pd.Series(SG_max_52[1]);
    SG_min_DoYs_series_52 = pd.Series(SG_min_52[0]);
    SG_min_series_52 = pd.Series(SG_min_52[1]);

    SG_max_min_53 = rc.my_peakdetect(y_axis=SG_pred_53, x_axis=X, delta=deltA);
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

    SG_max_min_71 = rc.my_peakdetect(y_axis=SG_pred_71, x_axis=X, delta=deltA);
    SG_max_71 =  SG_max_min_71[0]; SG_min_71 =  SG_max_min_71[1];
    SG_max_71 = rc.separate_x_and_y(m_list = SG_max_71);
    SG_min_71 = rc.separate_x_and_y(m_list = SG_min_71);
    SG_max_DoYs_series_71 = pd.Series(SG_max_71[0]);
    SG_max_series_71 = pd.Series(SG_max_71[1]);
    SG_min_DoYs_series_71 = pd.Series(SG_min_71[0]);
    SG_min_series_71 = pd.Series(SG_min_71[1]);

    SG_max_min_72 = rc.my_peakdetect(y_axis=SG_pred_72, x_axis=X, delta=deltA);
    SG_max_72 =  SG_max_min_72[0]; SG_min_72 =  SG_max_min_72[1];
    SG_max_72 = rc.separate_x_and_y(m_list = SG_max_72);
    SG_min_72 = rc.separate_x_and_y(m_list = SG_min_72);
    SG_max_DoYs_series_72 = pd.Series(SG_max_72[0]);
    SG_max_series_72 = pd.Series(SG_max_72[1]);
    SG_min_DoYs_series_72 = pd.Series(SG_min_72[0]);
    SG_min_series_72 = pd.Series(SG_min_72[1]);

    SG_max_min_73 = rc.my_peakdetect(y_axis=SG_pred_73, x_axis=X, delta=deltA);
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

    SG_max_min_91 = rc.my_peakdetect(y_axis=SG_pred_91, x_axis=X, delta=deltA);
    SG_max_91 =  SG_max_min_91[0]; SG_min_91 =  SG_max_min_91[1];
    SG_max_91 = rc.separate_x_and_y(m_list = SG_max_91);
    SG_min_91 = rc.separate_x_and_y(m_list = SG_min_91);
    SG_max_DoYs_series_91 = pd.Series(SG_max_91[0]);
    SG_max_series_91 = pd.Series(SG_max_91[1]);
    SG_min_DoYs_series_91 = pd.Series(SG_min_91[0]);
    SG_min_series_91 = pd.Series(SG_min_91[1]);

    SG_max_min_92 = rc.my_peakdetect(y_axis=SG_pred_92, x_axis=X, delta=deltA);
    SG_max_92 =  SG_max_min_92[0]; SG_min_92 =  SG_max_min_92[1];
    SG_max_92 = rc.separate_x_and_y(m_list = SG_max_92);
    SG_min_92 = rc.separate_x_and_y(m_list = SG_min_92);
    SG_max_DoYs_series_92 = pd.Series(SG_max_92[0]);
    SG_max_series_92 = pd.Series(SG_max_92[1]);
    SG_min_DoYs_series_92 = pd.Series(SG_min_92[0]);
    SG_min_series_92 = pd.Series(SG_min_92[1]);

    SG_max_min_93 = rc.my_peakdetect(y_axis=SG_pred_93, x_axis=X, delta=deltA);
    SG_max_93 =  SG_max_min_93[0]; SG_min_93 =  SG_max_min_93[1];
    SG_max_93 = rc.separate_x_and_y(m_list = SG_max_93);
    SG_min_93 = rc.separate_x_and_y(m_list = SG_min_93);
    SG_max_DoYs_series_93 = pd.Series(SG_max_93[0]);
    SG_max_series_93 = pd.Series(SG_max_93[1]);
    SG_min_DoYs_series_93 = pd.Series(SG_min_93[0]);
    SG_min_series_93 = pd.Series(SG_min_93[1]);

    ########################################################################################################
    ########################################################################################################

    plotting_dic = { "SG 31" : [SG_pred_31, SG_max_DoYs_series_31, SG_max_series_31],
                     "SG 32" : [SG_pred_32, SG_max_DoYs_series_32, SG_max_series_32],

                     "SG 51" : [SG_pred_51, SG_max_DoYs_series_51, SG_max_series_51],
                     "SG 52" : [SG_pred_52, SG_max_DoYs_series_52, SG_max_series_52],
                     "SG 53" : [SG_pred_53, SG_max_DoYs_series_53, SG_max_series_53],

                     "SG 71" : [SG_pred_71, SG_max_DoYs_series_71, SG_max_series_71],
                     "SG 72" : [SG_pred_72, SG_max_DoYs_series_72, SG_max_series_72],
                     "SG 73" : [SG_pred_73, SG_max_DoYs_series_73, SG_max_series_73],

                     "SG 91" : [SG_pred_91, SG_max_DoYs_series_91, SG_max_series_91],
                     "SG 92" : [SG_pred_92, SG_max_DoYs_series_92, SG_max_series_92],
                     "SG 93" : [SG_pred_93, SG_max_DoYs_series_93, SG_max_series_93]
                     }

    #############################################
    ###
    ###             plot
    ###
    #############################################
    #
    #   We have to change this part to make a perfect plot
    #        
    plot_title = county + ", " + plant + " (" + ID + "), delta = " + str(deltA)
    ax.set_ylim([-1.2, 1.2])
    # sb.set();
    
    # Aeolus throws error. TypeError: invalid type promotion
    # ax.scatter(date_df.Date, y, label="Raw data", s = 60); 
    ax.scatter(date_df.Date.values, y.values, label="Raw data", s = 60);

    for co, ite in enumerate(plotting_dic):
        lbl = ite + ", Peaks: " + str(len(plotting_dic[ite][2]))
        # ax.plot(X, plotting_dic[ite][0], label = lbl, c = eleven_colors[co])
        
        #  Aeolus throws error. TypeError: invalid type promotion
        # ax.plot(date_df.Date, plotting_dic[ite][0], label = lbl, c = eleven_colors[co])
        ax.plot(date_df.Date.values, plotting_dic[ite][0], label = lbl, c = eleven_colors[co])

        date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][1])]
        
        #  Aeolus throws error. TypeError: invalid type promotion
                 # plotting_dic[ite][1]
        # ax.scatter(date_df_specific.Date, plotting_dic[ite][2], s=100, marker='*', c = eleven_colors[co]);
        ax.scatter(date_df_specific.Date.values, plotting_dic[ite][2], s=100, marker='*', c = eleven_colors[co]);

    ax.set_title(plot_title);
    ax.set(ylabel=idx) # xlabel='Time', 
    ax.legend(loc="best");

#####################################################################################################################

# def subplots_savitzky_2_yrs(crr_fld, idx, deltA, SFYr):
#     #
#     #  This function is fine. But, we repace it with savitzky_2_yrs_panel()
#     #

#     if (not("human_system_start_time" in list(crr_fld.columns))):
#         crr_fld = rc.add_human_start_time(crr_fld)

#     eleven_colors = ["gray", "lightcoral", "red", "peru",
#                      "darkorange", "gold", "olive", "green",
#                      "blue", "violet", "deepskyblue"]

#     plant = crr_fld['CropTyp'].unique()[0]
#     # Take care of names, replace "/" and "," and " " by "_"
#     plant = plant.replace("/", "_")
#     plant = plant.replace(",", "_")
#     plant = plant.replace(" ", "_")
#     plant = plant.replace("__", "_")
    
#     county = crr_fld['county'].unique()[0]
#     ID = crr_fld['ID'].unique()[0]

#     y = crr_fld[idx]

#     #############################################
#     ###
#     ###             Smoothen
#     ###
#     #############################################
#     # differences are minor, but lets keep using Pythons function
#     # my_savitzky_pred = rc.savitzky_golay(y, window_size=Sav_win_size, order=sav_order)

#     SG_pred_31 = scipy.signal.savgol_filter(y, window_length= 3, polyorder=1)
#     SG_pred_32 = scipy.signal.savgol_filter(y, window_length= 3, polyorder=2)
    
#     SG_pred_51 = scipy.signal.savgol_filter(y, window_length= 5, polyorder=1)
#     SG_pred_52 = scipy.signal.savgol_filter(y, window_length= 5, polyorder=2)
#     SG_pred_53 = scipy.signal.savgol_filter(y, window_length= 5, polyorder=3)
    
#     SG_pred_71 = scipy.signal.savgol_filter(y, window_length= 7, polyorder=1)
#     SG_pred_72 = scipy.signal.savgol_filter(y, window_length= 7, polyorder=2)
#     SG_pred_73 = scipy.signal.savgol_filter(y, window_length= 7, polyorder=3)

#     SG_pred_91 = scipy.signal.savgol_filter(y, window_length= 9, polyorder=1)
#     SG_pred_92 = scipy.signal.savgol_filter(y, window_length= 9, polyorder=2)
#     SG_pred_93 = scipy.signal.savgol_filter(y, window_length= 9, polyorder=3)

   
#     # preds_df = pd.DataFrame(data = {'SG 31':SG_pred_31, 'SG 32':SG_pred_32, 
#     #                                 'SG 51':SG_pred_51, 'SG 52':SG_pred_52, 'SG 53':SG_pred_53, 
#     #                                 'SG 71':SG_pred_71, 'SG 72':SG_pred_72, 'SG 73':SG_pred_73, 
#     #                                 'SG 91':SG_pred_91, 'SG 92':SG_pred_92, 'SG 93':SG_pred_93}, 
#     #                         index = X)
#     # df2.set_index(keys = ['a'], drop=True,  inplace=True)
#     #############################################
#     ###
#     ###             find peaks
#     ###
#     #############################################
#     # X = crr_fld['doy']
#     X = rc.extract_XValues_of_2Yrs_TS(crr_fld, SF_yr = SFYr)

#     d = {'DoY': X, 'Date': pd.to_datetime(crr_fld.human_system_start_time.values).values}
#     date_df = pd.DataFrame(data=d)
#     """
#     Jul 1.
#     This function is written since Kirti said
#     we do not need to have parts of the next year. i.e. 
#     if we are looking at what is going on in a field in 2017,
#     we only need data since Aug. 2016 till the end of 2017.
#     We do not need anything in 2018.
#     """

#     SG_max_min_31 = rc.my_peakdetect(y_axis=SG_pred_31, x_axis=X, delta=deltA);
#     SG_max_31 =  SG_max_min_31[0]; SG_min_31 =  SG_max_min_31[1];
#     SG_max_31 = rc.separate_x_and_y(m_list = SG_max_31);
#     SG_min_31 = rc.separate_x_and_y(m_list = SG_min_31);
#     SG_max_DoYs_series_31 = pd.Series(SG_max_31[0]);
#     SG_max_series_31 = pd.Series(SG_max_31[1]);
#     SG_min_DoYs_series_31 = pd.Series(SG_min_31[0]);
#     SG_min_series_31 = pd.Series(SG_min_31[1]);


#     SG_max_min_32 = rc.my_peakdetect(y_axis=SG_pred_32, x_axis=X, delta=deltA);
#     SG_max_32 =  SG_max_min_32[0]; SG_min_32 =  SG_max_min_32[1];
#     SG_max_32 = rc.separate_x_and_y(m_list = SG_max_32);
#     SG_min_32 = rc.separate_x_and_y(m_list = SG_min_32);
#     SG_max_DoYs_series_32 = pd.Series(SG_max_32[0]);
#     SG_max_series_32 = pd.Series(SG_max_32[1]);
#     SG_min_DoYs_series_32 = pd.Series(SG_min_32[0]);
#     SG_min_series_32 = pd.Series(SG_min_32[1]);

#     ############
#     ############ window 5
#     ############

#     SG_max_min_51 = rc.my_peakdetect(y_axis=SG_pred_51, x_axis=X, delta=deltA);
#     SG_max_51 =  SG_max_min_51[0]; SG_min_51 =  SG_max_min_51[1];
#     SG_max_51 = rc.separate_x_and_y(m_list = SG_max_51);
#     SG_min_51 = rc.separate_x_and_y(m_list = SG_min_51);
#     SG_max_DoYs_series_51 = pd.Series(SG_max_51[0]);
#     SG_max_series_51 = pd.Series(SG_max_51[1]);
#     SG_min_DoYs_series_51 = pd.Series(SG_min_51[0]);
#     SG_min_series_51 = pd.Series(SG_min_51[1]);

#     SG_max_min_52 = rc.my_peakdetect(y_axis=SG_pred_52, x_axis=X, delta=deltA);
#     SG_max_52 =  SG_max_min_52[0]; SG_min_52 =  SG_max_min_52[1];
#     SG_max_52 = rc.separate_x_and_y(m_list = SG_max_52);
#     SG_min_52 = rc.separate_x_and_y(m_list = SG_min_52);
#     SG_max_DoYs_series_52 = pd.Series(SG_max_52[0]);
#     SG_max_series_52 = pd.Series(SG_max_52[1]);
#     SG_min_DoYs_series_52 = pd.Series(SG_min_52[0]);
#     SG_min_series_52 = pd.Series(SG_min_52[1]);

#     SG_max_min_53 = rc.my_peakdetect(y_axis=SG_pred_53, x_axis=X, delta=deltA);
#     SG_max_53 =  SG_max_min_53[0]; SG_min_53 =  SG_max_min_53[1];
#     SG_max_53 = rc.separate_x_and_y(m_list = SG_max_53);
#     SG_min_53 = rc.separate_x_and_y(m_list = SG_min_53);
#     SG_max_DoYs_series_53 = pd.Series(SG_max_53[0]);
#     SG_max_series_53 = pd.Series(SG_max_53[1]);
#     SG_min_DoYs_series_53 = pd.Series(SG_min_53[0]);
#     SG_min_series_53 = pd.Series(SG_min_53[1]);

#     ############
#     ############ window 7
#     ############

#     SG_max_min_71 = rc.my_peakdetect(y_axis=SG_pred_71, x_axis=X, delta=deltA);
#     SG_max_71 =  SG_max_min_71[0]; SG_min_71 =  SG_max_min_71[1];
#     SG_max_71 = rc.separate_x_and_y(m_list = SG_max_71);
#     SG_min_71 = rc.separate_x_and_y(m_list = SG_min_71);
#     SG_max_DoYs_series_71 = pd.Series(SG_max_71[0]);
#     SG_max_series_71 = pd.Series(SG_max_71[1]);
#     SG_min_DoYs_series_71 = pd.Series(SG_min_71[0]);
#     SG_min_series_71 = pd.Series(SG_min_71[1]);

#     SG_max_min_72 = rc.my_peakdetect(y_axis=SG_pred_72, x_axis=X, delta=deltA);
#     SG_max_72 =  SG_max_min_72[0]; SG_min_72 =  SG_max_min_72[1];
#     SG_max_72 = rc.separate_x_and_y(m_list = SG_max_72);
#     SG_min_72 = rc.separate_x_and_y(m_list = SG_min_72);
#     SG_max_DoYs_series_72 = pd.Series(SG_max_72[0]);
#     SG_max_series_72 = pd.Series(SG_max_72[1]);
#     SG_min_DoYs_series_72 = pd.Series(SG_min_72[0]);
#     SG_min_series_72 = pd.Series(SG_min_72[1]);

#     SG_max_min_73 = rc.my_peakdetect(y_axis=SG_pred_73, x_axis=X, delta=deltA);
#     SG_max_73 =  SG_max_min_73[0]; SG_min_73 =  SG_max_min_73[1];
#     SG_max_73 = rc.separate_x_and_y(m_list = SG_max_73);
#     SG_min_73 = rc.separate_x_and_y(m_list = SG_min_73);
#     SG_max_DoYs_series_73 = pd.Series(SG_max_73[0]);
#     SG_max_series_73 = pd.Series(SG_max_73[1]);
#     SG_min_DoYs_series_73 = pd.Series(SG_min_73[0]);
#     SG_min_series_73 = pd.Series(SG_min_73[1]);

#     ############
#     ############ window 9
#     ############

#     SG_max_min_91 = rc.my_peakdetect(y_axis=SG_pred_91, x_axis=X, delta=deltA);
#     SG_max_91 =  SG_max_min_91[0]; SG_min_91 =  SG_max_min_91[1];
#     SG_max_91 = rc.separate_x_and_y(m_list = SG_max_91);
#     SG_min_91 = rc.separate_x_and_y(m_list = SG_min_91);
#     SG_max_DoYs_series_91 = pd.Series(SG_max_91[0]);
#     SG_max_series_91 = pd.Series(SG_max_91[1]);
#     SG_min_DoYs_series_91 = pd.Series(SG_min_91[0]);
#     SG_min_series_91 = pd.Series(SG_min_91[1]);

#     SG_max_min_92 = rc.my_peakdetect(y_axis=SG_pred_92, x_axis=X, delta=deltA);
#     SG_max_92 =  SG_max_min_92[0]; SG_min_92 =  SG_max_min_92[1];
#     SG_max_92 = rc.separate_x_and_y(m_list = SG_max_92);
#     SG_min_92 = rc.separate_x_and_y(m_list = SG_min_92);
#     SG_max_DoYs_series_92 = pd.Series(SG_max_92[0]);
#     SG_max_series_92 = pd.Series(SG_max_92[1]);
#     SG_min_DoYs_series_92 = pd.Series(SG_min_92[0]);
#     SG_min_series_92 = pd.Series(SG_min_92[1]);

#     SG_max_min_93 = rc.my_peakdetect(y_axis=SG_pred_93, x_axis=X, delta=deltA);
#     SG_max_93 =  SG_max_min_93[0]; SG_min_93 =  SG_max_min_93[1];
#     SG_max_93 = rc.separate_x_and_y(m_list = SG_max_93);
#     SG_min_93 = rc.separate_x_and_y(m_list = SG_min_93);
#     SG_max_DoYs_series_93 = pd.Series(SG_max_93[0]);
#     SG_max_series_93 = pd.Series(SG_max_93[1]);
#     SG_min_DoYs_series_93 = pd.Series(SG_min_93[0]);
#     SG_min_series_93 = pd.Series(SG_min_93[1]);

#     ########################################################################################################
#     ########################################################################################################

#     plotting_dic = { "SG 31" : [SG_pred_31, SG_max_DoYs_series_31, SG_max_series_31],
#                      "SG 32" : [SG_pred_32, SG_max_DoYs_series_32, SG_max_series_32],

#                      "SG 51" : [SG_pred_51, SG_max_DoYs_series_51, SG_max_series_51],
#                      "SG 52" : [SG_pred_52, SG_max_DoYs_series_52, SG_max_series_52],
#                      "SG 53" : [SG_pred_53, SG_max_DoYs_series_53, SG_max_series_53],

#                      "SG 71" : [SG_pred_71, SG_max_DoYs_series_71, SG_max_series_71],
#                      "SG 72" : [SG_pred_72, SG_max_DoYs_series_72, SG_max_series_72],
#                      "SG 73" : [SG_pred_73, SG_max_DoYs_series_73, SG_max_series_73],

#                      "SG 91" : [SG_pred_91, SG_max_DoYs_series_91, SG_max_series_91],
#                      "SG 92" : [SG_pred_92, SG_max_DoYs_series_92, SG_max_series_92],
#                      "SG 93" : [SG_pred_93, SG_max_DoYs_series_93, SG_max_series_93]
#                      }

#     #############################################
#     ###
#     ###             plot
#     ###
#     #############################################
#     #
#     #   We have to change this part to make a perfect plot
#     #        
#     plot_title = county + ", " + plant + " (" + ID + "), delta = " + str(deltA)
#     # sb.set();

#     fig, ax = plt.subplots(figsize=(8,6));
#     ax.scatter(date_df.Date, y, label="Raw data", s = 60);

#     for co, ite in enumerate(plotting_dic):
#         lbl = ite + ", Peaks: " + str(len(plotting_dic[ite][2]))
#         # ax.plot(X, plotting_dic[ite][0], label = lbl, c = eleven_colors[co])
#         ax.plot(date_df.Date, plotting_dic[ite][0], label = lbl, c = eleven_colors[co])

#         date_df_specific = date_df[date_df.DoY.isin(plotting_dic[ite][1])]
#                  # plotting_dic[ite][1]
#         ax.scatter(date_df_specific.Date, plotting_dic[ite][2], s=100, marker='*', c = eleven_colors[co]);

#     ax.set_title(plot_title);
#     ax.set(xlabel='DoY', ylabel=idx)

#     ################################################
#     #
#     #    bare soil indices plots
#     #

#     # an_EE_TS_BSI = rc.initial_clean(df = crr_fld, column_to_be_cleaned='BSI')
#     # # an_EE_TS_NDWI = rc.initial_clean(df = crr_fld, column_to_be_cleaned='NDWI')
#     # an_EE_TS_PSRI = rc.initial_clean(df = crr_fld, column_to_be_cleaned='PSRI')
#     # an_EE_TS_LSWI = rc.initial_clean(df = crr_fld, column_to_be_cleaned='LSWI')

#     # ax.plot(an_EE_TS_BSI['doy'], an_EE_TS_BSI['BSI'], label="BSI")
#     # # ax.plot(x_NDWI, y_NDWI, label="NWDI")

#     # ax.plot(an_EE_TS_PSRI['doy'], an_EE_TS_PSRI['PSRI'], label="PSRI")
#     # ax.plot(an_EE_TS_LSWI['doy'], an_EE_TS_LSWI['LSWI'], label="LSWI")

#     ax.legend(loc="best");
#     return (fig)

def plot_TS(an_EE_TS_df, xp_axis='doy', yp_axis='NDVI'):
    year = int(an_EE_TS_df['year'].unique())
    plant = an_EE_TS_df['CropTyp'].unique()[0]
    county = an_EE_TS_df['county'].unique()[0]
    TRS = an_EE_TS_df['TRS'].unique()[0]

    an_EE_TS_df = remote_core.initial_clean(an_EE_TS_df)
    #
    # plot
    #
    plot_title = county + ", " + plant + ", " + str(year) + ", (" + TRS + ") delta = " + str()
    # sb.set();
    
    fig, ax = plt.subplots(figsize=(8,6));
    ax.plot(an_EE_TS_df[xp_axis], an_EE_TS_df[yp_axis], label="NDVI data");
    ax.set_title(plot_title);
    ax.legend(loc="best");

    return(TS_plot)

def subplots_savitzky(current_field, idx, deltA):

    eleven_colors = ["gray", "lightcoral", "red", "peru",
                     "darkorange", "gold", "olive", "green",
                     "blue", "violet", "deepskyblue"]

    plant = current_field['CropTyp'].unique()[0]
    # Take care of names, replace "/" and "," and " " by "_"
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")
    
    county = current_field['county'].unique()[0]
    ID = current_field['ID'].unique()[0]

    X = current_field['doy']
    y = current_field[idx]

    #############################################
    ###
    ###             Smoothen
    ###
    #############################################
    # differences are minor, but lets keep using Pythons function
    # my_savitzky_pred = rc.savitzky_golay(y, window_size=Sav_win_size, order=sav_order)

    SG_pred_31 = scipy.signal.savgol_filter(y, window_length = 3, polyorder = 1)
    SG_pred_32 = scipy.signal.savgol_filter(y, window_length = 3, polyorder = 2)
    
    SG_pred_51 = scipy.signal.savgol_filter(y, window_length = 5, polyorder = 1)
    SG_pred_52 = scipy.signal.savgol_filter(y, window_length = 5, polyorder = 2)
    SG_pred_53 = scipy.signal.savgol_filter(y, window_length = 5, polyorder = 3)
    
    SG_pred_71 = scipy.signal.savgol_filter(y, window_length = 7, polyorder = 1)
    SG_pred_72 = scipy.signal.savgol_filter(y, window_length = 7, polyorder = 2)
    SG_pred_73 = scipy.signal.savgol_filter(y, window_length = 7, polyorder = 3)

    SG_pred_91 = scipy.signal.savgol_filter(y, window_length = 9, polyorder = 1)
    SG_pred_92 = scipy.signal.savgol_filter(y, window_length = 9, polyorder = 2)
    SG_pred_93 = scipy.signal.savgol_filter(y, window_length = 9, polyorder = 3)

   
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

    SG_max_min_31 = rc.my_peakdetect(y_axis=SG_pred_31, x_axis=X, delta=deltA);
    SG_max_31 =  SG_max_min_31[0]; SG_min_31 =  SG_max_min_31[1];
    SG_max_31 = rc.separate_x_and_y(m_list = SG_max_31);
    SG_min_31 = rc.separate_x_and_y(m_list = SG_min_31);
    SG_max_DoYs_series_31 = pd.Series(SG_max_31[0]);
    SG_max_series_31 = pd.Series(SG_max_31[1]);
    SG_min_DoYs_series_31 = pd.Series(SG_min_31[0]);
    SG_min_series_31 = pd.Series(SG_min_31[1]);


    SG_max_min_32 = rc.my_peakdetect(y_axis=SG_pred_32, x_axis=X, delta=deltA);
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

    SG_max_min_51 = rc.my_peakdetect(y_axis=SG_pred_51, x_axis=X, delta=deltA);
    SG_max_51 =  SG_max_min_51[0]; SG_min_51 =  SG_max_min_51[1];
    SG_max_51 = rc.separate_x_and_y(m_list = SG_max_51);
    SG_min_51 = rc.separate_x_and_y(m_list = SG_min_51);
    SG_max_DoYs_series_51 = pd.Series(SG_max_51[0]);
    SG_max_series_51 = pd.Series(SG_max_51[1]);
    SG_min_DoYs_series_51 = pd.Series(SG_min_51[0]);
    SG_min_series_51 = pd.Series(SG_min_51[1]);

    SG_max_min_52 = rc.my_peakdetect(y_axis=SG_pred_52, x_axis=X, delta=deltA);
    SG_max_52 =  SG_max_min_52[0]; SG_min_52 =  SG_max_min_52[1];
    SG_max_52 = rc.separate_x_and_y(m_list = SG_max_52);
    SG_min_52 = rc.separate_x_and_y(m_list = SG_min_52);
    SG_max_DoYs_series_52 = pd.Series(SG_max_52[0]);
    SG_max_series_52 = pd.Series(SG_max_52[1]);
    SG_min_DoYs_series_52 = pd.Series(SG_min_52[0]);
    SG_min_series_52 = pd.Series(SG_min_52[1]);

    SG_max_min_53 = rc.my_peakdetect(y_axis=SG_pred_53, x_axis=X, delta=deltA);
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

    SG_max_min_71 = rc.my_peakdetect(y_axis=SG_pred_71, x_axis=X, delta=deltA);
    SG_max_71 =  SG_max_min_71[0]; SG_min_71 =  SG_max_min_71[1];
    SG_max_71 = rc.separate_x_and_y(m_list = SG_max_71);
    SG_min_71 = rc.separate_x_and_y(m_list = SG_min_71);
    SG_max_DoYs_series_71 = pd.Series(SG_max_71[0]);
    SG_max_series_71 = pd.Series(SG_max_71[1]);
    SG_min_DoYs_series_71 = pd.Series(SG_min_71[0]);
    SG_min_series_71 = pd.Series(SG_min_71[1]);

    SG_max_min_72 = rc.my_peakdetect(y_axis=SG_pred_72, x_axis=X, delta=deltA);
    SG_max_72 =  SG_max_min_72[0]; SG_min_72 =  SG_max_min_72[1];
    SG_max_72 = rc.separate_x_and_y(m_list = SG_max_72);
    SG_min_72 = rc.separate_x_and_y(m_list = SG_min_72);
    SG_max_DoYs_series_72 = pd.Series(SG_max_72[0]);
    SG_max_series_72 = pd.Series(SG_max_72[1]);
    SG_min_DoYs_series_72 = pd.Series(SG_min_72[0]);
    SG_min_series_72 = pd.Series(SG_min_72[1]);

    SG_max_min_73 = rc.my_peakdetect(y_axis=SG_pred_73, x_axis=X, delta=deltA);
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

    SG_max_min_91 = rc.my_peakdetect(y_axis=SG_pred_91, x_axis=X, delta=deltA);
    SG_max_91 =  SG_max_min_91[0]; SG_min_91 =  SG_max_min_91[1];
    SG_max_91 = rc.separate_x_and_y(m_list = SG_max_91);
    SG_min_91 = rc.separate_x_and_y(m_list = SG_min_91);
    SG_max_DoYs_series_91 = pd.Series(SG_max_91[0]);
    SG_max_series_91 = pd.Series(SG_max_91[1]);
    SG_min_DoYs_series_91 = pd.Series(SG_min_91[0]);
    SG_min_series_91 = pd.Series(SG_min_91[1]);

    SG_max_min_92 = rc.my_peakdetect(y_axis=SG_pred_92, x_axis=X, delta=deltA);
    SG_max_92 =  SG_max_min_92[0]; SG_min_92 =  SG_max_min_92[1];
    SG_max_92 = rc.separate_x_and_y(m_list = SG_max_92);
    SG_min_92 = rc.separate_x_and_y(m_list = SG_min_92);
    SG_max_DoYs_series_92 = pd.Series(SG_max_92[0]);
    SG_max_series_92 = pd.Series(SG_max_92[1]);
    SG_min_DoYs_series_92 = pd.Series(SG_min_92[0]);
    SG_min_series_92 = pd.Series(SG_min_92[1]);

    SG_max_min_93 = rc.my_peakdetect(y_axis=SG_pred_93, x_axis=X, delta=deltA);
    SG_max_93 =  SG_max_min_93[0]; SG_min_93 =  SG_max_min_93[1];
    SG_max_93 = rc.separate_x_and_y(m_list = SG_max_93);
    SG_min_93 = rc.separate_x_and_y(m_list = SG_min_93);
    SG_max_DoYs_series_93 = pd.Series(SG_max_93[0]);
    SG_max_series_93 = pd.Series(SG_max_93[1]);
    SG_min_DoYs_series_93 = pd.Series(SG_min_93[0]);
    SG_min_series_93 = pd.Series(SG_min_93[1]);

    ########################################################################################################
    ########################################################################################################

    plotting_dic = { "SG 31" : [SG_pred_31, SG_max_DoYs_series_31, SG_max_series_31],
                     "SG 32" : [SG_pred_32, SG_max_DoYs_series_32, SG_max_series_32],

                     "SG 51" : [SG_pred_51, SG_max_DoYs_series_51, SG_max_series_51],
                     "SG 52" : [SG_pred_52, SG_max_DoYs_series_52, SG_max_series_52],
                     "SG 53" : [SG_pred_53, SG_max_DoYs_series_53, SG_max_series_53],

                     "SG 71" : [SG_pred_71, SG_max_DoYs_series_71, SG_max_series_71],
                     "SG 72" : [SG_pred_72, SG_max_DoYs_series_72, SG_max_series_72],
                     "SG 73" : [SG_pred_73, SG_max_DoYs_series_73, SG_max_series_73],

                     "SG 91" : [SG_pred_91, SG_max_DoYs_series_91, SG_max_series_91],
                     "SG 92" : [SG_pred_92, SG_max_DoYs_series_92, SG_max_series_92],
                     "SG 93" : [SG_pred_93, SG_max_DoYs_series_93, SG_max_series_93]
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
        
    plot_title = county + ", " + plant + " (" + ID + "), delta = " + str(deltA)
    # sb.set();

    ##################################################################
    fig, ax = plt.subplots(figsize=(8,6));
    ax.grid(True)

    ax.scatter(X, y, label="Raw data", s=30);

    for co, ite in enumerate(plotting_dic):
        lbl = ite + ", Peaks: " + str(len(plotting_dic[ite][2]))
        ax.plot(X, plotting_dic[ite][0], label = lbl, c = eleven_colors[co])
        ax.scatter(plotting_dic[ite][1], plotting_dic[ite][2], s=100, marker='*', c = eleven_colors[co]);

    ax.set_title(plot_title);
    ax.set(xlabel='DoY', ylabel=idx)

    ################################################
    #
    #    bare soil indices plots
    #

    # an_EE_TS_BSI = rc.initial_clean(df = current_field, column_to_be_cleaned='BSI')
    # # an_EE_TS_NDWI = rc.initial_clean(df = current_field, column_to_be_cleaned='NDWI')
    # an_EE_TS_PSRI = rc.initial_clean(df = current_field, column_to_be_cleaned='PSRI')
    # an_EE_TS_LSWI = rc.initial_clean(df = current_field, column_to_be_cleaned='LSWI')

    # ax.plot(an_EE_TS_BSI['doy'], an_EE_TS_BSI['BSI'], label="BSI")
    # # ax.plot(x_NDWI, y_NDWI, label="NWDI")

    # ax.plot(an_EE_TS_PSRI['doy'], an_EE_TS_PSRI['PSRI'], label="PSRI")
    # ax.plot(an_EE_TS_LSWI['doy'], an_EE_TS_LSWI['LSWI'], label="LSWI")

    ax.legend(loc="best");
    return (fig)
