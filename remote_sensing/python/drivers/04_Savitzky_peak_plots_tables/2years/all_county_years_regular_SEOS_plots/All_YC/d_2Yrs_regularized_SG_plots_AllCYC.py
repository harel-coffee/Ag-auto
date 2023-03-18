####
#### July 2. This is a copy of the version we had from before. plotting one year.
#### Here we are extending it to 2 years. Since August of a given year to the end
#### of the next year.
####


import matplotlib.backends.backend_pdf
import csv
import numpy as np
import pandas as pd
# import geopandas as gpd
from IPython.display import Image
# from shapely.geometry import Point, Polygon
from math import factorial
import datetime
from datetime import date
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
register_matplotlib_converters()

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

####################################################################################
###
###                      Aeolus Core path
###
####################################################################################

sys.path.append('/home/hnoorazar/remote_sensing_codes/')

###
### Import remote cores
###
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

# given_county = "Grant"
# sos_thresh = 0.5
# eos_thresh = 0.5

regularized = True
minFinderDetla = 0.4

jumps = sys.argv[1]
indeks = sys.argv[2]
irrigated_only = int(sys.argv[3])
SF_year = int(sys.argv[4])
given_county = sys.argv[5]
SEOS_cut = int(sys.argv[6])

sos_thresh = int(SEOS_cut / 10)/10 # grab the first digit as SOS cut
eos_thresh = (SEOS_cut % 10) / 10  # grab the second digit as EOS cut

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
param_dir = "/home/hnoorazar/remote_sensing_codes/parameters/"
output_base = "/data/hydro/users/Hossein/remote_sensing/04_noJump_Regularized_plt_tbl_SOSEOS/"

if irrigated_only == True:
    output_Irr = "irrigated_only"
else:
    output_Irr = "non_irrigated_only"

regular_data_dir = "/data/hydro/users/Hossein/remote_sensing/03_Regularized_TS/70_cloud/2Yrs/"

regular_data_dir = regular_data_dir + "/noJump_Regularized/"
regular_output_dir = output_base + "/ALLYCF_plots_fine_gran/" + \
                     str(SF_year) + "_regular_" + output_Irr + "_" + indeks + \
                     "_SOS" + str(int(sos_thresh*10)) + "_EOS" + str(int(eos_thresh*10)) + "/"

f_name = "01_Regular_filledGap_" + given_county + "_SF_" + str(SF_year) + "_" + indeks + ".csv"

#####################################################################################

raw_dir = "/data/hydro/users/Hossein/remote_sensing/02_Eastern_WA_EE_TS/2Years/70_cloud/"
raw_f_name = "Eastern_WA_" + str(SF_year) + "_70cloud_selectors.csv"

#####################################################################################

data_dir = regular_data_dir
output_dir = regular_output_dir
plot_dir_base = regular_output_dir
print ("plot_dir_base is " + plot_dir_base)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir_base, exist_ok=True)

print ("_________________________________________________________")
print ("data dir is:")
print (data_dir)
print ("_________________________________________________________")
print ("output_dir is:")
print (output_dir)
print ("_________________________________________________________")

####################################################################################
###
###                   Read data
###
####################################################################################

a_df = pd.read_csv(data_dir + f_name, low_memory=False)
raw_df = pd.read_csv(raw_dir + raw_f_name, low_memory=False)

if 'Date' in a_df.columns:
    if type(a_df.Date.iloc[0]) == str:
        a_df['Date'] = pd.to_datetime(a_df.Date.values).values

##################################################################
##################################################################
####
####  plots has to be exact. So, we need 
####  to filter out NASS, and filter by last survey date
####
##################################################################
##################################################################

a_df = a_df[a_df['county'] == given_county.replace("_", " ")] # Filter Grant
a_df = rc.filter_out_NASS(a_df) # Toss NASS
a_df = rc.filter_by_lastSurvey(a_df, year = SF_year) # filter by last survey date
a_df['SF_year'] = SF_year

################################################################################
raw_df = raw_df[raw_df['county'] == given_county.replace("_", " ")] # Filter Grant
raw_df = rc.filter_out_NASS(raw_df) # Toss NASS
raw_df = rc.filter_by_lastSurvey(raw_df, year = SF_year) # filter by last survey date

################################################################################

if irrigated_only == True:
    a_df = rc.filter_out_nonIrrigated(a_df)
    output_Irr = "irrigated_only"
else:
    output_Irr = "non_irrigated_only"
    a_df = rc.filter_out_Irrigated(a_df)

######################

# The following columns do not exist in the old data
#
if not('DataSrc' in a_df.columns):
    print ("Data source is being set to NA")
    a_df['DataSrc'] = "NA"

if not('CovrCrp' in a_df.columns):
    print ("CovrCrp is being set to NA")
    a_df['CovrCrp'] = "NA"


if not('DataSrc' in raw_df.columns):
    print ("Data source is being set to NA")
    raw_df['DataSrc'] = "NA"
    
if not('CovrCrp' in raw_df.columns):
    print ("CovrCrp is being set to NA")
    raw_df['CovrCrp'] = "NA"


a_df = rc.initial_clean(df = a_df, column_to_be_cleaned = indeks)
raw_df = rc.initial_clean(df = raw_df, column_to_be_cleaned = indeks)

if not("human_system_start_time" in raw_df.columns):
    raw_df = rc.add_human_start_time_by_YearDoY(raw_df)

if 'Date' in raw_df.columns:
    if type(raw_df.Date.iloc[0]) == str:
        raw_df['Date'] = pd.to_datetime(raw_df.Date.values).values
else: 
    raw_df['Date'] = pd.to_datetime(raw_df.human_system_start_time.values).values


an_EE_TS = a_df.copy()
del(a_df)

### List of unique polygons
polygon_list = np.sort(an_EE_TS['ID'].unique())
print ("_____________________________________")
print("len(polygon_list)")
print (len(polygon_list))
print ("_____________________________________")

counter = 0

for a_poly in polygon_list:
    if (counter%100 == 0):
        print ("_____________________________________")
        print ("counter: " + str(counter))
        print (a_poly)

    curr_field_two_years = an_EE_TS[an_EE_TS['ID'] == a_poly].copy()
    
    #********#********#********#********#********#********
    #
    #    Set negative EVI values to zero. (we moved this line to step 00_remove_outliers. Lets see...)
    #
    ##********#********#********#********#********#********
    
    # curr_field_two_years.loc[curr_field_two_years[indeks] < 0 , indeks] = 0

    curr_raw = raw_df[raw_df['ID'] == a_poly].copy()

    #
    #  filter just one year to have a clean SOS, EOS stuff
    #
    # curr_field = curr_field[curr_field.image_year == SF_year]
    
    
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field_two_years.sort_values(by=['image_year', 'doy'], inplace=True)
    curr_raw.sort_values(by=['image_year', 'doy'], inplace=True)

    ################################################################
    ID = a_poly
    plant = curr_field_two_years['CropTyp'].unique()[0]
    plant = plant.replace("/", "_")
    plant = plant.replace(",", "_")
    plant = plant.replace(" ", "_")
    plant = plant.replace("__", "_")

    sub_out = plant + "/" # "/plant_based_plots/" + plant + "/"
    plot_path = plot_dir_base + sub_out
    plot_path = plot_path   # +  str(len(SG_max_DoYs_series)) + "_peaks/"
    os.makedirs(plot_path, exist_ok=True)
    # print ("plot_path is " + plot_path)

    
    fig, axs = plt.subplots(2, 2, figsize=(20,12),
                            sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.1, 'wspace': .1});

    (ax1, ax2), (ax3, ax4) = axs;
    ax1.grid(True); ax2.grid(True); ax3.grid(True); ax4.grid(True);

    rcp.SG_1yr_panels_clean_sciPy_My_Peaks_SOS_fineGranularity(twoYears_raw = curr_raw,
                                                               twoYears_regular = curr_field_two_years,
                                                               # dataAB = curr_field, 
                                                               idx = indeks, 
                                                               SG_params = [5, 1], 
                                                               SFYr = SF_year, ax = ax1, deltA = minFinderDetla,
                                                               onset_cut = sos_thresh, 
                                                               offset_cut = eos_thresh);

    rcp.SG_1yr_panels_clean_sciPy_My_Peaks_SOS_fineGranularity(twoYears_raw = curr_raw,
                                                               twoYears_regular = curr_field_two_years,
                                                               # dataAB = curr_field, 
                                                               idx = indeks, SG_params = [5, 3], 
                                                               SFYr = SF_year, ax = ax2, deltA = minFinderDetla,
                                                               onset_cut = sos_thresh, 
                                                               offset_cut = eos_thresh); 

    rcp.SG_1yr_panels_clean_sciPy_My_Peaks_SOS_fineGranularity(twoYears_raw = curr_raw,
                                                               twoYears_regular = curr_field_two_years,
                                                               # dataAB = curr_field, 
                                                               idx = indeks, SG_params=[7, 3],
                                                               SFYr = SF_year, ax=ax3, deltA = minFinderDetla,
                                                               onset_cut = sos_thresh, 
                                                               offset_cut = eos_thresh);

    rcp.SG_1yr_panels_clean_sciPy_My_Peaks_SOS_fineGranularity(twoYears_raw = curr_raw,
                                                               twoYears_regular = curr_field_two_years,
                                                               # dataAB = curr_field, 
                                                               idx = indeks, SG_params = [9, 3],
                                                               SFYr = SF_year, ax=ax4, deltA = minFinderDetla,
                                                               onset_cut = sos_thresh, 
                                                               offset_cut = eos_thresh)

    fig_name = plot_path + given_county + "_" + plant + "_SF_year_" + str(SF_year) + "_" + ID + '.png'

    os.makedirs(plot_path, exist_ok=True)

    plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight')
    plt.close('all')
    counter += 1


print ("done")
end_time = time.time()
print(end_time - start_time)


