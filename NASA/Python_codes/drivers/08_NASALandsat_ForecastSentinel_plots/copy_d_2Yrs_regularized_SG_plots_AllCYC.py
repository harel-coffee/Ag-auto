

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
###      Parameters                   
###
####################################################################################

regularized = True
indeks = "EVI"
irrigated_only = True
given_county = sys.argv[1]

SF_year = int(sys.argv[4])
SEOS_cut = int(sys.argv[6])

sos_thresh = int(SEOS_cut / 10)/10 # grab the first digit as SOS cut
eos_thresh = (SEOS_cut % 10) / 10  # grab the second digit as EOS cut

####################################################################################
###
###                   Aeolus Directories
###
####################################################################################

if irrigated_only == True:
    output_Irr = "irrigated_only"
else:
    output_Irr = "non_irrigated_only"

data_dir = "/data/hydro/users/Hossein/remote_sensing/03_Regularized_TS/70_cloud/2Yrs/noJump_Regularized/"

sentinel_raw_dir = "/data/hydro/users/Hossein/remote_sensing/02_Eastern_WA_EE_TS/2Years/70_cloud/"

#####################################################################################

os.makedirs(output_dir, exist_ok=True)

####################################################################################
###
###                   Read data
###
####################################################################################
f_name = "01_Regular_filledGap_" + given_county + "_SF_" + str(SF_year) + "_" + indeks + ".csv"
a_sentinel_df = pd.read_csv(data_dir + f_name, low_memory=False)

raw_f_name = "Eastern_WA_" + str(SF_year) + "_70cloud_selectors.csv"
sentinel_raw_df = pd.read_csv(sentinel_raw_dir + raw_f_name, low_memory=False)

if 'Date' in a_sentinel_df.columns:
    if type(a_sentinel_df.Date.iloc[0]) == str:
        a_sentinel_df['Date'] = pd.to_datetime(a_sentinel_df.Date.values).values

##################################################################
##################################################################
####
####  plots has to be exact. So, we need 
####  to filter out NASS, and filter by last survey date
####
##################################################################
##################################################################

a_sentinel_df = a_sentinel_df[a_sentinel_df['county'] == given_county.replace("_", " ")] # Filter Grant
a_sentinel_df = rc.filter_out_NASS(a_sentinel_df) # Toss NASS
a_sentinel_df = rc.filter_by_lastSurvey(a_sentinel_df, year = SF_year) # filter by last survey date
a_sentinel_df['SF_year'] = SF_year

################################################################################
sentinel_raw_df = sentinel_raw_df[sentinel_raw_df['county'] == given_county.replace("_", " ")] # Filter Grant
sentinel_raw_df = rc.filter_out_NASS(sentinel_raw_df) # Toss NASS
sentinel_raw_df = rc.filter_by_lastSurvey(sentinel_raw_df, year = SF_year) # filter by last survey date

################################################################################

if irrigated_only == True:
    a_sentinel_df = rc.filter_out_nonIrrigated(a_sentinel_df)
    output_Irr = "irrigated_only"
else:
    output_Irr = "non_irrigated_only"
    a_sentinel_df = rc.filter_out_Irrigated(a_sentinel_df)

######################

# The following columns do not exist in the old data
#
if not('DataSrc' in a_sentinel_df.columns):
    print ("Data source is being set to NA")
    a_sentinel_df['DataSrc'] = "NA"

if not('DataSrc' in sentinel_raw_df.columns):
    print ("Data source is being set to NA")
    sentinel_raw_df['DataSrc'] = "NA"

if not('CovrCrp' in a_sentinel_df.columns):
    print ("CovrCrp is being set to NA")
    a_sentinel_df['CovrCrp'] = "NA"
    
if not('CovrCrp' in sentinel_raw_df.columns):
    print ("CovrCrp is being set to NA")
    sentinel_raw_df['CovrCrp'] = "NA"


a_sentinel_df = rc.initial_clean(df = a_sentinel_df, column_to_be_cleaned = indeks)
sentinel_raw_df = rc.initial_clean(df = sentinel_raw_df, column_to_be_cleaned = indeks)

if not("human_system_start_time" in sentinel_raw_df.columns):
    sentinel_raw_df = rc.add_human_start_time_by_YearDoY(sentinel_raw_df)

if 'Date' in sentinel_raw_df.columns:
    if type(sentinel_raw_df.Date.iloc[0]) == str:
        sentinel_raw_df['Date'] = pd.to_datetime(sentinel_raw_df.Date.values).values
else: 
    sentinel_raw_df['Date'] = pd.to_datetime(sentinel_raw_df.human_system_start_time.values).values


an_EE_TS = a_sentinel_df.copy()
del(a_sentinel_df)

### List of unique polygons
polygon_list = np.sort(an_EE_TS['ID'].unique())
print ("_____________________________________")
print("len(polygon_list)")
print (len(polygon_list))
print ("_____________________________________")

counter = 0

for ID in polygon_list:
    if (counter%100 == 0):
        print ("_____________________________________")
        print ("counter: " + str(counter))
        print (ID)

    curr_field_two_years = an_EE_TS[an_EE_TS['ID'] == ID].copy()
    curr_raw = sentinel_raw_df[sentinel_raw_df['ID'] == ID].copy()
    curr_field_two_years.sort_values(by=['image_year', 'doy'], inplace=True)
    curr_raw.sort_values(by=['image_year', 'doy'], inplace=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(20,12),
                            sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.1, 'wspace': .1});

    (ax1, ax2), (ax3, ax4) = axs;
    ax1.grid(True); ax2.grid(True); ax3.grid(True); ax4.grid(True);
    rcp.panel_SOS_Sentinel4Landsat(twoYears_raw = curr_raw,
                                   twoYears_regular = curr_field_two_years,
                                   idx = indeks, SG_params=[7, 3],
                                   SFYr = SF_year, ax=ax3,
                                   onset_cut = sos_thresh, 
                                   offset_cut = eos_thresh);

    fig_name = plot_path + given_county + "_" + "_SF_year_" + str(SF_year) + "_" + ID + '.png'

    os.makedirs(plot_path, exist_ok=True)

    plt.savefig(fname = fig_name, dpi=100, bbox_inches='tight')
    plt.close('all')
    counter += 1


print ("done")
end_time = time.time()
print(end_time - start_time)


