# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="bC4yPD8Vv639"
# **Let the Fun Begin**
# [Colab resources are not guaranteed](https://research.google.com/colaboratory/faq.html#:~:text=Colab\%20is\%20able\%20to\%20provide,other\%20factors\%20vary\%20over\%20time.). One can, however, [subscribe and increase resources]((https://colab.research.google.com/signup)).
#
# <p>&nbsp;</p>
#
#
# split_large_feacturecollection_to_blocks
#
# %who

# %% id="KexkTWOCv98f"
try:
    import shutup
except ImportError:
  # !pip install shutup
  import shutup

shutup.please() # kill some of the messages

# %% colab={"base_uri": "https://localhost:8080/"} id="hzHm2AuTv_gf" outputId="3810d15d-3d59-4da4-9ec1-9d1cccb9f53b"
"""
Print Local Time 

colab runs on cloud. So, the time is not our local time.
This page is useful to determine how to do this.
"""
# !rm /etc/localtime
# !ln -s /usr/share/zoneinfo/US/Pacific /etc/localtime
# !date

# %% [markdown] id="oLscEeczwPwa"
# **geopandas and geemap must be installed every time!!!**

# %% id="Yw2k9M84wQW5"
import subprocess

try:
    import geemap
except ImportError:
    print('geemap not installed. Must be installed every tim to run this notebook. Installing ...')
    subprocess.check_call(["python", '-m', 'pip', 'install', 'geemap'])

    print('geopandas not installed. Must be installed every time to run this notebook. Installing ...')
    subprocess.check_call(["python", '-m', 'pip', 'install', 'geopandas'])
    subprocess.check_call(["python", '-m', 'pip', 'install', 'google.colab'])

# %% id="x97vA_z-wSXP"

# %% [markdown] id="A_ZM-E8owf9j"
# **Authenticate and import libraries**
#
# We have to import the libraries we need. Moreover, we need to Authenticate, every single time!

# %% id="9kO52mCdwga2"
import pandas as pd
import numpy as np
import geopandas as gpd

import folium, time, datetime, json, geemap, ee
from datetime import date

import scipy # we need this for savitzky-golay
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# %% [markdown] id="kZs1zgZXwtJg"
# ### **Mount Google Drive and import my Python modules**
#
# Here we are importing the Python functions that are written by me and are needed; ```NASA core``` and ```NASA plot core```.
#
# **Note:** These are on Google Drive now. Perhaps we can import them from GitHub.

# %% colab={"base_uri": "https://localhost:8080/"} id="Dq1XkGZqwtha" outputId="154662fb-a34f-4c13-f798-c592d2153f1a"
# Mount YOUR google drive in Colab
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0,"/content/drive/MyDrive/Colab Notebooks/")
import NASA_core as nc
import NASA_plot_core as ncp
import GEE_Python_core as gpc

# %% id="0ZtODYYXwv9-"
# **Change Current directory to the Colab folder on Google Drive**

import os
os.chdir("/content/drive/MyDrive/Colab Notebooks/") # Colab Notebooks
# # !ls

# %% [markdown] id="c6-1tln-w6p_"
# # Please tell me where to look for the shapefile!

# %% colab={"base_uri": "https://localhost:8080/", "height": 130} id="QEFiOBCcw29m" outputId="cab42719-f5c4-4c30-dda3-790a00c3db1e"
SF_name = "10_intersect_East_Irr_2008_2018_2cols"
shp_path = "/content/drive/MyDrive/NASA_trends/shapefiles/" + SF_name + "/" + SF_name + ".shp"

shp_path = "/content/drive/MyDrive/NASA_trends/shapefiles/" + \
            "Grant_4Fields_poly_wCentroids/Grant_4Fields_poly_wCentroids.shp"

"""
  we read our shapefile in to a geopandas data frame using 
  the geopandas.read_file method
  we'll make sure it's initiated in the EPSG 4326 CRS
"""
SF = gpd.read_file(shp_path, crs='EPSG:4326')

### for possible future use grab the data part of the shapefile
SF_data = SF[["ID", "Acres", "county", "CropTyp", \
              "DataSrc", "Irrigtn", "LstSrvD"]].copy()
SF_data.drop_duplicates(inplace=True)
print (SF_data.shape)

""" 
   Drop extra useless columns. Saves space.**
   Also, GEE behaves strangely. It has problem with Notes column before
"""
# SF = SF.drop(columns=["Notes", "TRS", "IntlSrD", "RtCrpTy", "Shp_Lng", "Shap_Ar","CropGrp","CropTyp","Acres","Irrigtn","LstSrvD","DataSrc","county","ExctAcr","cntrd_ln","cntrd_lt"])
SF = SF.drop(columns=["CropTyp","Acres","Irrigtn",
                      "LstSrvD","DataSrc","county","ExctAcr",
                      "cntrd_ln","cntrd_lt"])

SF.head(2)

# %% colab={"base_uri": "https://localhost:8080/"} id="kQNFWTyVw9jC" outputId="9b4c4241-543c-4bdf-db43-66e49eb59e25"
long_eq = "=============================================================================="
print (type(SF))
print (long_eq)
print (f"{SF.shape=}", )
print (long_eq)

# %% [markdown] id="cNmGfjZjxffG"
# **Form Geographical Regions**
#
#   - First, define a big region that covers Eastern Washington.
#   - Convert shapefile to ```ee.featurecollection.FeatureCollection```.
#

# %% id="0KokSLd-xLpd"
xmin = -125.0;
ymin = 45.0;
xmax = -116.0;
ymax = 49.0;

xmed = (xmin + xmax) / 2.0;
ymed = (ymin+ymax) / 2.0;

WA1 = ee.Geometry.Polygon([[xmin, ymin], [xmin, ymax], [xmed, ymax], [xmed, ymin], [xmin, ymin]]);
WA2 = ee.Geometry.Polygon([[xmed, ymin], [xmed, ymax], [xmax, ymax], [xmax, ymin], [xmed, ymin]]);
WA = [WA1,WA2];
big_rectangle = ee.FeatureCollection(WA);
SF = geemap.geopandas_to_ee(SF)

# %% id="iTZ_iV9wxhUE"
# Map = geemap.Map(center=[ymed, xmed], zoom=7)
# Map.addLayer(WA1, {'color': 'red'}, 'Western Half')
# Map.addLayer(WA2, {'color': 'red'}, 'Eastern Half')
# Map.addLayer(SF.head(5), {'color': 'blue'}, 'Fields')
# Map

# %% [markdown] id="ScwLLlr5xnbT"
# **Define Parameters**

# %% id="uoyMLZLSxjCS"
# start_date = "2017-01-01" # Date fromat for EE YYYY-MM-DD
# end_date = "2017-12-30"   # Date fromat for EE YYYY-MM-DD
start_date = str(2000) + '-01-01'; # YYYY-MM-DD
end_date =   str(2008) + '-01-01';   # YYYY-MM-DD

start_date = "2017-01-01" # Date fromat for EE YYYY-MM-DD
end_date = "2017-12-30"   # Date fromat for EE YYYY-MM-DD

"""
 sattelie_of_choice must be from
         'LANDSAT/LT04/C02/T1_L2' OR 
         'LANDSAT/LT05/C02/T1_L2' OR
         'LANDSAT/LE07/C02/T1_L2' OR
         'LANDSAT/LC08/C02/T1_L2'

        L4: life span is 7/16/1982 to 12/14/1993
        L5: life span is 1-March-1984 to 5-June-2013
        L7: life span is 15-April-1999 to 6-April-2022
        L8: life span is 11-Feb-2013 till now (June 2023)

To Do- Merge Sentinel into one function. (June 6 2023.)
"""
sattelie_of_choice = 'LANDSAT/LE07/C02/T1_L2'
cloud_perc = 70

# %% [markdown] id="2NGnUCf9xsRj"
# **Fetch data from GEE**
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 107} id="Pu--H6LvxpRC" outputId="104bb914-f8a3-4f2d-82a7-a8aaa595e459"
# %%time
# imageC = gpc.extract_sentinel_IC(big_rectangle, start_date, end_date, cloud_perc);
imageC = gpc.extract_satellite_IC(big_rectangle_featC=big_rectangle, 
                                  start_date = start_date, 
                                  end_date = end_date, 
                                  dataSource = sattelie_of_choice);
print ("The size of [imageC.size().getInfo()] is [{:.0f}].".format(imageC.size().getInfo()))

reduced = gpc.mosaic_and_reduce_IC_mean(imageC, SF, start_date, end_date)

needed_columns = ["ID", "EVI", 'NDVI', "system_start_time"]
reduced = geemap.ee_to_pandas(reduced, selectors=needed_columns)
reduced = reduced[needed_columns]

###
###  Sentinel
###
imageC_sentinel = gpc.extract_sentinel_IC(big_rectangle, start_date, end_date, cloud_perc);
print ("The size of [imageC.size().getInfo()] is [{:.0f}].".format(imageC_sentinel.size().getInfo()))

reduced_sentinel = gpc.mosaic_and_reduce_IC_mean(imageC_sentinel, SF, start_date, end_date)

"""
 Sometimes selectors=needed_columns below is not effective!!!
 So, for extra measurement we also do 
 A = A[needed_columns]
"""
# A = geemap.ee_to_pandas(reduced_sentinel, selectors=needed_columns)
# A = A[needed_columns]
# fileName = "Grant4Fields_Sentinel_" + start_date + "_" + end_date + ".csv"
# A.to_csv('/content/drive/MyDrive/colab_outputs/'+ fileName)
# A.head(2)

# %% [markdown] id="2SyH8eg6x5xe"
# # Export output to Google Drive
#
# We advise you to do it. If Python/CoLab kernel dies, then,
# previous steps should not be repeated.

# %% id="nbzxU6-7x0vn"
export_raw_data = False

if export_raw_data==True:
    outfile_name = "Grant_4Fields_poly_wCentroids_colab_outputLandSat7Jun62023Test2_" + start_date + "_" + end_date + ".csv"
    reduced.to_csv('/content/drive/MyDrive/colab_outputs/'+ outfile_name)

# %% [markdown] id="ItGdHJwyyB_K"
# # **Smooth the data**
#
# This is the end of Earh Engine Part. Below we start smoothing the data and carry on!
#
# First, all these steps can be done behind the scene. But doing them here, one at a time, has the advantage that if something goes wrong in the middle, then
# we do not lose the good stuff that was done earlier!
# For example, of one of the Python libraries/packages needs to be updated in the middle of the way
# we do not have to start doing everything from the beginning!
# <p>&nbsp;</p>
#
# Start with converting the type of ```reduced``` from ```ee.FeatureCollection``` to ```dataframe```.
#
# - For some reason when converting the ```ee.FeatureCollection``` to ```dataframe``` the function has a problem with the ```Notes``` column! So, I remove the unnecessary columns.
#
# **NA removal**
#
# Even though logically and intuitively all the bands should be either available or ```NA```, I have seen before that sometimes ```EVI``` is NA while ```NDVI``` is not. Therefore, I had to choose which VI we want to use so that we can clean the data properly. However, I did not see that here.  when I was testing this code for 4 fields.
#
# Another suprising observation was that the output of Colab had more data compared to its JS counterpart!!!
#
# # **Define the VI parameter we want to work with**

# %% id="UnI4YX-sy3bB"
VI_idx = "NDVI"

# %% colab={"base_uri": "https://localhost:8080/"} id="a84eRMqAx9Gg" outputId="bce4dcb6-3ec0-43d7-d575-7973d00810ec"
# reduced = reduced[reduced["system_start_time"].notna()]
reduced = reduced[reduced[VI_idx].notna()]
reduced.reset_index(drop=True, inplace=True)

## Add human readable time to the dataframe
reduced = nc.add_human_start_time_by_system_start_time(reduced)
reduced.head(2)
reduced.loc[0, "system_start_time"]

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="gwasOHCpyM5z" outputId="f08b0827-2099-445a-cf4f-367af27571c9"
# Pick a field
a_field = reduced[reduced.ID==reduced.ID.unique()[0]].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3),
                       sharex='col', sharey='row',
                       # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(axis='y', which="both")

ax.scatter(a_field['human_system_start_time'], a_field[VI_idx], s=40, c='#d62728');
ax.plot(a_field['human_system_start_time'], a_field[VI_idx], 
        linestyle='-',  linewidth=3.5, color="#d62728", alpha=0.8,
        label=f"raw {VI_idx}")
plt.ylim([-0.5, 1.2]);
ax.legend(loc="lower right");
# ax.set_title(a_field.CropTyp.unique()[0]);

# %% [markdown] id="4RS_yMcGyXaB"
# # Efficiency 
#
# Can we make this more efficient by doing the calculations in place as opposed to creating a new ```dataframe``` and copying stuff. Perhaps ```.map(.)``` too.
#
# **Remove outliers**

# %% id="q9VLleCHyTqM"
reduced["ID"] = reduced["ID"].astype(str)
IDs = np.sort(reduced["ID"].unique())

# %% colab={"base_uri": "https://localhost:8080/"} id="hW2gfGgayZmB" outputId="b4b676ae-e8e5-40ff-b095-cc1c9025c46f"
no_outlier_df = pd.DataFrame(data = None,
                         index = np.arange(reduced.shape[0]), 
                         columns = reduced.columns)
counter = 0
row_pointer = 0
for a_poly in IDs:
    if (counter % 1000 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = reduced[reduced["ID"]==a_poly].copy()
    # small fields may have nothing in them!
    if curr_field.shape[0] > 2:
        ##************************************************
        #
        #    Set negative index values to zero.
        #
        ##************************************************
        no_Outlier_TS = nc.interpolate_outliers_EVI_NDVI(outlier_input = curr_field, given_col = VI_idx)
        no_Outlier_TS.loc[no_Outlier_TS[VI_idx
                                        ] < 0 , VI_idx] = 0 

        """
        it is possible that for a field we only have x=2 data points
        where all the EVI/NDVI is outlier. Then, there is nothing to 
        use for interpolation. So, hopefully interpolate_outliers_EVI_NDVI is returning an empty data table.
        """ 
        if len(no_Outlier_TS) > 0:
            no_outlier_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
            counter += 1
            row_pointer += curr_field.shape[0]

# Sanity check. Will neved occur. At least should not!
no_outlier_df.drop_duplicates(inplace=True)

# %% [markdown] id="keCtT-no0ieV"
# **Remove the jumps**
#
# Maybe we can remove old/previous dataframes to free memory up!

# %% colab={"base_uri": "https://localhost:8080/"} id="4XbWi2MF0akF" outputId="03ac5075-36ab-4b3e-d194-571ca3a1bb5b"
noJump_df = pd.DataFrame(data = None,
                         index = np.arange(no_outlier_df.shape[0]), 
                         columns = no_outlier_df.columns)
counter, row_pointer = 0, 0

for a_poly in IDs:
    if (counter % 1000 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = no_outlier_df[no_outlier_df["ID"]==a_poly].copy()
    
    ################################################################
    # Sort by DoY (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)
    
    ################################################################

    no_Outlier_TS = nc.correct_big_jumps_1DaySeries_JFD(dataTMS_jumpie = curr_field, 
                                                        give_col = VI_idx, 
                                                        maxjump_perDay = 0.018)

    noJump_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
    counter += 1
    row_pointer += curr_field.shape[0]

noJump_df['human_system_start_time'] = pd.to_datetime(noJump_df['human_system_start_time'])

# Sanity check. Will neved occur. At least should not!
print ("Shape of noJump_df before dropping duplicates is {}.".format(noJump_df.shape)) 
noJump_df.drop_duplicates(inplace=True)
print ("Shape of noJump_df after dropping duplicates is {}.".format(noJump_df.shape))

del(no_Outlier_TS)

# %% [markdown] id="o8imjGKN0rdG"
# **Regularize**
#
# Here we regularize the data. "Regularization" means pick a value for every 10-days. Doing this ensures 
#
# 1.   all inputs have the same length, 
# 2.   by picking maximum value of a VI we are reducing the noise in the time-series by eliminating noisy data points. For example, snow or shaddow can lead to understimating the true VI.
#
# Moreover, here, I am keeping only 3 columns. As long as we have ```ID``` we can
# merge the big dataframe with the final result later, here or externally.
# This will reduce amount of memory needed. Perhaps I should do this
# right the beginning.

# %% colab={"base_uri": "https://localhost:8080/"} id="_uJicUns0lK0" outputId="e6619781-1726-4a42-a1e1-36be31150eb2"
# %%time

# define parameters
regular_window_size = 10
reg_cols = ['ID', 'human_system_start_time', VI_idx] # system_start_time list(noJump_df.columns)

st_yr = noJump_df.human_system_start_time.dt.year.min()
end_yr = noJump_df.human_system_start_time.dt.year.max()
no_days = (end_yr - st_yr + 1) * 366 # 14 years, each year 366 days!

no_steps = int(np.ceil(no_days / regular_window_size)) # no_days // regular_window_size

nrows = no_steps * len(IDs)
print('st_yr is {}.'.format(st_yr))
print('end_yr is {}.'.format(end_yr))
print('nrows is {}.'.format(nrows))
print (long_eq)


regular_df = pd.DataFrame(data = None,
                         index = np.arange(nrows), 
                         columns = reg_cols)
counter, row_pointer = 0, 0

for a_poly in IDs:
    if (counter % 1000 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = noJump_df[noJump_df["ID"]==a_poly].copy()
    ################################################################
    # Sort by date (sanitary check)
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)
    
    ################################################################
    regularized_TS = nc.regularize_a_field(a_df = curr_field, \
                                           V_idks = VI_idx, \
                                           interval_size = regular_window_size,\
                                           start_year = st_yr, \
                                           end_year = end_yr)
    
    regularized_TS = nc.fill_theGap_linearLine(a_regularized_TS = regularized_TS, V_idx = VI_idx)
    # if (counter == 0):
    #     print ("regular_df columns:",     regular_df.columns)
    #     print ("regularized_TS.columns", regularized_TS.columns)
    
    ################################################################
    # row_pointer = no_steps * counter
    
    """
       The reason for the following line is that we assume all years are 366 days!
       so, the actual thing might be smaller!
    """
    # why this should not work?: It may leave some empty rows in regular_df
    # but we drop them at the end.
    regular_df[row_pointer : (row_pointer+regularized_TS.shape[0])] = regularized_TS.values
    row_pointer += regularized_TS.shape[0]
    counter += 1

regular_df['human_system_start_time'] = pd.to_datetime(regular_df['human_system_start_time'])
regular_df.drop_duplicates(inplace=True)
regular_df.dropna(inplace=True)

# Sanity Check
regular_df.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
regular_df.reset_index(drop=True, inplace=True)

del(noJump_df)

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="VzU_nXps0ovH" outputId="ebb4549a-0eb4-49ae-bb07-f9c70813c1b9"
# Pick a field
a_field = regular_df[regular_df.ID==reduced.ID.unique()[0]].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3), sharex='col', sharey='row',
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'], 
        a_field[VI_idx], 
        linestyle='-', label=VI_idx, linewidth=3.5, color="dodgerblue", alpha=0.8)

ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);

# %% [markdown] id="u9xgFN960zr2"
# **Savitzky-Golay Smoothing**

# %% colab={"base_uri": "https://localhost:8080/"} id="iEdq5raA0w2t" outputId="e31c68fd-e306-4f8a-b9e2-1d0c13c975c8"
# %%time
counter = 0
window_len, polynomial_order = 7, 3

for a_poly in IDs:
    if (counter % 300 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = regular_df[regular_df["ID"]==a_poly].copy()
    
    # Smoothen by Savitzky-Golay
    SG = scipy.signal.savgol_filter(curr_field[VI_idx].values, window_length=window_len, polyorder=polynomial_order)
    SG[SG > 1 ] = 1 # SG might violate the boundaries. clip them:
    SG[SG < -1 ] = -1
    regular_df.loc[curr_field.index, VI_idx] = SG
    counter += 1

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="K1SHuof_02b6" outputId="7b134ed1-610a-4339-b057-c224fb39f3ec"
#  Pick a field
an_ID = reduced.ID.unique()[0]
a_field = regular_df[regular_df.ID==an_ID].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3),
                       sharex='col', sharey='row',
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'], 
        a_field[VI_idx], 
        linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8,
        label=f"smooth {VI_idx}")

# Raw data where we started from
raw = reduced[reduced.ID==an_ID].copy()
raw.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
ax.scatter(raw['human_system_start_time'], raw[VI_idx], s=15, c='#d62728', label=f"raw {VI_idx}");

ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} id="8SVIMiy004w2" outputId="dec75fdd-6e43-4bdb-9699-21cf41d032e1"
regular_df['human_system_start_time'] = pd.to_datetime(regular_df['human_system_start_time'])
# regular_df = pd.merge(regular_df, SF_data, on=['ID'], how='left') # we can do this later.
regular_df.reset_index(drop=True, inplace=True)
regular_df = nc.initial_clean(df=regular_df, column_to_be_cleaned = VI_idx
                              )
regular_df.head(2)

# %% [markdown] id="HzbcsPAQ084B"
# **Widen the data to use with ML**

# %% id="upZ5XzrY0787"
VI_colnames = [VI_idx + "_" + str(ii) for ii in range(1, 37)]
columnNames = ["ID", "year"] + VI_colnames

years = regular_df.human_system_start_time.dt.year.unique()
IDs = regular_df.ID.unique()
no_rows = len(IDs) * len(years)

data_wide = pd.DataFrame(columns=columnNames, index=range(no_rows))
data_wide.ID = list(IDs) * len(years)
data_wide.sort_values(by=["ID"], inplace=True)
data_wide.reset_index(drop=True, inplace=True)
data_wide.year = list(years) * len(IDs)

for an_ID in IDs:
    curr_field = regular_df[regular_df.ID == an_ID]
    curr_years = curr_field.human_system_start_time.dt.year.unique()
    for a_year in curr_years:
        curr_field_year = curr_field[curr_field.human_system_start_time.dt.year == a_year]

        data_wide_indx = data_wide[(data_wide.ID == an_ID) & (data_wide.year == a_year)].index

        if VI_idx == "EVI":
            data_wide.loc[data_wide_indx, "EVI_1":"EVI_36"] = curr_field_year.EVI.values[:36]
        elif VI_idx == "NDVI":
            data_wide.loc[data_wide_indx, "NDVI_1":"NDVI_36"] = curr_field_year.NDVI.values[:36]

# %% [markdown] id="T461-rF11DbT"
# # Please tell me where to look for the trained models and I will make you happy!

# %% id="1kdcR1bj1Ati"
model_dir = "/content/drive/MyDrive/NASA_trends/Models_Oct17/"


# %% id="eT-jcv-c1MBg"

# %% [markdown] id="QhLXtaty1MoW"
# # Functions we need
# We need the following two functions in case we want to use K-Nearest Neighbor or Deep Learning.
#
# -    Traditionnaly, images of dogs and cats are saved on the disk and read from there. Here I must try to figure out how to do them on the fly.

# %% id="grq_-P_O1JWd"
# We need this for KNN
def DTW_prune(ts1, ts2):
    d, _ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True)
    return d


# We need this for DL
# load and prepare the image
def load_image(filename):
    img = load_img(filename, target_size=(224, 224))  # load the image
    img = img_to_array(img)  # convert to array
    img = img.reshape(1, 224, 224, 3)  # reshape into a single sample with 3 channels
    img = img.astype("float32")  # center pixel data
    img = img - [123.68, 116.779, 103.939]
    return img



# %% id="ylxFGZbL1Jw_"
import pickle
model = "SVM"
winnerModel = "SG_NDVI_SVM_NoneWeight_00_Oct17_AccScoring_Oversample_SR3.sav"

if winnerModel.endswith(".sav"):
    # f_name = VI_idx + "_" + smooth + "_intersect_batchNumber" + batch_no + "_wide_JFD.csv"
    # data_wide = pd.read_csv(in_dir + f_name)
    # print("data_wide.shape: ", data_wide.shape)

    ML_model = pickle.load(open(model_dir + winnerModel, "rb"))
    predictions = ML_model.predict(data_wide.iloc[:, 2:])
    pred_colName = model + "_" + VI_idx  + "_preds"
    A = pd.DataFrame(columns=["ID", "year", pred_colName])
    A.ID = data_wide.ID.values
    A.year = data_wide.year.values
    A[pred_colName] = predictions
    predictions = A.copy()
    del A

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="-pu0gN0W1VrS" outputId="4e9aa2d7-6726-43b9-fe3b-4584d47559f9"
predictions = pd.merge(predictions, SF_data, on=['ID'], how='left')
predictions

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="7CUQS-_C1ZQ4" outputId="ba6135d2-0db0-417c-9198-5812b7770ef8"
#  Pick a field
an_ID = reduced.ID.unique()[0]
a_field = regular_df[regular_df.ID==an_ID].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3),
                       sharex='col', sharey='row',
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'], 
        a_field[VI_idx], 
        linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8,
        label=f"smooth {VI_idx}")

# Raw data where we started from
raw = reduced[reduced.ID==an_ID].copy()
raw.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
ax.scatter(raw['human_system_start_time'], raw[VI_idx], s=15, c='#d62728', label=f"raw {VI_idx}");

ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);

# %% id="oaj83_LS1b2U"
