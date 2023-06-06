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

# %% [markdown] id="vYD4w-2swrtt"
# **Let the Fun Begin**
# [Colab resources are not guaranteed](https://research.google.com/colaboratory/faq.html#:~:text=Colab\%20is\%20able\%20to\%20provide,other\%20factors\%20vary\%20over\%20time.). One can, however, [subscribe and increase resources]((https://colab.research.google.com/signup)).
#
# <p>&nbsp;</p>
#
#
# split_large_feacturecollection_to_blocks

# %% colab={"base_uri": "https://localhost:8080/"} id="LTm6bITYj_pn" outputId="27475b77-f50a-4e8f-cac8-332ba781951f"
# %who

# %% id="rNWduYF8ELxV"
try:
    import shutup
except ImportError:
  # !pip install shutup
  import shutup

shutup.please() # kill some of the messages

# %% id="zxAcDFMxDwIm" colab={"base_uri": "https://localhost:8080/"} outputId="c2b5e62c-8140-49a7-e82c-b7709d95ab4b"
"""
Print Local Time 

colab runs on cloud. So, the time is not our local time.
This page is useful to determine how to do this.
"""
# !rm /etc/localtime
# !ln -s /usr/share/zoneinfo/US/Pacific /etc/localtime
# !date

# %% [markdown] id="Nyg94VHsEK8B"
# **geopandas and geemap must be installed every time!!!**
#

# %% id="F9b71mZbEZQj"
import subprocess

try:
    import geemap
except ImportError:
    print('geemap not installed. Must be installed every tim to run this notebook. Installing ...')
    subprocess.check_call(["python", '-m', 'pip', 'install', 'geemap'])

    print('geopandas not installed. Must be installed every time to run this notebook. Installing ...')
    subprocess.check_call(["python", '-m', 'pip', 'install', 'geopandas'])
    subprocess.check_call(["python", '-m', 'pip', 'install', 'google.colab'])

# %% [markdown] id="StQCEezUEajV"
# **Authenticate and import libraries**
#
# We have to import the libraries we need. Moreover, we need to Authenticate, every single time!

# %% id="vWONKBNXEpCD"
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

# %% [markdown] id="Magk2Zr_ExC1"
# ### **Mount Google Drive and import my Python modules**
#
# Here we are importing the Python functions that are written by me and are needed; ```NASA core``` and ```NASA plot core```.
#
# **Note:** These are on Google Drive now. Perhaps we can import them from GitHub.

# %% id="IxshCK3dE8lf" colab={"base_uri": "https://localhost:8080/"} outputId="4e0f0226-4bc9-44ee-d193-44a83c75ceec"
# Mount YOUR google drive in Colab
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0,"/content/drive/MyDrive/Colab Notebooks/")
import NASA_core as nc
import NASA_plot_core as ncp
import GEE_Python_core as gpc

# %% [markdown] id="72Hu_ab3E93Z"
# **Change Current directory to the Colab folder on Google Drive**

# %% id="UEA2gsPJFJ9X"
import os
os.chdir("/content/drive/MyDrive/Colab Notebooks/") # Colab Notebooks
# # !ls

# %% [markdown] id="FIrLJ4-pFQEz"
# # Please tell me where to look for the shapefile!

# %% id="juqgx2EpFUU1" colab={"base_uri": "https://localhost:8080/", "height": 130} outputId="0ef8651b-1349-459e-bf71-3212ae1a6b9e"
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

# SF = SF.drop(columns=["Notes", "TRS", "IntlSrD", "RtCrpTy", "Shp_Lng", "Shap_Ar","CropGrp","CropTyp","Acres","Irrigtn","LstSrvD","DataSrc","county","ExctAcr","cntrd_ln","cntrd_lt"])
SF = SF.drop(columns=["CropTyp","Acres","Irrigtn",
                      "LstSrvD","DataSrc","county","ExctAcr",
                      "cntrd_ln","cntrd_lt"])

SF.head(2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} id="KUWLcca7sE_j" outputId="3bcdf9f4-3c8f-4275-aebc-cf38891bb035"
SF_data.head(2)

# %% id="EzT2oOS5Fi61" colab={"base_uri": "https://localhost:8080/"} outputId="ac9c482b-669a-41c1-87a9-54785c5625df"
long_eq = "=============================================================================="
print (type(SF))
print (long_eq)
print (f"{SF.shape=}", )
print (long_eq)

# %% [markdown] id="0SnBN0VCPT-7"
# # **Drop extra useless columns. Saves space.**
#
#  Also, GEE behaves strangely. It has problem with Notes column before

# %% [markdown] id="JVOsJFXkFoCH"
# **Form Geographical Regions**
#
#   - First, define a big region that covers Eastern Washington.
#   - Convert shapefile to ```ee.featurecollection.FeatureCollection```.
#

# %% id="Ihy6wVxAHe_A"
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

# %% [markdown] id="_WohFsaWI3_w"
# # Visualize the big region encompassing the Eastern Washington

# %% id="n_VdDpeqI66Q"
# Map = geemap.Map(center=[ymed, xmed], zoom=7)
# Map.addLayer(WA1, {'color': 'red'}, 'Western Half')
# Map.addLayer(WA2, {'color': 'red'}, 'Eastern Half')
# Map.addLayer(SF.head(5), {'color': 'blue'}, 'Fields')
# Map

# %% [markdown] id="OuKWgSEXJD7B"
# # Define Parameters

# %% id="Og1nGEviJJ1_"
# start_date = "2017-01-01" # Date fromat for EE YYYY-MM-DD
# end_date = "2017-12-30"   # Date fromat for EE YYYY-MM-DD
start_date = str(2000) + '-01-01'; # YYYY-MM-DD
end_date =   str(2008) + '-01-01';   # YYYY-MM-DD

start_date = "2017-01-01" # Date fromat for EE YYYY-MM-DD
end_date = "2017-12-30"   # Date fromat for EE YYYY-MM-DD


# start_date = "2017-01-01" # Date fromat for EE YYYY-MM-DD
# end_date = "2017-12-30"   # Date fromat for EE YYYY-MM-DD

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

# %% [markdown] id="rqJXjNo1JWmR"
# **Fetch data from GEE**

# %% colab={"base_uri": "https://localhost:8080/"} id="4fOcXB4cJPcC" outputId="47b2c3ab-c5aa-4f2a-d00a-7367c73e6b82"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 89} id="9BuuzAHTlr9G" outputId="2cbab48e-fdc6-44e5-adfc-e34575b83ff2"
# %%time
imageC_sentinel = gpc.extract_sentinel_IC(big_rectangle, start_date, end_date, cloud_perc);
print ("The size of [imageC.size().getInfo()] is [{:.0f}].".format(imageC_sentinel.size().getInfo()))

reduced_sentinel = gpc.mosaic_and_reduce_IC_mean(imageC_sentinel, SF, start_date, end_date)


needed_columns = ["ID", "EVI", 'NDVI', "system_start_time"]

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

# %% [markdown] id="3g13uBmAJaE1"
# # Export output to Google Drive
#
# We advise you to do it. If Python/CoLab kernel dies, then,
# previous steps should not be repeated.

# %% id="rZ-1w3Ehb3iV"
export_raw_data = True

if export_raw_data==True:
    outfile_name = "Grant_4Fields_poly_wCentroids_colab_outputLandSat7Jun62023Test2_" + start_date + "_" + end_date + ".csv"
    reduced.to_csv('/content/drive/MyDrive/colab_outputs/'+ outfile_name)


# %% [markdown] id="JWEL1ff9JzI-"
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

# %% [markdown] id="0-r6CjbMSBx-"
# **NA removal**
#
# Even though logically and intuitively all the bands should be either available or ```NA```, I have seen before that sometimes ```EVI``` is NA while ```NDVI``` is not. Therefore, I had to choose which VI we want to use so that we can clean the data properly. However, I did not see that here.  when I was testing this code for 4 fields.
#
# Another suprising observation was that the output of Colab had more data compared to its JS counterpart!!!

# %% id="sSDOXhv0SKNb"
# reduced = reduced[reduced["system_start_time"].notna()]
reduced = reduced[reduced["EVI"].notna()]
reduced.reset_index(drop=True, inplace=True)

# %% [markdown] id="Z8GEg8bVSKqn"
# Add human readable time to the dataframe

# %% colab={"base_uri": "https://localhost:8080/"} id="TrlVRbS8SPe-" outputId="cd304948-c3a5-4d8f-fc3c-6b4b8fb85c0b"
reduced = nc.add_human_start_time_by_system_start_time(reduced)
reduced.head(2)
reduced.loc[0, "system_start_time"]

# %% [markdown] id="v6ohoPCBSZVL"
# Make a plot for fun.

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="sqXm2ZKkSedK" outputId="f9f0cbd2-b6c6-4a62-8168-130f3322f871"
#  Pick a field
a_field = reduced[reduced.ID==reduced.ID.unique()[0]].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 3),
                       sharex='col', sharey='row',
                       # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(axis='y', which="both")

ax.scatter(a_field['human_system_start_time'], a_field["EVI"], s=40, c='#d62728');
ax.plot(a_field['human_system_start_time'], a_field["EVI"], 
        linestyle='-',  linewidth=3.5, color="#d62728", alpha=0.8,
        label="raw EVI")
plt.ylim([-0.5, 1.2]);
ax.legend(loc="lower right");
# ax.set_title(a_field.CropTyp.unique()[0]);

# %% [markdown] id="zTYd9dqLSsxn"
# # Efficiency 
#
# Can we make this more efficient by doing the calculations in place as opposed to creating a new ```dataframe``` and copying stuff. Perhaps ```.map(.)``` too.
#
# **Remove outliers**

# %% id="bAHLlnE7Szjw"
reduced["ID"] = reduced["ID"].astype(str)
IDs = np.sort(reduced["ID"].unique())
VI_idx = "NDVI"


# %% colab={"base_uri": "https://localhost:8080/"} id="FAxGeCA-TEoF" outputId="67b91036-c2de-4e3f-eeb2-2bebe85839cc"
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

# %% [markdown] id="L0Bu6xyNTIN7"
# **Remove the jumps**
#
# Maybe we can remove old/previous dataframes to free memory up!

# %% colab={"base_uri": "https://localhost:8080/"} id="bn2-bHZ4Tfk9" outputId="e6719b9f-0907-4d34-a615-f61cd8a3e157"
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
                                                        give_col = VI_idx
                                                        , 
                                                        maxjump_perDay = 0.018)

    noJump_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
    counter += 1
    row_pointer += curr_field.shape[0]

noJump_df['human_system_start_time'] = pd.to_datetime(noJump_df['human_system_start_time'])

# Sanity check. Will neved occur. At least should not!
print ("Shape of noJump_df before dropping duplicates is {}.".format(noJump_df.shape)) 
noJump_df.drop_duplicates(inplace=True)
print ("Shape of noJump_df after dropping duplicates is {}.".format(noJump_df.shape))

# %% id="mG6cq37a28cc"
del(no_Outlier_TS)

# %% [markdown] id="evXDiJKxTjEQ"
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

# %% colab={"base_uri": "https://localhost:8080/"} id="fynzOVKCT3qE" outputId="f5e74877-190b-493e-b84e-a16a2cc0447b"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="ykCINQmQU9xd" outputId="67cd988c-695f-487b-d470-cdf8f23fd51a"
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

# %% [markdown] id="IjmgbKcDVTiw"
# **Savitzky-Golay Smoothing**

# %% colab={"base_uri": "https://localhost:8080/"} id="oVhXa8DwVegh" outputId="3559aa32-a7c7-43ca-9f09-ad4777836fe6"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="7MwDu6q2Vpji" outputId="43a20c8f-50cb-4eb8-c4dd-7544e3100b61"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} id="SpyopmVQWiFD" outputId="8c955b32-c51c-4d28-c2fd-38520bfcf9a9"
regular_df['human_system_start_time'] = pd.to_datetime(regular_df['human_system_start_time'])
# regular_df = pd.merge(regular_df, SF_data, on=['ID'], how='left') # we can do this later.
regular_df.reset_index(drop=True, inplace=True)
regular_df = nc.initial_clean(df=regular_df, column_to_be_cleaned = VI_idx
                              )
regular_df.head(2)

# %% [markdown] id="RCVxZ2CJtUkR"
# **Widen the data to use with ML**

# %% id="GwjpZeyCtdHB"
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

# %% [markdown] id="AYMI0RwcWjc7"
# # Please tell me where to look for the trained models and I will make you happy!

# %% id="4HwojA6Ppui5"
model_dir = "/content/drive/MyDrive/NASA_trends/Models_Oct17/"


# %% [markdown] id="Y4HqLXrws4Da"
# # Functions we need
# We need the following two functions in case we want to use K-Nearest Neighbor or Deep Learning.
#
# -    Traditionnaly, images of dogs and cats are saved on the disk and read from there. Here I must try to figure out how to do them on the fly.

# %% id="XSOHwvICs8p5"
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



# %% id="3iZ64rAvv6Uf"
model = "SVM"
winnerModel = "SG_NDVI_SVM_NoneWeight_00_Oct17_AccScoring_Oversample_SR3.sav"

# %% id="XVEyEblRwaI8"
import pickle

# %% id="cHr4PiBJwfQU"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="ctpSsiQdEyqd" outputId="d641de78-0de2-4969-8b4d-e28b6b8b1d50"
predictions

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="QKBAJpkLwJLj" outputId="7eede014-7f14-4b69-e62f-69f83c9f7074"
predictions = pd.merge(predictions, SF_data, on=['ID'], how='left')
predictions

# %% colab={"base_uri": "https://localhost:8080/", "height": 291} id="k4JyjzKv43zL" outputId="ce933ec6-5267-40d4-c3fa-a52e27fbbdc3"
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

# %% id="VKskUUEQ47Oz"
