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

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/HNoorazar/Ag/blob/master/Perry_1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="AbobjLfNKB0O"
# #**There is no free lunch**
#
# [Colab resources are not guaranteed](https://research.google.com/colaboratory/faq.html#:~:text=Colab\%20is\%20able\%20to\%20provide,other\%20factors\%20vary\%20over\%20time.). One can, however, [subscribe and increase resources]((https://colab.research.google.com/signup)) at his disposal.
#
# Creating an App also needs $ and human time to create it. Please look at [Quotas and limits](https://cloud.google.com/appengine/docs/standard#:~:text=The%20standard%20environment%20gives%20you,suit%20your%20needs%2C%20see%20Quotas.) section as a starting point!
#
# I ran different counties as a separate jobs and it took hours. If they want to run all counties at once, the cost (computation or $) is even higher.
#
# <p>&nbsp;</p>

# %% [markdown] id="c9MAEcsBblEc"
# # Print Local Time 
#
# colab runs on cloud. So, the time is not our local time.
# This page is useful to determine how to do this.

# %% colab={"base_uri": "https://localhost:8080/"} id="y_XZYSOXbi2S" outputId="fabc2730-bd61-4456-99e3-c863c34af3b3"
# !rm /etc/localtime
# !ln -s /usr/share/zoneinfo/US/Pacific /etc/localtime
# !date

# %% [markdown] id="jX3TvMJBe3lZ"
# # **geopandas and geemap must be installed every time!!!**
#

# %% colab={"base_uri": "https://localhost:8080/"} id="xMCOiQUYe6XS" outputId="3e126c12-4eda-4dfb-ba61-da3ecfe3fda0"
# # !pip install geopandas geemap
# Installs geemap package
import subprocess

try:
    import geemap
except ImportError:
    print('geemap not installed. Must be installed every tim to run this notebook. Installing ...')
    subprocess.check_call(["python", '-m', 'pip', 'install', 'geemap'])

    print('geopandas not installed. Must be installed every time to run this notebook. Installing ...')
    subprocess.check_call(["python", '-m', 'pip', 'install', 'geopandas'])

# %% [markdown] id="4_4IBtM4ffpt"
# # **Authenticate and import libraries**
#
# We have to impor tthe libraries we need. Moreover, we need to Authenticate every single time!

# %% colab={"base_uri": "https://localhost:8080/", "height": 179} id="ZgTaTGby3lWZ" outputId="dcc40859-f667-4198-b00d-a81480861bde"
import numpy as np
import folium
import geopandas as gpd
import json, geemap, ee
import pandas as pd

import scipy # we need this for savitzky-golay

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from datetime import date
import datetime
import time

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# %% [markdown] id="wh4w6cCn71MV"
# ### **Mount Google Drive and import my Python modules**
#
# Here we are importing the Python functions that are written by me and are needed; ```NASA core``` and ```NASA plot core```.
#
# **Note:** These are on Google Drive now. Perhaps we can import them from GitHub.

# %% colab={"base_uri": "https://localhost:8080/"} id="RmVZ8vYg6zWz" outputId="599d99e2-9848-48d3-bed4-5294fe898985"
# Mount YOUR google drive in Colab
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0,"/content/drive/My Drive/Colab Notebooks/")
import NASA_core as nc
import NASA_plot_core as ncp
import GEE_Python_core as gpc

# %% [markdown] id="cErRXmUS2ips"
# # **Change Current directory to the Colab folder on Google Drive**

# %% id="rD5tBhc42ho2"
import os
os.chdir("/content/drive/My Drive/Colab Notebooks/") # Colab Notebooks
# # !ls

# %% id="wDKuzyJh64ZT"
# import os
# os.chdir("/content/drive/")
# # !ls

# os.chdir("/content/drive/MyDrive/NASA_trends/shapefiles/Grant2017/")
# # !ls
# os.getcwd()
# os.listdir("/content/drive/MyDrive/NASA_trends/shapefiles/")

# %% [markdown] id="3pa0Uyu9vnZ4"
# # Please tell me where to look for the shapefile!

# %% id="I3Lqds7fu0Jk"
# shp_path = "/Users/hn/Documents/01_research_data/NASA/shapefiles/Grant2017/Grant2017.shp"
# shp_path = "/content/My Drive/NASA_trends/shapefiles/Grant2017/Grant2017.shp"
shp_path = "/content/drive/MyDrive/NASA_trends/shapefiles/Grant2017/Grant2017.shp"
shp_path = "/content/drive/MyDrive/NASA_trends/shapefiles/Grant_4Fields_poly_wCentroids/Grant_4Fields_poly_wCentroids.shp"

# we read our shapefile in to a geopandas data frame using the geopandas.read_file method
# we'll make sure it's initiated in the EPSG 4326 CRS
Grant_4Fields_poly_wCentroids = gpd.read_file(shp_path, crs='EPSG:4326')

# define a helper function to put the geodataframe in the right format for constructing an ee object
# The following function and the immediate line after that works for 1 geometry. not all the fields in the shapefile.
# def shp_to_ee_fmt(geodf):
#     data = json.loads(geodf.to_json())
#     return data['features'][0]['geometry']['coordinates']
# Grant_4Fields_poly_wCentroids = ee.Geometry.MultiPolygon(shp_to_ee_fmt(Grant_4Fields_poly_wCentroids))

# Grant_4Fields_poly_wCentroids = ee.FeatureCollection(Grant_4Fields_poly_wCentroids)

# %% id="pFiiiP_YR5DI"
unwanted_columns = ['cntrd_ln', 'cntrd_lt', 'CropGrp', 'Shap_Ar', 'Shp_Lng', 'ExctAcr', 
                    'RtCrpTy', 'TRS', 'Notes', 'IntlSrD']
Grant_4Fields_poly_wCentroids = Grant_4Fields_poly_wCentroids.drop(columns=unwanted_columns)

# %% colab={"base_uri": "https://localhost:8080/", "height": 183} id="Un5vX6oF2qZx" outputId="dcfab0e5-92e4-4ced-ccc7-86dfa155167c"
print (type(Grant_4Fields_poly_wCentroids))
print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print ("Shape of Grant_4Fields_poly_wCentroids is", Grant_4Fields_poly_wCentroids.shape)
print ("==============================================================================")
Grant_4Fields_poly_wCentroids.head(2)

# %% [markdown] id="rgU4oL8K3MV7"
# # **Form Geographical Regions**
#
#   - First, define a big region that covers Eastern Washington.
#   - Convert shapefile to ```ee.featurecollection.FeatureCollection```.
#

# %% id="rgU1qGvx5jAM"
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
SF = geemap.geopandas_to_ee(Grant_4Fields_poly_wCentroids)


# %% [markdown] id="NdAnD4s4ASty"
# ## **WARNING!**
# For some reason the function ```feature2ee(.)``` does not work ***when*** it is imported from ```core``` module. (However, it works when it is directly written here!!!!) So, What the hell will happen with the rest of functions, e.g. smoothing functions, we want to use here?

# %% id="VlsU6nkK7Joz"
# was named "banke" in https://bikeshbade.com.np/tutorials/Detail/?title=Geo-pandas%20data%20frame%20to%20GEE%20feature%20collection%20using%20Python&code=13
# Grant_4Fields_poly_wCentroids_EEFC_from_Func = feature2ee(shp_path)
# Grant_4Fields_poly_wCentroids_EEFC_from_Func = gpc.feature2ee(shp_path)

# %% [markdown] id="E9hYtdeW5ezU"
# # **Visualize the big region encompassing the Eastern Washington**

# %% colab={"base_uri": "https://localhost:8080/", "height": 621, "referenced_widgets": ["f769e4f1def44dcba1ab317467029ad8", "4137e8d9335641469bc9f05a0d94deab", "f7469d998aa04f51ba8e17b73f7167fd", "19657da9b0184f259c5f6b19e61736cb", "60ce4794a2b4474eac37a9b837bfa862", "92d0475b5edb4add9789dd97ce560bad", "480cd7a5dfca45f3ae4cc4a958af5d06", "2595cf20f9da4f0fae8f09ad56ca2aa5", "c60a7fa072ce4f56ab251ddf3890e2b0", "77e14063b4f243dea2ad461299463343", "6c6f2eeb31624d47b5c1913f0e1adb75", "ab412a07d9ed4f1b8bfdba5244cc0ec4", "c87ec8d0e9a34c619e3ab11ebde86f65", "43b9394d2fde448e98762cd8a9290783", "7b6b402820da48b0b93755d4928a0225", "c7aa5f58777a4503aaf899233a3a8f05", "d3239937f43145f29f3e07751615172c", "d029807425524543ac4f8f925cb78eab", "8c4c9c5ed28940a0a48f1bd61fbb8826", "5f699bfc69794f0eadeb1b0ef9a95c49", "10b8145597c84aa09109491bfc1b6f27", "027961f5b9214e75943fcdcddcedcd0b", "96b87350f2ee4b229c99942703748dbd", "1e6ad4ded7a44d1fa64ec5c39c73ae6b", "87768fac5fef491e9c25ce5f0a22ce3b", "b4aa146c29e549069e5a6a329c4dc914", "4bd6f1ee0d994c0f93fbb2f6c3954e5e", "34d15103e4184ef3be4d903fac42a510"]} id="mVmBWoEb5fAm" outputId="e1451d02-883d-4bee-8a31-c40f52fb9bcd"
Map = geemap.Map(center=[ymed, xmed], zoom=7)
Map.addLayer(WA1, {'color': 'red'}, 'Western Half')
Map.addLayer(WA2, {'color': 'red'}, 'Eastern Half')
Map.addLayer(SF, {'color': 'blue'}, 'Fields')
Map

# %% [markdown] id="suZ4slkBhXrM"
# # **Define Parameters**

# %% id="1ZPc-HNuhatk"
start_date = "2017-01-01" # Date fromat for EE YYYY-MM-DD
end_date = "2017-12-30"   # Date fromat for EE YYYY-MM-DD
cloud_perc = 70

# %% colab={"base_uri": "https://localhost:8080/"} id="LpZWNdHzPubb" outputId="a1c6d30e-6905-49c5-8f16-ed932bf50141"
imageC = gpc.extract_sentinel_IC(big_rectangle, start_date, end_date, cloud_perc);
print ("The size of [imageC.size().getInfo()] is [{:.0f}].".format(imageC.size().getInfo()))
reduced = gpc.mosaic_and_reduce_IC_mean(imageC, SF, start_date, end_date)

# %% [markdown] id="h2ifGNw07cd6"
# # **Export output to Google Drive, only if you want!**

# %% id="TmOxua9os46Y"
export_raw_data = False

if export_raw_data==True:
  outfile_name = "Grant_4Fields_poly_wCentroids_colab_output"
  task = ee.batch.Export.table.toDrive(**{
                                      'collection': reduced,
                                      'description': outfile_name,
                                      'folder': "colab_outputs",
                                      'selectors':["ID", "Acres", "county", "CropTyp", "DataSrc", \
                                                    "Irrigtn", "LstSrvD", "EVI", 'NDVI', "system_start_time"],
                                      'fileFormat': 'CSV'})
  task.start()

  import time 
  while task.active():
    print('Polling for task (id: {}). Still breathing'.format(task.id))
    time.sleep(59)

# %% [markdown] id="_56djzTM9cdC"
# #**Smooth the data**
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

# %% colab={"base_uri": "https://localhost:8080/"} id="InWUd6YzwEFo" outputId="78cf261e-83ef-4150-fe40-f800b328e04e"
# See how long it takes to convert a FeatureCollection to dataframe!
# %%time
needed_columns = ["ID", "Acres", "county", "CropTyp", "DataSrc", \
                  "Irrigtn", "LstSrvD", "EVI", 'NDVI', "system_start_time"]
reduced = geemap.ee_to_pandas(reduced, selectors=needed_columns)

reduced = reduced[needed_columns]
reduced.head(2)

# %% [markdown] id="eYGNYXXFXduS"
# # Isolate the ```data``` part of the ```shapefile``` for future use.

# %% colab={"base_uri": "https://localhost:8080/"} id="PIHAOvE4XdN9" outputId="86488ced-eabb-437e-9ac6-277ede2b22e9"
SF_data = reduced[["ID", "Acres", "county", "CropTyp", \
                   "DataSrc", "Irrigtn", "LstSrvD"]].copy()
SF_data.drop_duplicates(inplace=True)
SF_data.shape

# %% [markdown] id="fK8dO0yzc94q"
# # NA removal
#
# Even though logically and intuitively all the bands should be either available or ```NA```, I have seen before that sometimes ```EVI``` is NA while ```NDVI``` is not. Therefore, I had to choose which VI we want to use so that we can clean the data properly. However, I did not see that here.  when I was testing this code for 4 fields.
#
# Another suprising observation was that the output of Colab had more data compared to its JS counterpart!!!

# %% id="Nf5Athf2jlRe"
# reduced = reduced[reduced["system_start_time"].notna()]
reduced = reduced[reduced["EVI"].notna()]
reduced.reset_index(drop=True, inplace=True)

# %% [markdown] id="SWE3Cw8ChnNs"
# # Add human readable time to the dataframe

# %% colab={"base_uri": "https://localhost:8080/"} id="1IhNPSPCjwK4" outputId="acb52a87-4264-4c7d-acfd-2038eda0902f"
reduced = nc.add_human_start_time_by_system_start_time(reduced)
reduced.head(2)
reduced.loc[0, "system_start_time"]

# %% [markdown] id="yEOetbtbiEZZ"
# # Make a plot for fun.

# %% colab={"base_uri": "https://localhost:8080/", "height": 211} id="maPiNLpubCtE" outputId="02f0bf7e-04b8-459f-a363-8ea67df05bfc"
#  Pick a field
a_field = reduced[reduced.ID==reduced.ID.unique()[0]].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(20, 3),
                       sharex='col', sharey='row',
                       # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);

ax.scatter(a_field['human_system_start_time'], a_field["EVI"], s=40, c='#d62728');
ax.plot(a_field['human_system_start_time'], a_field["EVI"], 
        linestyle='-',  linewidth=3.5, color="#d62728", alpha=0.8,
        label="raw EVI")
plt.ylim([-0.5, 1.2]);
ax.legend(loc="lower right");


# %% [markdown] id="sBUFPr-Zx2VH"
# # Efficiency 
#
# We can and should make this more efficient. Do some of the calculations in place as opposed to creating a new ```dataframe``` and copying stuff.
#
# #**Remove outliers**

# %% id="PschCyKp27bK"
reduced["ID"] = reduced["ID"].astype(str)
IDs = np.sort(reduced["ID"].unique())
indeks = "EVI"

# %% colab={"base_uri": "https://localhost:8080/"} id="D4JDoe0Sf3YO" outputId="448f2248-26b4-4386-b9cc-37a522e0a85a"
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
        """
         we are killing some of the ourliers here and put them
         in the normal range! do we want to do it here? No, lets do it later.
        """
        # curr_field.loc[curr_field[indeks] < 0 , indeks] = 0 
        no_Outlier_TS = nc.interpolate_outliers_EVI_NDVI(outlier_input = curr_field, given_col = indeks)
        no_Outlier_TS.loc[no_Outlier_TS[indeks] < 0 , indeks] = 0 

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

# %% [markdown] id="bygnLLy082lH"
# #**Remove the damn jumps**
#
# Maybe we can remove old/previous dataframes to free memory up!

# %% colab={"base_uri": "https://localhost:8080/"} id="9DEdxPAXgYsR" outputId="31ad9bb0-6f52-46ed-bab4-a6a2f80af74a"
noJump_df = pd.DataFrame(data = None,
                         index = np.arange(no_outlier_df.shape[0]), 
                         columns = no_outlier_df.columns)
counter = 0
row_pointer = 0

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
                                                        give_col = indeks, 
                                                        maxjump_perDay = 0.018)

    noJump_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
    counter += 1
    row_pointer += curr_field.shape[0]

noJump_df['human_system_start_time'] = pd.to_datetime(noJump_df['human_system_start_time'])

# Sanity check. Will neved occur. At least should not!
print ("Shape of noJump_df before dropping duplicates is {}.".format(noJump_df.shape)) 
noJump_df.drop_duplicates(inplace=True)
print ("Shape of noJump_df before dropping duplicates is {}.".format(noJump_df.shape))

# %% [markdown] id="RnrloDSm-7qr"
# # **Regularize**
#
# Here we regularize the data. "Regularization" means pick a value for every 10-days. Doing this ensures 1. all inputs have the same length, 2. by picking maximum value of a VI we are reducing the noise in the time-series by eliminating noisy data points. For example, snow or shaddow can lead to understimating the true VI.
#
# Moreover, here, I am keeping only 3 columns. As long as we have ```ID``` we can
# merge the big dataframe with the final result later, here or externally.
# This will reduce amount of memory needed. Perhaps I should do this
# right the beginning.

# %% colab={"base_uri": "https://localhost:8080/"} id="b4uaS8yf3qjc" outputId="6a91db93-b021-4275-bc1b-9c36ec345695"
# define parameters
regular_window_size = 10
reg_cols = ['ID', 'human_system_start_time', indeks] # system_start_time list(noJump_df.columns)

st_yr = noJump_df.human_system_start_time.dt.year.min()
end_yr = noJump_df.human_system_start_time.dt.year.max()
no_days = (end_yr - st_yr + 1) * 366 # 14 years, each year 366 days!

no_steps = int(np.ceil(no_days / regular_window_size)) # no_days // regular_window_size

nrows = no_steps * len(IDs)
print('st_yr is {}.'.format(st_yr))
print('end_yr is {}.'.format(end_yr))
print('nrows is {}.'.format(nrows))

# %% colab={"base_uri": "https://localhost:8080/"} id="WcwcEb4p3uB9" outputId="c7721882-eed1-407b-bd67-4fbad3952ff6"
regular_df = pd.DataFrame(data = None,
                         index = np.arange(nrows), 
                         columns = reg_cols)
counter = 0
row_pointer = 0

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
                                           V_idks = indeks, \
                                           interval_size = regular_window_size,\
                                           start_year = st_yr, \
                                           end_year = end_yr)
    
    regularized_TS = nc.fill_theGap_linearLine(a_regularized_TS = regularized_TS, V_idx = indeks)
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

    # right_pointer = row_pointer + min(no_steps, regularized_TS.shape[0])
    # print('right_pointer - row_pointer + 1 is {}!'.format(right_pointer - row_pointer + 1))
    # print('len(regularized_TS.values) is {}!'.format(len(regularized_TS.values)))
    # try:
    #     ### I do not know why the hell the following did not work for training set!
    #     ### So, I converted this to try-except statement! hopefully, this will
    #     ### work, at least as temporary remedy! Why it worked well with 2008-2021 but not 2013-2015
    #     regular_df[row_pointer: right_pointer] = regularized_TS.values
    # except:
    #     regular_df[row_pointer: right_pointer+1] = regularized_TS.values
    counter += 1

regular_df['human_system_start_time'] = pd.to_datetime(regular_df['human_system_start_time'])
regular_df.drop_duplicates(inplace=True)
regular_df.dropna(inplace=True)

# Sanity Check
regular_df.sort_values(by=["ID", 'human_system_start_time'], inplace=True)
regular_df.reset_index(drop=True, inplace=True)


# %% colab={"base_uri": "https://localhost:8080/", "height": 211} id="1B498-me7mMq" outputId="22db33e2-02c6-487a-cfba-ed9198bf23bd"
#  Pick a fields
a_field = regular_df[regular_df.ID==reduced.ID.unique()[0]].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(20, 3), sharex='col', sharey='row',
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'], 
        a_field["EVI"], 
        linestyle='-', label="EVI", linewidth=3.5, color="dodgerblue", alpha=0.8)

ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);

# %% [markdown] id="ram0TU5bEfqE"
# # **Savitzky-Golay Smoothing**

# %% colab={"base_uri": "https://localhost:8080/"} id="f4QosTqVBH2B" outputId="a3d0ca49-91af-4eae-907c-4bb6d4f51d20"
counter = 0
window_len = 7
polynomial_order = 3

for a_poly in IDs:
    if (counter % 300 == 0):
        print ("counter is [{:.0f}].".format(counter))
    curr_field = regular_df[regular_df["ID"]==a_poly].copy()
    
    # Smoothen by Savitzky-Golay
    SG = scipy.signal.savgol_filter(curr_field[indeks].values, window_length=window_len, polyorder=polynomial_order)
    SG[SG > 1 ] = 1 # SG might violate the boundaries. clip them:
    SG[SG < -1 ] = -1
    regular_df.loc[curr_field.index, indeks] = SG
    counter += 1

# %% colab={"base_uri": "https://localhost:8080/", "height": 211} id="yboIMfeWC_WH" outputId="b98d244d-2f33-47ad-ad52-96d346cf9bd2"
#  Pick a fields
an_ID = reduced.ID.unique()[0]
a_field = regular_df[regular_df.ID==an_ID].copy()
a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(20, 3),
                       sharex='col', sharey='row',
                       gridspec_kw={'hspace': 0.2, 'wspace': .05});
ax.grid(True);
ax.plot(a_field['human_system_start_time'], 
        a_field["EVI"], 
        linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8,
        label="smooth EVI")

# Raw data where we started from
raw = reduced[reduced.ID==an_ID].copy()
raw.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)
ax.scatter(raw['human_system_start_time'], raw["EVI"], s=15, c='#d62728', label="raw EVI");

ax.legend(loc="lower right");
plt.ylim([-0.5, 1.2]);

# %% [markdown] id="NSSjzF4dG62S"
# # **Season Detection**

# %% id="sQV7Xn8VF0Du"
# Define parameters
onset_cut, offset_cut = 0.3, 0.3

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} id="i7Y6kG3gUZfQ" outputId="4f1e2fb3-4467-42c1-b4fd-301a404e3460"
regular_df['human_system_start_time'] = pd.to_datetime(regular_df['human_system_start_time'])
regular_df = pd.merge(regular_df, SF_data, on=['ID'], how='left')
regular_df.reset_index(drop=True, inplace=True)
regular_df = nc.initial_clean(df=regular_df, column_to_be_cleaned = indeks)
regular_df.head(2)

# %% id="Su7EIy_pWpDb"

# %% id="rQG-Xmd-W63k"
ratio_name = indeks + "_ratio"
SEOS_output_columns = ['ID', 'human_system_start_time', indeks,
                       ratio_name, 'SOS', 'EOS', 'season_count']

min_year = regular_df.human_system_start_time.dt.year.min()
max_year = regular_df.human_system_start_time.dt.year.max()
no_years = max_year - min_year + 1
all_poly_and_SEOS = pd.DataFrame(data = None, 
                                 index = np.arange(4*no_years*len(regular_df)), 
                                 columns = SEOS_output_columns)
counter = 0
pointer_SEOS_tab = 0

regular_df = regular_df[SEOS_output_columns[0:3]]

# %% id="alEfdzWdXzNS"

# %% colab={"base_uri": "https://localhost:8080/"} id="PrCIxuGYY9cl" outputId="8892b354-5cba-4488-f4c9-3933e5ec363c"
for a_poly in IDs:
    if (counter % 1000 == 0):
        print ("_________________________________________________________")
        print ("counter: " + str(counter))
        print (a_poly)
    curr_field = regular_df[regular_df['ID']==a_poly].copy()
    curr_field.sort_values(by=['human_system_start_time'], inplace=True)
    curr_field.reset_index(drop=True, inplace=True)

    # extract unique years. Should be 2008 thru 2021.
    # unique_years = list(pd.DatetimeIndex(curr_field['human_system_start_time']).year.unique())
    unique_years = curr_field['human_system_start_time'].dt.year.unique()
    
    """
    detect SOS and EOS in each year
    """
    for yr in unique_years:
        curr_field_yr = curr_field[curr_field['human_system_start_time'].dt.year == yr].copy()

        # Orchards EVI was between more than 0.3
        y_orchard = curr_field_yr[curr_field_yr['human_system_start_time'].dt.month >= 5]
        y_orchard = y_orchard[y_orchard['human_system_start_time'].dt.month <= 10]
        y_orchard_range = max(y_orchard[indeks]) - min(y_orchard[indeks])

        if y_orchard_range > 0.3:
            #######################################################################
            ###
            ###             find SOS and EOS, and add them to the table
            ###
            #######################################################################
            curr_field_yr = nc.addToDF_SOS_EOS_White(pd_TS = curr_field_yr, 
                                                     VegIdx = indeks, 
                                                     onset_thresh = onset_cut, 
                                                     offset_thresh = offset_cut)

            ##
            ##  Kill false detected seasons 
            ##
            curr_field_yr = nc.Null_SOS_EOS_by_DoYDiff(pd_TS=curr_field_yr, min_season_length=40)

            #
            # extract the SOS and EOS rows 
            #
            SEOS = curr_field_yr[(curr_field_yr['SOS'] != 0) | curr_field_yr['EOS'] != 0]
            SEOS = SEOS.copy()
            # SEOS = SEOS.reset_index() # not needed really
            SOS_tb = curr_field_yr[curr_field_yr['SOS'] != 0]
            if len(SOS_tb) >= 2:
                SEOS["season_count"] = len(SOS_tb)
                # re-order columns of SEOS so they match!!!
                SEOS = SEOS[all_poly_and_SEOS.columns]
                all_poly_and_SEOS[pointer_SEOS_tab:(pointer_SEOS_tab+len(SEOS))] = SEOS.values
                pointer_SEOS_tab += len(SEOS)
            else:
                # re-order columns of fine_granular_table so they match!!!
                curr_field_yr["season_count"] = 1
                curr_field_yr = curr_field_yr[all_poly_and_SEOS.columns]

                aaa = curr_field_yr.iloc[0].values.reshape(1, len(curr_field_yr.iloc[0]))
                
                all_poly_and_SEOS.iloc[pointer_SEOS_tab:(pointer_SEOS_tab+1)] = aaa
                pointer_SEOS_tab += 1
        else: 
            """
             here are potentially apples, cherries, etc.
             we did not add EVI_ratio, SOS, and EOS. So, we are missing these
             columns in the data frame. So, use 666 as proxy
            """
            aaa = np.append(curr_field_yr.iloc[0], [666, 666, 666, 1])
            aaa = aaa.reshape(1, len(aaa))
            all_poly_and_SEOS.iloc[pointer_SEOS_tab:(pointer_SEOS_tab+1)] = aaa
            pointer_SEOS_tab += 1

        counter += 1

####################################################################################
###
###                   Write the outputs
###
####################################################################################
# replace the following line with dropna.
# all_poly_and_SEOS = all_poly_and_SEOS[0:(pointer_SEOS_tab)]
all_poly_and_SEOS.dropna(inplace=True)

# %% id="QiPJbTHhCO-v"
all_poly_and_SEOS = pd.merge(all_poly_and_SEOS, SF_data, on=['ID'], how='left')

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="Sf2mf_fXCRfH" outputId="fee6fc16-a064-4f87-bc8b-6f30f699c6b2"
all_poly_and_SEOS

# %% id="SIKinbsjCcth"
F1_SEOS = all_poly_and_SEOS[all_poly_and_SEOS.ID == IDs[0]]
F2_SEOS = all_poly_and_SEOS[all_poly_and_SEOS.ID == IDs[1]]
F3_SEOS = all_poly_and_SEOS[all_poly_and_SEOS.ID == IDs[2]]
F4_SEOS = all_poly_and_SEOS[all_poly_and_SEOS.ID == IDs[3]]

F1_TS = regular_df[regular_df.ID == IDs[0]]
F2_TS = regular_df[regular_df.ID == IDs[1]]
F3_TS = regular_df[regular_df.ID == IDs[2]]
F4_TS = regular_df[regular_df.ID == IDs[3]]

# %% colab={"base_uri": "https://localhost:8080/", "height": 390} id="Q9Z8PKp-Ck46" outputId="7c637dcf-72e5-4e80-a4e2-499f8d0d0882"
fig, axs = plt.subplots(2, 2, figsize=(30,6),
                        sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': .1});

(ax1, ax2), (ax3, ax4) = axs;
ax1.grid(True); ax2.grid(True);
ax3.grid(True); ax4.grid(True);

ax1.plot(F1_TS['human_system_start_time'], F1_TS["EVI"], linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8, label="EVI")
ax1.set_title(IDs[0] + ", "+ str(F1_SEOS.season_count.unique()[0]) + " seasons");
ax1.set(ylabel="EVI");

ax2.plot(F2_TS['human_system_start_time'], F2_TS["EVI"], linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8, label="EVI")
ax2.set_title(IDs[0] + ", "+ str(F2_SEOS.season_count.unique()[0]) + " seasons");
ax2.set(ylabel="EVI");

ax3.plot(F3_TS['human_system_start_time'], F3_TS["EVI"], linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8, label="EVI")
ax3.set_title(IDs[0] + ", "+ str(F3_SEOS.season_count.unique()[0]) + " seasons");
ax3.set(ylabel="EVI");

ax4.plot(F4_TS['human_system_start_time'], F4_TS["EVI"], linestyle='-',  linewidth=3.5, color="dodgerblue", alpha=0.8, label="EVI")
ax4.set_title(IDs[0] + ", "+ str(F4_SEOS.season_count.unique()[0]) + " seasons");
ax4.set(ylabel="EVI");

# %% colab={"base_uri": "https://localhost:8080/", "height": 37} id="rojxAYi7E9zP" outputId="e5a31158-013f-4932-d202-c2ab12af042a"

# %% id="ibp81BJiFZrr"

# %% id="v3DqotXyFp3v"
