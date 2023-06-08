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

# %% [markdown] id="C83zqLtO14eC"
# **Let the Fun Begin**
# [Colab resources are not guaranteed](https://research.google.com/colaboratory/faq.html#:~:text=Colab\%20is\%20able\%20to\%20provide,other\%20factors\%20vary\%20over\%20time.). One can, however, [subscribe and increase resources]((https://colab.research.google.com/signup)).
#
# <p>&nbsp;</p>
#
#
# split_large_feacturecollection_to_blocks

# %% colab={"base_uri": "https://localhost:8080/"} id="VnXIofeQ2G18" outputId="0fbf0310-3549-4199-e2b6-42359780be5f"
# %who

# %% colab={"base_uri": "https://localhost:8080/"} id="mrzgJSdo2KAP" outputId="2555a1b0-563b-48d4-ce3b-cc167020ad17"
try:
    import shutup
except ImportError:
  # !pip install shutup
  import shutup

shutup.please() # kill some of the messages

# %% colab={"base_uri": "https://localhost:8080/"} id="NxFOFNAQ2KQC" outputId="5d66fd7c-0355-4ecd-8a7a-14d6e9280e49"
"""
Print Local Time 

colab runs on cloud. So, the time is not our local time.
This page is useful to determine how to do this.
"""
# !rm /etc/localtime
# !ln -s /usr/share/zoneinfo/US/Pacific /etc/localtime # <--- The space after Pacific is needed
# !date

# %% [markdown] id="P1QEWzIE2Lpc"
# **geopandas and geemap must be installed every time!!!**

# %% colab={"base_uri": "https://localhost:8080/"} id="Ltd8-o6w2P_S" outputId="cead3bf9-2752-488a-ada5-95dd82abec1a"
import subprocess

try:
    import geemap
except ImportError:
    print('geemap not installed. Must be installed every tim to run this notebook. Installing ...')
    subprocess.check_call(["python", '-m', 'pip', 'install', 'geemap'])

    print('geopandas not installed. Must be installed every time to run this notebook. Installing ...')
    subprocess.check_call(["python", '-m', 'pip', 'install', 'geopandas'])
    subprocess.check_call(["python", '-m', 'pip', 'install', 'google.colab'])

# %% [markdown] id="B0brasq62QT6"
# **Authenticate and import libraries**
#
# We have to import the libraries we need. Moreover, we need to Authenticate, every single time!

# %% colab={"base_uri": "https://localhost:8080/"} id="1iqqx15m2XBq" outputId="30cd1a48-3d03-4281-92e2-5e95213d95af"
import pandas as pd
import numpy as np
import geopandas as gpd

import folium, time, datetime, json, geemap, ee
# from datetime import date

# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# %% [markdown] id="s9tT6Qpf2b6x"
# ### **Mount Google Drive and import my Python modules**
#
# Here we are importing the Python functions that are written by me and are needed; ```NASA core``` and ```NASA plot core```.
#
# **Note:** These are on Google Drive now. Perhaps we can import them from GitHub.

# %% colab={"base_uri": "https://localhost:8080/"} id="dr0Had6v2fc5" outputId="afb5ff02-ff0a-4178-e5e0-ed59af6d0ff2"
# Mount YOUR google drive in Colab
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0, "/content/drive/My Drive/Colab Notebooks/")
import NASA_core as nc
import NASA_plot_core as ncp
import GEE_Python_core as gpc

# %% [markdown] id="ngw31sTgxqJV"
# **Please tell me where to look for the shapefile and other directories**

# %% id="LsYrpCB2xqMd"
drive_pre     = "/content/drive/My Drive/"
shp_path_base = drive_pre + "NASA_trends/shapefiles/"
out_dir_base  = drive_pre + "colab_outputs/"
Colab_NB_dir  = drive_pre + "Colab Notebooks/"

import os
os.chdir(Colab_NB_dir)
# # !ls

# %% colab={"base_uri": "https://localhost:8080/"} id="QLnLaFTPFqjs" outputId="5d5026e4-78ad-4180-e6e3-9d85e295cc4c"
# %who

# %% colab={"base_uri": "https://localhost:8080/", "height": 165} id="LBF0GPOS2-wQ" outputId="85b3800a-cfae-4a7b-b6f1-b4f18039cf1a"
# %%time

SF_name = "10_intersect_East_Irr_2008_2018_2cols"
# SF_name = "Grant_4Fields_poly_wCentroids"
shp_path = shp_path_base + SF_name + "/" + SF_name + ".shp"

"""
  we read our shapefile in to a geopandas dataframe using 
  the geopandas.read_file method
  we'll make sure it's initiated in the EPSG 4326 CRS
"""
SF = gpd.read_file(shp_path, crs='EPSG:4326')

### for possible future use grab the data part of the shapefile
# SF_data = SF[["ID", "Acres", "county", "CropTyp", "DataSrc", "Irrigtn", "LstSrvD"]].copy()
SF_data = SF[["ID", "acreage"]].copy()
SF_data.drop_duplicates(inplace=True)
print (f"{SF_data.shape=}")

""" 
   Drop extra useless columns. Saves space.
   Also, GEE behaves strangely. It had problem with Notes column before
"""
# SF = SF.drop(columns=["Notes", "TRS", "IntlSrD", "RtCrpTy", "Shp_Lng", "Shap_Ar","CropGrp","CropTyp","Acres","Irrigtn","LstSrvD","DataSrc","county","ExctAcr","cntrd_ln","cntrd_lt"])
# SF = SF.drop(columns=["CropTyp","Acres","Irrigtn", "LstSrvD","DataSrc","county","ExctAcr", "cntrd_ln","cntrd_lt"])
SF = SF.drop(columns=["acreage"])
SF.head(2)

# %% [markdown] id="UHzat7iJ3DYq"
# **Form Geographical Regions**
#
#   - First, define a big region that covers Eastern Washington.
#   - Convert shapefile to ```ee.featurecollection.FeatureCollection```.
#

# %% id="2Ulg6lxU3Gmx"
xmin, xmax = -125.0, -116.0
ymin, ymax = 45.0, 49.0

xmed = (xmin + xmax) / 2.0;
ymed = (ymin+ymax) / 2.0;

WA1 = ee.Geometry.Polygon([[xmin, ymin], [xmin, ymax], [xmed, ymax], [xmed, ymin], [xmin, ymin]]);
WA2 = ee.Geometry.Polygon([[xmed, ymin], [xmed, ymax], [xmax, ymax], [xmax, ymin], [xmed, ymin]]);
WA = [WA1, WA2];
big_rectangle = ee.FeatureCollection(WA);

# %% colab={"base_uri": "https://localhost:8080/", "height": 621, "referenced_widgets": ["e40e9d64e44847aca5ac83e9e0aaaf72", "e9cf2424a403470189823bbb4a654c5f", "5049730e66034bf992b8f1f77acbc176", "97f45bcf5a114cac847a3021f28c408a", "2a47137e3f5047b58e420c6872480cf7", "46c416064689400c82d64decfb1f206c", "d7a1d3af96af44408a47cf55199dec0a", "de5b0318d18d4e468ee3d7a2385311b0", "032d63d14b90407dbcaee7106ff68a34", "eab8ad830ab84ff2a76248cf1634ed36", "5d01a0741f574774b0aedc080beddcbb", "9fb728f0eb7b4a529672c5788a27675f", "792ba332382b4c6693715c7579259e18", "562546b3116e405c97243f8321b36756", "9e903c9e5fd44a9b993c773407b9e9ca", "c7b5ff307edf4e20a3e1c2cca6b0bfa5", "e80ba9ae72224d5bbcc7ceb1aa1eaa4a", "c3320d2ce32941b882bf523325c59892", "40ddf95eda7e4e27aab68082e9ffa103", "b9f9dba8c363459c911e59247d02f00e", "5020ef6b32e74f57819b9e9e2b9eb01d", "0610c029354a49e380b69c92a3b65fd9", "6473db8bfa404b2d95a9363a05cdc5b5", "71338732b57e41faa466257187920e2f", "9df2eb0618514b94a66418195ae1bae0", "b876c550c72a406a8647ddbe68f85778", "281950503a784dc0bbaff36f7369f1ae"]} id="qEU9a1uWKy_F" outputId="65e05854-8cd1-4438-e593-d6e9126a99f5"
Map = geemap.Map(center=[ymed, xmed], zoom=7)
Map.addLayer(WA1, {'color': 'red'}, 'Western Half')
Map.addLayer(WA2, {'color': 'red'}, 'Eastern Half')
# Map.addLayer(curr_SF_geopandas, {'color': 'red'}, 'Fields')
Map

# %% [markdown] id="S2NoX5Wq3IKc"
# **Define Parameters for Landsat 7**
#
# We are interested in **1984-1993** and **1999-present**.
#
#   - ```L4```: life span is ```16-July-1982``` to ```14-Dec-1993```.
#   - ```L5```: life span is ```1-March-1984``` to ```5-June-2013```.
#   - ```L7```: life span is ```15-April-1999``` to ```6-April-2022```.
#   - ```L8```: life span is ```11-Feb-2013``` till now (```June 2023```).
#
#
#
#   - **To Do- Merge Sentinel into ```extract_satellite_IC()``` function. (June 6 2023.)**

# %% id="x5VWSY-A3Ksh"
"""
 sattelie_of_choice must be from
         'LANDSAT/LT04/C02/T1_L2' OR 
         'LANDSAT/LT05/C02/T1_L2' OR
         'LANDSAT/LE07/C02/T1_L2' OR
         'LANDSAT/LC08/C02/T1_L2'
"""
sattelie_of_choice7 = 'LANDSAT/LE07/C02/T1_L2'
sattelie_of_choice5 = 'LANDSAT/LT05/C02/T1_L2'
sattelie_of_choice4 = 'LANDSAT/LT04/C02/T1_L2'

start_date_L7 = str(1999) + '-01-01'; # YYYY-MM-DD
end_date_L7   = str(2009) + '-01-01'; # YYYY-MM-DD

start_date_L5_early = str(1984) + '-01-01'; # YYYY-MM-DD
end_date_L5_early   = str(1994) + '-01-01'; # YYYY-MM-DD

start_date_L5_late = str(1999) + '-01-01'; # YYYY-MM-DD
end_date_L5_late   = str(2009) + '-01-01'; # YYYY-MM-DD

start_date_L4 = str(1984) + '-01-01'; # YYYY-MM-DD
end_date_L4   = str(1994) + '-01-01'; # YYYY-MM-DD

# start_date = "2017-01-01" # Date fromat for EE YYYY-MM-DD
# end_date = "2017-12-30"   # Date fromat for EE YYYY-MM-DD


# %% id="7xGPvgJ2kB5B"
satellit_info = {"L4" : ['LANDSAT/LT04/C02/T1_L2', 1984, 1993], 
                 "L5_early" : ['LANDSAT/LT05/C02/T1_L2', 1984, 1993],
                 "L5_late" : ['LANDSAT/LT05/C02/T1_L2', 1999, 2007],
                 "L7" : ['LANDSAT/LE07/C02/T1_L2', 1999, 2007]}

# %% [markdown] id="-FcdEWnl3Mnz"
# **Fetch data from GEE**
#

# %% id="E-Y4UTwjE35b"
SFblocks = gpc.split_SF_to_blocks(SF, block_size=1000)

# %% id="Uj14GwO-3SDl" colab={"base_uri": "https://localhost:8080/"} outputId="138d2801-282c-4b65-836e-f97fc6b2af41"
# %%time
# imageC = gpc.extract_sentinel_IC(big_rectangle, start_date, end_date, cloud_perc);
export_raw_data = True
counter=0
for a_sat_key in list(satellit_info.keys()):
    a_sat_info = satellit_info[a_sat_key]
    satelliteChoice = a_sat_info[0]
    start_year = a_sat_info[1]
    end_year = a_sat_info[2]
    # for a_year in range(start_year, end_year+1):
    for a_year in reversed(range(start_year, end_year+1)):
        for a_SFblocks_key in list(SFblocks.keys()):
            curr_SF_geopandas = geemap.geopandas_to_ee(SFblocks[a_SFblocks_key])
            st_date = str(a_year) + '-01-01'
            ed_date = str(a_year+1) + '-01-01'
            imageC = gpc.extract_satellite_IC(big_rectangle_featC = big_rectangle, 
                                              start_date = st_date, end_date = ed_date, 
                                              dataSource = satelliteChoice);

            reduced = gpc.mosaic_and_reduce_IC_mean(imageC, curr_SF_geopandas, 
                                                    st_date, ed_date)
            
            """
              The reason I have try-except below is that
              it seems some years there is no data. 
              Landsat 4 has been alive since 1982 but I am not getting 
              data in 1984. I am not sure what is happening.
            """
            try:
              needed_columns = ["ID", "EVI", 'NDVI', "system_start_time"]
              reduced = geemap.ee_to_pandas(reduced, selectors=needed_columns)
              reduced = reduced[needed_columns]
              if export_raw_data==True:
                EVI = reduced[["ID", "EVI", "system_start_time"]]
                NDVI = reduced[["ID", "NDVI", "system_start_time"]]

                EVI.dropna(subset=["EVI"], inplace=True)
                NDVI.dropna(subset=["NDVI"], inplace=True)

                outfile_name = "block" + str(a_SFblocks_key) + "_" + a_sat_key + "_" + st_date + "_" + ed_date + "_EVI.csv"
                EVI.to_csv(out_dir_base + "/intersection_pre2008/" + outfile_name)

                outfile_name = "block" + str(a_SFblocks_key) + "_" + a_sat_key + "_" + st_date + "_" + ed_date + "_NDVI.csv"
                NDVI.to_csv(out_dir_base + "/intersection_pre2008/" + outfile_name)
            except:
              if counter%10==0:
                print(f"{a_sat_key = }, {a_year = }, {a_SFblocks_key = }")
                counter += 1

        # print ("The size of [imageC.size().getInfo()] is [{:.0f}].".format(imageC.size().getInfo()))

# %% id="ii27AjLbCkqP"

# %% id="Qd4Ydg1bCku8"

# %% id="KMxJpGHACkv9"

# %% colab={"base_uri": "https://localhost:8080/"} id="aBbWhwmjhxT0" outputId="f5cc0104-a26e-4c2d-c74c-9110e0b468b1"
a_sat_key = list(satellit_info.keys())[0]
a_sat_info = satellit_info[a_sat_key]
satelliteChoice = a_sat_info[0]
start_year = a_sat_info[1]
end_year = a_sat_info[2]
print (f"{satelliteChoice=}")
print (f"{start_year=}")
print (f"{end_year=}")

a_year = start_year
st_date = str(a_year) + '-01-01'
ed_date = str(a_year+10) + '-01-01'

a_SFblocks_key = list(SFblocks.keys())[0]
curr_SF_geopandas = geemap.geopandas_to_ee(SFblocks[a_SFblocks_key][0:100])
st_date = str(a_year) + '-01-01'
ed_date = str(a_year+10) + '-01-01'
imageC = gpc.extract_satellite_IC(big_rectangle_featC=big_rectangle, 
                                  start_date = st_date, 
                                  end_date   = ed_date, 
                                  dataSource = satelliteChoice);

reduced = gpc.mosaic_and_reduce_IC_mean(imageC, curr_SF_geopandas, st_date, ed_date)
needed_columns = ["ID", "EVI", 'NDVI', "system_start_time"]
reduced = geemap.ee_to_pandas(reduced, selectors=needed_columns)
reduced = reduced[needed_columns]

# %% id="j2eqdiKDmCzf"
outfile_name = "block" + str(a_SFblocks_key) + "_" + a_sat_key + "_" + st_date + "_" + ed_date + ".csv"
reduced.to_csv(out_dir_base + "/intersection_pre2008/" + outfile_name)

# %% colab={"base_uri": "https://localhost:8080/", "height": 621, "referenced_widgets": ["44d3ec52bc8549b8992fdc80ab69dcd9", "6f5a6b01277749ed9dae040f2a46c989", "2dd7e79038714ca0b8a7113d714a2334", "997f6eb69e4d4ea4a25a051b15c5451f", "0fc0c82305ee4ca6992e3df11c4e4644", "ef5cd76321724d99936af7ef4a77177d", "e515484818f04da59b09dd5028ca8d08", "ff920d7e03404519887476cdb25e4f3f", "a689cc9727b44569b5a2401b1ba64ff2", "b8863b3ec30841389abb33f249929318", "9d1e26d35a3942eabf1ec2cbaee8a8de", "fdc400c6d7fd4b6f834d7e06cb591b5a", "4bb73780ea9c45b1b3d1e1db859bd530", "7538c6da7eda46fe863f034151d7b42d", "b62cb44046b448ce9206f41418257c7d", "f5db02c6fdac4a288ff7f0ff79568018", "901645ca98034f42bf51797d0913b9fa", "e6a7c51dd2004a3baa77926d3e7d3ba0", "243f433defb94c069e1ec8e4f648dc33", "6a9219cadb204b879a4f0b966f04d88d", "9a9f47cc161c4a74b0840d212d2521bc", "8bae6a4c569a48ce8d9d1264086fdc06", "f4555ed00c4743b0ba6e9bf3478e4101", "25153c14de6f42d88a7a5f1a4373d76d", "0019c9d0d626414e82c72cb2da1b38ef", "df583c04afaa42b591212052dd82ce9d"]} id="QbQuSDOo8jWn" outputId="04559e06-506e-4034-fb78-69f322111372"
Map = geemap.Map(center=[ymed, xmed], zoom=7)
#Map.addLayer(WA1, {'color': 'red'}, 'Western Half')
# Map.addLayer(WA2, {'color': 'red'}, 'Eastern Half')
Map.addLayer(curr_SF_geopandas, {'color': 'red'}, 'Fields')
Map

# %% [markdown] id="547FvT1M3Tqi"
# # Export output to Google Drive
#
# We advise you to do it. If Python/CoLab kernel dies, then,
# previous steps should not be repeated.

# %% id="Gwgkie_h3bkj"

# %% id="eR-D7DA53gjk"
