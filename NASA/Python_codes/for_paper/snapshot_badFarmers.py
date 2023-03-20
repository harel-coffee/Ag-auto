# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %%
import datetime
from datetime import date
import datetime
import time

import sys
import os, os.path
from os import listdir
from os.path import isfile, join
import numpy as np
import re # regular expression

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sb

import rasterio 
from rasterio.mask import mask
from rasterio.plot import show

# import glob # did NOT installed. need it here?
import shapefile # did NOT installed


import rioxarray as rxr
import xarray as xr
import fiona
# import geopandas as gpd
import earthpy as et # need it here?
import earthpy.plot as ep # need it here?

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as ncp


# %%
TOA_or_corrected="corrected"

# %%
param_dir = "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/"
SF_dir = "/Users/hn/Documents/01_research_data/NASA/shapefiles/00000_train_SF_NASSout_Irr_CorrectYr/"

raster_dir_main = "/Users/hn/Documents/01_research_data/NASA/snapshots/01_raster_GEE/"
plot_path = "/Users/hn/Documents/01_research_data/NASA/for_paper/plots/badFarmers/"
os.makedirs(plot_path, exist_ok=True)


# %%
def generateNames(county):
    TOA_or_corrected = 'corrected'

    if county == "Monterey2014":
        raster_dir = raster_dir_main + "snapshot_Monterey/"
    elif county == "AdamBenton2016":
        raster_dir = raster_dir_main + "snapshot_AdamBenton2016/"

    elif county == "FranklinYakima2018":
        raster_dir = raster_dir_main + "snapshot_FranklinYakima2018/"

    elif county == "Grant2017":
        raster_dir = raster_dir_main + "snapshot_Grant2017/"

    elif county == "Walla2015":
        raster_dir = raster_dir_main + "snapshot_Walla2015/"
        
    if TOA_or_corrected == "TOA":
        Tiff_files = [x for x in os.listdir(raster_dir) if x.endswith(".tif")]
        raster_files = [s for s in Tiff_files if "TOA" in s]
        raster_files = np.sort(raster_files)
    else:
        Tiff_files = [x for x in os.listdir(raster_dir) if x.endswith(".tif")]
        raster_files = [s for s in Tiff_files if "L2C2" in s]
        raster_files = np.sort(raster_files)

    SF_Name = "badFarmers/badFarmers.shp"
    SF = shapefile.Reader(SF_dir + SF_Name)
    Fiona_SF = fiona.open(SF_dir + SF_Name)
    SF_CRS = Fiona_SF.crs['init'].lower()

    return (SF_Name, SF, Fiona_SF, SF_CRS, raster_dir, raster_files)


def detect_countyName(year):
    if year=="2015":
        county="Walla2015"
    elif year=="2016":
        county="AdamBenton2016"
    elif year=="2017":
        county="Grant2017"
    elif year=="2018":
        county="FranklinYakima2018"
    return county
    


# %%
SF_Name = "badFarmers/badFarmers.shp"
SF = shapefile.Reader(SF_dir + SF_Name)
Fiona_SF = fiona.open(SF_dir + SF_Name)
SF_CRS = Fiona_SF.crs['init'].lower()

# %%
field_IDs = ["162687_WSDA_SF_2015",
             "57593_WSDA_SF_2016",
             "60617_WSDA_SF_2016",
             # "35065_WSDA_SF_2018", 
             "39244_WSDA_SF_2018", # ? 
             # "40865_WSDA_SF_2018",
             "46239_WSDA_SF_2018"]

# %%
field_IDs_dict = {"57593_WSDA_SF_2016": ["2016-05-10", "2016-07-13", "2016-08-14", "2016-09-15"],
                  "162687_WSDA_SF_2015":["2015-05-01", "2015-08-21", "2015-09-29", "2015-11-16"],
                  "60617_WSDA_SF_2016":["2016-05-10", "2016-06-27", "2016-09-15", "2016-10-01"],
                  "39244_WSDA_SF_2018":["2018-07-10", "2018-07-26", "2018-09-28", "2018-10-30"],
                  "46239_WSDA_SF_2018":["2018-04-30", "2018-06-24", "2018-09-28", "2018-11-08"]
                 }

fields_from_keys=sorted(list(field_IDs_dict.keys()))

# %%
import matplotlib
matplotlib.rcParams['figure.figsize'] = [20, 5]

# %%
from matplotlib import pyplot

# %%
size = 20
title_FontSize = 10
legend_FontSize = 8
tick_FontSize = 12
label_FontSize = 14

params = {'legend.fontsize': 'medium',
          'figure.figsize': (20, 5),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size * 0.75,
          'ytick.labelsize': size * 0.75,
          'axes.titlepad': 10}

#
#  Once set, you cannot change them, unless restart the notebook
#
plt.rc('font', family = 'Palatino')
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelleft'] = True
plt.rcParams.update(params)

# %%
# %%time
for ii in range(len(SF)):
    curr_ID=SF.records()[ii]['ID']
    curr_county_name = detect_countyName(curr_ID.split("_")[-1])
    SF_Name, SF, Fiona_SF, SF_CRS, raster_dir, raster_files = generateNames(curr_county_name)

    curr_poly = SF.shapeRecords()[ii].shape.__geo_interface__
    curr_crop = SF.records()[ii]['CropTyp']
    curr_crop = curr_crop.replace(" ", "_")

    curr_ctr_lat = SF.records()[ii]['ctr_lat']
    curr_ctr_long = SF.records()[ii]['ctr_long']
    curr_surv = SF.records()[ii]['LstSrvD']

    n_columns = 4
    n_rows = 1

    subplot_size = 5
    plot_width = n_columns*subplot_size
    plot_length = n_rows*5
    print (plot_width, plot_length)
    # fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(plot_width, plot_length))
    fig, axes = pyplot.subplots(n_rows, n_columns, figsize=(plot_width, plot_length))
    # fig.set_size_inches(plot_width, plot_length)

    for count, file in enumerate(raster_files):
        curr_raster_file = raster_dir + file;
        curr_rasterio_im = rasterio.open(curr_raster_file);

        out_img, out_transform = mask(dataset = curr_rasterio_im, 
                                      shapes = [curr_poly], 
                                      crop = True)

        curr_time = int(file.split("_")[0]) / 1000
        # convert epoch time to human time
        curr_time = time.strftime('%Y-%m-%d', time.localtime(curr_time))

        if curr_time in field_IDs_dict[curr_ID]:        
            col_idx = field_IDs_dict[curr_ID].index(curr_time)
            curr_ax = axes[col_idx]
            curr_ax.axis("off")

            # show(out_img, ax=curr_ax, title=curr_time)
            show(out_img, ax=curr_ax)
            curr_ax.set_title(curr_time, fontsize=20)

    # Title of the figure
    curr_crop_name = curr_crop.lower().replace(" ", "_").replace(",", "").replace("/", "_")
    # fig.set_size_inches(plot_width, plot_length)
    figure_title = curr_crop_name + " [" + \
                   curr_ID + ": " + str(curr_ctr_lat) + ", " + str(curr_ctr_long) + \
                   "]" + ", [" + curr_surv + "]"

    short_ID = curr_ID.split("_")[0] + "_" + curr_ID.split("_")[-1]
    fig_name = plot_path + short_ID + "_badFarmer"  +'.pdf'

    plt.savefig(fname=fig_name, dpi=400, bbox_inches='tight')
    plt.close('all')

# %%

# %%

# %%

# %%
