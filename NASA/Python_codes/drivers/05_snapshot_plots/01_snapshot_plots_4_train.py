import datetime
from datetime import date
import datetime
import time

import os, os.path, sys
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
county = sys.argv[1]           # [AdamBenton2016, FranklinYakima2018, Grant2017, Monterey2014, Walla2015]
TOA_or_corrected = sys.argv[2] # [TOA, corrected]
print (county)
print (TOA_or_corrected)
####################################################################################
###
###                   Aeolus Directories
###
####################################################################################
param_dir = "/data/hydro/users/Hossein/NASA/000_shapefile_data_part/"

raster_dir = "/data/hydro/users/Hossein/NASA/01_raster_GEE/"
SF_dir = "/data/hydro/users/Hossein/NASA/000_shapefiles_train_irr_NASSout_SurvCorrect/"

snapshot_panel_dir = "/data/hydro/users/Hossein/NASA/06_snapshot_plots/"
os.makedirs(snapshot_panel_dir, exist_ok=True)
#####################################################
########
######## read the files
########
if county == "Monterey2014":
    SF = shapefile.Reader(SF_dir + "clean_Monterey_centroids/clean_Monterey_centroids.shp")

    # we can check the projectios of shapefiles and rasterfiles are identical!!!!
    Fiona_SF = fiona.open(SF_dir + "clean_Monterey_centroids/clean_Monterey_centroids.shp")
    SF_CRS = Fiona_SF.crs['init'].lower()
    raster_dir = raster_dir + "snapshot_Monterey/"

elif county == "AdamBenton2016":
    SF_Name = "AdamBenton2016_irr_NASSout_survCorrect_centroids/AdamBenton2016_irr_NASSout_survCorrect_centroids.shp"
    SF = shapefile.Reader(SF_dir + SF_Name)
    Fiona_SF = fiona.open(SF_dir + SF_Name)
    SF_CRS = Fiona_SF.crs['init'].lower()
    raster_dir = raster_dir + "snapshot_AdamBenton2016/"

elif county == "FranklinYakima2018":
    SF_Name = "FranklinYakima2018_irr_NASSout_survCorrect_centroids/FranklinYakima2018_irr_NASSout_survCorrect_centroids.shp"
    SF = shapefile.Reader(SF_dir + SF_Name)
    Fiona_SF = fiona.open(SF_dir + SF_Name)
    SF_CRS = Fiona_SF.crs['init'].lower()
    raster_dir = raster_dir + "snapshot_FranklinYakima2018/"

elif county == "Grant2017":
    SF_Name = "Grant2017_irr_NASSout_survCorrect_centroids/Grant2017_irr_NASSout_survCorrect_centroids.shp"
    SF = shapefile.Reader(SF_dir + SF_Name)
    Fiona_SF = fiona.open(SF_dir + SF_Name)
    SF_CRS = Fiona_SF.crs['init'].lower()
    raster_dir = raster_dir + "snapshot_Grant2017/"

elif county == "Walla2015":
    SF_Name = "Walla2015_irr_NASSout_survCorrect_centroids/Walla2015_irr_NASSout_survCorrect_centroids.shp"
    SF = shapefile.Reader(SF_dir + SF_Name)
    Fiona_SF = fiona.open(SF_dir + SF_Name)
    SF_CRS = Fiona_SF.crs['init'].lower()
    raster_dir = raster_dir + "snapshot_Walla2015/"


if TOA_or_corrected == "TOA":
    Tiff_files = [x for x in os.listdir(raster_dir) if x.endswith(".tif")]
    raster_files = [s for s in Tiff_files if "TOA" in s]
    raster_files = np.sort(raster_files)
else:
    Tiff_files = [x for x in os.listdir(raster_dir) if x.endswith(".tif")]
    raster_files = [s for s in Tiff_files if "L2C2" in s]
    raster_files = np.sort(raster_files)


# first fix polygon from shapefile
for ii in range(len(SF)):
    curr_poly = SF.shapeRecords()[ii].shape.__geo_interface__
    curr_ID = SF.records()[ii]['ID']
    curr_crop = SF.records()[ii]['CropTyp']

    curr_ctr_lat = SF.records()[ii]['ctr_lat']
    curr_ctr_long = SF.records()[ii]['ctr_long']

    if county != "Monterey2014":
        curr_surv = SF.records()[ii]['LstSrvD']
    else:
        curr_surv = "2014"


    ######## First count the number of all black images to set figure size properly
    black_count = 0
    for count, file in enumerate(raster_files):
        curr_raster_file = raster_dir + file;
        curr_rasterio_im = rasterio.open(curr_raster_file);
        
        # Sanity check: Projection of SF file and rasterfile
        assert(str(curr_rasterio_im.crs).lower() == Fiona_SF.crs['init'].lower())
        
        out_img, out_transform = mask(dataset = curr_rasterio_im, 
                                      shapes = [curr_poly], 
                                      crop = True)
        
        if sum(sum(sum(out_img))) == 0:
            black_count += 1

    max_no_pictures = len(raster_files) - black_count
    n_columns = 10
    n_rows = int(np.ceil(max_no_pictures/n_columns))

    subplot_size = 5
    plot_width = n_columns*subplot_size
    plot_length = n_rows*5
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(plot_width, plot_length))

    col_idx = 0
    row_idx = 0

    independent_count = 0

    for count, file in enumerate(raster_files):
        curr_raster_file = raster_dir + file;
        curr_rasterio_im = rasterio.open(curr_raster_file);
        
        # Sanity check: Projection of SF file and rasterfile
        assert(str(curr_rasterio_im.crs).lower() == Fiona_SF.crs['init'].lower())
        
        out_img, out_transform = mask(dataset = curr_rasterio_im, 
                                      shapes = [curr_poly], 
                                      crop = True)
            
        # if image is completely black, skip it:
        if sum(sum(sum(out_img))) != 0:
            # form the title!
            curr_time = int(file.split("_")[0]) / 1000
            # convert epoch time to human time
            curr_time = time.strftime('%Y-%m-%d', time.localtime(curr_time))

            row_idx = independent_count // 10 # this is the same as floor(count/10).
            col_idx = independent_count % 10  # remainder of the division. (0-9 goes to 0-9, 10-19 is mapped to 0-9, ...)
            curr_ax = axes[row_idx][col_idx]
            # show(out_img, ax=curr_ax, title=curr_time)
            show(out_img, ax=curr_ax)
            curr_ax.set_title(curr_time, fontsize=20)
            independent_count += 1

    # Title of the figure
    curr_crop_name = curr_crop.lower().replace(" ", "_").replace(",", "").replace("/", "_")
    
    figure_title = county + ", " + curr_crop_name + " [" + \
                   curr_ID + ": " + str(curr_ctr_lat) + ", " + str(curr_ctr_long) + "]" + ", [" + curr_surv + "]"

    fig.suptitle(figure_title, fontsize=26, fontweight='bold', y=0.95)
    # plt.title('figure_title', fontsize=25, fontweight="bold", y=2)
                  
    
    plot_path = snapshot_panel_dir + curr_crop_name + "/"
    os.makedirs(plot_path, exist_ok=True)

    fig_name = plot_path + curr_ID + "_" + str(curr_ctr_lat) + "_" + str(curr_ctr_long) + "_" + TOA_or_corrected +'.pdf'

    plt.savefig(fname=fig_name, dpi=100, bbox_inches='tight')
    plt.close('all')

end_time = time.time()
print ("it took {:.0f} minutes to run this code.".format((end_time - start_time)/60))


