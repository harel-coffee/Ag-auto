# Per_MS Meeting

library(scales)
library(lattice)
library(jsonlite)
library(raster)

library(data.table)
library(shiny)
library(shinydashboard)
library(shinyBS)
library(maps)
library(rgdal)    # for readOGR and others
library(sp)       # for spatial objects
library(leaflet)  # for interactive maps (NOT leafletR here)
library(dplyr)    # for working with data frames
library(ggplot2)  # for plotting
library(reshape2)
library(RColorBrewer)
# library(plotly)
# library(Hmisc)

data_dir <- "/data/hnoorazar/remote_sensing_pre_MS/"




# shape_dir <- "/data/hnoorazar/remote_sensing_pre_MS/Min_double_crops/"
# data_dir <- paste0("/Users/hn/Desktop/Desktop/Ag/",
#                    "check_point/pre_microsoft_meeting/", 
#                    "filtered_shape_files/simple/Min_double_crops")

# mins_file <- rgdal::readOGR(dsn=path.expand(paste0(shape_dir, 
#                                                    "/Min_DoubleCrop.shp")),
#                             layer = "Min_DoubleCrop", 
#                             GDAL1_integer64_policy = TRUE)

# Min_sp <- spTransform(mins_file, CRS("+init=epsg:4326"))


##############################
############################## TRUELY SURVEYED 2018
##############################
shape_dir <- "/data/hnoorazar/remote_sensing_pre_MS/double_2018/"

mins_file <- rgdal::readOGR(dsn=path.expand(paste0(shape_dir, 
                                                   "/double_2018.shp")),
                            layer = "double_2018", 
                            GDAL1_integer64_policy = TRUE)

Min_sp <- spTransform(mins_file, CRS("+init=epsg:4326"))
##############################
##############################
##############################

centroids <- read.csv(paste0(data_dir, "double_crop_centroid.csv"),
                     as.is=TRUE) %>% data.table()

#######################################################
#######################################################
#######################################################
