rm(list=ls())
library(data.table)
library(dplyr)
library(foreign)
library(rgdal)
# library(sp) # rgdal appears to load this already


base_dir <- "/Users/hn/Documents/01_research_data/Ag_check_point/remote_sensing/03_cleaned_shapeFiles/"
weird_dir <- paste0(base_dir, "WSDACrop_2012_2018_weird_projection/")
latLong_dir <- paste0(base_dir, "WSDACrop_2012_2018_lat_long/")

##########################################
#######
#######
#######
##########################################
WSDACrop <- rgdal::readOGR(paste0(weird_dir, "/WSDACrop_2012_2018_weird_projection.shp"),
                         layer = "WSDACrop_2012_2018_weird_projection", 
                         GDAL1_integer64_policy = TRUE)

WSDACrop <- WSDACrop[grepl('Grant', WSDACrop$county), ]

writeOGR(obj = WSDACrop, 
         dsn = paste0(base_dir, "/Grant_2012_2018_weird/"), 
         layer="Grant_2012_2018_weird", 
         driver="ESRI Shapefile")

##########################################
#######
#######
#######
##########################################
WSDACrop <- rgdal::readOGR(paste0(latLong_dir, "/WSDACrop_2012_2018_lat_long.shp"),
                         layer = "WSDACrop_2012_2018_lat_long", 
                         GDAL1_integer64_policy = TRUE)

WSDACrop <- WSDACrop[grepl('Grant', WSDACrop$county), ]

writeOGR(obj = WSDACrop, 
         dsn = paste0(base_dir, "/Grant_2012_2018_LatLong/"), 
         layer="Grant_2012_2018_LatLong", 
         driver="ESRI Shapefile")


