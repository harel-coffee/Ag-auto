rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
# library(sf)
library(foreign)


data_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/Eastern_2018/"

WSDA_2018 <- readOGR(paste0(data_dir, "/Eastern_2018.shp"),
                     layer = "Eastern_2018", 
                     GDAL1_integer64_policy = TRUE)


LOI = WSDA_2018[WSDA_2018@data$ID == "45393_WSDA_SF_2018", ]
rgeos::gCentroid(LOI, byid=TRUE)


LOI = WSDA_2018[WSDA_2018@data$ID == "1051_WSDA_SF_2018", ]
rgeos::gCentroid(LOI, byid=TRUE)
