##########################################################################################
rm(list=ls())
library(data.table)
library(dplyr)
library(foreign)
library(rgdal)


shape_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                    "remote_sensing/00_shapeFiles/02_correct_years/", 
                    "05_filtered_shapefiles/Grant/Grant_2016/")


TS_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                 "remote_sensing/01_NDVI_TS/Grant/No_EVI/", 
                 "Grant_10_cloud/Grant_2016/")

Grant_2016_TS <- read.csv(paste0(TS_dir, "Grant_2016_TS.csv"), as.is=TRUE)
Grant_2016_sf <- rgdal::readOGR(paste0(shape_dir, "/Grant_2016.shp"),
                                       layer = "Grant_2016", 
                                       GDAL1_integer64_policy = TRUE)


Grant_2016_TS <- within(Grant_2016_TS, remove(system.index))
setnames(Grant_2016_TS, old=c(".geo"), new=c("geo"))

Grant_2016_TS_unique <- within(Grant_2016_TS, remove(B2, B3, B4, B8, NDVI, doy))
Grant_2016_TS_unique <- unique(Grant_2016_TS_unique)
Grant_2016_TS_unique <- tibble::rowid_to_column(Grant_2016_TS_unique, "identifier")
find_1st_potato <- Grant_2016_TS_unique %>% filter(CropTyp == "Potato" & Acres == 121)
