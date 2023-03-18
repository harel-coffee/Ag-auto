rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
# library(sf)
library(foreign)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)
##########
########## Directories
##########

data_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                    "/remote_sensing/00_shapeFiles/02_correct_years/", 
                    "03_correct_years_separate/lat_long_projections/")
param_dir = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"

double_crop_potential_plants = read.csv(paste0(param_dir, "double_crop_potential_plants.csv"), as.is=TRUE)


WSDACrop_2015 <- readOGR(paste0(data_dir, 
                                "WSDACrop_2015/WSDACrop_2015.shp"),
                        layer = "WSDACrop_2015", 
                        GDAL1_integer64_policy = TRUE)

WSDACrop_2016 <- readOGR(paste0(data_dir, 
                                "WSDACrop_2016/WSDACrop_2016.shp"),
                        layer = "WSDACrop_2016", 
                        GDAL1_integer64_policy = TRUE)

WSDACrop_2017 <- readOGR(paste0(data_dir, 
                                "WSDACrop_2017/WSDACrop_2017.shp"),
                        layer = "WSDACrop_2017", 
                        GDAL1_integer64_policy = TRUE)

WSDACrop_2018 <- readOGR(paste0(data_dir, 
                                "WSDACrop_2018/WSDACrop_2018.shp"),
                        layer = "WSDACrop_2018", 
                        GDAL1_integer64_policy = TRUE)

WSDACrop_2015 <- WSDACrop_2015[WSDACrop_2015@data$CropTyp %in% double_crop_potential_plants$Crop_Type, ]
WSDACrop_2016 <- WSDACrop_2016[WSDACrop_2016@data$CropTyp %in% double_crop_potential_plants$Crop_Type, ]
WSDACrop_2017 <- WSDACrop_2017[WSDACrop_2017@data$CropTyp %in% double_crop_potential_plants$Crop_Type, ]
WSDACrop_2018 <- WSDACrop_2018[WSDACrop_2018@data$CropTyp %in% double_crop_potential_plants$Crop_Type, ]

################################################################################################
###
###
###

WSDACrop_2015 <- WSDACrop_2015[grepl('Grant', WSDACrop_2015$county), ]
WSDACrop_2016 <- WSDACrop_2016[grepl('Grant', WSDACrop_2016$county), ]
WSDACrop_2017 <- WSDACrop_2017[grepl('Grant', WSDACrop_2017$county), ]
WSDACrop_2018 <- WSDACrop_2018[grepl('Grant', WSDACrop_2018$county), ]


dim(WSDACrop_2018@data)
dim(WSDACrop_2017@data)
dim(WSDACrop_2016@data)
dim(WSDACrop_2015@data)

write_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                    "/remote_sensing/00_shapeFiles/02_correct_years/", 
                    "05_filtered_shapefiles/Grant/potential_doubles/")

if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = WSDACrop_2015, 
         dsn = paste0(write_dir, "/Grant_potentials_2015/"), 
         layer="Grant_2015", 
         driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2016, 
         dsn = paste0(write_dir, "/Grant_potentials_2016/"), 
         layer="Grant_2016", 
         driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2017, 
         dsn = paste0(write_dir, "/Grant_potentials_2017/"), 
         layer="Grant_2017", 
         driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2018, 
         dsn = paste0(write_dir, "/Grant_potentials_2018/"), 
         layer="Grant_2018", 
         driver="ESRI Shapefile")

