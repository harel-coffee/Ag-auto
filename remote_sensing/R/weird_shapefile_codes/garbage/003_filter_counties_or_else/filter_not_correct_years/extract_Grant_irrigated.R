rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
# library(sf)
library(foreign)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)
##########
########## Directories
##########

data_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                   "remote_sensing/00_shapeFiles/01_not_correct_years/", 
                   "01_true_shapefiles_separate_years/")

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

################################################################################################
###
###
###

WSDACrop_2015 <- WSDACrop_2015[grepl('Grant', WSDACrop_2015$county), ]
WSDACrop_2016 <- WSDACrop_2016[grepl('Grant', WSDACrop_2016$county), ]
WSDACrop_2017 <- WSDACrop_2017[grepl('Grant', WSDACrop_2017$county), ]
WSDACrop_2018 <- WSDACrop_2018[grepl('Grant', WSDACrop_2018$county), ]

WSDACrop_2015 <- filter_out_non_irrigated_shapefile(WSDACrop_2015)
WSDACrop_2016 <- filter_out_non_irrigated_shapefile(WSDACrop_2016)
WSDACrop_2017 <- filter_out_non_irrigated_shapefile(WSDACrop_2017)
WSDACrop_2018 <- filter_out_non_irrigated_shapefile(WSDACrop_2018)


dim(WSDACrop_2018@data)
dim(WSDACrop_2017@data)
dim(WSDACrop_2016@data)
dim(WSDACrop_2015@data)


write_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                    "remote_sensing/00_shapeFiles/", 
                    "02_not_correct_years_irrigated/")

if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = WSDACrop_2015, 
         dsn = paste0(write_dir, "/Grant_2015_irrigated/"), 
         layer="Grant_2015_irrigated", 
         driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2016, 
         dsn = paste0(write_dir, "/Grant_2016_irrigated/"), 
         layer="Grant_2016_irrigated", 
         driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2017, 
         dsn = paste0(write_dir, "/Grant_2017_irrigated/"), 
         layer="Grant_2017_irrigated", 
         driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2018, 
         dsn = paste0(write_dir, "/Grant_2018_irrigated/"), 
         layer="Grant_2018_irrigated", 
         driver="ESRI Shapefile")


