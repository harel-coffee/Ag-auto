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

WSDACrop_2012 <- readOGR(paste0(data_dir, 
                                "WSDACrop_2012/WSDACrop_2012.shp"),
                        layer = "WSDACrop_2012", 
                        GDAL1_integer64_policy = TRUE)

WSDACrop_2013 <- readOGR(paste0(data_dir, 
                                "WSDACrop_2013/WSDACrop_2013.shp"),
                        layer = "WSDACrop_2013", 
                        GDAL1_integer64_policy = TRUE)

WSDACrop_2014 <- readOGR(paste0(data_dir, 
                                "WSDACrop_2014/WSDACrop_2014.shp"),
                        layer = "WSDACrop_2014", 
                        GDAL1_integer64_policy = TRUE)

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

WSDACrop_2012 <- WSDACrop_2012[grepl('Grant', WSDACrop_2012$county), ]
WSDACrop_2013 <- WSDACrop_2013[grepl('Grant', WSDACrop_2013$county), ]
WSDACrop_2014 <- WSDACrop_2014[grepl('Grant', WSDACrop_2014$county), ]
WSDACrop_2015 <- WSDACrop_2015[grepl('Grant', WSDACrop_2015$county), ]
WSDACrop_2016 <- WSDACrop_2016[grepl('Grant', WSDACrop_2016$county), ]
WSDACrop_2017 <- WSDACrop_2017[grepl('Grant', WSDACrop_2017$county), ]
WSDACrop_2018 <- WSDACrop_2018[grepl('Grant', WSDACrop_2018$county), ]


dim(WSDACrop_2018@data)
dim(WSDACrop_2017@data)
dim(WSDACrop_2016@data)
dim(WSDACrop_2015@data)
dim(WSDACrop_2014@data)
dim(WSDACrop_2013@data)
dim(WSDACrop_2012@data)


write_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                    "/remote_sensing/00_shapeFiles/02_correct_years/", 
                    "05_filtered_shapefiles/Grant/")

if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = WSDACrop_2012, 
         dsn = paste0(write_dir, "/Grant_2012/"), 
         layer="Grant_2012", 
         driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2013, 
         dsn = paste0(write_dir, "/Grant_2013/"), 
         layer="Grant_2013", 
         driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2014, 
         dsn = paste0(write_dir, "/Grant_2014/"), 
         layer="Grant_2014", 
         driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2015, 
         dsn = paste0(write_dir, "/Grant_2015/"), 
         layer="Grant_2015", 
         driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2016, 
         dsn = paste0(write_dir, "/Grant_2016/"), 
         layer="Grant_2016", 
         driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2017, 
         dsn = paste0(write_dir, "/Grant_2017/"), 
         layer="Grant_2017", 
         driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2018, 
         dsn = paste0(write_dir, "/Grant_2018/"), 
         layer="Grant_2018", 
         driver="ESRI Shapefile")


########################################################################

# commented out on March 19

# WSDACrop_2018 <- pick_proper_cols_w_notes(WSDACrop_2018)
# WSDACrop_2017 <- pick_proper_cols_w_notes(WSDACrop_2017)
# WSDACrop_2016 <- pick_proper_cols_w_notes(WSDACrop_2016)
# WSDACrop_2015 <- pick_proper_cols_w_notes(WSDACrop_2015)
# WSDACrop_2014 <- pick_proper_cols_w_notes(WSDACrop_2014)
# WSDACrop_2013 <- pick_proper_cols_w_notes(WSDACrop_2013)
# WSDACrop_2012 <- pick_proper_cols_w_notes(WSDACrop_2012)

# WSDACrop_2012_2018 <- rbind(WSDACrop_2012, WSDACrop_2013,
#                             WSDACrop_2014, WSDACrop_2015,
#                             WSDACrop_2016, WSDACrop_2017,
#                             WSDACrop_2018)

# rm(WSDACrop_2012, WSDACrop_2013,
#    WSDACrop_2014, WSDACrop_2015,
#    WSDACrop_2016, WSDACrop_2017,
#    WSDACrop_2018)

# writeOGR(obj = WSDACrop_2012_2018, 
#          dsn = paste0(write_dir, "/Grant_2012_2018/"), 
#          layer="WSDACrop_2012_2018_lat_long", 
#          driver="ESRI Shapefile")


