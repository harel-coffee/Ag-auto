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
data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                  "00_shapeFiles/01_not_correct_years/01_true_shapefiles_separate_years/")

# WSDACrop_2012 <- readOGR(paste0(data_dir, 
#                                 "WSDACrop_2012/WSDACrop_2012.shp"),
#                         layer = "WSDACrop_2012", 
#                         GDAL1_integer64_policy = TRUE)

# WSDACrop_2013 <- readOGR(paste0(data_dir, 
#                                 "WSDACrop_2013/WSDACrop_2013.shp"),
#                         layer = "WSDACrop_2013", 
#                         GDAL1_integer64_policy = TRUE)

# WSDACrop_2014 <- readOGR(paste0(data_dir, 
#                                 "WSDACrop_2014/WSDACrop_2014.shp"),
#                         layer = "WSDACrop_2014", 
#                         GDAL1_integer64_policy = TRUE)

WSDACrop_2015 <- readOGR(paste0(data_dir, 
                                "WSDACrop_2015/WSDACrop_2015.shp"),
                        layer = "WSDACrop_2015", 
                        GDAL1_integer64_policy = TRUE)

WSDACrop_2016 <- readOGR(paste0(data_dir, 
                                "WSDACrop_2016/WSDACrop_2016.shp"),
                        layer = "WSDACrop_2016", 
                        GDAL1_integer64_policy = TRUE)

WSDACrop_2018 <- readOGR(paste0(data_dir, 
                                "WSDACrop_2018/WSDACrop_2018.shp"),
                        layer = "WSDACrop_2018", 
                        GDAL1_integer64_policy = TRUE)

dim(WSDACrop_2018@data)
dim(WSDACrop_2016@data)
dim(WSDACrop_2015@data)
# dim(WSDACrop_2014@data)
# dim(WSDACrop_2013@data)
# dim(WSDACrop_2012@data)

########################################################################################
######
######     add the goddamn numeric identifier.
######
########################################################################################
# WSDACrop_2012 <- add_identifier(dt_df=WSDACrop_2012, year="2012")
# WSDACrop_2013 <- add_identifier(dt_df=WSDACrop_2013, year="2013")
# WSDACrop_2014 <- add_identifier(dt_df=WSDACrop_2014, year="2014")
WSDACrop_2015 <- add_identifier(dt_df=WSDACrop_2015, year="2015")
WSDACrop_2016 <- add_identifier(dt_df=WSDACrop_2016, year="2016")
# WSDACrop_2017 <- add_identifier(dt_df=WSDACrop_2017, year="2017") # is in another file since data changed from Perry side
# WSDACrop_2018 <- add_identifier(dt_df=WSDACrop_2018, year="2018")

########################################################################################

write_dir <- data_dir

if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

# writeOGR(obj = WSDACrop_2012, 
#          dsn = paste0(write_dir, "/WSDACrop_2012/"), 
#          layer="WSDACrop_2012", 
#          driver="ESRI Shapefile")

# writeOGR(obj = WSDACrop_2013, 
#          dsn = paste0(write_dir, "/WSDACrop_2013/"), 
#          layer="WSDACrop_2013", 
#          driver="ESRI Shapefile")

# writeOGR(obj = WSDACrop_2014, 
#          dsn = paste0(write_dir, "/WSDACrop_2014/"), 
#          layer="WSDACrop_2014", 
#          driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2015, 
         dsn = paste0(write_dir, "/WSDACrop_2015/"), 
         layer="WSDACrop_2015", 
         driver="ESRI Shapefile")

# writeOGR(obj = WSDACrop_2016, 
#          dsn = paste0(write_dir, "/WSDACrop_2016/"), 
#          layer="WSDACrop_2016", 
#          driver="ESRI Shapefile")

# writeOGR(obj = WSDACrop_2017, 
#          dsn = paste0(write_dir, "/WSDACrop_2017/"), # is in another file since data changed from Perry side
#          layer="WSDACrop_2017", 
#          driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2018, 
         dsn = paste0(write_dir, "/WSDACrop_2018/"), 
         layer="WSDACrop_2018", 
         driver="ESRI Shapefile")


