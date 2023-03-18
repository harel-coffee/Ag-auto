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



coi = c("Grant")

##########
########## 2018
##########
data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                   "00_shapeFiles/02_correct_years/03_correct_years_separate/", 
                   "lat_long_projections/")

base_write <- paste0("/Users/hn/Documents/01_research_data//", 
                     "remote_sensing/00_shapeFiles/02_correct_years/", 
                     "05_filtered_shapefiles/batches/potential_fields/")

param_dir = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"

WSDACrop_2018 <- readOGR(paste0(data_dir, "WSDACrop_2018/WSDACrop_2018.shp"),
                    layer = "WSDACrop_2018", 
                    GDAL1_integer64_policy = TRUE)

WSDACrop_2018$Notes <- tolower(WSDACrop_2018$Notes)

WSDACrop_2018 <- WSDACrop_2018[WSDACrop_2018@data$county %in% coi, ]
WSDACrop_2018 <- toss_Nass(WSDACrop_2018)


##########
########## 2017
##########
data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                   "00_shapeFiles/02_correct_years/03_correct_years_separate/", 
                   "lat_long_projections/")

base_write <- paste0("/Users/hn/Documents/01_research_data//", 
                     "remote_sensing/00_shapeFiles/02_correct_years/", 
                     "05_filtered_shapefiles/batches/potential_fields/")

param_dir = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"

WSDACrop_2017 <- readOGR(paste0(data_dir, "WSDACrop_2017/WSDACrop_2017.shp"),
                    layer = "WSDACrop_2017", 
                    GDAL1_integer64_policy = TRUE)

WSDACrop_2017$Notes <- tolower(WSDACrop_2017$Notes)
WSDACrop_2017 <- WSDACrop_2017[WSDACrop_2017@data$county %in% coi, ]
WSDACrop_2017 <- toss_Nass(WSDACrop_2017)



##########
########## 2016
##########

data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                   "00_shapeFiles/02_correct_years/03_correct_years_separate/", 
                   "lat_long_projections/")

base_write <- paste0("/Users/hn/Documents/01_research_data//", 
                     "remote_sensing/00_shapeFiles/02_correct_years/", 
                     "05_filtered_shapefiles/batches/potential_fields/")

param_dir = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"

WSDACrop_2016 <- readOGR(paste0(data_dir, "WSDACrop_2016/WSDACrop_2016.shp"),
                    layer = "WSDACrop_2016", 
                    GDAL1_integer64_policy = TRUE)

WSDACrop_2016$Notes <- tolower(WSDACrop_2016$Notes)
WSDACrop_2016 <- WSDACrop_2016[WSDACrop_2016@data$county %in% coi, ]
WSDACrop_2016 <- toss_Nass(WSDACrop_2016)

##########
########## 2015
##########

data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                   "00_shapeFiles/02_correct_years/03_correct_years_separate/", 
                   "lat_long_projections/")

base_write <- paste0("/Users/hn/Documents/01_research_data//", 
                     "remote_sensing/00_shapeFiles/02_correct_years/", 
                     "05_filtered_shapefiles/batches/potential_fields/")

param_dir = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"

WSDACrop_2015 <- readOGR(paste0(data_dir, "WSDACrop_2015/WSDACrop_2015.shp"),
                    layer = "WSDACrop_2015", 
                    GDAL1_integer64_policy = TRUE)

WSDACrop_2015$Notes <- tolower(WSDACrop_2015$Notes)
WSDACrop_2015 <- WSDACrop_2015[WSDACrop_2015@data$county %in% coi, ]

WSDACrop_2015 <- toss_Nass(WSDACrop_2015)


nrow(WSDACrop_2015)
nrow(WSDACrop_2016)
nrow(WSDACrop_2017)
nrow(WSDACrop_2018)

# 2017 is the largest





