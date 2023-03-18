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


data_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                   "Ag_check_point/remote_sensing/00_shapeFiles/", 
                   "02_correct_years/05_filtered_shapefiles/Grant/", 
                   "Grant_2018/")

grant_2018 <- readOGR(paste0(data_dir, "Grant_2018.shp"),
                        layer = "Grant_2018", 
                        GDAL1_integer64_policy = TRUE)

a_field <- grant_2018[3, ]

write_dir <- paste0("/Users/hn/Desktop/")

writeOGR(obj = a_field, 
         dsn = paste0(write_dir, "/a_Grant_2018/"), 
         layer="a_Grant_2018", 
         driver="ESRI Shapefile")