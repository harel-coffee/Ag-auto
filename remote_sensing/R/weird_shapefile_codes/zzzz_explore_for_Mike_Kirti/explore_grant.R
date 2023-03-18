rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
# library(sp) # rgdal appears to load this already
library(foreign)


base_dir <- "/Users/hn/Documents/01_research_data/Ag_check_point/remote_sensing/03_cleaned_shapeFiles/"
weird_dir <- paste0(base_dir, "Grant_2012_2018_weird/")
latLong_dir <- paste0(base_dir, "Grant_2012_2018_LatLong/")

##########################################
#######
#######    Werid part
#######
##########################################
grant_SF_weird <- rgdal::readOGR(paste0(weird_dir),
                           layer = "Grant_2012_2018_weird", 
                           GDAL1_integer64_policy = TRUE)

top_three_weird <- grant_SF_weird[1:3, ]

write_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                    "Ag_check_point/remote_sensing/", 
                    "03_cleaned_shapeFiles/filtered_some_fields/")

if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = top_three_weird, 
         dsn = paste0(write_dir, "/grant_three_weird/"), 
         layer="grant_three_weird", 
         driver="ESRI Shapefile")

first_weird <- grant_SF_weird[1, ]
writeOGR(obj = first_weird, 
         dsn = paste0(write_dir, "/first_weird/"), 
         layer="first_weird", 
         driver="ESRI Shapefile")


second_weird <- grant_SF_weird[2, ]
writeOGR(obj = second_weird, 
         dsn = paste0(write_dir, "/second_weird/"), 
         layer="second_weird", 
         driver="ESRI Shapefile")


third_weird <- grant_SF_weird[3, ]
writeOGR(obj = first_weird, 
         dsn = paste0(write_dir, "/third_weird/"), 
         layer="third_weird", 
         driver="ESRI Shapefile")
##########################################
#######
#######    lat long part
#######
##########################################
grant_SF_latlong <- rgdal::readOGR(paste0(latLong_dir),
                           layer = "Grant_2012_2018_LatLong", 
                           GDAL1_integer64_policy = TRUE)

top_three_latlong <- grant_SF_latlong[1:3, ]

write_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                    "Ag_check_point/remote_sensing/", 
                    "03_cleaned_shapeFiles/filtered_some_fields/")

if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = top_three_latlong, 
         dsn = paste0(write_dir, "/grant_three_latlong/"), 
         layer="grant_three_latlong", 
         driver="ESRI Shapefile")


first_grant_latlong <- grant_SF_latlong[1, ]
second_grant_latlong <- grant_SF_latlong[2, ]
third_grant_latlong <- grant_SF_latlong[3, ]


writeOGR(obj = first_grant_latlong, 
         dsn = paste0(write_dir, "/first_grant_latlong/"), 
         layer="first_grant_latlong", 
         driver="ESRI Shapefile")

writeOGR(obj = second_grant_latlong, 
         dsn = paste0(write_dir, "/second_grant_latlong/"), 
         layer="second_grant_latlong", 
         driver="ESRI Shapefile")

writeOGR(obj = third_grant_latlong, 
         dsn = paste0(write_dir, "/third_grant_latlong/"), 
         layer="third_grant_latlong", 
         driver="ESRI Shapefile")
