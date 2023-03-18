rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
library(sf)
library(foreign)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)


##########
########## Directories
##########

data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                    "00_shapeFiles/01_not_correct_years/", 
                    "01_true_shapefiles_separate_years/WSDACrop_2017/")

WSDA_2017 <- readOGR(paste0(data_dir, "/WSDACrop_2017.shp"),
                     layer = "WSDACrop_2017", 
                     GDAL1_integer64_policy = TRUE)

WSDA_2017 <- filter_out_non_irrigated_shapefile(WSDA_2017) # 81682 rows
nrow(WSDA_2017)

##########
########## write TRUE shapefiles
##########

write_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                    "00_shapeFiles/0002_final_shapeFiles/0001_irrigated/")
if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = WSDA_2017, 
         dsn = paste0(write_dir, "/WSDA_Irrigated_2017/"), 
         layer="WSDA_Irrigated_2017", 
         driver="ESRI Shapefile")

############################################################################
############################################################################
############################################################################

Eastern_no_Grant <- pick_eastern_counties_noGrant(WSDA_2017)
Grant <- WSDA_2017[grepl('Grant', WSDA_2017$county), ]

nrow(WSDA_2017)
nrow(Grant)
nrow(Eastern_no_Grant)

write_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                    "00_shapeFiles/0002_final_shapeFiles/0002_Eastern_Irrigated/")
if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = Eastern_no_Grant, 
         dsn = paste0(write_dir, "/Eastern_noGrant_Irrigated_2017/"), 
         layer="Eastern_noGrant_Irrigated_2017", 
         driver="ESRI Shapefile")

# 74064 rows
Eastern <- raster::bind(Eastern_no_Grant, Grant)

write_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                    "00_shapeFiles/0002_final_shapeFiles/0002_Eastern_Irrigated/")
if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = Eastern, 
         dsn = paste0(write_dir, "/Eastern_Irrigated_2017/"), 
         layer="Eastern_Irrigated_2017", 
         driver="ESRI Shapefile")


################################
write_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                    "00_shapeFiles/0002_final_shapeFiles/")
if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = Grant, 
         dsn = paste0(write_dir, "/Grant_Irrigated_2017/"), 
         layer="Grant_Irrigated_2017", 
         driver="ESRI Shapefile")



