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

# write_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
#                     "00_shapeFiles/0002_final_shapeFiles/")
# if (dir.exists(file.path(write_dir)) == F){
#   dir.create(path=file.path(write_dir), recursive=T)
# }

##########
########## Directories
##########

data_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                   "remote_sensing/00_shapeFiles/01_not_correct_years/", 
                   "01_true_shapefiles_separate_years/")

years = c("2015", "2016", "2018")

for (yr in years){
  WSDACrop <- readOGR(paste0(data_dir, "WSDACrop_", yr, "/WSDACrop_", yr, ".shp"),
                      layer = paste0("WSDACrop_", yr), 
                      GDAL1_integer64_policy = TRUE)
  
  ####################################################################################
  ######
  ###### Transform projections
  ######
  ####################################################################################

  WSDACrop <- transfer_projection_to_lat_long(WSDACrop)
  nrow(WSDACrop) # 2012: 170621

  ####################################################################################
  ######
  ######   Filter Irrigated
  ###### 
  ####################################################################################

  WSDACrop <- filter_out_non_irrigated_shapefile(WSDACrop)
  nrow(WSDACrop) # 2012: 64849

  #################################
  #########
  #########    Write Irrigated shapefiles 
  #########
  #################################

  Irrigated_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                          "remote_sensing/00_shapeFiles/0002_final_shapeFiles/", 
                          "0001_Irrigated/")
  
  if (dir.exists(file.path(Irrigated_dir)) == F){
    dir.create(path=file.path(Irrigated_dir), recursive=T)
   }
  
  writeOGR(obj = WSDACrop, 
           dsn = paste0(Irrigated_dir, "/WSDA_Irrigated_", yr, "/"), 
           layer = paste0("WSDA_Irrigated_", yr), 
           driver="ESRI Shapefile")

  ####################################################################################
  #####
  ######   Filter Eastern Counties
  ###### 
  ####################################################################################
  nrow(WSDACrop) # 2012: 64849
  WSDACrop <- pick_eastern_counties(WSDACrop)
  nrow(WSDACrop) # 2012: 59443

  Irrigated_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                          "remote_sensing/00_shapeFiles/0002_final_shapeFiles/", 
                          "0002_Eastern_Irrigated/")
  if (dir.exists(file.path(Irrigated_dir)) == F){
    dir.create(path=file.path(Irrigated_dir), recursive=T)
   }

  writeOGR(obj = WSDACrop, 
           dsn = paste0(Irrigated_dir, "/Eastern_Irrigated_", yr, "/"), 
           layer = paste0("Eastern_Irrigated_", yr), 
           driver="ESRI Shapefile")
  ####################################################################################
  ######
  ######   Filter Grant
  ###### 
  ####################################################################################
  Grant <- WSDACrop[grepl('Grant', WSDACrop$county), ]
  nrow(Grant) # 2012: 10861

  writeOGR(obj = Grant, 
         dsn = paste0(Irrigated_dir, "/Grant_Irrigated_", yr, "/"), 
         layer= paste0("Grant_Irrigated_", yr), 
         driver="ESRI Shapefile")


  ###################################################################################
  ######
  ######   Filter Eastern Counties no Grant
  ###### 
  ####################################################################################
  WSDACrop <- pick_eastern_counties_noGrant(WSDACrop)
  nrow(WSDACrop) # 2012: 48582

  writeOGR(obj = WSDACrop, 
         dsn = paste0(Irrigated_dir, "/Eastern_noGrant_Irrigated_", yr, "/"), 
         layer= paste0("Eastern_noGrant_Irrigated_", yr), 
         driver="ESRI Shapefile")

}




