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
  
  WSDACrop@data$Notes <- tolower(WSDACrop@data$Notes)
  WSDACrop@data$CropTyp <- tolower(WSDACrop@data$CropTyp)
  WSDACrop@data$DataSrc <- tolower(WSDACrop@data$DataSrc)
  WSDACrop@data$Irrigtn <- tolower(WSDACrop@data$Irrigtn)

  if ("Organic" %in% colnames(WSDACrop@data)){
    WSDACrop@data <- within(WSDACrop@data, remove("Organic"))
  }
 
  if ("CovrCrp" %in% colnames(WSDACrop@data)){
    WSDACrop@data <- within(WSDACrop@data, remove("CovrCrp"))
  }

  if ("TtlAcrs" %in% colnames(WSDACrop@data)){
    setnames(WSDACrop@data, old=c("TtlAcrs"), new=c("Acres"))
  }
  
  ####################################################################################
  ######
  ###### Transform projections
  ######
  ####################################################################################

  WSDACrop <- transfer_projection_to_lat_long(WSDACrop)


  ####################################################################################
  #####
  ######   Filter Eastern Counties
  ###### 
  ####################################################################################
  nrow(WSDACrop) # 2015: 191258
  
  WSDACrop <- pick_eastern_counties(WSDACrop)
  
  nrow(WSDACrop) # 2015: 162943

  write_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                      "remote_sensing/00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/")

  if (dir.exists(file.path(write_dir)) == F){
    dir.create(path=file.path(write_dir), recursive=T)
   }

  writeOGR(obj = WSDACrop, 
           dsn = paste0(write_dir, "/Eastern_", yr, "/"), 
           layer = paste0("Eastern_", yr), 
           driver="ESRI Shapefile")
}




