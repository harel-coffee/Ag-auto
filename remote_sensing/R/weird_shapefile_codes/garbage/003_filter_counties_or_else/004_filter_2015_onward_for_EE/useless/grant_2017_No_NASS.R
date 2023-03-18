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
data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                   "00_shapeFiles/02_correct_years/03_correct_years_separate/", 
                   "lat_long_projections/")

base_write <- paste0("/Users/hn/Documents/01_research_data//", 
                     "remote_sensing/00_shapeFiles/02_correct_years/", 
                     "05_filtered_shapefiles/batches/potential_fields/")

param_dir = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"

WSDACrop <- readOGR(paste0(data_dir, "WSDACrop_2017/WSDACrop_2017.shp"),
                    layer = "WSDACrop_2017", 
                    GDAL1_integer64_policy = TRUE)
WSDACrop$Notes <- tolower(WSDACrop$Notes)

# colnames(WSDACrop@data)[colnames(WSDACrop@data)=="Source"] <- "DataSrc"

# writeOGR(obj = WSDACrop,
#          dsn = paste0(data_dir, "WSDACrop_2017/"), 
#          layer="WSDACrop_2017", 
#          driver="ESRI Shapefile")
############################################################################
#######
#######         Counties of interest
#######
coi = c("Grant")

WSDACrop <- WSDACrop[WSDACrop@data$county %in% coi, ]
unique(WSDACrop@data$DataSrc)
#
# potential cultivars for double cropping
#
# WSDACrop <- WSDACrop[WSDACrop@data$CropTyp %in% double_crop_potential_plants$Crop_Type, ]
# dim(WSDACrop) # 

sfe_1 <- WSDACrop[WSDACrop@data$DataSrc != "NASS", ]
sfe_2 <- WSDACrop[WSDACrop@data$DataSrc == "NASS", ]
nrow(sfe_1@data)
nrow(sfe_2@data)
nrow(WSDACrop@data)


WSDACrop <- toss_Nass(WSDACrop)

NASS_dir <- paste0("/Users/hn/Documents/01_research_data", 
                   "/remote_sensing/00_shapeFiles/02_correct_years/", 
                   "05_filtered_shapefiles/Grant2017_NoNass_All_Plants/")
if (dir.exists(file.path(NASS_dir)) == F){
  dir.create(path=file.path(NASS_dir), recursive=T)
}

writeOGR(obj = WSDACrop,
         dsn = NASS_dir, 
         layer="Grant2017_No_Nass", 
         driver="ESRI Shapefile")















