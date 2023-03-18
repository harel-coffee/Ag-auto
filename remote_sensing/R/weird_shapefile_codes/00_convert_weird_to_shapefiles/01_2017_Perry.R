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
                   "00_shapeFiles/00_WSDA_original_weird_files/WSDACrop2017/")

WWA <- readOGR(paste0(data_dir, "/WWA.shp"),
               layer = "WWA", 
               GDAL1_integer64_policy = TRUE)

Palouse <- readOGR(paste0(data_dir, "/Palouse.shp"),
               layer = "Palouse", 
               GDAL1_integer64_policy = TRUE)

NE <- readOGR(paste0(data_dir, "/NE.shp"),
               layer = "NE", 
               GDAL1_integer64_policy = TRUE)

ColumbiaBasin <- readOGR(paste0(data_dir, "/ColumbiaBasin.shp"),
               layer = "ColumbiaBasin", 
               GDAL1_integer64_policy = TRUE)

Central <- readOGR(paste0(data_dir, "/Central.shp"),
               layer = "Central", 
               GDAL1_integer64_policy = TRUE)

WSDA_2017 <- raster::bind(WWA, Palouse, NE, ColumbiaBasin, Central)
########################################################################################
######
######     add the goddamn numeric identifier.
######
########################################################################################

WSDA_2017 <- add_identifier(dt_df=WSDA_2017, year="2017")

########################################################################################
WSDA_2017@data <- within(WSDA_2017@data, remove("CropType_1", "OBJECTID", "OID_"))

setnames(WSDA_2017@data, old=c("SHAPE_Area", "SHAPE_Leng", "County", "Irrigation", "CropGroup", "CropType", "DataSource", "InitialSur", "LastSurvey", "ExactAcres", "RotationCr"), 
                         new=c("Shap_Ar",    "Shp_Lng",    "county", "Irrigtn",    "CropGrp",   "CropTyp",  "DataSrc",    "IntlSrD",    "LstSrvD" ,   "ExctAcr",    "RtCrpTy"))

WSDA_2017@data$Notes <- tolower(WSDA_2017@data$Notes)
WSDA_2017@data$CropTyp <- tolower(WSDA_2017@data$CropTyp)
WSDA_2017@data$DataSrc <- tolower(WSDA_2017@data$DataSrc)
WSDA_2017@data$Irrigtn <- tolower(WSDA_2017@data$Irrigtn)

# WSDA_2017@data$year <- paste0("2017_shapeFile")

WSDA_2017 <- transfer_projection_to_lat_long(WSDA_2017) # 206277 rows

##########
########## write TRUE shapefiles
##########

write_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                    "00_shapeFiles/01_not_correct_years/", 
                    "01_true_shapefiles_separate_years/")

if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = WSDA_2017, 
         dsn = paste0(write_dir, "/WSDACrop_2017/"), 
         layer = "WSDACrop_2017", 
         driver = "ESRI Shapefile")



