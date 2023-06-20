
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


data_dir = "/Users/hn/Documents/01_research_data/NASA/shapefiles/2022_survey/"


SSF_name = "Selected_Fields_Survery_WSDA2020"
WSDA <- readOGR(paste0(data_dir, "2022_survey_Supriya/", SSF_name, ".shp"),
                layer = SSF_name, 
                GDAL1_integer64_policy = TRUE)

# ASF - Amin Shapefile
ASF_name <- "Dcrop_FieldSurveyLocation_WSU_Students_2022"
ASF <- readOGR(paste0(data_dir, ASF_name, "/", ASF_name, ".shp"),
               layer = ASF_name, 
               GDAL1_integer64_policy = TRUE)


WSDA@data <- within(WSDA@data, remove("FourthSu_1", "FourthSu_2", 
                                      "Notes3", "ThirdField", "CropGroup",
                                      "Shape_Le_1", "Shape_Area", "OBJECTID_1"))

writeOGR(obj = WSDA, 
         dsn = paste0(data_dir, "/WSU_Students_2022_GEE/"), 
         layer="WSU_Students_2022_GEE", 
         driver="ESRI Shapefile")



WSDA_data <- WSDA@data

write.csv(WSDA_data, 
          paste0(data_dir, "WSU_Students_2022_GEE/", "WSU_Students_2022_GEE_data.csv"), row.names = F)
