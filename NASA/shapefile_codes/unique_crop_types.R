####
#### This module generates #s in the following link ()
#### https://docs.google.com/document/d/18KX24FkL70_Xhxagwx9EBRWeQmz-Ud-iuTXqnf9YXnk/edit?usp=sharing
####

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

##############################################################################

data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                   "00_shapeFiles/01_not_correct_years/01_true_shapefiles_separate_years/")

##############################################################################
crop_types = c("CropTypes")
for (yr in c(2012:2018)){
    WSDA <- readOGR(paste0(data_dir, "WSDACrop_", yr, "/WSDACrop_", yr, ".shp"),
                            layer = paste0("WSDACrop_", yr), 
                            GDAL1_integer64_policy = TRUE)
    WSDA <- WSDA@data
    WSDA$CropTyp <- tolower(WSDA$CropTyp)
    cc <- unique(WSDA$CropTyp)
    crop_types <- c(crop_types, cc)
}

# monteray
Mont <- readOGR("/Users/hn/Documents/01_research_data/NASA/shapefiles/Monterey/2014_Crop_Monterey_CDL.shp",
                       layer = "2014_Crop_Monterey_CDL", 
                       GDAL1_integer64_policy = TRUE)

Mont <- Mont@data
Mont <- Mont %>% filter(County == "Monterey") %>% data.table()
setnames(Mont, old=c("Crop2014", "Acres", "County"), new=c("CropTyp", "ExctAcr", "county"))
Mont$CropTyp <- tolower(Mont$CropTyp)

crop_types <- c(crop_types, unique(Mont$CropTyp))

crop_types <- sort(unique(crop_types))

crop_types <- crop_types[!crop_types %in% c("CropTypes")]
crop_types <- tolower(crop_types)
crop_types <- sort(unique(crop_types))
write.csv(crop_types, 
          file = "/Users/hn/Documents/01_research_data/NASA/comprehensive_cropTypes.csv",
          row.names=FALSE)


