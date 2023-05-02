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

data_dir <- paste0("/Users/hn/Documents/01_research_data/NASA/shapefiles/00_WSDA_separateYears/")

##############################################################################

for (yr in c(2008:2018)){
    WSDA <- readOGR(paste0(data_dir, "/WSDA_EW_", yr, ".shp"),
                            layer = paste0("WSDA_EW_", yr), 
                            GDAL1_integer64_policy = TRUE)
    WSDA <- WSDA@data
    WSDA$CropType <- tolower(WSDA$CropType)
    write.csv(WSDA, file = paste0(data_dir, "WSDA_DataTable_", yr, ".csv"), row.names=FALSE)
}

# Monteray
# Mont <- readOGR("/Users/hn/Documents/01_research_data/NASA/shapefiles/Monterey/2014_Crop_Monterey_CDL.shp",
#                        layer = "2014_Crop_Monterey_CDL", 
#                        GDAL1_integer64_policy = TRUE)

# Mont <- Mont@data
# Mont <- Mont %>% filter(County == "Monterey") %>% data.table()
# setnames(Mont, old=c("Crop2014", "Acres", "County"), new=c("CropTyp", "ExctAcr", "county"))
# Mont$CropTyp <- tolower(Mont$CropTyp)

# crop_types <- c(crop_types, unique(Mont$CropTyp))

# crop_types <- sort(unique(crop_types))

# crop_types <- crop_types[!crop_types %in% c("CropTypes")]
# crop_types <- tolower(crop_types)
# crop_types <- sort(unique(crop_types))
# write.csv(crop_types, 
#           file = "/Users/hn/Documents/01_research_data/NASA/comprehensive_cropTypes.csv",
#           row.names=FALSE)


