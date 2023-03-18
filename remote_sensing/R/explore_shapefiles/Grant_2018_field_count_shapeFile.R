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
                   "/00_shapeFiles/01_not_correct_years/", 
                   "01_true_shapefiles_separate_years/")

##############################################################################

WSDA_2018 <- readOGR(paste0(data_dir, "WSDACrop_2018/WSDACrop_2018.shp"),
                    layer = "WSDACrop_2018", 
                     GDAL1_integer64_policy = TRUE)

WSDA_2018 <- WSDA_2018@data

Grant_2018 <- WSDA_2018[grepl("Grant", WSDA_2018$county), ]


Grant_2018_double_by_notes <- filter_double_by_Notes(Grant_2018)
dim(Grant_2018_double_by_notes) # 328


Grant_2018_double_by_notes_No_NASS <- Grant_2018_double_by_notes %>%
                                      filter(DataSrc != "NASS") %>%
                                      data.table()
dim(Grant_2018_double_by_notes_No_NASS)


Grant2018_Irrigated_double_by_notes_NASS_out <- Grant2018_Irrigated_double_by_notes %>%
                                                filter(DataSrc != "NASS") %>%
                                                data.table()
sum(Grant2018_Irrigated_double_by_notes_NASS_out$ExctAcr)


######
######
######

param_dir <- "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"
double_crop_potential_plants <- read.csv(paste0(param_dir, "double_crop_potential_plants.csv"))

Grant_2018_double_by_notes_poten <- Grant_2018_double_by_notes %>% 
                                   filter(CropTyp %in% double_crop_potential_plants$Crop_Type)

dim(Grant_2018_double_by_notes_poten)

Grant_2018_double_by_notes_poten_NoNass <- Grant_2018_double_by_notes_poten %>% 
                                           filter(DataSrc != "NASS")
dim(Grant_2018_double_by_notes_poten_NoNass)



