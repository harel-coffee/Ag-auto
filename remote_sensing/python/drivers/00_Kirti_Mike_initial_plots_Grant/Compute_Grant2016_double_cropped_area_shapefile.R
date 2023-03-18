
##########################################################################################
######
######   Compute_Grant_double_cropped_area_2016_shapefile
######
######
##########################################################################################
rm(list=ls())
library(data.table)
library(dplyr)
library(foreign)
library(rgdal)

data_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                   "remote_sensing/00_shapeFiles/02_correct_years/", 
                   "05_filtered_shapefiles/Grant/Grant_2016/")

param_dir <- "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"


Grant_2016 <- rgdal::readOGR(paste0(data_dir, "/Grant_2016.shp"),
                             layer = "Grant_2016", 
                             GDAL1_integer64_policy = TRUE)
Grant_2016 <- Grant_2016@data

##########################################################################################
###
###  filter double cropped
###
##########################################################################################
Grant_2016$Notes <- tolower(Grant_2016$Notes)

Grant_2016_double <- Grant_2016[grepl('double', Grant_2016$Notes), ]
Grant_2016_dbl <- Grant_2016[grepl('dbl', Grant_2016$Notes), ]

Grant_2016_double <- rbind(Grant_2016_double, Grant_2016_dbl)
Grant_2016_double <- data.table(Grant_2016_double)

sum(Grant_2016_double$ExctAcr)

grant_2016_double_acr_by_cultivar <- Grant_2016_double %>% 
                                     group_by(county, year, CropTyp) %>% 
                                     summarise(acreage_by_plant=sum(ExctAcr)) %>% 
                                     data.table()

write.table(grant_2016_double_acr_by_cultivar, 
            file = paste0(data_dir, "grant_2016_all_Notes_double_acr_by_cultivar.csv"),
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

double_crop_potential_plants <- read.csv(paste0(param_dir, "double_crop_potential_plants.csv"), 
                                         as.is=TRUE)

Grant_2016_double <- Grant_2016_double %>% 
                     filter(CropTyp %in% double_crop_potential_plants$Crop_Type) %>%
                     data.table()

sum(Grant_2016_double$ExctAcr) # 6138.336



grant_2016_2crop_acr_by_cultivar <- Grant_2016_double %>% 
                                    group_by(county, year, CropTyp) %>% 
                                    summarise(acreage_by_plant=sum(ExctAcr)) %>% 
                                    data.table()

write.table(grant_2016_2crop_acr_by_cultivar, 
            file = paste0(data_dir, "grant_2016_potential_2Crop_Notes_acr_by_cultivar.csv"),
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")



