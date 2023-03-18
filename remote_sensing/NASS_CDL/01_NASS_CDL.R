rm(list=ls())
library(foreign)
library(data.table)
library(dplyr)
library(rgdal)
library(sp)


download_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                       "remote_sensing/NASS_CDL/download_by_cdlTools/")

WA_2019 <- cdlTools::getCDL(x = 'Washington', year = 2019, location = download_dir, ssl.verifypeer = FALSE)
WA_2018 <- cdlTools::getCDL(x = 'Washington', year = 2018, location = download_dir, ssl.verifypeer = FALSE)
WA_2017 <- cdlTools::getCDL(x = 'Washington', year = 2017, location = download_dir, ssl.verifypeer = FALSE)

# you can get counts with table
WA_2019_table <- table(raster::getValues(WA_2019[[1]]))
WA_2018_table <- table(raster::getValues(WA_2018[[1]]))
WA_2017_table <- table(raster::getValues(WA_2017[[1]]))


crop_count_2019 <- data.frame( crop = cdlTools::updateNamesCDL(names(WA_2019_table)), 
                               WA_2019_table , 
                               stringsAsFactors=FALSE)

crop_count_2018 <- data.frame( crop = cdlTools::updateNamesCDL(names(WA_2018_table)), 
                               WA_2018_table , 
                               stringsAsFactors=FALSE)

crop_count_2017 <- data.frame( crop = cdlTools::updateNamesCDL(names(WA_2017_table)), 
                               WA_2017_table , 
                               stringsAsFactors=FALSE)


crop_count_2019$crop <- tolower(crop_count_2019$crop)
crop_count_2018$crop <- tolower(crop_count_2018$crop)
crop_count_2017$crop <- tolower(crop_count_2017$crop)

crop_count_2019 <- crop_count_2019[grepl('dbl', crop_count_2019$crop), ]
crop_count_2018 <- crop_count_2018[grepl('dbl', crop_count_2018$crop), ]
crop_count_2017 <- crop_count_2017[grepl('dbl', crop_count_2017$crop), ]


crop_count_2019
crop_count_2018
crop_count_2017

sum(crop_count_2019$Freq)
sum(crop_count_2018$Freq)
sum(crop_count_2017$Freq)


