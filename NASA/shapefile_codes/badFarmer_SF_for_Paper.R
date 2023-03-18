

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



data_dir <- "/Users/hn/Documents/01_research_data/NASA/shapefiles/00000_train_SF_NASSout_Irr_CorrectYr/"

field_IDs = c("57593_WSDA_SF_2016",
              "162687_WSDA_SF_2015",
              "60617_WSDA_SF_2016",
              "39244_WSDA_SF_2018", 
              "46239_WSDA_SF_2018")


SF_Name = "Walla2015_irr_NASSout_survCorrect_centroids/Walla2015_irr_NASSout_survCorrect_centroids.shp"
WSDA_2015 <- readOGR(paste0(data_dir, SF_Name),
                     layer = "Walla2015_irr_NASSout_survCorrect_centroids", 
                     GDAL1_integer64_policy = TRUE)

SF_Name = "AdamBenton2016_irr_NASSout_survCorrect_centroids/AdamBenton2016_irr_NASSout_survCorrect_centroids.shp"
WSDA_2016 <- readOGR(paste0(data_dir, SF_Name),
                     layer = "AdamBenton2016_irr_NASSout_survCorrect_centroids", 
                     GDAL1_integer64_policy = TRUE)

SF_Name = "Grant2017_irr_NASSout_survCorrect_centroids/Grant2017_irr_NASSout_survCorrect_centroids.shp"
WSDA_2017 <- readOGR(paste0(data_dir, SF_Name),
                     layer = "Grant2017_irr_NASSout_survCorrect_centroids", 
                     GDAL1_integer64_policy = TRUE)

SF_Name = "FranklinYakima2018_irr_NASSout_survCorrect_centroids/FranklinYakima2018_irr_NASSout_survCorrect_centroids.shp"
WSDA_2018 <- readOGR(paste0(data_dir, SF_Name),
                     layer = "FranklinYakima2018_irr_NASSout_survCorrect_centroids", 
                     GDAL1_integer64_policy = TRUE)



WSDA_2015 <- WSDA_2015[WSDA_2015@data$ID %in% field_IDs, ]
WSDA_2016 <- WSDA_2016[WSDA_2016@data$ID %in% field_IDs, ]
WSDA_2017 <- WSDA_2017[WSDA_2017@data$ID %in% field_IDs, ]
WSDA_2018 <- WSDA_2018[WSDA_2018@data$ID %in% field_IDs, ]

badFarmers <- rbind(WSDA_2015, WSDA_2016, WSDA_2017, WSDA_2018)

writeOGR(obj = badFarmers, 
         dsn = paste0(data_dir, "/badFarmers/"), 
         layer="badFarmers", 
         driver="ESRI Shapefile")




