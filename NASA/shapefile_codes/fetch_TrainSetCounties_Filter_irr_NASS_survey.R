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

data_dir <- paste0("/Users/hn/Documents/01_research_data/NASA/shapefiles/")
write_dir <- paste0("/Users/hn/Documents/01_research_data/NASA/shapefiles/")
##############################################################################

Grant <- readOGR(paste0(data_dir, "Grant2017/Grant2017.shp"),
                       layer = "Grant2017", 
                       GDAL1_integer64_policy = TRUE)

Grant <- Grant[Grant@data$DataSrc != "nass", ]
Grant <- filter_out_non_irrigated_shapefile(Grant)
Grant <- Grant[grepl('2017', Grant@data$LstSrvD), ]

writeOGR(obj = Grant, 
         dsn = paste0(write_dir, "/", "Grant2017_irr_NASSout_survCorrect"), 
         layer = "Grant2017_irr_NASSout_survCorrect", 
         driver = "ESRI Shapefile")


AdamBenton2016 <- readOGR(paste0(data_dir, "AdamBenton2016/AdamBenton2016.shp"),
                       layer = "AdamBenton2016", 
                       GDAL1_integer64_policy = TRUE)

AdamBenton2016 <- AdamBenton2016[AdamBenton2016@data$DataSrc != "nass", ]
AdamBenton2016 <- filter_out_non_irrigated_shapefile(AdamBenton2016)
AdamBenton2016 <- AdamBenton2016[grepl('2016', AdamBenton2016@data$LstSrvD), ]

writeOGR(obj = AdamBenton2016, 
         dsn = paste0(write_dir, "/", "AdamBenton2016_irr_NASSout_survCorrect"), 
         layer = "AdamBenton2016_irr_NASSout_survCorrect", 
         driver = "ESRI Shapefile")



Franklin <- readOGR(paste0(data_dir, "FranklinYakima2018/FranklinYakima2018.shp"),
                       layer = "FranklinYakima2018", 
                       GDAL1_integer64_policy = TRUE)

Franklin <- Franklin[Franklin@data$DataSrc != "nass", ]
Franklin <- filter_out_non_irrigated_shapefile(Franklin)
Franklin <- Franklin[grepl('2018', Franklin@data$LstSrvD), ]

writeOGR(obj = Franklin, 
         dsn = paste0(write_dir, "/", "FranklinYakima2018_irr_NASSout_survCorrect"), 
         layer = "FranklinYakima2018_irr_NASSout_survCorrect", 
         driver = "ESRI Shapefile")



Walla <- readOGR(paste0(data_dir, "Walla2015/Walla2015.shp"),
                       layer = "Walla2015", 
                       GDAL1_integer64_policy = TRUE)

Walla <- Walla[Walla@data$DataSrc != "nass", ]
Walla <- filter_out_non_irrigated_shapefile(Walla)
Walla <- Walla[grepl('2015', Walla@data$LstSrvD), ]
writeOGR(obj = Walla, 
         dsn = paste0(write_dir, "/", "Walla2015_irr_NASSout_survCorrect"), 
         layer = "Walla2015_irr_NASSout_survCorrect", 
         driver = "ESRI Shapefile")

