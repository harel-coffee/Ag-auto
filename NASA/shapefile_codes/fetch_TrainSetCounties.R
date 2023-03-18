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
                  "00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/")

write_dir <- paste0("/Users/hn/Documents/01_research_data/NASA/shapefiles/")
##############################################################################

WSDA <- readOGR(paste0(data_dir, "Eastern_", 2017, "/Eastern_", 2017, ".shp"),
                       layer = paste0("Eastern_", 2017), 
                       GDAL1_integer64_policy = TRUE)


Grant <- WSDA[grepl('Grant', WSDA$county), ]

writeOGR(obj = Grant, 
         dsn = paste0(write_dir, "/", "Grant2017"), 
         layer = "Grant2017", 
         driver = "ESRI Shapefile")


WSDA <- readOGR(paste0(data_dir, "Eastern_", 2016, "/Eastern_", 2016, ".shp"),
                       layer = paste0("Eastern_", 2016), 
                       GDAL1_integer64_policy = TRUE)

Adam <- WSDA[grepl('Adam', WSDA$county), ]
Benton <- WSDA[grepl('Benton', WSDA$county), ]
Adam <- raster::bind(Adam, Benton)

writeOGR(obj = Adam, 
         dsn = paste0(write_dir, "/", "AdamBenton2016"), 
         layer = "AdamBenton2016", 
         driver = "ESRI Shapefile")

Adam <- Adam@data
write.csv(Adam, 
          "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/AdamBenton2016.csv", row.names = F)


WSDA <- readOGR(paste0(data_dir, "Eastern_", 2018, "/Eastern_", 2018, ".shp"),
                       layer = paste0("Eastern_", 2018), 
                       GDAL1_integer64_policy = TRUE)

Franklin <- WSDA[grepl('Franklin', WSDA$county), ]
Yakima <- WSDA[grepl('Yakima', WSDA$county), ]
Franklin <- raster::bind(Franklin, Yakima)

writeOGR(obj = Franklin, 
         dsn = paste0(write_dir, "/", "FranklinYakima2018"), 
         layer = "FranklinYakima2018", 
         driver = "ESRI Shapefile")


Franklin <- Franklin@data
write.csv(Franklin, 
          "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/FranklinYakima2018.csv", row.names = F)



# which year is the walla walla surveyed most
for (yr in c(2015, 2016, 2017, 2018)){
    WSDA <- readOGR(paste0(data_dir, "Eastern_", yr, "/Eastern_", yr, ".shp"),
                    layer = paste0("Eastern_", yr), 
                     GDAL1_integer64_policy = TRUE)
    walla <- WSDA[grepl('Walla Walla', WSDA$county), ]
    walla <- walla@data
    walla <- filter_lastSrvyDate_DataTable(walla, yr)
    print (paste0("no. fields in year ", yr, " is ", length(unique(walla$ID))))
}


WSDA <- readOGR(paste0(data_dir, "Eastern_", 2015, "/Eastern_", 2015, ".shp"),
                       layer = paste0("Eastern_", 2015), 
                       GDAL1_integer64_policy = TRUE)

Walla <- WSDA[grepl('Walla Walla', WSDA$county), ]

writeOGR(obj = Walla, 
         dsn = paste0(write_dir, "/", "Walla2015"), 
         layer = "Walla2015", 
         driver = "ESRI Shapefile")


Walla <- Walla@data
write.csv(Walla, 
          "/Users/hn/Documents/01_research_data/NASA/data_part_of_shapefile/Walla2015.csv", row.names = F)

