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

data_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                    "remote_sensing/00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/Eastern_2017/")

WSDA_2017 <- readOGR(paste0(data_dir, "/Eastern_2017.shp"),
                     layer = "Eastern_2017", 
                     GDAL1_integer64_policy = TRUE)

sort(colnames(WSDA_2017@data))

WSDA_2017@data$Notes <- tolower(WSDA_2017@data$Notes)
WSDA_2017@data$CropTyp <- tolower(WSDA_2017@data$CropTyp)
WSDA_2017@data$DataSrc <- tolower(WSDA_2017@data$DataSrc)
WSDA_2017@data$Irrigtn <- tolower(WSDA_2017@data$Irrigtn)

Grant <- WSDA_2017[grepl('Grant', WSDA_2017$county), ]

IDs <- c("107433_WSDA_SF_2017", "99423_WSDA_SF_2017", "108447_WSDA_SF_2017")

Grant <- Grant[Grant@data$ID %in% IDs, ]


write_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                    "remote_sensing/00_shapeFiles/Microsof_shapeFiles/")

if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = Grant, 
         dsn = paste0(write_dir, "/threeFieldsGrant_2017/"), 
         layer="threeFieldsGrant_2017", 
         driver="ESRI Shapefile")


