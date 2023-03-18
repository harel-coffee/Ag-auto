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
                   "remote_sensing/00_shapeFiles/0002_final_shapeFiles/", 
                   "000_Eastern_WA/Eastern_2017/")

Eastern_2017 <- readOGR(paste0(data_dir, "/Eastern_2017.shp"),
                     layer = "Eastern_2017", 
                     GDAL1_integer64_policy = TRUE)

sort(colnames(Eastern_2017@data))

Eastern_2017@data$Notes[10000]
Eastern_2017@data$CropTyp[1]
Eastern_2017@data$DataSrc[1]
Eastern_2017@data$Irrigtn[1]

Eastern_2017_sample <- Eastern_2017[1:100, ]

write_dir <- paste0("/Users/hn/Documents/01_research_data/", 
                    "remote_sensing/00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/")

if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = Eastern_2017_sample, 
         dsn = paste0(write_dir, "/small_shape_4_codeTest/"), 
         layer="Eastern_2017_sample", 
         driver="ESRI Shapefile")


