
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

in_dir_base <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                      "00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/")
hack_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/Hackathon_ShapeFile/"

################################################################################
yr = 2017
in_dir <- paste0(in_dir_base, "Eastern_", yr, "/")
Eastern_2017 <- readOGR(paste0(in_dir, "Eastern_", yr, ".shp"),
                        layer = paste0("Eastern_", yr), 
                        GDAL1_integer64_policy = TRUE)

Grant_2017 <- Eastern_2017[grepl("Grant", Eastern_2017$county), ]

# toss NASS and filter by last survey date
Grant_2017 <- filter_lastSrvyDate(dt=Grant_2017, year=yr)
Grant_2017 <- Grant_2017[Grant_2017@data$DataSrc != "nass", ]

################################################################################

yr = 2018
in_dir <- paste0(in_dir_base, "Eastern_", yr, "/")
Eastern <- readOGR(paste0(in_dir, "Eastern_", yr, ".shp"),
                   layer = paste0("Eastern_", yr), 
                   GDAL1_integer64_policy = TRUE)

Yakima <- Eastern[grepl("Yakima", Eastern$county), ]
dim(Yakima)

# toss NASS and filter by last survey date
Yakima <- filter_lastSrvyDate(dt=Yakima, year=yr)
dim(Yakima)

Yakima <- Yakima[Yakima@data$DataSrc != "nass", ]
dim(Yakima)


################################################################################


Grant_Yakima <- raster::bind(Grant_2017, Yakima)


write_dir <- hack_dir

if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

writeOGR(obj = Grant_Yakima, 
         dsn = paste0(write_dir, "/Grant_Yakima/"), 
         layer="Grant_Yakima", 
         driver="ESRI Shapefile")


