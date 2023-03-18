rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
library(sf)
library(foreign)

dir <- "/Users/hn/Documents/01_research_data/remote_sensing/00_shapeFiles/02_correct_years/05_filtered_shapefiles/Grant/all_fields/"

Grant_2015 <- readOGR(paste0(dir, "Grant_2015/Grant_2015.shp"),
              layer = "Grant_2015", 
              GDAL1_integer64_policy = TRUE)

Grant_2016 <- readOGR(paste0(dir, "Grant_2016/Grant_2016.shp"),
                      layer = "Grant_2016", 
                      GDAL1_integer64_policy = TRUE)

Grant_2017 <- readOGR(paste0(dir, "Grant_2017/Grant_2017.shp"),
                      layer = "Grant_2017", 
                      GDAL1_integer64_policy = TRUE)

Grant_2018 <- readOGR(paste0(dir, "Grant_2018/Grant_2018.shp"),
                      layer = "Grant_2018", 
                      GDAL1_integer64_policy = TRUE)

setnames(Grant_2015@data, old=c("DataSrc"), new=c("Source"))
setnames(Grant_2016@data, old=c("DataSrc"), new=c("Source"))
setnames(Grant_2017@data, old=c("DataSrc"), new=c("Source"))
setnames(Grant_2018@data, old=c("DataSrc"), new=c("Source"))

common_cols <- c("county", "CropTyp", "ExctAcr", "ID", "IntlSrD", 
                 "Irrigtn", "LstSrvD", "Notes", "RtCrpTy", "Shap_Ar", "Shp_Lng", "Source", "TRS", "year")

Grant_2015 <- Grant_2015[, (names(Grant_2015) %in% common_cols)]
Grant_2016 <- Grant_2016[, (names(Grant_2016) %in% common_cols)]
Grant_2017 <- Grant_2017[, (names(Grant_2017) %in% common_cols)]
Grant_2018 <- Grant_2018[, (names(Grant_2018) %in% common_cols)]

Grant_2015_2018 <- bind(Grant_2015, Grant_2016, Grant_2017, Grant_2018)


write_dir <- dir

if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}


writeOGR(obj = Grant_2015_2018, 
         dsn = paste0(write_dir, "/Grant_2015_2018/"), 
         layer="Grant_2015_2018", 
         driver="ESRI Shapefile")


