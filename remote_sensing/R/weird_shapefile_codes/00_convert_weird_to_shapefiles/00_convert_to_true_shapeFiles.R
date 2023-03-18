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
                   "/remote_sensing/00_shapeFiles/00_WSDA_original_weird_files/")
                     

weird_2012_2018_dir <- paste0(data_dir, "/2012_2018_shapefile.gdb")

# list the layer names in their to read desired layer
# ogrInfo(weird_2012_2018_dir, require_geomType="wkbPolygon")
ogrListLayers(weird_2012_2018_dir);
# ogrInfo(weird_2012_2018_dir)

gdb <- path.expand(weird_2012_2018_dir)
# WSDACrop_2012 <- readOGR(gdb, "WSDACrop_2012")
# WSDACrop_2013 <- readOGR(gdb, "WSDACrop_2013")
# WSDACrop_2014 <- readOGR(gdb, "WSDACrop_2014")

WSDACrop_2015 <- readOGR(gdb, "WSDACrop_2015")

# multiple incompatible geometries
WSDACrop_2016 <- readOGR(gdb, "WSDACrop_2016")
# WSDACrop_2018 <- readOGR(gdb, "WSDACrop_2018")


########################################################################################
######
######     add the goddamn numeric identifier.
######
########################################################################################

# WSDACrop_2012 <- add_identifier(dt_df=WSDACrop_2012, year="2012")
# WSDACrop_2013 <- add_identifier(dt_df=WSDACrop_2013, year="2013")
# WSDACrop_2014 <- add_identifier(dt_df=WSDACrop_2014, year="2014")

WSDACrop_2015 <- add_identifier(dt_df=WSDACrop_2015, year="2015")
WSDACrop_2016 <- add_identifier(dt_df=WSDACrop_2016, year="2016")

# WSDACrop_2017 <- add_identifier(dt_df=WSDACrop_2017, year="2017")
# WSDACrop_2018 <- add_identifier(dt_df=WSDACrop_2018, year="2018")

########################################################################################

# After updating R we receive some error about polygons. So, we need to add (require_geomType="wkbPolygon") at the end.
# WSDACrop_2018 <- readOGR(gdb, "WSDACrop_2018", require_geomType="wkbPolygon") 

# setnames(WSDACrop_2012@data, old=c("Rt1CrpT", "County", "SHAPE_Area", "SHAPE_Length"), 
#                              new=c("RtCrpTy", "county", "Shap_Ar", "Shp_Lng"))

# setnames(WSDACrop_2013@data, old=c("County", "Shape_Length", "Shape_Area"), 
#                              new=c("county", "Shp_Lng", "Shap_Ar"))

# setnames(WSDACrop_2014@data, old=c("Rt1CrpT", "LstSrvy", "County"), 
#                              new=c("RtCrpTy", "LstSrvD", "county"))

setnames(WSDACrop_2015@data, old=c("SHAPE_Area", "SHAPE_Length", "County", "RotCropType", "Acres"), 
                             new=c("Shap_Ar", "Shp_Lng", "county", "RtCrpTy", "TotalAcres"))

setnames(WSDACrop_2016@data, old=c("SHAPE_Area", "SHAPE_Length", "County"), 
                             new=c("Shap_Ar", "Shp_Lng", "county"))

# setnames(WSDACrop_2017@data, old=c("SHAPE_Area", "SHAPE_Length", "RotationCropType", "County", "Acres"), 
#                              new=c("Shap_Ar", "Shp_Lng", "RtCrpTy", "county", "TotalAcres"))

setnames(WSDACrop_2018@data, old=c("RttnCrp", "County"), 
                             new=c("RtCrpTy", "county"))

# WSDACrop_2012@data$year <- paste0("2012_shapeFile")
# WSDACrop_2013@data$year <- paste0("2013_shapeFile")
# WSDACrop_2014@data$year <- paste0("2014_shapeFile")
WSDACrop_2015@data$year <- paste0("2015_shapeFile")
WSDACrop_2016@data$year <- paste0("2016_shapeFile")
# WSDACrop_2017@data$year <- paste0("2017_shapeFile")
# WSDACrop_2018@data$year <- paste0("2018_shapeFile")
##########
########## write TRUE shapefiles
##########

write_dir <- paste0(data_dir, "/01_true_shapefiles_separate_years/")
if (dir.exists(file.path(write_dir)) == F){
  dir.create(path=file.path(write_dir), recursive=T)
}

# writeOGR(obj = WSDACrop_2012, 
#          dsn = paste0(write_dir, "/WSDACrop_2012/"), 
#          layer="WSDACrop_2012", 
#          driver="ESRI Shapefile")

# writeOGR(obj = WSDACrop_2013, 
#          dsn = paste0(write_dir, "/WSDACrop_2013/"), 
#          layer="WSDACrop_2013", 
#          driver="ESRI Shapefile")

# writeOGR(obj = WSDACrop_2014, 
#          dsn = paste0(write_dir, "/WSDACrop_2014/"), 
#          layer="WSDACrop_2014", 
#          driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2015, 
         dsn = paste0(write_dir, "/WSDACrop_2015/"), 
         layer="WSDACrop_2015", 
         driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2016, 
         dsn = paste0(write_dir, "/WSDACrop_2016/"), 
         layer="WSDACrop_2016", 
         driver="ESRI Shapefile")

# writeOGR(obj = WSDACrop_2017, 
#          dsn = paste0(write_dir, "/WSDACrop_2017/"), 
#          layer="WSDACrop_2017", 
#          driver="ESRI Shapefile")

writeOGR(obj = WSDACrop_2018, 
         dsn = paste0(write_dir, "/WSDACrop_2018/"), 
         layer="WSDACrop_2018", 
         driver="ESRI Shapefile")

