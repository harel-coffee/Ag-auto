#######################################################
#####
#####    Aug. 20, 2021
#####
#####    Goal: 
#####        Pick up 4 fields (2 double-cropped and 2 single-cropped)
#####        to look at their NDVI/EVI.
#####
#####   Why? I do not know.
##### 
#####

rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
library(sf)
library(foreign)



##############################################################################################################
#
#    Directory setup
#
SF_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                 "00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/Eastern_2017/")


output_dir <- paste0("/Users/hn/Documents/01_research_data/NASA/", 
                     "shapefiles/01_4_locations_investigation/")

if (dir.exists(output_dir) == F) {dir.create(path = output_dir, recursive = T)}

##############################################################################################################
#
#    Parameter setup
#

IDs <- c("104670_WSDA_SF_2017", "105429_WSDA_SF_2017", # single
         "106054_WSDA_SF_2017", "106534_WSDA_SF_2017") # double

IDs <- c("104563_WSDA_SF_2017", "102385_WSDA_SF_2017", # single
         "102309_WSDA_SF_2017", "103372_WSDA_SF_2017") # double

# good_IDs (bigger acrages)
IDs <- c("105429_WSDA_SF_2017", "104563_WSDA_SF_2017", # single
         "102309_WSDA_SF_2017", "106054_WSDA_SF_2017") # double

##############################################################################################################
#
#    Body
#

SF <- readOGR(paste0(SF_dir, "Eastern_2017.shp"),
                     layer = "Eastern_2017", 
                     GDAL1_integer64_policy = TRUE)

selected_SF <- SF[SF@data$ID %in% IDs, ]

SF_centers <- rgeos::gCentroid(selected_SF, byid=TRUE)
crs <- CRS("+proj=lcc 
            +lat_1=45.83333333333334 
            +lat_2=47.33333333333334 
            +lat_0=45.33333333333334 
            +lon_0=-120.5 +datum=WGS84")

centroid_coord <- spTransform(SF_centers, CRS("+proj=longlat +datum=WGS84"))
centroid_coord_dt <- data.table(centroid_coord@coords)
setnames(centroid_coord_dt, old=c("x", "y"), new=c("centroid_long", "centroid_lat"))

centroid_coord_dt$ID <- selected_SF@data$ID

selected_SF@data <- dplyr::left_join(x = selected_SF@data,
                                     y = centroid_coord_dt, 
                                     by = "ID")

writeOGR(obj = selected_SF, 
         dsn = paste0(output_dir, "/Grant_4Fields_poly_wCentroids/"), 
         layer="Grant_4Fields_poly_wCentroids", 
         driver="ESRI Shapefile")


write.csv(centroid_coord_dt, 
          file=paste0(output_dir, "4centroids.csv"), 
          row.names=FALSE)




