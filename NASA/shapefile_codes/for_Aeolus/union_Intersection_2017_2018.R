#######################################################
#####
#####    Oct. 6, 2021
#####
#####    Goal: 
#####        Merge/intersect shapefiles over years and get
#####        smaller fields, if their boundaries change over time.
##### 
#####

rm(list=ls())

.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(FNN)
# library(rgdal)
library(rgeos)
library(sp)
library(sf, lib.loc = "~/.local/lib/R3.5.1")
library(foreign)
library(raster) # For convenience, shapefile function and show method for Spatial objects
library(magrittr)
library(EnvStats, lib.loc = "~/.local/lib/R3.5.1")


source_path = "/home/hnoorazar/remote_sensing_codes/remote_core.R"
source(source_path)
options(digit=9)
options(digits=9)

##############################################################################################################
#
#    Directory setup
#
SF_dir <- paste0("/data/hydro/users/Hossein/NASA/000_Eastern_WA/")


output_dir <- SF_dir
if (dir.exists(output_dir) == F) {dir.create(path = output_dir, recursive = T)}

##############################################################################################################
#
#    Parameter setup
#
##############################################################################################################
#
#    Body
#
SF_2018 <- readOGR(paste0(SF_dir, "Eastern_2018/", "Eastern_2018.shp"),
                          layer = "Eastern_2018", 
                          GDAL1_integer64_policy = TRUE)

SF_2017 <- readOGR(paste0(SF_dir, "Eastern_2017/", "Eastern_2017.shp"),
                          layer = "Eastern_2017", 
                          GDAL1_integer64_policy = TRUE)

SF_2018 <- filter_out_non_irrigated_shapefile(SF_2018)
SF_2017 <- filter_out_non_irrigated_shapefile(SF_2017)

#########************************************************************************************
#
#     Method 1
#

#
# https://stackoverflow.com/questions/15075361/how-to-perform-a-vector-overlay-of-two-spatialpolygonsdataframe-objects
#
gFragment <- function(X, Y) {

  # the following 3 lines are from 
# https://gis.stackexchange.com/questions/223252/how-to-overcome-invalid-input-geom-and-self-intersection-when-intersecting-shape
  # while this function is from: 
# https://stackoverflow.com/questions/15075361/how-to-perform-a-vector-overlay-of-two-spatialpolygonsdataframe-objects
  # Lambert equal area with origin on the center of your area
  prj <- '+proj=laea +lat_0=10 +lon_0=-81 +ellps=WGS84 +units=m +no_defs'
  X <- X %>%
       spTransform(CRS(prj)) %>%
       gBuffer(byid=TRUE, width=0)

  Y <- Y %>%
       spTransform(CRS(prj)) %>%
       gBuffer(byid=TRUE, width=0)

  aa <- gIntersection(X, Y, byid = TRUE)
  bb <- gDifference(X, gUnionCascaded(Y), byid = T)
  cc <- gDifference(Y, gUnionCascaded(X), byid = T)
  ## Note: testing for NULL is needed in case any of aa, bb, or cc is empty,
  ## as when X & Y don't intersect, or when one is fully contained in the other
  SpatialPolygons(c(if(is.null(aa)) NULL else aa@polygons,
                    if(is.null(bb)) NULL else bb@polygons,
                    if(is.null(cc)) NULL else cc@polygons)) 
}

## Try it out
Fragments <- gFragment(SF_2018, SF_2017)

print (class(Fragments))

writeOGR(obj = Fragments, 
         dsn = paste0(output_dir, "/Intersect_EasternIrr_2017_2018/"), 
         layer="Intersect_EasternIrr_2017_2018", 
         driver="ESRI Shapefile")
