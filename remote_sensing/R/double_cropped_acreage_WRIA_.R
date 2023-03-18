rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
library(sf)
library(foreign)

##############################################################################################################
######
###### read WRIA shapeFile
######

data_dir <- paste0("/Users/hn/Documents/01_research_data/WR_WAT_WRIA.gdb/")
ogrListLayers(data_dir);
gdb <- path.expand(data_dir)

WRIA <- readOGR(gdb, "Water_Resource_Inventory_Areas")

##############################################################################################################
#
#    Directory setup
#
SF_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                 "00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/")

double_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                     "01_NDVI_TS/70_Cloud/00_Eastern_WA_withYear/2Years/", 
                     "06_list_of_2cropped_IDs_4_website/")


output_dir <- paste0("/Users/hn/Documents/00_GitHub/ag_papers/remote_sensing/WRIA_acreage_for_fabio/")
if (dir.exists(output_dir) == F) {dir.create(path = output_dir, recursive = T)}

##############################################################################################################
#
#    Parameter setup
#
years <- c(2016, 2017, 2018)

##############################################################################################################
#
#    Body
#
for (year in years){
  SF_name = paste0("Eastern_", year)
  double_name = paste0("doubleCroppedFields_", year,
                       "_onlyAnnuals_JustIrr_LastSrvFalse_NASSIn_SG_win7_Order3")

  SF <- readOGR(paste0(SF_dir, SF_name, "/", SF_name, ".shp"),
                layer = SF_name, 
                GDAL1_integer64_policy = TRUE)

  double_data <- data.table(read.csv(paste0(double_dir, double_name, ".csv"), as.is=TRUE))
  unique_IDs <- unique(double_data$ID)

  SF <- SF[SF@data$ID %in% unique_IDs, ]

  ####### Find Centroids
  SF_centers <- rgeos::gCentroid(SF, byid=TRUE)

  ####### the following is not necessary in our case. All are on the same page!
  crs <- CRS("+proj=lcc 
             +lat_1=45.83333333333334 
             +lat_2=47.33333333333334 
             +lat_0=45.33333333333334 
             +lon_0=-120.5 +datum=WGS84")

  centroid_coord <- spTransform(SF_centers, 
                                CRS("+proj=longlat +datum=WGS84"))

  centroid_coord_dt <- data.table(centroid_coord@coords)
  setnames(centroid_coord_dt, old=c("x", "y"), new=c("long", "lat"))

  #####
  ##### We need to assign each field('s centroid) to proper WRIA
  #####
  
  # coordinates(centroid_coord_dt) <- ~ long + lat  # This converts datatable to SpatialPoints
  # proj4string(centroid_coord_dt) <- proj4string(WRIA)
  # over(centroid_coord_dt, WRIA)


}



