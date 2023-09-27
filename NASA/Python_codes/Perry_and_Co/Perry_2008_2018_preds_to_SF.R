library(data.table)
library(rgdal)
library(dplyr)
library(sp)
library(sf)



SF_in_dir <- "/Users/hn/Documents/01_research_data/NASA/shapefiles/10_intersect_East_Irr_2008_2018_2cols/"

file = "10_intersect_East_Irr_2008_2018_2cols"
SF <- readOGR(paste0(SF_in_dir, file, ".shp"),
                layer = file, 
                GDAL1_integer64_policy = TRUE)
  
SF_centroids <- rgeos::gCentroid(SF, byid=TRUE)
SF_centroids_dt <- data.table(SF_centroids@coords)
SF@data$ctr_lat = SF_centroids_dt$y
SF@data$ctr_long = SF_centroids_dt$x

preds_dir = "/Users/hn/Documents/01_research_data/NASA/Perry_2008_2018/"
preds = read.csv(paste0(preds_dir, "/Perry_2008_2018_preds.csv"))

setnames(preds, old = c('NDVI_SG_DL_p3'), new = c('NDVI_SG_DL'))


writeOGR(obj=SF,
         dsn = paste0(preds_dir, file), 
         layer= paste0(file, "_centroids"),
         driver="ESRI Shapefile")


##############

# data_dir = "/Users/hn/Documents/01_research_data/NASA/shapefiles/08_intersect_East_Irr_2008_2018/"
# WSDA_2008_2018 <- readOGR(paste0(data_dir, "Intersect_East_Irr_cleanCols_2008_2018.shp"),
#                           layer = "Intersect_East_Irr_cleanCols_2008_2018", 
#                           GDAL1_integer64_policy = TRUE)

  
# SF_centroids <- rgeos::gCentroid(WSDA_2008_2018, byid=TRUE)
# SF_centroids_dt <- data.table(SF_centroids@coords)
# WSDA_2008_2018@data$ctr_lat = SF_centroids_dt$y
# WSDA_2008_2018@data$ctr_long = SF_centroids_dt$x
