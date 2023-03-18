


library(data.table)
library(rgdal)
library(dplyr)
library(sp)
library(sf)



data_dir <- "/Users/hn/Documents/01_research_data/NASA/shapefiles/"
files <- c("AdamBenton2016", "FranklinYakima2018", "Grant2017", "clean_Monterey", "Walla2015")

for (file in files){
  SF <- readOGR(paste0(data_dir, file, "/", file, ".shp"),
                layer = file, 
                GDAL1_integer64_policy = TRUE)
  
  SF_centroids <- rgeos::gCentroid(SF, byid=TRUE)
  SF_centroids_dt <- data.table(SF_centroids@coords)
  SF@data$ctr_lat = SF_centroids_dt$y
  SF@data$ctr_long = SF_centroids_dt$x

  writeOGR(obj=SF, 
           dsn = paste0(data_dir, file, "_centroids"), 
           layer= paste0(file, "_centroids"),
           driver="ESRI Shapefile")
}



data_dir <- "/Users/hn/Documents/01_research_data/NASA/shapefiles/train_SF_NASSout_Irr_CorrectYr/"
files <- c("AdamBenton2016_irr_NASSout_survCorrect", 
           "FranklinYakima2018_irr_NASSout_survCorrect", 
           "Grant2017_irr_NASSout_survCorrect",
           "Walla2015_irr_NASSout_survCorrect")

for (file in files){
  SF <- readOGR(paste0(data_dir, file, "/", file, ".shp"),
                layer = file, 
                GDAL1_integer64_policy = TRUE)
  
  SF_centroids <- rgeos::gCentroid(SF, byid=TRUE)
  SF_centroids_dt <- data.table(SF_centroids@coords)
  SF@data$ctr_lat = SF_centroids_dt$y
  SF@data$ctr_long = SF_centroids_dt$x

  writeOGR(obj=SF, 
           dsn = paste0(data_dir, file, "_centroids"), 
           layer= paste0(file, "_centroids"),
           driver="ESRI Shapefile")
}

