############      packages      #############

library(data.table)
library(rgdal)
library(dplyr)
library(sp)
library(sf)


# a0000000c <-  sf::st_read(dsn = paste0(shape_file_dir,"file.gdb"))
# a0000000c <-  rgdal::readOGR(dsn = paste0(shape_file_dir, "file.gdb"))
# a0000000c <- foreign::read.dbf(dsn = paste0(shape_file_dir, "file.gdb"))
# a0000000c <- read.gdbtable(dsn= paste0(shape_file_dir, "file.gdb"))

############      Directories      #############

data_dir <- paste0("/Users/hn/Desktop/Desktop/Ag/", 
                   "check_point/remote_sensing/")

shape_file_dir <- paste0(data_dir, 
                         "2018_weird_shape_file/2018WSDACrop.gdb") 

##############################################
############                     #############
############   read fips data    #############

some_counties <- read.csv(paste0(data_dir, 
                                 "counties_to_look_at.csv"),
                          as.is=TRUE)

ogrListLayers(shape_file_dir)
gdb <- path.expand(shape_file_dir)
WSDACrop <- readOGR(gdb, "WSDACrop_2018")

##############################################
############                     #############
############  add LastSurveyYear #############

WSDACrop@data$LastSurveyYear <- substr(WSDACrop@data$LastSurveyDate, 1, 4)

WSDACrop2018 <- WSDACrop[WSDACrop@data$LastSurveyYear == "2018", ]

#################################################
############                        #############
############  subset double crops   #############

WSDACrop2018_doublecrop <- subset(WSDACrop2018, 
                                  RotationCrop != " ")

WSDACrop2018_doublecrop <- WSDACrop2018_doublecrop[grepl('double', WSDACrop2018_doublecrop$Notes), ]
###############################################
############                      #############
############   subset some crops  #############

some_crops <- c("Alfalfa Hay", "Alfalfa Seed",
                "Alfalfa/Grass Hay", 
                "Apple", "Blueberry", 
                "Cherry", 
                "Grape, Juice", "Grape, Table",
                "Grape, Wine", "Potato")

WSDACrop2018_some_crops <- subset(WSDACrop2018, 
                                  CropType %in% some_crops)

###################################################
############                          #############
############   save filtered shapes   #############

writeOGR(obj=WSDACrop2018_doublecrop, 
         dsn = paste0("/Users/hn/Desktop/Desktop/", 
                      "Ag/check_point/pre_microsoft_meeting/", 
                      "filtered_shape_files"), 
         layer="WSDACrop2018_doublecrop/double_crops/", 
         driver="ESRI Shapefile")

writeOGR(obj = WSDACrop2018_some_crops, 
         dsn = paste0("/Users/hn/Desktop/Desktop/", 
                      "Ag/check_point/pre_microsoft_meeting/", 
                      "filtered_shape_files/some_crops/"), 
         layer="WSDACrop2018_some_crops", 
         driver="ESRI Shapefile")

raster::shapefile(WSDACrop2018_some_crops, 
         filename=paste0("/Users/hn/Desktop/Desktop/", 
                         "Ag/check_point/pre_microsoft_meeting/", 
                         "filtered_shape_files/new_some_crops/",
                         "WSDACrop2018_some_crops"),
         overwrite=TRUE)

simple_some_crops <- rmapshaper::ms_simplify(WSDACrop2018_some_crops)
simple_doublecrop <- rmapshaper::ms_simplify(WSDACrop2018_doublecrop)

writeOGR(obj=simple_doublecrop, 
         dsn = paste0("/Users/hn/Desktop/Desktop/", 
                      "Ag/check_point/pre_microsoft_meeting/", 
                      "filtered_shape_files/simple/double_crops/"), 
         layer="WSDACrop2018_doublecrop", 
         driver="ESRI Shapefile")

writeOGR(obj = simple_some_crops, 
         dsn = paste0("/Users/hn/Desktop/Desktop/", 
                      "Ag/check_point/pre_microsoft_meeting/", 
                      "filtered_shape_files/simple/some_crops/"), 
         layer="WSDACrop2018_some_crops", 
         driver="ESRI Shapefile")
###################################################
############                          #############
############   Get Polygon Centroids  #############

double_crops_centroids <- rgeos::gCentroid(WSDACrop2018_doublecrop, 
                                           byid=TRUE)

some_crops_centroids <- rgeos::gCentroid(WSDACrop2018_some_crops, 
                                         byid=TRUE)
####
#### Convert to Lat Long
####

crs <- CRS("+proj=lcc 
           +lat_1=45.83333333333334 
           +lat_2=47.33333333333334 
           +lat_0=45.33333333333334 
           +lon_0=-120.5 +datum=WGS84")

centroid_coord <- spTransform(double_crops_centroids, 
                              CRS("+proj=longlat +datum=WGS84"))
#######################################################################
double_crops_centroids_coord <- data.table(double_crops_centroids@coords)
some_crops_centroids_coord <- data.table(some_crops_centroids@coords)

setnames(double_crops_centroids_coord, 
         old=c("x", "y"), 
         new=c("lat", "long"))

setnames(some_crops_centroids_coord, 
         old=c("x", "y"), 
         new=c("lat", "long"))

double_crops_centroids_coord$lat <- double_crops_centroids_coord$lat/10000
some_crops_centroids_coord$lat <- some_crops_centroids_coord$lat/10000

double_crops_centroids_coord$long <- double_crops_centroids_coord$long/1000
some_crops_centroids_coord$long <- some_crops_centroids_coord$long/1000

cols <- c("lat", "long")
double_crops_centroids_coord[,(cols) := round(.SD,5), .SDcols=cols]
some_crops_centroids_coord[,(cols) := round(.SD,5), .SDcols=cols]


write.table(some_crops_centroids_coord, 
            file = paste0("/Users/hn/Desktop/", 
                          "Desktop/Ag/check_point/", 
                          "pre_microsoft_meeting/", 
                          "some_crops_centroids_coord.csv"), 
            row.names=FALSE, na="",col.names=TRUE, sep=",")


write.table(double_crops_centroids_coord, 
            file = paste0("/Users/hn/Desktop/", 
                          "Desktop/Ag/check_point/", 
                          "pre_microsoft_meeting/", 
                          "double_crops_centroids_coord.csv"), 
            row.names=FALSE, na="",col.names=TRUE, sep=",")

