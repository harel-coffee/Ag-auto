############      packages      #############
rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
library(sf)


data_dir <- paste0("/Users/hn/Desktop/Desktop/Ag/", 
                   "check_point/pre_microsoft_meeting/")

mins_file_dir <- paste0(data_dir, "/Mins_files/wsda2018shp/")
mins_file <- readOGR(paste0(mins_file_dir, "/WSDACro_2018.shp"),
                     layer = "WSDACro_2018", 
                     GDAL1_integer64_policy = TRUE)


mins_file_2018_surveyed <- mins_file[grepl('2018', mins_file$LastSurvey), ]

writeOGR(obj=mins_file_2018_surveyed, 
         dsn = paste0("/Users/hn/Desktop/Desktop/", 
                      "Ag/check_point/pre_microsoft_meeting/", 
                      "2018_surveyed"), 
         layer="2018_surveyed", 
         driver="ESRI Shapefile")


####################################
############
############ Double Cropping
############

doublecrop_2018 <- subset(mins_file_2018_surveyed, RotationCr != " ")

doublecrop_2018 <- doublecrop_2018[grepl('double', doublecrop_2018$Notes), ]

writeOGR(obj=doublecrop_2018, 
         dsn = paste0("/Users/hn/Desktop/Desktop/", 
                      "Ag/check_point/pre_microsoft_meeting/", 
                      "filtered_shape_files/double_2018/"), 
         layer="double_2018", 
         driver="ESRI Shapefile")


####################################
############
############ FOUR different Crops
############
some_crops <- c("Apple", "Potato", "Grape, Wine", "Alfalfa Hay")

apple_2018 <- mins_file_2018_surveyed[grepl('Apple', mins_file_2018_surveyed$CropType), ]
potato_2018 <- mins_file_2018_surveyed[grepl('Potato', mins_file_2018_surveyed$CropType), ]
wine_grape_2018 <- mins_file_2018_surveyed[grepl('Grape, Wine', mins_file_2018_surveyed$CropType), ]
alfalfa_hay_2018 <- mins_file_2018_surveyed[grepl('Alfalfa Hay', mins_file_2018_surveyed$CropType), ]


some_crops_2018 <- rbind(apple_2018, potato_2018, 
                         wine_grape_2018, alfalfa_hay_2018)

writeOGR(obj=some_crops_2018, 
         dsn = paste0("/Users/hn/Desktop/Desktop/", 
                      "Ag/check_point/pre_microsoft_meeting/", 
                      "filtered_shape_files/some_crops_2018/"), 
         layer="some_crops_2018", 
         driver="ESRI Shapefile")

###### pick up fifty fields

# B <- subset(apple_2018, OBJECTID==19) we can filter by object ID

fifty_apple <- apple_2018[1:50, ]
fifty_potato <- potato_2018[1:50, ]
fifty_grape <- wine_grape_2018[1:50, ]
fifty_alfalfa<- alfalfa_hay_2018[1:50, ]
fifty_double <- doublecrop_2018[1:50, ]


writeOGR(obj=fifty_apple, 
         dsn = paste0("/Users/hn/Desktop/Desktop/", 
                      "Ag/check_point/pre_microsoft_meeting/", 
                      "filtered_shape_files/", 
                      "fifty_field_from_each_type/fifty_apple/"), 
         layer="fifty_apple", 
         driver="ESRI Shapefile")

writeOGR(obj=fifty_potato, 
         dsn = paste0("/Users/hn/Desktop/Desktop/", 
                      "Ag/check_point/pre_microsoft_meeting/", 
                      "filtered_shape_files/", 
                      "fifty_field_from_each_type/fifty_potato/"), 
         layer="fifty_potato", 
         driver="ESRI Shapefile")

writeOGR(obj=fifty_grape, 
         dsn = paste0("/Users/hn/Desktop/Desktop/", 
                      "Ag/check_point/pre_microsoft_meeting/", 
                      "filtered_shape_files/", 
                      "fifty_field_from_each_type/fifty_grape/"), 
         layer="fifty_grape", 
         driver="ESRI Shapefile")

writeOGR(obj=fifty_alfalfa, 
         dsn = paste0("/Users/hn/Desktop/Desktop/", 
                      "Ag/check_point/pre_microsoft_meeting/", 
                      "filtered_shape_files/", 
                      "fifty_field_from_each_type/fifty_alfalfa/"),  
         layer="fifty_alfalfa", 
         driver="ESRI Shapefile")

writeOGR(obj=fifty_double, 
         dsn = paste0("/Users/hn/Desktop/Desktop/", 
                      "Ag/check_point/pre_microsoft_meeting/", 
                      "filtered_shape_files/", 
                      "fifty_field_from_each_type/fifty_double/"), 
         layer="fifty_double", 
         driver="ESRI Shapefile")

#####
##### Double crop centroids
#####
doublecrop_2018_center <- rgeos::gCentroid(doublecrop_2018, byid=TRUE)

crs <- CRS("+proj=lcc 
           +lat_1=45.83333333333334 
           +lat_2=47.33333333333334 
           +lat_0=45.33333333333334 
           +lon_0=-120.5 +datum=WGS84")

centroid_coord <- spTransform(doublecrop_2018_center, 
                              CRS("+proj=longlat +datum=WGS84"))

centroid_coord_dt <- data.table(centroid_coord@coords)

write.table(centroid_coord_dt, 
            paste0(data_dir, "double_crop_centroid.csv"), 
            row.names = FALSE, col.names = TRUE, sep=",")


#####
##### fifty field centroids
#####

fifty_apple_center <- rgeos::gCentroid(fifty_apple, byid=TRUE)
fifty_potato_center <- rgeos::gCentroid(fifty_potato, byid=TRUE)
fifty_grape_center <- rgeos::gCentroid(fifty_grape, byid=TRUE)
fifty_alfa_center <- rgeos::gCentroid(fifty_alfalfa, byid=TRUE)
fifty_double_center <- rgeos::gCentroid(fifty_double, byid=TRUE)


apple_center <- spTransform(fifty_apple_center, 
                           CRS("+proj=longlat +datum=WGS84"))
apple_center <- data.table(apple_center@coords)
apple_center
# apple_center = -119.393766 46.2298139
############################################################

potato_center <- spTransform(fifty_potato_center, 
                           CRS("+proj=longlat +datum=WGS84"))
potato_center <- data.table(potato_center@coords)
potato_center # -119.312542 45.9642573
############################################################
grape_center <- spTransform(fifty_grape_center, 
                           CRS("+proj=longlat +datum=WGS84"))
grape_center <- data.table(grape_center@coords)
grape_center # -119.043813 45.9749662

############################################################
alfa_center <- spTransform(fifty_alfa_center, 
                           CRS("+proj=longlat +datum=WGS84"))
alfa_center <- data.table(alfa_center@coords)
alfa_center # -119.550878 46.2703536
############################################################
double_center <- spTransform(fifty_double_center, 
                           CRS("+proj=longlat +datum=WGS84"))
double_center <- data.table(double_center@coords)
double_center # -119.639383 46.2450093

############################################################

