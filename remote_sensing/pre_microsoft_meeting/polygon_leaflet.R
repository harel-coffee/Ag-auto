library(leaflet)

library(data.table)
library(rgdal)
library(dplyr)
library(sp)
library(sf)

############################################################
###############                              ###############
###############      directories             ###############
###############                              ###############

data_dir <- paste0("/Users/hn/Desktop/Desktop/Ag/", 
                   "check_point/pre_microsoft_meeting/")

simle_dir <- paste0(data_dir, "filtered_shape_files/simple/")

shape_file_dir <- paste0(data_dir, "2018WSDACrop.gdb") 

############################################################
###############                              ###############
###############      read shapefile          ###############
###############                              ###############

ogrListLayers(shape_file_dir)
gdb <- path.expand(shape_file_dir)
WSDACrop <- readOGR(gdb, "WSDACrop_2018")

########################################################################
double_crops <- rgdal::readOGR(dsn = paste0(simle_dir, 
                                            "double_crops/", 
                                            "WSDACrop2018_doublecrop.dbf"))


########################################################################
#########
######### filter double croup by Note column 
#########
WSDACrop <- WSDACrop[grepl('double', WSDACrop$Notes), ]

########################################################################
##################
################## one try to plot 
##################
double_by_Note <- spTransform(WSDACrop, CRS("+init=epsg:4326"))

leaflet() %>%
addProviderTiles(providers$OpenTopoMap, # Esri.WorldStreetMap
                options= providerTileOptions(opacity = 0.99))%>%
# addProviderTiles("CartoDB.Positron", 
#                  options= providerTileOptions(opacity = 0.99)) %>%
addPolygons(data = double_by_Note,
            stroke = TRUE, 
            fillOpacity = 0.5, 
            smoothFactor = 0.5)
##################
################## one try to plot 
##################
double_by_both <- spTransform(double_crops, CRS("+init=epsg:4326"))

leaflet() %>%
addProviderTiles(providers$OpenTopoMap, # Esri.WorldStreetMap
                options= providerTileOptions(opacity = 0.99))%>%
# addProviderTiles("CartoDB.Positron", 
#                  options= providerTileOptions(opacity = 0.99)) %>%
addPolygons(data = double_by_both,
            stroke = TRUE, 
            fillOpacity = 0.5, 
            smoothFactor = 0.5)

#############################
us_state_dir <- paste0(data_dir, "/cb_2018_us_state_20m/")

states <- readOGR(paste0(us_state_dir, "/cb_2018_us_state_20m.shp"),
                  layer = "cb_2018_us_state_20m", 
                  GDAL1_integer64_policy = TRUE)


nWStates <- subset(states, states$STUSPS %in% c("WA","OR","ID"))

leaflet(neStates) %>%
addPolygons(color = "#444444", weight = 1, smoothFactor = 0.5,
            opacity = 1.0, fillOpacity = 0.5,
            fillColor = ~colorQuantile("YlOrRd", ALAND)(ALAND),
            highlightOptions = highlightOptions(color = "white", weight = 2,
                                                bringToFront = TRUE))

##################################################################
########### 
###########        Min's shapefile
###########
mins_file_dir <- paste0(data_dir, "wsda2018shp/")
mins_file <- readOGR(paste0(mins_file_dir, "/WSDACro_2018.shp"),
                     layer = "WSDACro_2018", 
                     GDAL1_integer64_policy = TRUE)

mins_file_double_crop <- subset(mins_file, 
                                RotationCr != " ")

mins_file_double_crop <- mins_file_double_crop[grepl('double', 
                                               mins_file_double_crop$Notes), ]

writeOGR(obj=mins_file_double_crop, 
         dsn = paste0("/Users/hn/Desktop/Desktop/", 
                      "Ag/check_point/pre_microsoft_meeting/", 
                      "filtered_shape_files/not_simple/Min_double_crops/"), 
         layer="Min_DoubleCrop", 
         driver="ESRI Shapefile")

###### simplify to reduce the size for faster reading
simple_Min_double_crop <- rmapshaper::ms_simplify(mins_file_double_crop)

writeOGR(obj=simple_Min_double_crop, 
         dsn = paste0("/Users/hn/Desktop/Desktop/", 
                      "Ag/check_point/pre_microsoft_meeting/", 
                      "filtered_shape_files/simple/Min_double_crops/"), 
         layer="Min_DoubleCrop", 
         driver="ESRI Shapefile")


Min_sp <- spTransform(simple_Min_double_crop, CRS("+init=epsg:4326"))

leaflet() %>%
addProviderTiles(providers$OpenTopoMap, # Esri.WorldStreetMap
                options= providerTileOptions(opacity = 0.99))%>%
# addProviderTiles("CartoDB.Positron", 
#                  options= providerTileOptions(opacity = 0.99)) %>%
addPolygons(data = Min_sp,
            stroke = TRUE, 
            fillOpacity = 0.5, 
            smoothFactor = 0.5)

min_double_crops_centroids <- rgeos::gCentroid(mins_file_double_crop, 
                                                byid=TRUE)

