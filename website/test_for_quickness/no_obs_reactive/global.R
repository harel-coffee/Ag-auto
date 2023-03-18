# Water Rights

#===============
# LOAD PACKAGES
#===============
# library(tidyverse)
# library(maptools)

library(scales)
library(lattice)
library(jsonlite)
library(raster)

library(data.table)
library(shiny)
library(shinydashboard)
library(shinyBS)
library(maps)
library(rgdal)    # for readOGR and others
library(sp)       # for spatial objects
library(leaflet)  # for interactive maps (NOT leafletR here)
library(dplyr)    # for working with data frames
library(ggplot2)  # for plotting
library(reshape2)
library(RColorBrewer)
#####################################################
#####
#####       Read Shape files
#####
######################################################
wtr_right_dir <- "/data/hnoorazar/water_right/"

shapefile_dir <- paste0(wtr_right_dir, "shapefiles/")
shapefile_dir <- paste0(wtr_right_dir, "simple_shapefiles/")

data_dir <- paste0(wtr_right_dir, "data/")
######################################################
shapefile_dir <- "/data/hnoorazar/water_right/shapefiles/"

shapefile_dir <- paste0("/Users/hn/Desktop/", 
                        "Desktop/Ag/check_point/", 
                        "water_right/shapefiles/")

shapefile_dir <- paste0("/Users/hn/Desktop/", 
                        "Desktop/Ag/check_point", 
                        "/water_right/simple_shapefiles/")

data_dir <- paste0("/Users/hn/Desktop/Desktop/",
                   "Ag/check_point/water_right/data/"
                   )

##################
##
## all streams
##
##################
# start_time <- Sys.time()
# all_streams_sp <- rgdal::readOGR(dsn=path.expand(
#                                           paste0(shapefile_dir, 
#                                                  "streams_Okanogan/")),
#                                 layer = "streams_Okanogan");
# print (Sys.time() - start_time)


##################
##
## Rivers
##
##################

##################
##
## all basins
##
##################

all_basins_sp <- rgdal::readOGR(dsn=path.expand(
                                        paste0(shapefile_dir, 
                                              "all_basins/")),
                                layer = "all_basins");
##################
##
## all subbasins
##
##################
all_subbasins_sp <- rgdal::readOGR(dsn=path.expand(
                                          paste0(shapefile_dir, 
                                                 "all_subbasins/")),
                                layer = "all_subbasins");

#####################################################
#####
#####       Water Right data
#####
######################################################
####################
#
# spatial_wtr_right
#
####################
spatial_wtr_right <- readRDS(paste0(data_dir,
                            "water_right_attributes_not_jiggled.rds")) %>% 
                     data.table()

spatial_wtr_right <- na.omit(spatial_wtr_right, cols=c("subbasin"))

spatial_wtr_right$colorr <- "#ffff00"
spatial_wtr_right <- data.table(spatial_wtr_right)
# old_names <- c("WaRecID", "Source_Lat", "Source_Lon", 
#                "PriorityDa", 
#                "Subbasin", "WRIA_NM", "Source_NM") 

# new_names <- c("WR_Doc_ID", "lat", "long", 
#                "right_date", 
#                "subbasin", "county_type", "stream")

# setnames(spatial_wtr_right, old=old_names, new=new_names)
spatial_wtr_right <- data.table(spatial_wtr_right)
####################
#
# Places of use
#
####################
places_of_use <- readRDS(paste0(data_dir, 
                                "places_of_use.rds")) %>% 
                 data.table()
places_of_use$colorr <- "#ffff00"
places_of_use <- data.table(places_of_use)

place_of_use_sp <- rgdal::readOGR(dsn=path.expand(
                                  paste0(shapefile_dir, 
                                         "place_of_use/")),
                                  layer = "place_of_use");

# curr_spatial <- spatial_wtr_right
all_basins <- sort(unique(spatial_wtr_right$WRIA_NM))

subbasins <- c("Ahtanum Creek", 
               "Lmumu-Burbank",
               "Lower Yakima tributaries",
               "tributaries", 
               "Satus Creek",
               "Taneum-Manastash",
               "Toppenish Creek",
               "Wilson-Cherry")

Upper_Yakima_center <- c(46.98, -120.6)
Lower_Yakima_center <- c(46.47, -120.349)
Naches_center <- c(46.69, -120.73)
Wenatchee_center <- c(47.59, -120.58)
Methow_center <- c(48.32, -120.13)
Okanogan_center <- c(48.53, -119.56)
Walla_Walla_center <- c(46.08, -118.34)

########################################
######
######       Functions
######
########################################

###########################
######
######   construct map
######
###########################
base_map_sat <- leaflet() %>%
                addTiles(urlTemplate = paste0("http://server.", 
                                              "arcgisonline.com/", 
                                              "ArcGIS/rest/services", 
                                              "/World_Imagery/", 
                                              "MapServer", 
                                              "/tile/{z}/{y}/{x}"),

                         attribution = paste0('Maps by ', 
                                              '<a href="http://',
                                              'www.mapbox.com/">', 
                                              'Mapbox</a>'),
                         layerId = "Satellite",
                         options= providerTileOptions(opacity = 0.9))

base_map_st <- leaflet() %>%
               addTiles(urlTemplate = paste0("//{s}.tiles.mapbox.", 
                                             "com/v3/jcheng.map-", 
                                             "5ebohr46/{z}/{x}/{y}", 
                                             ".png"),
                        attribution = paste0('Maps by <a href="http://', 
                                      'www.mapbox.com/">Mapbox</a>')
                        )


start_time <- Sys.time()

base_map_sat_st <- leaflet() %>%
                   addTiles(urlTemplate = paste0("http://server.", 
                                                 "arcgisonline.com/", 
                                                 "ArcGIS/rest/services", 
                                                 "/World_Imagery/", 
                                                 "MapServer", 
                                                 "/tile/{z}/{y}/{x}"),

                            attribution = paste0('Maps by ', 
                                                '<a href="http://',
                                                'www.mapbox.com/">', 
                                                'Mapbox</a>'),
                           layerId = "Satellite",
                           options= providerTileOptions(opacity = 0.9)) %>%
                   addTiles(urlTemplate = paste0("//{s}.tiles.mapbox.", 
                                                 "com/v3/jcheng.map-", 
                                                 "5ebohr46/{z}/{x}/{y}", 
                                                 ".png"),
                             attribution = paste0('Maps by <a href="http://', 
                                                 'www.mapbox.com/">Mapbox</a>'),
                             options= providerTileOptions(opacity = 0.4)
                            ) # %>%
                   # setView(lat = 46, lng =-121, zoom = 6) %>%
                   # addPolylines(data = all_streams_sp,
                   #              stroke = TRUE,
                   #              fillOpacity = 0.5, 
                   #              smoothFactor = 0.5, 
                   #              weight = 1, 
                   #              color = "#80BFFD", 
                   #              group ="rivers")

# base_map_sat_st <- addPolylines(map = base_map_sat_st, 
#                                 data = all_streams_sp,
#                                 stroke = TRUE,
#                                 fillOpacity = 0.5, 
#                                 smoothFactor = 0.5, 
#                                 weight = 1, 
#                                 color = "#80BFFD", 
#                                 group ="rivers")
print(" ")
print("plotting base map takes: ")
print (Sys.time() - start_time)

build_map <- function(data_dt, sub_bas){
  if (is.null(sub_bas)){
    map <- leaflet() %>%
           addTiles(urlTemplate = paste0("http://server.", 
                                         "arcgisonline.com/", 
                                         "ArcGIS/rest/services", 
                                         "/World_Imagery/", 
                                         "MapServer", 
                                         "/tile/{z}/{y}/{x}"),

                       attribution = paste0('Maps by ', 
                                          '<a href="http://',
                                          'www.mapbox.com/">', 
                                          'Mapbox</a>'),
                       layerId = "Satellite",
                       options= providerTileOptions(opacity = 0.9)) %>%
                       setView(lat = 46, lng =-121, zoom = 5)

    } else {
      data_dt <- data_dt %>%
                 filter(subbasin %in% sub_bas) %>%
                 data.table()

      if (unique(data_dt$WRIA_NM) == "Walla Walla"){
        # Walla_Walla_center <- c(46.08, -118.34)
        mean_lat <- 46.08
        mean_long <- -118.34

        } else if (unique(data_dt$WRIA_NM) == "Upper Yakima"){
          # Upper_Yakima_center <- c(46.98, -120.6)
          mean_lat <- 46.98 
          mean_long <- -120.6
      
        } else if (unique(data_dt$WRIA_NM) == "Lower Yakima"){
          # Lower_Yakima_center <- c(46.47, -120.349)
          mean_lat <- 46.47
          mean_long <- -120.349
        
        } else if (unique(data_dt$WRIA_NM) == "Naches"){
          # Naches_center <- c(46.69, -120.73)
          mean_lat <- 46.69
          mean_long <- -120.73
        
        } else if (unique(data_dt$WRIA_NM) == "Wenatchee"){
          # Wenatchee_center <- c(47.59, -120.58)
          mean_lat <- 47.59
          mean_long <- -120.58
        
        } else if (unique(data_dt$WRIA_NM) == "Methow"){
          # Methow_center <- c(48.32, -120.13)
          mean_lat <- 48.32
          mean_long <- -120.13
        
        } else if (unique(data_dt$WRIA_NM) == "Okanogan"){
          # Okanogan_center <- c(48.53, -119.56)
          mean_lat <- 48.53
          mean_long <- -119.56
      }

      curr_basins_sp <- all_basins_sp[all_basins_sp$WRIA_NM %in% unique(data_dt$WRIA_NM), ]
      map <- base_map_sat_st %>% 
             setView(lat = mean_lat, lng = mean_long, zoom = 7) %>%
             addCircleMarkers(data = data_dt, 
                              lng = ~ long, lat = ~lat,
                              label = ~ popup,
                              radius = 3,
                              color = ~ colorr,
                              stroke  = FALSE,
                              fillOpacity = .95 
                              ) %>%
             addPolygons(# map = map, 
                         data = curr_basins_sp, 
                         fill = F, 
                         weight = 7, 
                         color = "red",
                         group ="Outline")
        
      curr_subbasins_sp <- all_subbasins_sp[all_subbasins_sp$Subbasin %in% sub_bas, ]
      map <- addPolygons(map = map, 
                         data = curr_subbasins_sp,
                         fill = F, 
                         weight = 1.5, 
                         color = "yellow", 
                         group ="Outline")
    }

  return(map)
}
