# Lagoon

library(scales)
library(lattice)
# library(ggmap)
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
# library(plotly)
# library(Hmisc)

data_dir <- "/data/hnoorazar/codling_moth/"
map_dir <- "/data/hnoorazar/lagoon/map_files/"
plot_dir <- "/data/hnoorazar/lagoon/plots/"

######################################
###################################### clear above
######################################
##########################
########################## For Analog Map
##########################

#########################################################
# read county shapefile
shapefile_dir <- paste0(map_dir, "shape_files/tl_2017_us_county/")
shapefile_dir <- paste0(map_dir, "shape_files/tl_2017_us_county_simple/")

counties <- rgdal::readOGR(dsn=path.expand(shapefile_dir), 
                           layer = "tl_2017_us_county")

# Extract just the three states OR: 41, WA:53, ID: 16
counties <- counties[counties@data$STATEFP %in% c("16", "41", "53"), ]

####################################################################
#
#  Read these to put borders on counties. The way I did this for 
# analog was different. Do you want to make them similar?
#
skagit <- readOGR(paste0(map_dir, "Skagit.geo.json"), "OGRGeoJSON") 
snohomish <- readOGR(paste0(map_dir, "Snohomish.geo.json"), "OGRGeoJSON")
whatcom <- readOGR(paste0(map_dir, "/Whatcom.geo.json"), "OGRGeoJSON")

spatial_lagoon <- readRDS(paste0(map_dir, 
                                 "lagoon_spatial_coordinate.rds")) %>% 
                  group_by(location, lat, long, cluster)

st_cnty_names <- read.csv(paste0(map_dir, 
                                 "17_counties_fips_unique.csv"),
                          header=T,
                          as.is=T) %>% 
                 data.table()

#####################################################################


