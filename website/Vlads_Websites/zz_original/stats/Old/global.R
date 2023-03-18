# stats dashboard

###  Packages  ####
library(shiny)
library(ggplot2) # , lib.loc = "r_lib"
library(leaflet) # , lib.loc = "r_lib"
library(shinyBS)
library(shinydashboard)
library(stringr)
library(lubridate)
library(rgdal)
library(Cairo)
# library(readr, lib.loc="/home/hnoorazar/R/x86_64-redhat-linux-gnu-library/3.3") # , lib.loc = "r_lib"
options(shiny.usecairo=TRUE)

###########################
###                    ####
###    Import Spatial  ####
###                    ####
###########################

plot_dir = "/data/VladOles/hardiness/plots/"

stats <- read.csv('/data/VladOles/hardiness/CH_stats_by_loc.csv')
stats$location <- paste(stats$lat, stats$long)
stats$lat <- as.numeric(stats$lat)
stats$long <- as.numeric(stats$long)
obs_hist <- 'obs_hist'
models <- append(obs_hist,  setdiff(unique(stats$model), obs_hist))
time_frames <- c('1979-2016', '2025-2049', '2050-2074', '2075-2099')
scenario_types <- c('RCP4.5', 'RCP8.5')

###
### Server
###
# skagit <- readOGR("geo/Skagit.geo.json", "OGRGeoJSON") # Skagit County
# snohomish <- readOGR("geo/Snohomish.geo.json", "OGRGeoJSON") # Snohomish County
# whatcom <- readOGR("geo/Whatcom.geo.json", "OGRGeoJSON") # Whatcom County
# 
# # Bind all counties ####
# counties <- rbind(skagit, snohomish, whatcom, makeUniqueIDs = TRUE)

