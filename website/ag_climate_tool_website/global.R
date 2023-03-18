#global.R holds the main data processing and setup for the application
#packages
library(corpcor)
library(data.table)
library(dplyr)
library(ggplot2)
library(grid)
library(gridExtra)
library(leaflet)
library(lubridate)
library(maps)
library(maptools)
library(ncdf4)
library(plotly)
library(rbokeh)
library(rgdal)
library(rgeos)
library(RSQLite)
library(shiny)
library(shinyBS)
library(shinydashboard)
library(sp)
library(stringr)
library(tidyr)

# Project package
library(cbcct)
source("plot_gdd.R")
source("plot_temp.R")
source("plot_precip.R")

#checkpoint for final version (uncomment when done)
#library(checkpoint)
#checkpoint(format(date(), "%Y-%m-%d"))

###########################
# Settings
###########################
#Colors and theming
circle_color = '#981e32'
r = 2000 #radius

#precip_colors <- c("#333333", "#333333", "#333333", "#333333", "#333333")
#gdd_colors <- c("#333333", "#333333", "#333333", "#333333", "#333333")
precip_colors <- c("#1808FF", "#0734E8", "#0578FF", "#07AAE8", "#07BBDD")
gdd_colors <- c("#FF200D", "#E83D0C", "#FF5E00", "#E87D0C", "#FFA70D")
color_scheme <- cbcct::palette3

#today's date
#today = as.Date("2016-08-02")
today = Sys.Date()
if(hour(Sys.time()) <= 6) today <- today - 1 #if before 6:59am, use yesterday

this_year <- as.numeric(format(today, "%Y"))
print(this_year)
last_day_of_year <- as.Date(paste(this_year, "12-31", sep="-"))

#Data location
data_dir = "data/"

#####################
# Initial Data
#####################

historical_model <- "VIC_Binary_CONUS"
historical_rcp <- "historical"
historical_year_range <- c(1979, 2016)

future_model <- "CNRM-CM5"
future_rcp <- "rcp85"
future_year_range <- c(2006, 2099)
mahalanobis_vars <- "all"

loadingProgress <- 0

#Spatial data
spatial_historical <- readRDS("data/distance/historical.rds")
analogue_runs <- list(
  "All Variables" = "all",
  "Temperature Variables" = "temp",
  "Precipitation Variables" = "precip"
)
default_analogue_run = "all"

#Years available
#future_years = c(2035,2085)
historical_years = seq(1979,2015,1)

###########################################
#Load Initial Data for use in all locations
###########################################

#Locations and Crops
data(crops)
selected_crops <- crops
data(crop_locations)
data(crb_locations)
elevlocs <- readRDS("data/elevlocs.rds")
county <- readRDS("data/latloncounty.rds")

matchdat <- readRDS("data/matches/agclimate_match.rds")
match_varnames <- c("Propensity Score", "Hours <0F", "Hours 0-3F", "Hours 3-6F", "Hours 6-9F",
                    "Hours 9-12F", "Hours 12-15F", "Hours 15-18F", "Hours 18-21F",
                    "Hours 21-24F", "Hours 24-27F", "Hours 27-30F", "Hours 30-33F",
                    "Hours 33-36F", "Hours 36-39F", "Hours >39F", "GS Days", 
                    "Rain", "Rain Sq.", "Clay", "Silt", "Sand", "County", "State")



default_crop_name <- "Barley"
crop_names <- data.table("id" = crops$id, "name" = crops$name)
crop_names <- na.omit(crop_names)$name

irrigation <-read.table("data/CropParam.txt", header=TRUE, sep=",")
irrigation <- data.table(irrigation)
irrigation$lat = substr(irrigation$filename, 12, 19)
irrigation$long = substr(irrigation$filename, 21, 30)

###temporary hack
#avail_locations <- irrigation[, c("lat", "long"), with=FALSE]
#avail_locations <- avail_locations[!duplicated(avail_locations), ]
#avail_locations$filter = paste0(avail_locations$lat, avail_locations$long)
#print(avail_locations[!duplicated(avail_locations$filter), ])
#crop_locations$filter = paste0(crop_locations$latitude, crop_locations$longitude)
#crop_locations <- subset(crop_locations, filter %in% avail_locations$filter)
###temporary hack

###temporary hack
#new way of filtering based on state, not irrigation status ##
stateswanted <- c("Oregon", "Washington", "Idaho")

alldata <- read.table("data/CropParam.txt", header=TRUE, sep=",")
alldata <- data.table(alldata) 
alldata$lat = substr(alldata$filename, 12, 19)
alldata$long = substr(alldata$filename, 21, 30)

#filter by state <- identify places we don't want, so if crop_locations matches that, we don't include them (this is becacuse CropParamCRB.txt doesn't include western spots)
inUnwantedstates <- alldata %>% filter(!(statename %in% stateswanted))

#make lat long list to cull "crop_locations" data from 
Notavail_locations <- inUnwantedstates[, c("lat", "long")]  #This code was causing an error: , with=FALSE
Notavail_locations <- Notavail_locations[!duplicated(Notavail_locations), ]
Notavail_locations$filter = paste0(Notavail_locations$lat, Notavail_locations$long)
#print(avail_locations[!duplicated(avail_locations$filter), ])
crop_locations$filter = paste0(crop_locations$latitude, crop_locations$longitude)
crop_locations <- subset(crop_locations, !(filter %in% Notavail_locations$filter))
###temporary hack



#GEO Layers
states <- readOGR(paste0(data_dir, "geo/states/cb_2014_us_state_20m.shp"), layer = "cb_2014_us_state_20m", verbose = FALSE)
cbStates <- subset(states, states$STUSPS %in% c("WA", "OR", "ID"))
#waCounties <- readOGR(paste0(data_dir, "geo/WACounties/cb_2014_53_cousub_500k.shp"), layer = "cb_2014_53_cousub_500k", verbose = FALSE)
#orCounties <- readOGR(paste0(data_dir, "geo/ORCounties/cb_2014_41_cousub_500k.shp"), layer = "cb_2014_41_cousub_500k", verbose = FALSE)
#idCounties <- readOGR(paste0(data_dir, "geo/IDCounties/cb_2014_16_cousub_500k.shp"), layer = "cb_2014_16_cousub_500k", verbose = FALSE)

#Historical Data for Spatial Analogues
spatial_historical <- readRDS(paste0(data_dir, "/distance/historical.rds"))

#CodlingMoth Data
data_sets <- list()


