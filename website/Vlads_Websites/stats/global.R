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

# stats <- read.csv('/data/hnoorazar/hardiness/data/CH_stats_by_loc.csv')
# stats$location <- paste(stats$lat, stats$long)
# stats$lat <- as.numeric(stats$lat)
# stats$long <- as.numeric(stats$long)
# obs_hist <- 'obs_hist'

obs_hist <- 'obs_hist'
data_dir <- "/data/hnoorazar/hardiness/data/"
cabernet_stats <- read.csv(paste0(data_dir, 'cabernet_sauvignon_CH_stats_by_loc.csv'))
# done once in my local machine
# cabernet_stats$location <- paste(cabernet_stats$lat, cabernet_stats$long) 
cabernet_stats$lat <- as.numeric(cabernet_stats$lat)
cabernet_stats$long <- as.numeric(cabernet_stats$long)


chardonnay_stats <- read.csv(paste0(data_dir,'chardonnay_CH_stats_by_loc.csv'))
# done once in my local machine
# chardonnay_stats$location <- paste(chardonnay_stats$lat, chardonnay_stats$long)
chardonnay_stats$lat <- as.numeric(chardonnay_stats$lat)
chardonnay_stats$long <- as.numeric(chardonnay_stats$long)

merlot_stats <- read.csv(paste0(data_dir,'merlot_CH_stats_by_loc.csv'))
merlot_stats$lat <- as.numeric(merlot_stats$lat)
merlot_stats$long <- as.numeric(merlot_stats$long)

reisling_stats <- read.csv(paste0(data_dir,'reisling_CH_stats_by_loc.csv'))
reisling_stats$lat <- as.numeric(reisling_stats$lat)
reisling_stats$long <- as.numeric(reisling_stats$long)



###########
########### State boundries:
###########
shapefile_dir<-"/data/hnoorazar/codling_moth/shape_files/tl_2017_us_county_simple/"
counties <- rgdal::readOGR(dsn=path.expand(shapefile_dir), 
                           layer = "tl_2017_us_county")

# Extract just the three states OR: 41, WA:53, ID: 16
counties <- counties[counties@data$STATEFP %in% c("16", "41", "53", "30"), ]

#
# Compute states like so, to put border around states
states <- aggregate(counties[, "STATEFP"], 
                    by = list(ID = counties@data$STATEFP), 
                    FUN = unique, dissolve = T)

#
#   Rocky borders
#
# rocky_dir <- "/data/hnoorazar/codling_moth/shape_files/simple_Rocky_Mountain/"
# rocky <- rgdal::readOGR(dsn=path.expand(rocky_dir), 
#                         layer = "simple_Rocky_Mountain")

rocky_dir <- "/data/hnoorazar/codling_moth/shape_files/Rocky_Mountain/"
rocky <- rgdal::readOGR(dsn=path.expand(rocky_dir), 
                        layer = "Rocky_Mountain")



