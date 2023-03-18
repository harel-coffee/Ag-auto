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