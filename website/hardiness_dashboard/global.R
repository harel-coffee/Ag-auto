# hardiness dashboard

###  Packages  ####
library(shiny)
library(cowplot) # , lib.loc = "r_lib"
library(ggplot2) # , lib.loc = "r_lib"
library(leaflet) # , lib.loc = "r_lib"
library(zoo) # , lib.loc = "r_lib"
library(shinyBS)
library(shinydashboard)
library(purrr)
library(tidyr)
library(dplyr)
library(tibble)
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

plot_dir = "/data/hnoorazar/hardiness/plots/"
observed_plot_dir <- paste0(plot_dir, "observed/")

spatial_hardiness_locs <- readRDS(paste0("/data/hnoorazar/hardiness/data/cm_spatial_hardiness.rds")) %>% 
                          group_by(location, lat, long)

###
### Server
###
skagit <- readOGR("geo/Skagit.geo.json", "OGRGeoJSON") # Skagit County
snohomish <- readOGR("geo/Snohomish.geo.json", "OGRGeoJSON") # Snohomish County
whatcom <- readOGR("geo/Whatcom.geo.json", "OGRGeoJSON") # Whatcom County

# Bloom Pruet

# Bind all counties ####
counties <- rbind(skagit, snohomish, whatcom, makeUniqueIDs = TRUE)

