# Water Rights

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

######################################################
RD <- c("1916-06-30", "1884-10-30", 
        "1905-05-10", "1905-05-10",
        "1905-05-10", "1974-08-02",
        "1933-08-25", "1902-06-30", 
        "2009-07-30", "2009-07-30")

lat <- c(47.10483, 47.10483, 47.10483,
         47.10483, 47.10483, 47.10483,
         47.33486, 47.33486, 47.33486, 47.33486)

long <- c(-121.1577, -121.2309, -121.0622,
          -121.3069, -121.2470, -121.2208,
          -121.2534, -121.0608, -121.2736,
          -120.9735)

WRS <- c("surfaceWater", "surfaceWater", "surfaceWater", 
         "surfaceWater", "surfaceWater", "surfaceWater", 
         "groundwater", "groundwater", "groundwater",
         "groundwater")


spatial_wtr_right = data.table(right_date = RD,
                               lat = lat,
                               long = long,
                               WaRecRCWCl = WRS
                               )
spatial_wtr_right$popup <- 1

spatial_wtr_right$color <- "#ffff00"

# d <- "/Users/hn/Desktop/water_right_attributes_jiggled.rds"
# spatial_wtr_right <- readRDS(d) %>% data.table()
# spatial_wtr_right$color <- "#ffff00"

