rm(list=ls())
library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)
library(ggpubr)

options(digits=9)
options(digit=9)


cod_param_dir <- "/Users/hn/Documents/GitHub/Ag/codling_moth/code/parameters/"
chill_param_dir <- "/Users/hn/Documents/GitHub/Ag/chilling/parameters/"

cod_locations <- "LocationGroups.csv"
limited_cities <- "limited_locations.csv"

cod_locations <- read.csv(paste0(cod_param_dir, cod_locations), as.is=TRUE)
limited_cities <- read.csv(paste0(chill_param_dir, limited_cities), as.is=TRUE)

setnames(cod_locations, old=c("latitude", "longitude"), new=c("lat", "long"))
cod_locations <- within(cod_locations, remove(locationGroup))

####################################################################################
cod_locations$location <- paste0(cod_locations$lat, "_", cod_locations$long)
limited_cities$location <- paste0(limited_cities$lat, "_", limited_cities$long)

common_cities <- limited_cities %>% filter(location %in% cod_locations$location)
####################################################################################
lost_cities <- limited_cities %>% 
               filter(!(location %in% common_cities$location))%>% 
               data.table()

##########
########## Find closest locations
##########
closest_locations <- data.table()
ct <- "Hillsboro"
for (ct in unique(lost_cities$city)){
  cod_locations_copy <- cod_locations
  curr_city <- lost_cities %>% filter(city == ct) %>% data.table()
  cod_locations_copy$latDiff <- cod_locations_copy$lat - curr_city$lat
  cod_locations_copy$longDiff <- cod_locations_copy$long - curr_city$long
  cod_locations_copy$distance <- sqrt(cod_locations_copy$latDiff^2 + cod_locations_copy$longDiff^2)
  minim_row <- which.min(cod_locations_copy[["distance"]])
  minim_row <- cod_locations_copy[minim_row, ]
  minim_row$city <- ct
  closest_locations <- rbind(closest_locations, minim_row)
}

write.table(closest_locations, 
            file = paste0(chill_param_dir, "close_locs_4_bloom.csv"),
            row.names=FALSE, na="",col.names=TRUE, sep=",")
bloom_locs <- closest_locations %>% filter(distance < 0.5)
bloom_locs <- subset(bloom_locs, select=c(city, lat, long, location))
bloom_locs <- rbind(bloom_locs, common_cities)


write.table(bloom_locs, 
            file = paste0(chill_param_dir, "bloom_limited_cities.csv"),
            row.names=FALSE, na="",col.names=TRUE, sep=",")

