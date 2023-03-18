rm(list=ls())

library(ggmap)
library(ggpubr)
library(lubridate)
library(purrr)
library(scales)
library(tidyverse)
library(maps)
library(data.table)
library(dplyr)
library(sp)
options(digits=9)
options(digit=9)


# chill_core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_core.R"
# source(chill_core_path)

# data_dir = "/Users/hn/Documents/01_research_data/chilling/01_data/02/"
# param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"
# LocationGroups_NoMontana <- read.csv(paste0(param_dir, "LocationGroups_NoMontana.csv"), header=T, sep=",", as.is=T)

LocationGroups <- read.csv("/Users/hn/Documents/01_research_data/codling_moth_2021/LocationGroups.csv",
                            header=T, sep=",", as.is=T)

elev_dir <- "/Users/hn/Documents/01_research_data/large_4_GitHub/Min_DB/all_elevations/"

# this did not lead to anything. I donno how to get my hands on the data
# Elevs <- raster(paste0(elev_dir, "/PNW_3arcsec_Hi_Res/pnwdem3s.flt")) 

Elevs_ascii <- read.asciigrid(paste0(elev_dir, "vicpnwdem.asc"))
DF <- as.data.frame(Elevs_ascii)
names(DF)[1] <- "elevation"
names(DF)[2] <- "long"
names(DF)[3] <- "lat"
DF <- data.table(DF)
DF$location <- paste0(DF$lat, "_", DF$long)
LocationGroups$location <- paste0(LocationGroups$latitude, "_", LocationGroups$longitude)
DF_subset_AllourLocs <- DF %>% filter(location %in% LocationGroups$location) %>% data.table()


LocationGroups <- within(LocationGroups, remove(latitude, longitude))
DF_subset_AllourLocs <- left_join(DF_subset_AllourLocs, LocationGroups, by="location")

DF_subset_AllourLocs[, .(eleveation_mean = mean(elevation)), by = c("locationGroup")]
DF_subset_AllourLocs[, .(eleveation_min = min(elevation)), by = c("locationGroup")]
DF_subset_AllourLocs[, .(eleveation_max = max(elevation)), by = c("locationGroup")]

###############################################################################################
binary_core_path = "/Users/hn/Documents/00_GitHub/Ag/read_binary_core/read_binary_core.R"
source(binary_core_path)

observed_dir <- "/Users/hn/Documents/01_research_data/codling_moth_2021/00_binary_Aeolus/historical/"
all_data=data.table()

counter=0
for (loc in LocationGroups$location){
  if (file.exists((paste0(observed_dir, "data_", loc)))) {
    counter=counter + 1
  }
}


for (loc in LocationGroups$location){
  data_dt <- read_binary(paste0(observed_dir, "data_", loc), hist=TRUE, no_vars=8)
  data_dt$location <- loc
  data_dt$t_mean <- (data_dt$tmax + data_dt$tmin)/2
  all_data = rbind(all_data, data_dt)
}

all_data <- left_join(all_data, LocationGroups, by="location")
all_data_noNA <- all_data %>% drop_na(t_mean)
all_data_noNA[, .(tmean_mean = mean(t_mean)), by = c("locationGroup")]


all_data_noNA[, .(precip_mean = mean(precip)), by = c("locationGroup")]


all_data_noNA[, annual_cum_precip := cumsum(precip), by=list(year, location)]

all_data_noNA[, annual_cum_precip := cumsum(precip), by=list(year, location)]


data_dt <- all_data_noNA %>% filter(month==12 & day == 30) %>% data.table()
data_dt[, .(precip_mean = mean(annual_cum_precip)), by = c("locationGroup")]










# loc <- "Omak_48.40625_-119.53125"
# data_dt <- read_binary(paste0(observed_dir, loc), hist=TRUE, no_vars=8)
# data_dt <- put_chill_season(data_dt, chill_start = "sept")
# data_dt <- data_dt %>% filter(!(month %in% c(4, 5, 6, 7, 8))) %>% data.table()

# # remove the incomplete chill seasons
# data_dt <- data_dt %>% filter(chill_season != min(data_dt$chill_season))
# data_dt <- data_dt %>% filter(chill_season != max(data_dt$chill_season)) %>% data.table()

# #
# # add DoY to the data
# #
# data_dt$DoWintSeason <- 1
# data_dt[, DoWintSeason := cumsum(DoWintSeason), by=list(chill_season)]

# data_dt[, annual_cum_precip := cumsum(precip), by=list(chill_season)]

# #
# #  pick only the last day of the chill seasons
# #
# data_dt <- data_dt %>% filter(month==3 & day == 31) %>% data.table()
# mean(data_dt$annual_cum_precip)

# ###
# ### annual
# ###
# data_dt <- read_binary(paste0(observed_dir, loc), hist=TRUE, no_vars=8)
# #
# # compute cumulative sum
# #
# data_dt[, annual_cum_precip := cumsum(precip), by=list(year)]
# data_dt <- data_dt %>% filter(month==12 & day == 31) %>% data.table()
# mean(data_dt$annual_cum_precip)


