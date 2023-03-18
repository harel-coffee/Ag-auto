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


chill_core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_core.R"
source(chill_core_path)

data_dir = "/Users/hn/Documents/01_research_data/chilling/01_data/02/"
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"

LocationGroups_NoMontana <- read.csv(paste0(param_dir, "LocationGroups_NoMontana.csv"), 
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

DF_subset_AllourLocs <- DF %>% filter(location %in% LocationGroups_NoMontana$location) %>% data.table()


four_locations <- read.csv(paste0(param_dir, "4_locations.csv"), as.is=TRUE)
four_locations$location <- paste0(four_locations$lat, "_", four_locations$long)
four_locations <- within(four_locations, remove(lat, long))

four_elevations <- DF_subset_AllourLocs %>% filter(location %in% four_locations$location)
four_elevations <- left_join(four_elevations, four_locations, by="location")



###############################################################################################
binary_core_path = "/Users/hn/Documents/00_GitHub/Ag/read_binary_core/read_binary_core.R"
source(binary_core_path)

observed_dir <- "/Users/hn/Documents/01_research_data/chilling/01_data/binary_observed_for_topography/"

"Eugene_44.03125_-123.09375"
"Omak_48.40625_-119.53125"
"Walla_46.03125_-118.34375"
"Yakima_46.59375_-120.53125"

loc <- "Yakima_46.59375_-120.53125"
data_dt <- read_binary(paste0(observed_dir, loc), hist=TRUE, no_vars=8)
data_dt$t_mean <- (data_dt$tmax + data_dt$tmin)/2
data_dt <- put_chill_season(data_dt, chill_start = "sept")

# remove the incomplete chill seasons
data_dt <- data_dt %>% filter(chill_season != min(data_dt$chill_season))
data_dt <- data_dt %>% filter(chill_season != max(data_dt$chill_season)) %>% data.table()
data_dt <- data_dt %>% filter(!(month %in% c(4, 5, 6, 7, 8))) %>% data.table()

data_dt <- data_dt[, .(Tmean_per_Chill_Yr = mean(t_mean)), by = c("chill_season")]
mean(data_dt$Tmean_per_Chill_Yr)

data_dt <- read_binary(paste0(observed_dir, loc), hist=TRUE, no_vars=8)
data_dt$t_mean <- (data_dt$tmax + data_dt$tmin)/2
data_dt <- data_dt[, .(Tmean_per_Yr = mean(t_mean)), by = c("year")]
mean(data_dt$Tmean_per_Yr)

######
###### precip
######
"Eugene_44.03125_-123.09375"
"Omak_48.40625_-119.53125"
"Walla_46.03125_-118.34375"
"Yakima_46.59375_-120.53125"

###
### winter chill season
###
loc <- "Omak_48.40625_-119.53125"
data_dt <- read_binary(paste0(observed_dir, loc), hist=TRUE, no_vars=8)
data_dt <- put_chill_season(data_dt, chill_start = "sept")
data_dt <- data_dt %>% filter(!(month %in% c(4, 5, 6, 7, 8))) %>% data.table()

# remove the incomplete chill seasons
data_dt <- data_dt %>% filter(chill_season != min(data_dt$chill_season))
data_dt <- data_dt %>% filter(chill_season != max(data_dt$chill_season)) %>% data.table()

#
# add DoY to the data
#
data_dt$DoWintSeason <- 1
data_dt[, DoWintSeason := cumsum(DoWintSeason), by=list(chill_season)]

data_dt[, annual_cum_precip := cumsum(precip), by=list(chill_season)]

#
#  pick only the last day of the chill seasons
#
data_dt <- data_dt %>% filter(month==3 & day == 31) %>% data.table()
mean(data_dt$annual_cum_precip)

###
### annual
###
data_dt <- read_binary(paste0(observed_dir, loc), hist=TRUE, no_vars=8)
#
# compute cumulative sum
#
data_dt[, annual_cum_precip := cumsum(precip), by=list(year)]
data_dt <- data_dt %>% filter(month==12 & day == 31) %>% data.table()
mean(data_dt$annual_cum_precip)


