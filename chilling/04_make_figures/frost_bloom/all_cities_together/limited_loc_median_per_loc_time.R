rm(list=ls())
library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)
library(ggpubr)

options(digits=9)
options(digit=9)

# compute medians per location, time_periods

dues <- c("Dec", "Jan", "Feb")
due <- dues[3]
DoY_Map <- read.csv("/Users/hn/Documents/GitHub/Ag/DoY_Map.csv", as.is=TRUE)
for (due in dues){
  #######################################################################################
  param_dir <- "/Users/hn/Documents/GitHub/Ag/chilling/parameters/"
  LOI <- data.table(read.csv(paste0(param_dir, "limited_locations.csv"), as.is=T))

  #######################################################################################
  # Read Data
  
  data_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/"
  data_dir <- paste0(data_dir, due, "/")

  first_frost <- data.table(readRDS(paste0(data_dir, "first_frost_till_", due, ".rds")))
  fifth_frost <- data.table(readRDS(paste0(data_dir, "fifth_frost_till_", due, ".rds")))

  first_frost <- first_frost %>% filter(year != 1949) %>% data.table()
  fifth_frost <- fifth_frost %>% filter(year != 1949) %>% data.table()
  
  first_frost <- pick_single_cities_by_location(dt=first_frost, city_info=LOI)
  fifth_frost <- pick_single_cities_by_location(dt=fifth_frost, city_info=LOI)

  first_frost_medians <- first_frost %>%
                         group_by(time_period, city, emission) %>%
                         summarise(median = median(chill_dayofyear)) %>%
                         data.table()

  fifth_frost_medians <- fifth_frost %>%
                         group_by(time_period, city, emission) %>%
                         summarise(median = median(chill_dayofyear)) %>%
                         data.table()
  
  write.table(first_frost_medians_merged, 
              file = paste0(data_dir, "first_city_loc_time.csv"), 
              row.names=FALSE, na="", col.names=T, sep=",")

  write.table(fifth_frost_medians_merged, 
              file = paste0(data_dir, "fifth_city_loc_time.csv"), 
              row.names=FALSE, na="", col.names=T, sep=",")
}



