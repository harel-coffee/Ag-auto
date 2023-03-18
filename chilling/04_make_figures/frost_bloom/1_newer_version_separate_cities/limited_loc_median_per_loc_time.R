rm(list=ls())
library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)
library(ggpubr)

options(digits=9)
options(digit=9)

source_path_1 = "/Users/hn/Documents/GitHub/Ag/chilling/chill_core.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/chilling/chill_plot_core.R"
source(source_path_1)
source(source_path_2)

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

  first_frost <- first_frost %>% filter(time_period %in% c("1979-2015", "2026-2050", "2051-2075", "2076-2099"))
  fifth_frost <- fifth_frost %>% filter(time_period %in% c("1979-2015", "2026-2050", "2051-2075", "2076-2099"))
  
  first_frost <- pick_single_cities_by_location(dt=first_frost, city_info=LOI)
  fifth_frost <- pick_single_cities_by_location(dt=fifth_frost, city_info=LOI)

  first_frost_medians <- first_frost %>%
                         group_by(time_period, city, emission) %>%
                         summarise(median_chill_DoY = median(chill_dayofyear)) %>%
                         data.table()

  fifth_frost_medians <- fifth_frost %>%
                         group_by(time_period, city, emission) %>%
                         summarise(median_chill_DoY = median(chill_dayofyear)) %>%
                         data.table()
  
  out_dir <- paste0(data_dir, "cleaner/"); print(out_dir)
  if (dir.exists(file.path(out_dir)) == F) {
    dir.create(path = file.path(out_dir), recursive = T)
  }
  write.table(first_frost_medians, 
              file = paste0(out_dir, "first_city_loc_time.csv"), 
              row.names=FALSE, na="", col.names=T, sep=",")

  write.table(fifth_frost_medians, 
              file = paste0(out_dir, "fifth_city_loc_time.csv"), 
              row.names=FALSE, na="", col.names=T, sep=",")
}



