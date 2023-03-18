rm(list=ls())
library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)
library(ggpubr)

options(digits=9)
options(digit=9)

source_path_1 = "/Users/hn/Documents/GitHub/Ag/chilling/4th_draft/chill_core.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/chilling/4th_draft/chill_plot_core.R"
source(source_path_1)
source(source_path_2)
##########################################################################################

data_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/bloom_new_params/"
plot_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/bloom_new_params/"

##########################################################################################
file_name <- "bloom_limited_cities.rds"
all_data <- data.table(readRDS(paste0(data_dir, file_name)))
all_data <- merge(all_data, cities, all.X=TRUE, by="location")

emissions <- c("RCP 8.5", "RCP 4.5") 
all_cities <- unique(all_data$city)
apple_types <- c("cripps_pink", "gala", "red_deli")

em = "RCP 8.5"
ct = "Richland"

for (em in emissions){
  for (ct in all_cities){
    curr_data <- all_data %>% filter(city == ct & emission==em) %>% data.table()
    
    cripps_pink <- subset(curr_data, select=c(time_period, emission, city, 
                                              dayofyear, cripps_pink, year, model))

    gala <- subset(curr_data, select=c(time_period, emission, city, 
                                       dayofyear, gala, year, model))

    red_deli <- subset(curr_data, select=c(time_period, emission, city, 
                                           dayofyear, red_deli, year, model))

  }
}









