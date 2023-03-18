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
for (due in dues){
  #######################################################################################
  
  #######################################################################################
  # Read Data
  
  data_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/"
  data_dir <- paste0(data_dir, due, "/")

  first_frost <- data.table(readRDS(paste0(data_dir, "first_frost_till_", due, ".rds")))
  fifth_frost <- data.table(readRDS(paste0(data_dir, "fifth_frost_till_", due, ".rds")))
  
  first_frost$extended_DoY <- first_frost$chill_dayofyear + 243
  fifth_frost$extended_DoY <- fifth_frost$chill_dayofyear + 243
  saveRDS(first_frost, paste0(data_dir, "first_frost_till_", due, ".rds"))
  saveRDS(fifth_frost, paste0(data_dir, "fifth_frost_till_", due, ".rds"))

  first_frost <- data.table(readRDS(paste0(data_dir, due, "_cleaner/first_frost_till_", due, ".rds")))
  fifth_frost <- data.table(readRDS(paste0(data_dir, due, "_cleaner/fifth_frost_till_", due, ".rds")))
  
  first_frost$extended_DoY <- first_frost$chill_dayofyear + 243
  fifth_frost$extended_DoY <- fifth_frost$chill_dayofyear + 243
  saveRDS(first_frost, paste0(data_dir, due, "_cleaner/first_frost_till_", due, ".rds"))
  saveRDS(fifth_frost, paste0(data_dir, due, "_cleaner/fifth_frost_till_", due, ".rds"))


}

