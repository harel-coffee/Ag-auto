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

  first_frost_meds <- data.table(readRDS(paste0(data_dir, "first_frost_medians_", "till_", due, ".rds")))
  fifth_frost_meds <- data.table(readRDS(paste0(data_dir, "fifth_frost_medians_", "till_", due, ".rds")))
  first_frost_meds <- within(first_frost_meds, remove(extended_DoY))
  fifth_frost_meds <- within(fifth_frost_meds, remove(extended_DoY))
  
  first_frost_meds$extended_DoY_median <- first_frost_meds$median + 243
  fifth_frost_meds$extended_DoY_median <- fifth_frost_meds$median + 243
  saveRDS(first_frost_meds, paste0(data_dir, "first_frost_medians_", "till_", due, ".rds"))
  saveRDS(fifth_frost_meds, paste0(data_dir, "fifth_frost_medians_", "till_", due, ".rds"))

}

