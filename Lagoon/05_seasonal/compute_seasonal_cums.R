rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
options(digit=9)
options(digits=9)

source_path_1 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)
############################################################################

data_base <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/"
in_dir_ext <- c("precip", "rain", "snow", "runbase")

AV_fileNs <- c("month_all_last_days", "month_cum_rain", 
               "month_cum_snow", "monthly_cum_runbase")
ii <- 1

for (ii in 1:4){
  dat_dir <- paste0(data_base, in_dir_ext[ii], "/")
  print(in_dir_ext[ii])
  data <- data.table(readRDS(paste0(dat_dir, AV_fileNs[ii], ".rds")))
  data <- seasonal_cum(data_tb = data, material=in_dir_ext[ii])
  saveRDS(data, 
          paste0(dat_dir, "seasonal_cum_", in_dir_ext[ii], ".rds"))

}


