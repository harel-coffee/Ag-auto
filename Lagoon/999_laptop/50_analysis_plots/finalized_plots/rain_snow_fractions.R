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
data_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/"
in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/rain/"

############################################################################

rain_AV_fileNs <- c("ann_cum_rain", "wtr_yr_cum_rain", "seasonal_cum_rain")
timeP_ty_middN <- c("annual", "wtr_yr", "seasonal")
av_tg_col_pref <- c("annual_cum_", "annual_cum_", "seasonal_cum_")
av_titles <- c("ann. cum. ", "wtr. yr. cum. ", "seasonal cum. ")

cluster_types <- c("least precip", "lesser precip", "less precip", "most precip")
emissions <- c("RCP 4.5", "RCP 8.5")

############################################################################
timeP_ty <- 1
clust_g <- cluster_types[1]
############################################################################
precip_y_lab <- "cum. precip. (mm)"
rain_y_lab <- "rain fraction of precip."
snow_y_lab <- "snow fraction of precip."

for (timeP_ty in 1:3){ # annual or wtr_yr or seasonal?
  plot_dir <- paste0(data_dir, timeP_ty_middN[timeP_ty], "/fractions/")
  if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}

  precip_tg_col <- paste0(av_tg_col_pref[timeP_ty], "precip")
  rain_tg_col <- paste0(av_tg_col_pref[timeP_ty], "rain")
  AVs <- readRDS(paste0(in_dir, rain_AV_fileNs[timeP_ty], ".rds")) %>% data.table()
  
  AVs$rain_fraction <- AVs[, get(rain_tg_col)] / AVs[, get(precip_tg_col)]
  AVs$snow_fraction <- 1 - AVs$rain_fraction

  for (em in emissions){
  	curr_AVs_em <- AVs %>% filter(emission == em) %>% data.table()
  }
}







