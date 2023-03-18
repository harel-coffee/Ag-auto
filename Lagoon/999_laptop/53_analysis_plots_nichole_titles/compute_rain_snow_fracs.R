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
in_dir <- paste0(data_dir, "rain/")
out_dir <- paste0(data_dir, "rain_snow_fractions/")
if (dir.exists(out_dir) == F) {dir.create(path = out_dir, recursive = T)}
############################################################################

rain_AV_fileNs <- c("ann_cum_rain", "wtr_yr_cum_rain", 
                    "seasonal_cum_rain", "month_cum_rain")
timeP_ty_middN <- c("annual", "wtr_yr", "seasonal", "monthly")
av_tg_col_pref <- c("annual_cum_", "annual_cum_", "seasonal_cum_", "monthly_cum_")
av_titles <- c("ann. cum. ", "wtr. yr. cum. ", "seasonal cum. ", "monthly cum. ")
############################################################################
timeP_ty <- 1
############################################################################

for (timeP_ty in 1:4){ # annual or wtr_yr or seasonal or monthly?
  precip_tg_col <- paste0(av_tg_col_pref[timeP_ty], "precip")
  rain_tg_col <- paste0(av_tg_col_pref[timeP_ty], "rain")
  snow_tg_col <- paste0(av_tg_col_pref[timeP_ty], "snow")
  AVs <- readRDS(paste0(in_dir, rain_AV_fileNs[timeP_ty], ".rds")) %>% data.table()
  if (timeP_ty_middN[timeP_ty] == "seasonal"){
    # we are missing cum. precip. in order to compute
    # fractions! so, we need to add them here.
    seasonal_precip <- readRDS(paste0(data_dir, "/precip/seasonal_cum_precip.rds"))
    AVs <- merge(AVs, seasonal_precip)
    AVs$seasonal_cum_snow <- AVs$seasonal_cum_precip - AVs$seasonal_cum_rain
  }

  AVs$rain_fraction <- AVs[, get(rain_tg_col)] / AVs[, get(precip_tg_col)]
  AVs$snow_fraction <- AVs[, get(snow_tg_col)] / AVs[, get(precip_tg_col)]
  # do not do the following, it is possible that
  # all precip falls as snow. so, if NaN is produced for rain
  # we will by mistake produce NaN for snow as well!!!
  # AVs$snow_fraction <- 1 - AVs$rain_fraction
  saveRDS(AVs, paste0(out_dir, timeP_ty_middN[timeP_ty], "_fracs.rds"))
}







