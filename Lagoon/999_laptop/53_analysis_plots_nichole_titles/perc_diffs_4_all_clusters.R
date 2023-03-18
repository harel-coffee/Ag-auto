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

bias_dir_ext <- "/02_med_diff_med_obs/"
unbias_dir_ext <- "/02_med_diff_med_no_bias/"

precip_AV_fileNs <- c("ann_all_last_days", "wtr_yr_sept_all_last_days")
rain_AV_fileNs <- c("ann_cum_rain", "wtr_yr_cum_rain")
snow_AV_fileNs <- c("ann_cum_snow", "wtr_yr_cum_snow")
runoff_AV_fileNs <- c("ann_cum_runbase", "wtr_yr_cum_runbase")

cluster_types <- c("least precip", "lesser precip", "less precip", "most precip")
timeP_ty_middN <- c("ann", "wtr_yr")

av_tg_col_pref <- c("annual_cum_", "annual_cum_")

av_titles <- c("ann. cum. ", "wtr. yr. cum. ")
emissions <- c("RCP 4.5", "RCP 8.5")

dt_type <-  in_dir_ext[1]
in_dir <- paste0(data_base, dt_type, "/")
timeP_ty <- 1
clust_g <- cluster_types[1]

for (dt_type in in_dir_ext){ # precip or rain or runoff?
  in_dir <- paste0(data_base, dt_type, "/")
  for (timeP_ty in 1:3){ # annual or chunk or wtr_yr?

    # if (dt_type=="precip"){
    #  files <- precip_AV_fileNs
    #  AV_y_lab <- "cum. precip. (mm)"
    #  AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], dt_type)
    #  AV_title <- paste0(av_titles[timeP_ty], "precip.")

    #  } else if (dt_type=="rain"){
    #   files <- rain_AV_fileNs
    #   AV_y_lab <- "cum. rain (mm)"
    #   AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], dt_type)
    #   AV_title <- paste0(av_titles[timeP_ty], "rain.")

    #  } else if (dt_type=="snow"){
    #   files <- snow_AV_fileNs
    #   AV_y_lab <- "cum. snow (mm)"
    #   AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], dt_type)
    #   AV_title <- paste0(av_titles[timeP_ty], "snow.")

    #  }else if (dt_type=="runbase"){
    #   files <- runoff_AV_fileNs
    #   AV_y_lab <- "cum. runoff (mm)"
    #   AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], "runbase")
    #   AV_title <- paste0(av_titles[timeP_ty], "runoff.")
    # }
    unbias_diff <- readRDS(paste0(in_dir, unbias_dir_ext, 
                                  "detail_med_diff_med_", 
                                  timeP_ty_middN[timeP_ty], "_", 
                                  dt_type, ".rds")) %>% data.table()

    unbias_diff_45 <- unbias_diff %>% filter(emission=="RCP 4.5") %>% data.table()
    unbias_diff_85 <- unbias_diff %>% filter(emission=="RCP 8.5") %>% data.table()
    rm(unbias_diff)

    ################# plotting area below
    #########
    ######### unbiased Percentage diffs
    #########
    if (dt_type=="runbase"){
      x <- "runoff"
      } else {
      x <- dt_type
    }

    if (timeP_ty_middN[timeP_ty]=="ann"){
      y <- "annual"
       } else if (timeP_ty_middN[timeP_ty]=="chunk"){
      y <- "seasonal"
       } else{
      y <- "water year"
    }

    box_title <- paste0("unbiased differences (", y, ", ", x, ".)")
    box_subtitle <- "for each model median is\ntaken over years, separately"

    unbias_perc_diff_45 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = unbias_diff_45,
                                                             y_lab = "differences (%)",
                                                             tgt_col = "perc_diff",
                                                             ttl = box_title, 
                                                             subttl = box_subtitle)
    unbias_perc_diff_45 <- unbias_perc_diff_45 + ggtitle(box_title)

    unbias_perc_diff_85 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = unbias_diff_85,
                                                             y_lab = "differences (%)",
                                                             tgt_col = "perc_diff",
                                                             ttl = box_title, 
                                                             subttl = box_subtitle)
    unbias_perc_diff_85 <- unbias_perc_diff_85 + ggtitle(box_title)

    plot_dir <- paste0(in_dir, "new_2_", dt_type, "/perc_changes/")
    if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
    
    ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_unbiased_diffs_RCP45.png"),
           plot = unbias_perc_diff_45, 
           width = 6, height = 2.5, units = "in", 
           dpi=400, device = "png",
           path = plot_dir)
    
    ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_unbiased_diffs_RCP85.png"),
           plot = unbias_perc_diff_85, 
           width = 6, height = 2.5, units = "in", 
           dpi = 400, device = "png",
           path = plot_dir)
    
    print (plot_dir)
    ################# plotting area above  
  }
}


