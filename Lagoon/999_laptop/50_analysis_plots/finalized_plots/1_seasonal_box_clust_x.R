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

precip_AV_fileNs <- c("seasonal_cum_precip")
rain_AV_fileNs <- c("seasonal_cum_rain")
snow_AV_fileNs <- c("seasonal_cum_snow")
runoff_AV_fileNs <- c("seasonal_cum_runbase")

season_types <- c("fall", "winter", "spring", "summer")
timeP_ty_middN <- c("seasonal")

av_tg_col_pref <- c("seasonal_cum_")
av_titles <- c("seasonal cum. ")
emissions <- c("RCP 4.5", "RCP 8.5")

dt_type <-  in_dir_ext[1]
in_dir <- paste0(data_base, dt_type, "/")
timeP_ty <- 1
season_g <- season_types[1]

for (dt_type in in_dir_ext){ # precip or rain or runoff?
  in_dir <- paste0(data_base, dt_type, "/")
  for (timeP_ty in 1:1){ # annual or chunk or wtr_yr?

    if (dt_type=="precip"){
     files <- precip_AV_fileNs
     AV_y_lab <- "cum. precip. (mm)"
     AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], dt_type)
     AV_title <- paste0(av_titles[timeP_ty], "precip.")

     } else if (dt_type=="rain"){
      files <- rain_AV_fileNs
      AV_y_lab <- "cum. rain (mm)"
      AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], dt_type)
      AV_title <- paste0(av_titles[timeP_ty], "rain.")

     } else if (dt_type=="snow"){
      files <- snow_AV_fileNs
      AV_y_lab <- "cum. snow (mm)"
      AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], dt_type)
      AV_title <- paste0(av_titles[timeP_ty], "snow.")
     } else if (dt_type=="runbase"){
      files <- runoff_AV_fileNs
      AV_y_lab <- "cum. runoff (mm)"
      AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], "runbase")
      AV_title <- paste0(av_titles[timeP_ty], "runoff.")
    }

    AVs <- readRDS(paste0(in_dir, files[timeP_ty], ".rds")) %>% data.table()
    AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
    AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table(); rm(AVs)

    for (season_g in season_types){
      subttl <- paste0("(", season_g, " season)")
      curr_AVs_85 <- AVs_85 %>% filter(season == season_g) %>% data.table()
      curr_AVs_45 <- AVs_45 %>% filter(season == season_g) %>% data.table()
      
      #########
      ######### Actual value plots
      #########
      AV_box_85 <- seasonal_cum_box_clust_x(dt = curr_AVs_85, tgt_col = AV_tg_col,
                                            y_lab = paste0("seasonal ", AV_y_lab))
      AV_box_85 <- AV_box_85 + ggtitle(label= paste0(AV_title, " ", subttl)) 
      # , subtitle=subttl

      AV_box_45 <- seasonal_cum_box_clust_x(dt = curr_AVs_45, tgt_col = AV_tg_col,
                                            y_lab = paste0("seasonal ", AV_y_lab))
      AV_box_45 <- AV_box_45 + ggtitle(label= paste0(AV_title, " ", subttl))
      # , subtitle=subttl

      plot_dir <- paste0(in_dir, "new_2_", dt_type, "/seasonal/clust_x/")
      if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}

      ggsave(filename = paste0(gsub("\ ", "_", season_g), "_", 
                               timeP_ty_middN[timeP_ty], "_RCP85.png"),
             plot = AV_box_85, 
             width = 6, height = 2.5, units = "in", 
             dpi = 400, device = "png",
             path = plot_dir)

      ggsave(filename = paste0(gsub("\ ", "_", season_g), "_", 
                               timeP_ty_middN[timeP_ty], "_RCP45.png"),
             plot = AV_box_45, 
             width = 6, height = 2.5, units = "in", 
             dpi = 400, device = "png",
             path = plot_dir)
    }
    print (plot_dir)
  }
}


