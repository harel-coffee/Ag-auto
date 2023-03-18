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

source_path_1 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)
############################################################################

data_base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/rain_snow_fractions/"
AV_fileNs <- c("annual_fracs", "wtr_yr_fracs")
timeP_ty_middN <- c("ann", "wtr_yr")
timeP_ty <- 1

for (timeP_ty in 1:2){ # annual or wtr_yr?
  ###############################################################
  # set up title stuff
  # 
  if (timeP_ty_middN[timeP_ty]== "ann"){
     title_time <- "calendar year"
     } else if (timeP_ty_middN[timeP_ty]== "wtr_yr"){
     title_time <- "water year"
  }
  AV_y_lab <- "cum. precip. (mm)"
  AV_title <- paste0("ann. precip.", " (", title_time, ")")
  AV_tg_col <- "annual_cum_precip"
  
  ###############################################################
  ##################################################################################
  AVs <- readRDS(paste0(data_base, AV_fileNs[timeP_ty], ".rds")) %>% data.table()
  AVs <- subset(AVs, select = c("location", "cluster", "year", "time_period", 
                                "model", "emission",
                                "annual_cum_precip", "rain_fraction", "snow_fraction"))

  AVs <- remove_observed(AVs)
  AVs <- remove_current_timeP(AVs) # remove 2006-2025  
  AVs <- convert_5_numeric_clusts_to_alphabet(data_tb = AVs) # update clusters labels

  AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
  AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table(); rm(AVs)
  
  quans_85 <- find_quantiles(AVs_85, tgt_col= AV_tg_col, time_type="annual")
  quans_45 <- find_quantiles(AVs_45, tgt_col= AV_tg_col, time_type="annual")

  AV_box_85 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = AVs_85, y_lab = AV_y_lab, 
                                                 tgt_col = AV_tg_col) + 
               ggtitle(AV_title) +
               coord_cartesian(ylim = c(quans_85[1], quans_85[2]))

  AV_box_45 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = AVs_45, y_lab = AV_y_lab, 
                                                 tgt_col = AV_tg_col) + 
               ggtitle(AV_title) + 
               coord_cartesian(ylim = c(quans_45[1], quans_45[2]))

  ###################################
  #####
  ##### fraction plots
  #####
  ###################################
  ######################################################################
  ######################################################################
  box_title <- paste0("fraction of precip. fell as rain (", title_time, ")")

  quans_85 <- 100 * find_quantiles(AVs_85, tgt_col= "rain_fraction", time_type="annual")
  quans_45 <- 100 * find_quantiles(AVs_45, tgt_col= "rain_fraction", time_type="annual")
  
  rain_frac_85 <- annual_fraction(data_tb = AVs_85,
                                  y_lab = "rain fraction (%)", 
                                  tgt_col="rain_fraction") +
                  ggtitle(box_title) + 
                  coord_cartesian(ylim = c(max(-2, quans_85[1]), 
                                           min(quans_85[2], 110)))  
  rain_85 <- ggarrange(plotlist = list(AV_box_85, rain_frac_85),
                       ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")
  ######
  ######
  ######
  rain_frac_45 <- annual_fraction(data_tb = AVs_45,
                                  y_lab = "rain fraction (%)", 
                                  tgt_col="rain_fraction") +
                  ggtitle(box_title) + 
                  coord_cartesian(ylim = c(max(-2, quans_45[1]), 
                                           min(quans_45[2], 110)))

  rain_45 <- ggarrange(plotlist = list(AV_box_45, rain_frac_45),
                       ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")
  ##################################################################################
  ##################################################################################
  # box_title <- paste0("snow fraction", " (", title_time, ")")
  # quans_85 <- 100 * find_quantiles(AVs_85, tgt_col= "snow_fraction", time_type="annual")
  # quans_45 <- 100 * find_quantiles(AVs_45, tgt_col= "snow_fraction", time_type="annual")

  # snow_frac_85 <- annual_fraction(data_tb = AVs_85,
  #                                 y_lab = "snow fraction (%)", 
  #                                 tgt_col="snow_fraction") +
  #                 ggtitle(box_title) + 
  #                 coord_cartesian(ylim = c(quans_85[1], 110))

  # snow_85 <- ggarrange(plotlist = list(AV_box_85, snow_frac_85),
  #                      ncol=1, nrow=2, common.legend=TRUE, 
  #                      legend="bottom")

  # snow_frac_45 <- annual_fraction(data_tb = AVs_45,
  #                                 y_lab = "snow fraction (%)", 
  #                                 tgt_col="snow_fraction") +
  #                 ggtitle(box_title) + 
  #                 coord_cartesian(ylim = c(quans_45[1], 110))

  # snow_45 <- ggarrange(plotlist = list(AV_box_45, snow_frac_45),
  #                      ncol=1, nrow=2, common.legend=TRUE, 
  #                     legend="bottom")
  
  ###################################
  #####
  ##### save plots
  #####
  ###################################
    
  plot_dir <- paste0(data_base, "narrowed_rain_snow_fractions", "/", timeP_ty_middN[timeP_ty], "/")
  if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
  print (plot_dir)

  ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_rain_45.png"),
         plot = rain_45, width=5.5, height=3, units = "in", 
         dpi=400, device = "png", path = plot_dir)

  ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_rain_85.png"),
         plot = rain_85, width=5.5, height=3, units = "in", 
         dpi=400, device = "png", path = plot_dir)

  # ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_snow_45.png"),
  #        plot = snow_45, width=5.5, height=3, units = "in", 
  #        dpi=400, device = "png", path = plot_dir)

  # ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_snow_85.png"),
  #        plot = snow_85, width=5.5, height=3, units = "in", 
  #        dpi=400, device = "png", path = plot_dir)
  # print(paste0(timeP_ty_middN[timeP_ty], "_snow_85.png"))
}


