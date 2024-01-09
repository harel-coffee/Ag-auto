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
base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
data_base <- paste0(base, "rain_snow_fractions/")
AV_fileNs <- c("wtr_yr_fracs") # "annual_fracs",
timeP_ty_middN <- c("wtr_yr") # "ann", 
timeP_ty <- 1

##########################################
#
# unbias diff data directory
#
diff_dir <- paste0(base, "precip/02_med_diff_med_no_bias/")

for (timeP_ty in 1:1){ # annual or wtr_yr?
  ###############################################################
  # set up title stuff
  # 
  # if (timeP_ty_middN[timeP_ty]== "ann"){
  #    title_time <- "calendar year"
  #    } else if (timeP_ty_middN[timeP_ty]== "wtr_yr"){
  #    title_time <- "water year"
  # }
  AV_y_lab <- "precipitation (mm)"
  AV_title <- paste0("annual precipitation for historical", 
                     " and three future time frames")
  AV_tg_col <- "annual_cum_precip"
  
  ###############################################################
  ###############################################################
  AVs <- readRDS(paste0(data_base, AV_fileNs[timeP_ty], ".rds")) %>% 
         data.table()
  unbias_diff <- readRDS(paste0(diff_dir, "detail_med_diff_med_", 
                                timeP_ty_middN[timeP_ty], 
                                "_precip.rds")) %>% data.table()

  AVs <- subset(AVs, select = c("location", "cluster", "year", 
                                "time_period", "model",
                                "emission", "annual_cum_precip",
                                "rain_fraction", "snow_fraction"))

  AVs <- remove_observed(AVs)
  AVs <- remove_current_timeP(AVs) # remove 2006-2025
  AVs <- convert_5_numeric_clusts_to_alphabet(data_tb = AVs)

  unbias_diff <- remove_observed(unbias_diff)
  # remove 2006-2025
  unbias_diff <- remove_current_timeP(unbias_diff) 
  unbias_diff <- convert_5_numeric_clusts_to_alphabet(unbias_diff) 

  AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
  AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table()
  rm(AVs)

  unbias_diff_45 <- unbias_diff %>% filter(emission=="RCP 4.5") %>% 
                    data.table()
  unbias_diff_85 <- unbias_diff %>% filter(emission=="RCP 8.5") %>% 
                    data.table()
  rm(unbias_diff)
  ##################################
  #####
  ##### AVs plots
  
  quans_85 <- find_quantiles(AVs_85, tgt_col= AV_tg_col, 
                             time_type="annual")
  quans_45 <- find_quantiles(AVs_45, tgt_col= AV_tg_col, 
                             time_type="annual")

  AV_box_85 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = AVs_85, 
                                            y_lab = AV_y_lab, 
                                            tgt_col = AV_tg_col) + 
               ggtitle(AV_title) +
               coord_cartesian(ylim = c(max(0, quans_85[1]), quans_85[2]))

  AV_box_45 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = AVs_45, 
                                                 y_lab = AV_y_lab, 
                                                 tgt_col = AV_tg_col) + 
               ggtitle(AV_title) + 
               coord_cartesian(ylim=c(max(0, quans_45[1]), quans_45[2]))

  ###################################
  #####
  ##### fraction plots
  #####
  ###################################
  box_title <- paste0("proportion (%) of annual", 
                      " precipitation in rain form")

  quans_85 <- 100 * find_quantiles(AVs_85, tgt_col="rain_fraction", 
                                   time_type="annual")
  quans_45 <- 100 * find_quantiles(AVs_45, tgt_col="rain_fraction", 
                                   time_type="annual")  
  rain_frac_85 <- annual_fraction(data_tb = AVs_85,
                                  y_lab = "rain portion (%)", 
                                  tgt_col="rain_fraction") +
                  ggtitle(box_title) + 
                  coord_cartesian(ylim = c(max(-2, quans_85[1]), 
                                           min(quans_85[2], 110)))
  rain_frac_45 <- annual_fraction(data_tb = AVs_45,
                                  y_lab = "rain portion (%)", 
                                  tgt_col="rain_fraction") +
                  ggtitle(box_title) + 
                  coord_cartesian(ylim = c(max(-2, quans_45[1]), 
                                           min(quans_45[2], 110)))
  #################################################################
  ###################################
  #####
  ##### difference plots
  #####
  ###################################
  box_title <- paste0("% difference between future and historical", 
                      " annual precipitation")

  quans_85 <- find_quantiles(unbias_diff_85, tgt_col= "perc_diff", time_type="annual")
  quans_45 <- find_quantiles(unbias_diff_45, tgt_col= "perc_diff", time_type="annual")

  unbias_perc_diff_85<-ann_wtrYr_chunk_cum_box_cluster_x(dt=unbias_diff_85,
                                                         y_lab="differences (%)",
                                                         tgt_col="perc_diff",
                                                         ttl=box_title, 
                                                         subttl=box_subtitle) + 
                         ggtitle(box_title) +
                         coord_cartesian(ylim = c(quans_85[1], quans_85[2]))

  unbias_perc_diff_45<-ann_wtrYr_chunk_cum_box_cluster_x(dt=unbias_diff_45,
                                                         y_lab="differences (%)",
                                                         tgt_col="perc_diff",
                                                         ttl=box_title, 
                                                         subttl = box_subtitle) + 
                         ggtitle(box_title) + 
                         coord_cartesian(ylim = c(quans_45[1], quans_45[2]))
  ###########################################
  #####
  ##### save plots
  #####
  ###########################################
  #
  # just plots
  #
  ###########################################
  #
  # just fractions
  #
  ##########################################
  plot_base <- paste0(base, "plots/")
  just_frac_dir <- paste0(plot_base, "precip/", 
                          timeP_ty_middN[timeP_ty], "/just_frac/")
  if (dir.exists(just_frac_dir) == F) {
    dir.create(path = just_frac_dir, recursive = T)}
  print (just_frac_dir)

  ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_rain_45.png"),
         plot = rain_frac_45, 
         width=5.5, height=1.5, units = "in", 
         dpi=600, device = "png", path = just_frac_dir)

  ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_rain_85.png"),
         plot = rain_frac_85, 
         width=5.5, height=1.5, units = "in", 
         dpi=600, device = "png", path = just_frac_dir)
  ##########################################
  #
  # just AVs
  #
  ##########################################
  just_AVs_dir <- paste0(plot_base, "precip/", 
                          timeP_ty_middN[timeP_ty], "/just_AVs/")
  if (dir.exists(just_AVs_dir) == F) {
    dir.create(path = just_AVs_dir, recursive=T)}
  print (just_AVs_dir)

  ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_rain_45.png"),
         plot = AV_box_45, 
         width=5.5, height=1.5, units = "in", 
         dpi=600, device = "png", path = just_AVs_dir)

  ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_rain_85.png"),
         plot = AV_box_85, 
         width=5.5, height=1.5, units = "in", 
         dpi=600, device = "png", path = just_AVs_dir)
  ##########################################
  #
  # just unbiass diff
  #
  ##########################################
  just_diff_dir <- paste0(plot_base, "precip/", 
                          timeP_ty_middN[timeP_ty], 
                          "/just_unbiass_diff/")
  if (dir.exists(just_diff_dir) == F) {
    dir.create(path = just_diff_dir, recursive = T)}
  print (just_diff_dir)

  ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_rain_45.png"),
         plot = unbias_perc_diff_45, 
         width=5.5, height=1.5, units = "in", 
         dpi=600, device = "png", path = just_diff_dir)

  ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_rain_85.png"),
         plot = unbias_perc_diff_85, 
         width=5.5, height=1.5, units = "in", 
         dpi=600, device = "png", path = just_diff_dir)
  ###################################
  # 
  # 3 in 1
  #
  plot_3_in_1_dir <- paste0(plot_base, "precip/", 
                     timeP_ty_middN[timeP_ty], "/3_in_1/")
  if (dir.exists(plot_3_in_1_dir) == F) {
  	dir.create(path = plot_3_in_1_dir, recursive = T)}
  print (plot_3_in_1_dir)

  rain_45 <- ggarrange(plotlist = list(AV_box_45, 
                                       unbias_perc_diff_45,
                                       rain_frac_45),
                       ncol = 1, nrow = 3, 
                       common.legend = TRUE, legend="bottom")

  rain_85 <- ggarrange(plotlist = list(AV_box_85, 
                                       unbias_perc_diff_85,
                                       rain_frac_85),
                       ncol = 1, nrow = 3, 
                       common.legend = TRUE, legend="bottom")
  
  ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_rain_45.png"),
         plot = rain_45, 
         width=5.5, height=5, units = "in", 
         dpi=600, device = "png", path = plot_3_in_1_dir)

  ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_rain_85.png"),
         plot = rain_85, 
         width=5.5, height=5, units = "in", 
         dpi=600, device = "png", path = plot_3_in_1_dir)
  ###################################
  # 
  #    precip_diff
  #
  plot_precip_diff_dir <- paste0(plot_base, "precip/", 
                                 timeP_ty_middN[timeP_ty], 
                                 "/precip_&_diffs/")
  if (dir.exists(plot_precip_diff_dir) == F) {
    dir.create(path = plot_precip_diff_dir, recursive = T)}
  print (plot_precip_diff_dir)

  rain_45 <- ggarrange(plotlist = list(AV_box_45, unbias_perc_diff_45),
                       ncol = 1, nrow = 2, 
                       common.legend = TRUE, legend="bottom")

  rain_85 <- ggarrange(plotlist = list(AV_box_85, unbias_perc_diff_85),
                       ncol = 1, nrow = 2, 
                       common.legend = TRUE, legend="bottom")
  
  ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_rain_45.png"),
         plot = rain_45, width=5.5, height=3.5, units = "in", 
         dpi=600, device = "png", path = plot_precip_diff_dir)

  ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_rain_85.png"),
         plot = rain_85, width=5.5, height=3.5, units = "in", 
         dpi=600, device = "png", path = plot_precip_diff_dir)
  ###################################
  # 
  # Precip and fractions
  #
  plot_precip_frac_dir <- paste0(plot_base, "precip/", 
                                 timeP_ty_middN[timeP_ty], 
                                 "/precip_&_fraction/")
  if (dir.exists(plot_precip_frac_dir) == F) {
    dir.create(path = plot_precip_frac_dir, recursive = T)}
  print (plot_precip_frac_dir)

  rain_45 <- ggarrange(plotlist = list(AV_box_45, rain_frac_45),
                       ncol = 1, nrow = 2, 
                       common.legend = TRUE, legend="bottom")

  rain_85 <- ggarrange(plotlist = list(AV_box_85, rain_frac_85),
                       ncol = 1, nrow = 2, 
                       common.legend = TRUE, legend="bottom")
  
  ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_rain_45.png"),
         plot = rain_45, width=5.5, height=3.5, units = "in", 
         dpi=600, device = "png", path = plot_precip_frac_dir)

  ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_rain_85.png"),
         plot = rain_85, width=5.5, height=3.5, units = "in", 
         dpi=600, device = "png", path = plot_precip_frac_dir)

}
