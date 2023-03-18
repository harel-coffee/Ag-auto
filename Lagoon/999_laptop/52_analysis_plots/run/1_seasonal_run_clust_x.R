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

b <- "/Users/hn/Documents/GitHub/Ag/Lagoon/"
source_path_1 <- paste0(b, "core_lagoon.R")
source_path_2 <- paste0(b, "/core_plot_lagoon.R")
source(source_path_1)
source(source_path_2)
#######################################################################

data_base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
plot_base <- paste0(data_base, "plots/runoff/seasonal/clust_x/")
in_dir_ext <- c("runbase")
unbias_dir_ext <- "/02_med_diff_med_no_bias/"

runoff_AV_fileNs <- c("seasonal_cum_runbase")
season_types <- c("fall", "winter", "spring", "summer")
timeP_ty_middN <- c("seasonal")

av_tg_col_pref <- c("seasonal_cum_")
av_titles <- c("seasonal ")

dt_type <-  in_dir_ext[1]
in_dir <- paste0(data_base, dt_type, "/")
timeP_ty <- 1
season_g <- season_types[1]

for (dt_type in in_dir_ext){
  in_dir <- paste0(data_base, dt_type, "/")
  for (timeP_ty in 1:1){
    files <- runoff_AV_fileNs
    AV_y_lab <- "runoff (mm)"
    AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], "runbase")

    AVs <- readRDS(paste0(in_dir, files[timeP_ty], ".rds")) %>%
           data.table()
    unbias_diff<-readRDS(paste0(in_dir, 
                                unbias_dir_ext, 
                                "detail_med_diff_med_", 
                                timeP_ty_middN[timeP_ty], 
                                "_", dt_type, ".rds")) %>% 
                   data.table()
    
    AVs <- na.omit(AVs)
    unbias_diff <- na.omit(unbias_diff)
    
    AVs <- remove_observed(AVs)
    unbias_diff <- remove_observed(unbias_diff)
    AVs <- remove_current_timeP(AVs)# remove 2006-2025
    unbias_diff <- remove_current_timeP(unbias_diff)
    
    # update clusters labels
    AVs <- convert_5_numeric_clusts_to_alphabet(AVs)
    unbias_diff <- convert_5_numeric_clusts_to_alphabet(unbias_diff)

    AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
    AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table()

    unbias_diff_45 <- unbias_diff %>% 
                      filter(emission=="RCP 4.5") %>% 
                      data.table()
    unbias_diff_85 <- unbias_diff %>% 
                      filter(emission=="RCP 8.5") %>% 
                      data.table(); 
    rm(AVs, unbias_diff)

    for (season_g in season_types){
      subttl <- paste0(" (", season_g, ")")
      curr_AVs_85 <- AVs_85 %>% 
                     filter(season == season_g) %>% 
                     data.table()
      curr_AVs_45 <- AVs_45 %>% 
                     filter(season == season_g) %>% 
                     data.table()

      curr_diff_45 <- unbias_diff_45 %>% 
                      filter(season == season_g) %>% 
                      data.table()
      curr_diff_85 <- unbias_diff_85 %>% 
                      filter(season == season_g) %>% 
                      data.table()      
      #########
      ######### Actual value plots
      #########
      AV_title <- paste0(season_g, 
                         " runoff for historical and three", 
                         " future time frames")

      quans_85 <- find_quantiles(curr_AVs_85, 
                                 tgt_col= AV_tg_col, 
                                 time_type="seasonal")
      quans_45 <- find_quantiles(curr_AVs_45, 
                                 tgt_col= AV_tg_col, 
                                 time_type="seasonal")
  
      AV_box_85 <- seasonal_cum_box_clust_x(dt = curr_AVs_85, 
                                            tgt_col = AV_tg_col,
                                            y_lab = AV_y_lab) +
                   ggtitle(label= AV_title) +
            coord_cartesian(ylim = c(max(0, quans_85[1]), quans_85[2]))

      AV_box_45 <- seasonal_cum_box_clust_x(dt = curr_AVs_45, 
                                            tgt_col = AV_tg_col,
                                            y_lab = AV_y_lab)+
                   ggtitle(label= AV_title) +
            coord_cartesian(ylim = c(max(0, quans_45[1]), quans_45[2]))
      #########
      ######### difference plot
      #########
      box_title <- "% difference between future"
      box_title <- paste0(box_title, 
                          " and historical ", 
                          season_g, 
                          " runoff")

      quans_85 <- find_quantiles(curr_diff_85, 
                                 tgt_col="perc_diff", 
                                 time_type="seasonal")
      quans_45 <- find_quantiles(curr_diff_45, 
                                 tgt_col="perc_diff", 
                                 time_type="seasonal")
      unbias_perc_diff_85 <- seasonal_cum_box_clust_x(dt = curr_diff_85,
                                              y_lab = "differences (%)",
                                              tgt_col = "perc_diff") + 
                             ggtitle(box_title) +
                      coord_cartesian(ylim = c(quans_85[1], 
                                               quans_85[2]))

      unbias_perc_diff_45 <- seasonal_cum_box_clust_x(dt = curr_diff_45,
                                              y_lab = "differences (%)",
                                              tgt_col = "perc_diff") + 
                             ggtitle(box_title) + 
                     coord_cartesian(ylim = c(quans_45[1], 
                                              quans_45[2]))
      ###################################
      #####
      ##### arrange plots
      #####
      ###################################
      RCP45 <- ggarrange(plotlist = list(AV_box_45, unbias_perc_diff_45),
                         ncol = 1, nrow = 2, 
                         common.legend = TRUE, legend="bottom")

      RCP85 <- ggarrange(plotlist = list(AV_box_85, unbias_perc_diff_85),
                         ncol = 1, nrow = 2, 
                         common.legend = TRUE, legend="bottom")
      ###################################
      #####
      ##### save plots
      #####
      ###################################
      ###################################
      #####
      ##### runoff and diffs
      #####
      ###################################
      run_n_diff <- paste0(plot_base, "run_n_diff/")
      if (dir.exists(run_n_diff) == F) {
        dir.create(path = run_n_diff, recursive = T)}

      ggsave(filename = paste0(gsub("\ ", "_", season_g), "_45.png"),
             plot = RCP45,
             width = 5.5, height=3.5, units = "in", 
             dpi=600, device = "png", path = run_n_diff)

      ggsave(filename = paste0(gsub("\ ", "_", season_g), "_85.png"),
             plot = RCP85,
             width = 5.5, height = 3.5, units = "in",
             dpi=600, device = "png", path = run_n_diff)
      ###################################
      #####
      ##### actual values
      #####
      ###################################
      just_AVs <- paste0(plot_base, "just_AVs/")
      if (dir.exists(just_AVs) == F) {
        dir.create(path = just_AVs, recursive = T)}

      ggsave(filename = paste0(gsub("\ ", "_", season_g), "_45.png"),
             plot = AV_box_45,
             width = 5.5, height=1.5, units = "in", 
             dpi=600, device = "png", path = just_AVs)

      ggsave(filename = paste0(gsub("\ ", "_", season_g), "_85.png"),
             plot = AV_box_85,
             width = 5.5, height = 1.5, units = "in",
             dpi=600, device = "png", path = just_AVs)
      ###################################
      #####
      ##### differences
      #####
      ###################################
      just_diff <- paste0(plot_base, "just_diff/")
      if (dir.exists(just_diff) == F) {
        dir.create(path = just_diff, recursive = T)}

      ggsave(filename = paste0(gsub("\ ", "_", season_g), "_45.png"),
             plot=unbias_perc_diff_45,
             width=5.5, height=1.5, units = "in", 
             dpi=600, device = "png", path = just_diff)

      ggsave(filename = paste0(gsub("\ ", "_", season_g), "_85.png"),
             plot = unbias_perc_diff_85,
             width=5.5, height=1.5, units = "in",
             dpi=600, device = "png", path = just_diff)
    }
  }
}


