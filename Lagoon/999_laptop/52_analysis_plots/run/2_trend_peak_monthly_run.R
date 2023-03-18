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
start_time <- Sys.time()
data_base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
plot_base <- paste0(data_base, "plots/runoff/monthly/")
in_dir_ext <- c("runbase")
unbias_dir_ext <- "/02_med_diff_med_no_bias/"
runoff_AV_fileNs <- c("monthly_cum_runbase")
timeP_ty_middN <- c("month")
av_tg_col_pref <- c("monthly_cum_")
av_titles <- c("monthly ")

dt_type <- in_dir_ext[1]
timeP_ty <- 1

for (dt_type in in_dir_ext){
  in_dir <- paste0(data_base, dt_type, "/")
  for (timeP_ty in 1:1){
    files <- runoff_AV_fileNs
    AV_y_lab <- "runoff (mm)"
    AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], "runbase")
    AV_title <- paste0(av_titles[timeP_ty], "runoff")

    AVs <- readRDS(paste0(in_dir, files[timeP_ty], ".rds")) %>% 
           data.table()
    unbias_diff <- readRDS(paste0(in_dir, unbias_dir_ext, 
                                  "detail_med_diff_med_", 
                                  timeP_ty_middN[timeP_ty], "_", 
                                  dt_type, ".rds")) %>% 
                   data.table()
    AVs <- na.omit(AVs); unbias_diff <- na.omit(unbias_diff)
    AVs <- remove_observed(AVs); AVs <- remove_current_timeP(AVs)
    unbias_diff <- remove_observed(unbias_diff)
    unbias_diff <- remove_current_timeP(unbias_diff)
    
    # update clusters labels
    AVs <- convert_5_numeric_clusts_to_alphabet(AVs)
    unbias_diff <- convert_5_numeric_clusts_to_alphabet(unbias_diff)

    cluster_types <- unique(AVs$cluster)
    clust_g <- cluster_types[1]
    #
    # remove those rows whose perc diff is more than 1000%
    #
    unbias_diff <- unbias_diff %>% 
                   filter(perc_diff<600)%>%
                   data.table()
    unbias_diff <- unbias_diff %>% 
                   filter(perc_diff>-600)%>% 
                   data.table()
    AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
    AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table()
    unbias_diff_45 <- unbias_diff %>% 
                      filter(emission=="RCP 4.5") %>% 
                      data.table()
    unbias_diff_85 <- unbias_diff %>% 
                      filter(emission=="RCP 8.5") %>% 
                      data.table()
    rm(AVs, unbias_diff)

    for (clust_g in cluster_types){
      curr_AVs_45 <- AVs_45 %>% 
                     filter(cluster == clust_g) %>% 
                     data.table()
      curr_AVs_85 <- AVs_85 %>% 
                     filter(cluster == clust_g) %>% 
                     data.table()      
      curr_unbias_diff_45 <- unbias_diff_45 %>% 
                             filter(cluster == clust_g) %>% 
                             data.table()
      curr_unbias_diff_85 <- unbias_diff_85 %>% 
                             filter(cluster == clust_g) %>% 
                             data.table()
      #########
      ######### Actual value plots
      #########
      quans_85 <- find_quantiles(data_table=curr_AVs_85, 
                                 tgt_col= AV_tg_col, 
                                 time_type="monthly")
      quans_45 <- find_quantiles(data_table=curr_AVs_45, 
                                 tgt_col= AV_tg_col, 
                                 time_type="monthly")
      AV_box_85 <- box_trend_monthly_cum(dt=curr_AVs_85, 
                                         p_type = "box",
                                         y_lab = AV_y_lab, 
                                         tgt_col = AV_tg_col) + 
                   ggtitle(paste0("monthly runoff for historical", 
                                     " and three future time frames")) + 
                   coord_cartesian(ylim = c(quans_85[1], quans_85[2]))

      AV_box_45 <- box_trend_monthly_cum(dt = curr_AVs_45, 
                                         p_type = "box",
                                         y_lab = AV_y_lab, 
                                         tgt_col = AV_tg_col) + 
                      ggtitle(paste0("monthly runoff for historical", 
                                     " and three future time frames")) + 
              coord_cartesian(ylim = c(quans_45[1], quans_45[2]))
      #########
      ######### unbiased Percentage diffs
      #########
      box_title <- "% difference between future and historical"
      box_title <-  paste0(box_title, " monthly runoff")
      
      quans_85 <- find_quantiles(curr_unbias_diff_85, 
                                 tgt_col = "perc_diff", 
                                 time_type = "monthly")
      quans_45 <- find_quantiles(curr_unbias_diff_45, 
                                 tgt_col = "perc_diff", 
                                 time_type = "monthly")

      unbias_perc_diff_85 <- box_trend_monthly_cum(dt=curr_unbias_diff_85,
                                                   p_type="box",
                                                y_lab="differences (%)",
                                                tgt_col="perc_diff")+
                      ggtitle(box_title) +
                      coord_cartesian(ylim = c(quans_85[1], quans_85[2]))
      unbias_perc_diff_45 <- box_trend_monthly_cum(dt = curr_unbias_diff_45, 
                                            p_type="box",
                                            y_lab = "differences (%)",
                                            tgt_col = "perc_diff") + 
                      ggtitle(box_title) + 
                      coord_cartesian(ylim = c(quans_45[1], quans_45[2]))
      ###################################
      #####
      ##### arrange plots
      #####
      ###################################
      run_n_diff_45 <- ggarrange(plotlist = list(AV_box_45, 
                                        unbias_perc_diff_45),
                                  ncol = 1, nrow = 2,
                                  common.legend = TRUE, 
                                  legend="bottom")

      run_n_diff_85 <- ggarrange(plotlist = list(AV_box_85, 
                                        unbias_perc_diff_85),
                                  ncol = 1, nrow = 2, 
                                  common.legend = TRUE, 
                                  legend="bottom")
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
      ggsave(filename = paste0(gsub("\ ", "_", clust_g), 
                               "_unbiased_45.png"),
             plot = run_n_diff_45,
             width=8, height = 3.5, units = "in", 
             dpi=600, device = "png", path = run_n_diff)
      
      ggsave(filename = paste0(gsub("\ ", "_", clust_g),
                               "_unbiased_85.png"),
             plot = run_n_diff_85, 
             width=8, height=3.5, units = "in", 
             dpi=600, device = "png", path = run_n_diff)
      ###################################
      #####
      ##### actual values
      #####
      ###################################
      just_AVs <- paste0(plot_base, "just_AVs/")
      if (dir.exists(just_AVs) == F) {
        dir.create(path = just_AVs, recursive = T)}

      ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_45.png"),
             plot = AV_box_45,
             width = 5.5, height=1.5, units = "in", 
             dpi=600, device = "png", path = just_AVs)

      ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_85.png"),
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

      ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_45.png"),
             plot=unbias_perc_diff_45,
             width=8, height=1.5, units = "in", 
             dpi=600, device = "png", path = just_diff)

      ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_85.png"),
             plot = unbias_perc_diff_85,
             width=8, height=1.5, units = "in",
             dpi=600, device = "png", path = just_diff)
    }
  }
}

print (Sys.time() - start_time)


