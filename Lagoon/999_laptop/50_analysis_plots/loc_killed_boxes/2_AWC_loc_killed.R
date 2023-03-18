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

bias_dir_ext <- "/02_med_diff_med_obs/loc_killed/"
unbias_dir_ext <- "/02_med_diff_med_no_bias/loc_killed/"

precip_AV_fileNs <- c("ann_all_last_days",
                      "Sept_March_all_last_days",
                      "wtr_yr_sept_all_last_days")

rain_AV_fileNs <- c("ann_cum_rain", 
                    "Sept_March_cum_rain",
                    "wtr_yr_cum_rain")

snow_AV_fileNs <- c("ann_cum_snow", 
                    "Sept_March_cum_snow",
                    "wtr_yr_cum_snow")

runoff_AV_fileNs <- c("ann_cum_runbase", 
                      "chunk_cum_runbase",
                      "wtr_yr_cum_runbase")

cluster_types <- c("least precip", "lesser precip", "less precip", "most precip")
timeP_ty_middN <- c("ann", "chunk", "wtr_yr")

av_tg_col_pref <- c("annual_cum_", "chunk_cum_", "annual_cum_")

av_titles <- c("ann. cum. ", "Sept. - Mar. cum. ", "wtr. yr. cum. ")
emissions <- c("RCP 4.5", "RCP 8.5")

dt_type <-  in_dir_ext[1]
in_dir <- paste0(data_base, dt_type, "/")
timeP_ty <- 1
clust_g <- cluster_types[1]

for (dt_type in in_dir_ext){ # precip or rain or runoff?
  in_dir <- paste0(data_base, dt_type, "/")

  for (timeP_ty in 1:3){ # annual or chunk or wtr_yr?

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
    bias_diff <- readRDS(paste0(in_dir, 
                                bias_dir_ext, 
                                "loc_killed_med_diff_med_", 
                                timeP_ty_middN[timeP_ty], "_", 
                                dt_type, ".rds")) %>% data.table()
    setnames(bias_diff, 
             old = c("med_of_diffs_of_meds", "perc_med_of_diffs_of_meds"),
             new = c("diff", "perc_diff"))
    
    unbias_diff <- readRDS(paste0(in_dir,
                                  unbias_dir_ext, 
                                  "loc_killed_med_diff_med_", 
                                  timeP_ty_middN[timeP_ty], "_", 
                                  dt_type, ".rds")) %>% data.table()
    setnames(unbias_diff, 
             old = c("med_of_diffs_of_meds", "perc_med_of_diffs_of_meds"),
             new = c("diff", "perc_diff"))
    ##################################################################
    #
    # KILL effect of loc?
    #
    ##################################################################
    AVs <- AVs %>% filter(time_period != "2006-2025") %>% data.table()
    AVs <- AVs %>%
           group_by(model, time_period, emission, cluster) %>%
           transmute(AV = median(get(AV_tg_col)))%>%
           unique() %>%
           data.table()

    ##################################################################
    AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
    AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table()
    bias_diff_45 <- bias_diff %>% filter(emission=="RCP 4.5") %>% data.table()
    bias_diff_85 <- bias_diff %>% filter(emission=="RCP 8.5") %>% data.table()
    unbias_diff_45 <- unbias_diff %>% filter(emission=="RCP 4.5") %>% data.table()
    unbias_diff_85 <- unbias_diff %>% filter(emission=="RCP 8.5") %>% data.table()
    rm(AVs, unbias_diff, bias_diff)

    for (clust_g in cluster_types){
      curr_AVs_85 <- AVs_85 %>% filter(cluster == clust_g) %>% data.table()
      curr_AVs_45 <- AVs_45 %>% filter(cluster == clust_g) %>% data.table()
      
      curr_unbias_diff_45 <- unbias_diff_45 %>% filter(cluster == clust_g) %>% data.table()
      curr_unbias_diff_85 <- unbias_diff_85 %>% filter(cluster == clust_g) %>% data.table()
      
      curr_bias_diff_45 <- bias_diff_45 %>% filter(cluster == clust_g) %>% data.table()
      curr_bias_diff_85 <- bias_diff_85 %>% filter(cluster == clust_g) %>% data.table()
      
      # AV_title <- paste0(AV_title, " (", clust_g, ". cites)")
      #########
      ######### Actual value plots
      #########
      AV_box_85 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = curr_AVs_85, 
                                                     y_lab = AV_y_lab, 
                                                     tgt_col = "AV")
      AV_box_85 <- AV_box_85 + ggtitle(AV_title)

      AV_box_45 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = curr_AVs_45, 
                                                     y_lab = AV_y_lab, 
                                                     tgt_col = "AV")
      AV_box_45 <- AV_box_45 + ggtitle(AV_title)
      
      #########
      ######### unbiased Magnitude diffs
      #########
      mag_y_lab <- "magnitude of differences"
      box_title <- "unbiased differences"
      box_subtitle <- "for each model median is\ntaken over years, separately"
      
      unbias_mag_diff_45 <- ann_wtrYr_chunk_cum_box_cluster_x(dt=curr_unbias_diff_45,
                                                              y_lab=mag_y_lab,
                                                              tgt_col = "diff",
                                                              ttl=box_title, 
                                                              subttl=box_subtitle)
      unbias_mag_diff_45 <- unbias_mag_diff_45 + ggtitle(box_title)

      
      unbias_mag_diff_85 <- ann_wtrYr_chunk_cum_box_cluster_x(dt=curr_unbias_diff_85,
                                                              y_lab=mag_y_lab,
                                                              tgt_col = "diff",
                                                              ttl=box_title, 
                                                              subttl=box_subtitle)
      unbias_mag_diff_85 <- unbias_mag_diff_85 + ggtitle(box_title)

      #########
      ######### biased Magnitude diffs
      #########
      mag_y_lab <- "magnitude of differences"
      box_title <- "biased differences"
      box_subtitle <- "for each model median is\ntaken over years, separately"
      bias_mag_diff_45 <- ann_wtrYr_chunk_cum_box_cluster_x(dt=curr_bias_diff_45,
                                                            y_lab=mag_y_lab,
                                                            tgt_col = "diff",
                                                            ttl=box_title, 
                                                            subttl=box_subtitle)
      bias_mag_diff_45 <- bias_mag_diff_45 + ggtitle(box_title)

      bias_mag_diff_85 <- ann_wtrYr_chunk_cum_box_cluster_x(dt=curr_bias_diff_85,
                                                            y_lab = "magnitude of differences",
                                                            tgt_col = "diff",
                                                            ttl=box_title, 
                                                            subttl=box_subtitle)
      bias_mag_diff_85 <- bias_mag_diff_85 + ggtitle(box_title)

      #########
      ######### unbiased Percentage diffs
      #########
      box_title <- "unbiased differences"
      box_subtitle <- "for each model median is\ntaken over years, separately"

      unbias_perc_diff_45 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = curr_unbias_diff_45,
                                                               y_lab = "differences (%)",
                                                               tgt_col = "perc_diff",
                                                               ttl = box_title, 
                                                               subttl = box_subtitle)
      unbias_perc_diff_45 <- unbias_perc_diff_45 + ggtitle(box_title)

      unbias_perc_diff_85 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = curr_unbias_diff_85,
                                                               y_lab = "differences (%)",
                                                               tgt_col = "perc_diff",
                                                               ttl = box_title, 
                                                               subttl = box_subtitle)
      unbias_perc_diff_85 <- unbias_perc_diff_85 + ggtitle(box_title)

      #########
      ######### biased Percentage diffs
      #########
      perc_y_lab <- "differences (%)"
      box_title <- "biased differences"
      box_subtitle <- "for each model median is\ntaken over years, separately"

      bias_perc_diff_45 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = curr_bias_diff_45,
                                                             y_lab = "differences (%)",
                                                             tgt_col = "perc_diff",
                                                             ttl = box_title, 
                                                             subttl = box_subtitle)
      bias_perc_diff_45 <- bias_perc_diff_45 + ggtitle(box_title)


      bias_perc_diff_85 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = curr_bias_diff_85,
                                                             y_lab = "differences (%)",
                                                             tgt_col = "perc_diff",
                                                             ttl = box_title, 
                                                             subttl = box_subtitle)
      bias_perc_diff_85 <- bias_perc_diff_85 + ggtitle(box_title)

      unbiased_RCP45 <- ggarrange(plotlist = list(AV_box_45, unbias_mag_diff_45,
                                                  unbias_perc_diff_45),
                                  ncol = 3, nrow = 1, widths = c(1.25, 1, 1),
                                  common.legend = TRUE, 
                                  legend="bottom")
      biased_RCP45 <- ggarrange(plotlist = list(AV_box_45, bias_mag_diff_45,
                                                           bias_perc_diff_45),
                                  ncol = 3, nrow = 1, widths = c(1.25, 1, 1),
                                  common.legend = TRUE, 
                                  legend="bottom")

      unbiased_RCP85 <- ggarrange(plotlist = list(AV_box_85, unbias_mag_diff_85, 
                                                             unbias_perc_diff_85),
                                  ncol = 3, nrow = 1, widths = c(1.25, 1, 1),
                                  common.legend = TRUE, 
                                  legend="bottom")

      biased_RCP85 <- ggarrange(plotlist = list(AV_box_85, bias_mag_diff_85,
                                                bias_perc_diff_85),
                                ncol = 3, nrow = 1, widths = c(1.25, 1, 1),
                                common.legend = TRUE, 
                                legend="bottom")
      
      plot_dir <- paste0(in_dir, "new_2_", dt_type, "/loc_killed/")
      if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
      
      ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_", 
                               timeP_ty_middN[timeP_ty], "_unbiased_RCP45.png"),
             plot = unbiased_RCP45, 
             width = 6, height = 2.5, units = "in", 
             dpi=400, device = "png",
             path = plot_dir)
      
      ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_", 
                               timeP_ty_middN[timeP_ty], "_biased_RCP45.png"),
             plot = biased_RCP45, 
             width = 6, height = 2.5, units = "in", 
             dpi=400, device = "png",
             path = plot_dir)
      
      ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_", 
                               timeP_ty_middN[timeP_ty], "_unbiased_RCP85.png"),
             plot = unbiased_RCP85, 
             width = 6, height = 2.5, units = "in", 
             dpi = 400, device = "png",
             path = plot_dir)
      
      ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_", 
                               timeP_ty_middN[timeP_ty], "_biased_RCP85.png"),
             plot = biased_RCP85, 
             width = 6, height = 2.5, units = "in", 
             dpi = 400, device = "png",
             path = plot_dir)
      
      print (plot_dir)
    }
  }
}


