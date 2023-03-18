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
start_time <- Sys.time()
data_base <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/"

in_dir_ext <- c("precip", "rain", "runbase", "snow")

bias_dir_ext <- "/02_med_diff_med_obs/"
unbias_dir_ext <- "/02_med_diff_med_no_bias/"

precip_AV_fileNs <- c("month_all_last_days")
rain_AV_fileNs <- c("month_cum_rain")
snow_AV_fileNs <- c("month_cum_snow")
runoff_AV_fileNs <- c("monthly_cum_runbase")

cluster_types <- c("least precip", "lesser precip", "less precip", "most precip")
timeP_ty_middN <- c("month")

av_tg_col_pref <- c("monthly_cum_")
av_titles <- c("monthly cum. ")
month_names <- c("01_jan", "02_feb", "03_mar", "04_apr", "05_may", "06_jun", 
                 "07_july", "08_aug", "09_sept", "10_oct", "11_nov", "12_dec")

dt_type <- in_dir_ext[1]
timeP_ty <- 1
clust_g <- cluster_types[1]

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

     }else if (dt_type=="runbase"){
      files <- runoff_AV_fileNs
      AV_y_lab <- "cum. runoff (mm)"
      AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], "runbase")
      AV_title <- paste0(av_titles[timeP_ty], "runoff.")
    }

    AVs <- readRDS(paste0(in_dir, files[timeP_ty], ".rds")) %>% data.table()
    bias_diff <- readRDS(paste0(in_dir, bias_dir_ext, "detail_med_diff_med_", 
                                timeP_ty_middN[timeP_ty], "_", 
                                dt_type, ".rds")) %>% data.table()

    unbias_diff <- readRDS(paste0(in_dir, unbias_dir_ext, "detail_med_diff_med_", 
                                  timeP_ty_middN[timeP_ty], "_", 
                                  dt_type, ".rds")) %>% data.table()

    AVs <- AVs %>% filter(time_period != "2006-2025") %>% data.table()  
    bias_diff <- bias_diff %>% filter(time_period != "2006-2025") %>% data.table()
    unbias_diff <- unbias_diff %>% filter(time_period != "2006-2025") %>% data.table()
    #
    # remove those rows whose perc diff is more than 1000%
    #
    bias_diff <- bias_diff %>% filter(perc_diff < 600) %>% data.table()
    unbias_diff <- unbias_diff %>% filter(perc_diff < 600) %>% data.table()

    bias_diff <- bias_diff %>% filter(perc_diff > -600) %>% data.table()
    unbias_diff <- unbias_diff %>% filter(perc_diff > -600) %>% data.table()
  
    plot_dir <- paste0(in_dir, "new_2_", dt_type, "/monthly", "_", dt_type, "/peak/")
    if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
    print (plot_dir)

    AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
    AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table()
    bias_diff_45 <- bias_diff %>% filter(emission=="RCP 4.5") %>% data.table()
    bias_diff_85 <- bias_diff %>% filter(emission=="RCP 8.5") %>% data.table()
    unbias_diff_45 <- unbias_diff %>% filter(emission=="RCP 4.5") %>% data.table()
    unbias_diff_85 <- unbias_diff %>% filter(emission=="RCP 8.5") %>% data.table()
    rm(AVs, unbias_diff, bias_diff)

    for (clust_g in cluster_types){
      curr_AVs_45 <- AVs_45 %>% filter(cluster == clust_g) %>% data.table()
      curr_AVs_85 <- AVs_85 %>% filter(cluster == clust_g) %>% data.table()
      
      curr_unbias_diff_45 <- unbias_diff_45 %>% filter(cluster == clust_g) %>% data.table()
      curr_unbias_diff_85 <- unbias_diff_85 %>% filter(cluster == clust_g) %>% data.table()
      
      curr_bias_diff_45 <- bias_diff_45 %>% filter(cluster == clust_g) %>% data.table()
      curr_bias_diff_85 <- bias_diff_85 %>% filter(cluster == clust_g) %>% data.table()

      #########
      ######### Actual value plots
      #########
      assign(x = "AV_box_85",
             value = {box_trend_monthly_cum(dt = curr_AVs_85, p_type="box",
                                            y_lab = AV_y_lab, tgt_col = AV_tg_col) + 
                      ggtitle(paste0("monthy cum. ", dt_type, "."))
                      })

      assign(x = "AV_box_45",
             value = {box_trend_monthly_cum(dt = curr_AVs_45, p_type="box",
                                            y_lab = AV_y_lab, tgt_col = AV_tg_col) + 
                      ggtitle(paste0("monthy cum. ", dt_type, "."))
                      })

      #########
      ######### unbiased Magnitude diffs
      #########
      mag_y_lab <- "magnitude of diffs."
      box_title <- "unbiased differences"
      box_subtitle <- "for each model median is\ntaken over years, separately"

      assign(x = "unbias_mag_diff_45",
             value = {box_trend_monthly_cum(dt=curr_unbias_diff_45, p_type="box",
                                            y_lab=mag_y_lab, tgt_col = "diff") + 
                      ggtitle(box_title)
                      })

      assign(x = "unbias_mag_diff_85",
             value = {box_trend_monthly_cum(dt=curr_unbias_diff_85, p_type="box",
                                            y_lab=mag_y_lab, tgt_col = "diff") + 
                      ggtitle(box_title)
                      })
      #########
      ######### biased Magnitude diffs
      #########
      mag_y_lab <- "magnitude of diffs."
      box_title <- "biased diffs."
      box_subtitle <- "for each model median is\ntaken over years, separately"

      assign(x = "bias_mag_diff_45",
             value = {box_trend_monthly_cum(dt=curr_bias_diff_45, p_type="box",
                                            y_lab=mag_y_lab, tgt_col = "diff") + 
                      ggtitle(box_title)
                      })

      assign(x = "bias_mag_diff_85",
             value = {box_trend_monthly_cum(dt=curr_bias_diff_85, p_type="box",
                                            y_lab = "magnitude of diffs.",
                                            tgt_col = "diff")+ ggtitle(box_title)
                      })

      #########
      ######### unbiased Percentage diffs
      #########
      box_title <- "unbiased differences"
      box_subtitle <- "for each model median is\ntaken over years, separately"

      assign(x = "unbias_perc_diff_45",
             value = {box_trend_monthly_cum(dt = curr_unbias_diff_45, p_type="box",
                                            y_lab = "differences (%)",
                                            tgt_col = "perc_diff") + 
                      ggtitle(box_title)
                      })

      assign(x = "unbias_perc_diff_85",
             value = {box_trend_monthly_cum(dt = curr_unbias_diff_85, p_type="box",
                                            y_lab = "differences (%)", 
                                            tgt_col = "perc_diff") + 
                      ggtitle(box_title)
                      })

      #########
      ######### biased Percentage diffs
      #########
      perc_y_lab <- "differences (%)"; box_title <- "biased differences"
      box_subtitle <- "for each model median is\ntaken over years, separately"

      assign(x = "bias_perc_diff_45",
             value = {box_trend_monthly_cum(dt = curr_bias_diff_45, p_type="box",
                                            y_lab = "differences (%)",
                                            tgt_col = "perc_diff") + 
                      ggtitle(box_title)
                      })

      assign(x = "bias_perc_diff_85",
             value = {box_trend_monthly_cum(dt = curr_bias_diff_85, p_type="box",
                                            y_lab = "differences (%)",
                                            tgt_col = "perc_diff") + 
                      ggtitle(box_title)
                      })

      unbiased_RCP45 <- ggarrange(plotlist = list(AV_box_45, 
                                                  unbias_mag_diff_45,
                                                  unbias_perc_diff_45),
                         ncol = 1, nrow = 3,
                         common.legend = TRUE, legend="bottom")

      biased_RCP45 <- ggarrange(plotlist = list(AV_box_45, bias_mag_diff_45, 
                                                bias_perc_diff_45),
                                ncol = 1, nrow = 3, common.legend = TRUE, legend="bottom")

      unbiased_RCP85 <- ggarrange(plotlist = list(AV_box_85, unbias_mag_diff_85, 
                                                  unbias_perc_diff_85),
                                  ncol = 1, nrow = 3, common.legend = TRUE, legend="bottom")
      
      biased_RCP85 <- ggarrange(plotlist = list(AV_box_85, bias_mag_diff_85, 
                                                bias_perc_diff_85),
                                ncol = 1, nrow = 3, common.legend = TRUE, legend="bottom")

      ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_", 
                               timeP_ty_middN[timeP_ty], "_unbiased_RCP45.png"),
             plot = unbiased_RCP45, 
             width = 9, height = 6, units = "in", 
             dpi=400, device = "png",
             path = plot_dir)

      ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_", 
                               timeP_ty_middN[timeP_ty], "_biased_RCP45.png"),
             plot = biased_RCP45, 
             width = 9, height = 6, units = "in", 
             dpi=400, device = "png",
             path = plot_dir)
      
      ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_", 
                               timeP_ty_middN[timeP_ty], "_unbiased_RCP85.png"),
             plot = unbiased_RCP85, 
             width = 9, height = 6, units = "in", 
             dpi=400, device = "png",
             path = plot_dir)

      ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_", 
                               timeP_ty_middN[timeP_ty], "_biased_RCP85.png"),
             plot = biased_RCP85, 
             width = 9, height = 6, units = "in", 
             dpi=400, device = "png",
             path = plot_dir)
    }
  }
}

print (Sys.time() - start_time)