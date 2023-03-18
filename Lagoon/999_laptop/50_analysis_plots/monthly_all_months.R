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

in_dir_ext <- c("snow") # "precip", "rain", "runbase", 

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
     files <- precip_AV_fileNs; AV_y_lab <- "cum. precip. (mm)"
     AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], dt_type)
     AV_title <- paste0(av_titles[timeP_ty], "precip.")

     } else if (dt_type=="rain"){
      files <- rain_AV_fileNs; AV_y_lab <- "cum. rain (mm)"
      AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], dt_type)
      AV_title <- paste0(av_titles[timeP_ty], "rain.")

     } else if (dt_type=="snow"){
      files <- snow_AV_fileNs; AV_y_lab <- "cum. snow (mm)"
      AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], dt_type)
      AV_title <- paste0(av_titles[timeP_ty], "snow.")

     }else if (dt_type=="runbase"){
      files <- runoff_AV_fileNs; AV_y_lab <- "cum. runoff (mm)"
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
    unbias_diff[is.na(unbias_diff)] <- -666
    bias_diff[is.na(bias_diff)] <- -666

    bias_diff <- bias_diff %>% filter(perc_diff < 600 | perc_diff==-666) %>% data.table()
    unbias_diff <- unbias_diff %>% filter(perc_diff < 600 | perc_diff==-666) %>% data.table()

    bias_diff <- bias_diff %>% filter(perc_diff > -600 | perc_diff==-666) %>% data.table()
    unbias_diff <- unbias_diff %>% filter(perc_diff > -600 | perc_diff==-666) %>% data.table()

    bias_diff$perc_diff[bias_diff$perc_diff == -666] <- NaN
    unbias_diff$perc_diff[unbias_diff$perc_diff == -666] <- NaN

    AVs_orig <- AVs; bias_diff_orig <- bias_diff; unbias_diff_orig <- unbias_diff
    
    for (month_no in 1:12){
      plot_dir <- paste0(in_dir, "new_2_", dt_type, "/monthly", "_", dt_type, "/new_arr/")
      if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
      print (plot_dir)
      print (month_no)

      AVs <- AVs_orig %>% filter(month == month_no)%>% data.table()
      bias_diff <- bias_diff_orig %>% filter(month == month_no)%>% data.table()
      unbias_diff <- unbias_diff_orig %>% filter(month %in% month_no)%>% data.table()
      
      AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
      AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table()
      bias_diff_45 <- bias_diff %>% filter(emission=="RCP 4.5") %>% data.table()
      bias_diff_85 <- bias_diff %>% filter(emission=="RCP 8.5") %>% data.table()
      unbias_diff_45 <- unbias_diff %>% filter(emission=="RCP 4.5") %>% data.table()
      unbias_diff_85 <- unbias_diff %>% filter(emission=="RCP 8.5") %>% data.table()
      rm(AVs, unbias_diff, bias_diff)

      for (clust_g in cluster_types){
        print (clust_g)
        print (Sys.time() - start_time)
        curr_AVs_85 <- AVs_85 %>% filter(cluster == clust_g) %>% data.table()
        curr_AVs_45 <- AVs_45 %>% filter(cluster == clust_g) %>% data.table()
        
        curr_unbias_diff_45 <- unbias_diff_45 %>% filter(cluster == clust_g) %>% data.table()
        curr_unbias_diff_85 <- unbias_diff_85 %>% filter(cluster == clust_g) %>% data.table()
        
        curr_bias_diff_45 <- bias_diff_45 %>% filter(cluster == clust_g) %>% data.table()
        curr_bias_diff_85 <- bias_diff_85 %>% filter(cluster == clust_g) %>% data.table()

        #########
        ######### Actual value plots
        #########
        assign(x = paste(gsub(" ", "_", clust_g), month_no, "AV_box_85", sep = "_"),
               value = {Nov_Dec_cum_box(dt = curr_AVs_85, y_lab = AV_y_lab, 
                                        tgt_col = AV_tg_col) + ggtitle(paste0("monthy cum. ", dt_type, "."))
                                     })

        # AV_box_85 <- Nov_Dec_cum_box(dt = curr_AVs_85, y_lab = AV_y_lab, 
        #                              tgt_col = AV_tg_col)+ ggtitle(paste0("monthy cum. ", dt_type, "."))
        # AV_box_45 <- Nov_Dec_cum_box(dt = curr_AVs_45, y_lab = AV_y_lab, 
        #                              tgt_col = AV_tg_col) + ggtitle(paste0("monthy cum. ", dt_type, "."))

        assign(x = paste(gsub(" ", "_", clust_g), month_no, "AV_box_45", sep = "_"),
               value = {Nov_Dec_cum_box(dt = curr_AVs_45, y_lab = AV_y_lab, 
                                        tgt_col = AV_tg_col) + ggtitle(paste0("monthy cum. ", dt_type, "."))
                                     })

        #########
        ######### unbiased Magnitude diffs
        #########
        mag_y_lab <- "magnitude of diffs."
        box_title <- "unbiased differences"
        box_subtitle <- "for each model median is\ntaken over years, separately"

        assign(x = paste(gsub(" ", "_", clust_g), month_no, "unbias_mag_diff_45", sep = "_"),
               value = {Nov_Dec_Diffs(dt=curr_unbias_diff_45, y_lab=mag_y_lab,
                                            tgt_col = "diff", ttl=box_title, 
                                            subttl=box_subtitle) + ggtitle(box_title)
                                     })

        assign(x = paste(gsub(" ", "_", clust_g), month_no, "unbias_mag_diff_85", sep = "_"),
               value = {Nov_Dec_Diffs(dt=curr_unbias_diff_85, y_lab=mag_y_lab,
                                      tgt_col = "diff", ttl=box_title, 
                                      subttl=box_subtitle)+ ggtitle(box_title)
                                     })

        # unbias_mag_diff_45 <- Nov_Dec_Diffs(dt=curr_unbias_diff_45, y_lab=mag_y_lab,
        #                                     tgt_col = "diff", ttl=box_title, 
        #                                     subttl=box_subtitle) + ggtitle(box_title)

        # unbias_mag_diff_85 <- Nov_Dec_Diffs(dt=curr_unbias_diff_85, y_lab=mag_y_lab,
        #                                     tgt_col = "diff", ttl=box_title, 
        #                                     subttl=box_subtitle)+ ggtitle(box_title)

        #########
        ######### biased Magnitude diffs
        #########
        mag_y_lab <- "magnitude of diffs."
        box_title <- "biased diffs."
        box_subtitle <- "for each model median is\ntaken over years, separately"

        assign(x = paste(gsub(" ", "_", clust_g), month_no, "bias_mag_diff_45", sep = "_"),
               value = {Nov_Dec_Diffs(dt=curr_bias_diff_45, y_lab=mag_y_lab,
                                      tgt_col = "diff", ttl=box_title, 
                                      subttl=box_subtitle)+ ggtitle(box_title)
                                     })

        assign(x = paste(gsub(" ", "_", clust_g), month_no, "bias_mag_diff_85", sep = "_"),
               value = {Nov_Dec_Diffs(dt=curr_bias_diff_85,
                                      y_lab = "magnitude of diffs.",
                                      tgt_col = "diff", ttl=box_title, 
                                      subttl=box_subtitle)+ ggtitle(box_title)
                                     })

        # bias_mag_diff_45 <- Nov_Dec_Diffs(dt=curr_bias_diff_45, y_lab=mag_y_lab,
        #                                   tgt_col = "diff", ttl=box_title, 
        #                                   subttl=box_subtitle)+ ggtitle(box_title)

        # bias_mag_diff_85 <- Nov_Dec_Diffs(dt=curr_bias_diff_85,
        #                                   y_lab = "magnitude of diffs.",
        #                                   tgt_col = "diff", ttl=box_title, 
        #                                   subttl=box_subtitle)+ ggtitle(box_title)

        #########
        ######### unbiased Percentage diffs
        #########
        box_title <- "unbiased differences"
        box_subtitle <- "for each model median is\ntaken over years, separately"

        assign(x = paste(gsub(" ", "_", clust_g), month_no, "unbias_perc_diff_45", sep = "_"),
               value = {Nov_Dec_Diffs(dt = curr_unbias_diff_45,
                                      y_lab = "differences (%)",
                                      tgt_col = "perc_diff", ttl = box_title, 
                                      subttl = box_subtitle) + ggtitle(box_title)
                                     })

        # unbias_perc_diff_45 <- Nov_Dec_Diffs(dt = curr_unbias_diff_45,
        #                                      y_lab = "differences (%)",
        #                                      tgt_col = "perc_diff", ttl = box_title, 
        #                                      subttl = box_subtitle) + ggtitle(box_title)

        assign(x = paste(gsub(" ", "_", clust_g), month_no, "unbias_perc_diff_85", sep = "_"),
               value = {Nov_Dec_Diffs(dt = curr_unbias_diff_85,
                                      y_lab = "differences (%)",
                                      tgt_col = "perc_diff", ttl = box_title, 
                                      subttl = box_subtitle)+ ggtitle(box_title)
                                     })

        # unbias_perc_diff_85 <- Nov_Dec_Diffs(dt = curr_unbias_diff_85,
        #                                      y_lab = "differences (%)",
        #                                      tgt_col = "perc_diff", ttl = box_title, 
        #                                      subttl = box_subtitle)+ ggtitle(box_title)

        #########
        ######### biased Percentage diffs
        #########
        perc_y_lab <- "differences (%)"; box_title <- "biased differences"
        box_subtitle <- "for each model median is\ntaken over years, separately"

        assign(x = paste(gsub(" ", "_", clust_g), month_no, "bias_perc_diff_45", sep = "_"),
               value = {Nov_Dec_Diffs(dt = curr_bias_diff_45,
                                      y_lab = "differences (%)",
                                      tgt_col = "perc_diff", ttl = box_title, 
                                      subttl = box_subtitle)+ ggtitle(box_title)
                                     })

        # bias_perc_diff_45 <- Nov_Dec_Diffs(dt = curr_bias_diff_45,
        #                                    y_lab = "differences (%)",
        #                                    tgt_col = "perc_diff", ttl = box_title, 
        #                                    subttl = box_subtitle)+ ggtitle(box_title)
        
        assign(x = paste(gsub(" ", "_", clust_g), month_no, "bias_perc_diff_85", sep = "_"),
               value = {Nov_Dec_Diffs(dt = curr_bias_diff_85,
                                      y_lab = "differences (%)",
                                      tgt_col = "perc_diff", ttl = box_title, 
                                      subttl = box_subtitle)+ ggtitle(box_title)
                                     })
        
        # bias_perc_diff_85 <- Nov_Dec_Diffs(dt = curr_bias_diff_85,
        #                                    y_lab = "differences (%)",
        #                                    tgt_col = "perc_diff", ttl = box_title, 
        #                                    subttl = box_subtitle)+ ggtitle(box_title)

      }
    }
  }
  
  least_unbiased_RCP45 <- ggarrange(plotlist = list(
                                                    least_precip_9_AV_box_45,
                                                    least_precip_10_AV_box_45,
                                                    least_precip_11_AV_box_45,
                                                    least_precip_12_AV_box_45,
                                                    least_precip_1_AV_box_45, 
                                                    least_precip_2_AV_box_45,
                                                    least_precip_3_AV_box_45,
                                                    least_precip_4_AV_box_45,
                                                    least_precip_5_AV_box_45,
                                                    least_precip_6_AV_box_45,
                                                    least_precip_7_AV_box_45,
                                                    least_precip_8_AV_box_45,
                                                    #
                                                    # mag of diffs
                                                    #
                                                     
                                                    least_precip_9_unbias_mag_diff_45, 
                                                    least_precip_10_unbias_mag_diff_45, 
                                                    least_precip_11_unbias_mag_diff_45, 
                                                    least_precip_12_unbias_mag_diff_45,
                                                    least_precip_1_unbias_mag_diff_45, 
                                                    least_precip_2_unbias_mag_diff_45, 
                                                    least_precip_3_unbias_mag_diff_45, 
                                                    least_precip_4_unbias_mag_diff_45, 
                                                    least_precip_5_unbias_mag_diff_45, 
                                                    least_precip_6_unbias_mag_diff_45, 
                                                    least_precip_7_unbias_mag_diff_45, 
                                                    least_precip_8_unbias_mag_diff_45,
                                                    #
                                                    # perc of diffs
                                                    #
                                                     
                                                    least_precip_9_unbias_perc_diff_45, 
                                                    least_precip_10_unbias_perc_diff_45, 
                                                    least_precip_11_unbias_perc_diff_45, 
                                                    least_precip_12_unbias_perc_diff_45,
                                                    least_precip_1_unbias_perc_diff_45,
                                                    least_precip_2_unbias_perc_diff_45, 
                                                    least_precip_3_unbias_perc_diff_45, 
                                                    least_precip_4_unbias_perc_diff_45, 
                                                    least_precip_5_unbias_perc_diff_45, 
                                                    least_precip_6_unbias_perc_diff_45, 
                                                    least_precip_7_unbias_perc_diff_45, 
                                                    least_precip_8_unbias_perc_diff_45
                                                    ),
                                    ncol = 12, nrow = 3,
                                    common.legend = TRUE, legend="bottom")

  least_biased_RCP45 <- ggarrange(plotlist = list(
                                                  least_precip_9_AV_box_45,
                                                  least_precip_10_AV_box_45,
                                                  least_precip_11_AV_box_45,
                                                  least_precip_12_AV_box_45,
                                                  least_precip_1_AV_box_45, 
                                                  least_precip_2_AV_box_45,
                                                  least_precip_3_AV_box_45,
                                                  least_precip_4_AV_box_45,
                                                  least_precip_5_AV_box_45,
                                                  least_precip_6_AV_box_45,
                                                  least_precip_7_AV_box_45,
                                                  least_precip_8_AV_box_45,
                                                  #
                                                  # mag of diffs
                                                  #
                                                   
                                                  least_precip_9_bias_mag_diff_45, 
                                                  least_precip_10_bias_mag_diff_45, 
                                                  least_precip_11_bias_mag_diff_45, 
                                                  least_precip_12_bias_mag_diff_45,
                                                  least_precip_1_bias_mag_diff_45, 
                                                  least_precip_2_bias_mag_diff_45, 
                                                  least_precip_3_bias_mag_diff_45, 
                                                  least_precip_4_bias_mag_diff_45, 
                                                  least_precip_5_bias_mag_diff_45, 
                                                  least_precip_6_bias_mag_diff_45, 
                                                  least_precip_7_bias_mag_diff_45, 
                                                  least_precip_8_bias_mag_diff_45,
                                                  #
                                                  # perc of diffs
                                                  #
                                                  
                                                  least_precip_9_bias_perc_diff_45, 
                                                  least_precip_10_bias_perc_diff_45, 
                                                  least_precip_11_bias_perc_diff_45, 
                                                  least_precip_12_bias_perc_diff_45,
                                                  least_precip_1_bias_perc_diff_45,
                                                  least_precip_2_bias_perc_diff_45, 
                                                  least_precip_3_bias_perc_diff_45, 
                                                  least_precip_4_bias_perc_diff_45, 
                                                  least_precip_5_bias_perc_diff_45, 
                                                  least_precip_6_bias_perc_diff_45, 
                                                  least_precip_7_bias_perc_diff_45, 
                                                  least_precip_8_bias_perc_diff_45
                                                    ),
                                  ncol = 12, nrow = 3,
                                  common.legend = TRUE, legend="bottom")

  least_unbiased_RCP85 <- ggarrange(plotlist = list(
                                                    least_precip_9_AV_box_85,
                                                    least_precip_10_AV_box_85,
                                                    least_precip_11_AV_box_85,
                                                    least_precip_12_AV_box_85,
                                                    least_precip_1_AV_box_85, 
                                                    least_precip_2_AV_box_85,
                                                    least_precip_3_AV_box_85,
                                                    least_precip_4_AV_box_85,
                                                    least_precip_5_AV_box_85,
                                                    least_precip_6_AV_box_85,
                                                    least_precip_7_AV_box_85,
                                                    least_precip_8_AV_box_85,
                                                    #
                                                    # mag of diffs
                                                    #
                                                    
                                                    least_precip_9_unbias_mag_diff_85, 
                                                    least_precip_10_unbias_mag_diff_85, 
                                                    least_precip_11_unbias_mag_diff_85, 
                                                    least_precip_12_unbias_mag_diff_85,
                                                    least_precip_1_unbias_mag_diff_85, 
                                                    least_precip_2_unbias_mag_diff_85, 
                                                    least_precip_3_unbias_mag_diff_85, 
                                                    least_precip_4_unbias_mag_diff_85, 
                                                    least_precip_5_unbias_mag_diff_85, 
                                                    least_precip_6_unbias_mag_diff_85, 
                                                    least_precip_7_unbias_mag_diff_85, 
                                                    least_precip_8_unbias_mag_diff_85, 
                                                    #
                                                    # perc of diffs
                                                    #
                                                     
                                                    least_precip_9_unbias_perc_diff_85, 
                                                    least_precip_10_unbias_perc_diff_85, 
                                                    least_precip_11_unbias_perc_diff_85, 
                                                    least_precip_12_unbias_perc_diff_85,
                                                    least_precip_1_unbias_perc_diff_85,
                                                    least_precip_2_unbias_perc_diff_85, 
                                                    least_precip_3_unbias_perc_diff_85, 
                                                    least_precip_4_unbias_perc_diff_85, 
                                                    least_precip_5_unbias_perc_diff_85, 
                                                    least_precip_6_unbias_perc_diff_85, 
                                                    least_precip_7_unbias_perc_diff_85, 
                                                    least_precip_8_unbias_perc_diff_85
                                                    ),
                                    ncol = 12, nrow = 3,
                                    common.legend = TRUE, legend="bottom")

  least_biased_RCP85 <- ggarrange(plotlist = list(
                                                  least_precip_9_AV_box_85,
                                                  least_precip_10_AV_box_85,
                                                  least_precip_11_AV_box_85,
                                                  least_precip_12_AV_box_85,
                                                  least_precip_1_AV_box_85, 
                                                  least_precip_2_AV_box_85,
                                                  least_precip_3_AV_box_85,
                                                  least_precip_4_AV_box_85,
                                                  least_precip_5_AV_box_85,
                                                  least_precip_6_AV_box_85,
                                                  least_precip_7_AV_box_85,
                                                  least_precip_8_AV_box_85,
                                                  #
                                                  # mag of diffs
                                                  #
                                                  
                                                  least_precip_9_bias_mag_diff_85, 
                                                  least_precip_10_bias_mag_diff_85, 
                                                  least_precip_11_bias_mag_diff_85, 
                                                  least_precip_12_bias_mag_diff_85,
                                                  least_precip_1_bias_mag_diff_85, 
                                                  least_precip_2_bias_mag_diff_85, 
                                                  least_precip_3_bias_mag_diff_85, 
                                                  least_precip_4_bias_mag_diff_85, 
                                                  least_precip_5_bias_mag_diff_85, 
                                                  least_precip_6_bias_mag_diff_85, 
                                                  least_precip_7_bias_mag_diff_85, 
                                                  least_precip_8_bias_mag_diff_85, 
                                                  #
                                                  # perc of diffs
                                                  #
                                                  
                                                  least_precip_9_bias_perc_diff_85, 
                                                  least_precip_10_bias_perc_diff_85, 
                                                  least_precip_11_bias_perc_diff_85, 
                                                  least_precip_12_bias_perc_diff_85,
                                                  least_precip_1_bias_perc_diff_85,
                                                  least_precip_2_bias_perc_diff_85, 
                                                  least_precip_3_bias_perc_diff_85, 
                                                  least_precip_4_bias_perc_diff_85, 
                                                  least_precip_5_bias_perc_diff_85, 
                                                  least_precip_6_bias_perc_diff_85, 
                                                  least_precip_7_bias_perc_diff_85, 
                                                  least_precip_8_bias_perc_diff_85
                                                  ),
                                  ncol = 12, nrow = 3,
                                  common.legend = TRUE, legend="bottom")

  lesser_unbiased_RCP45 <- ggarrange(plotlist = list(
                                                    lesser_precip_9_AV_box_45,
                                                    lesser_precip_10_AV_box_45,
                                                    lesser_precip_11_AV_box_45,
                                                    lesser_precip_12_AV_box_45,
                                                    lesser_precip_1_AV_box_45, 
                                                    lesser_precip_2_AV_box_45,
                                                    lesser_precip_3_AV_box_45,
                                                    lesser_precip_4_AV_box_45,
                                                    lesser_precip_5_AV_box_45,
                                                    lesser_precip_6_AV_box_45,
                                                    lesser_precip_7_AV_box_45,
                                                    lesser_precip_8_AV_box_45,
                                                    #
                                                    # mag of diffs
                                                    #
                                                     
                                                    lesser_precip_9_unbias_mag_diff_45, 
                                                    lesser_precip_10_unbias_mag_diff_45, 
                                                    lesser_precip_11_unbias_mag_diff_45, 
                                                    lesser_precip_12_unbias_mag_diff_45,
                                                    lesser_precip_1_unbias_mag_diff_45, 
                                                    lesser_precip_2_unbias_mag_diff_45, 
                                                    lesser_precip_3_unbias_mag_diff_45, 
                                                    lesser_precip_4_unbias_mag_diff_45, 
                                                    lesser_precip_5_unbias_mag_diff_45, 
                                                    lesser_precip_6_unbias_mag_diff_45, 
                                                    lesser_precip_7_unbias_mag_diff_45, 
                                                    lesser_precip_8_unbias_mag_diff_45,
                                                    #
                                                    # perc of diffs
                                                    #
                                                    
                                                    lesser_precip_9_unbias_perc_diff_45, 
                                                    lesser_precip_10_unbias_perc_diff_45, 
                                                    lesser_precip_11_unbias_perc_diff_45, 
                                                    lesser_precip_12_unbias_perc_diff_45,
                                                    lesser_precip_1_unbias_perc_diff_45,
                                                    lesser_precip_2_unbias_perc_diff_45, 
                                                    lesser_precip_3_unbias_perc_diff_45, 
                                                    lesser_precip_4_unbias_perc_diff_45, 
                                                    lesser_precip_5_unbias_perc_diff_45, 
                                                    lesser_precip_6_unbias_perc_diff_45, 
                                                    lesser_precip_7_unbias_perc_diff_45, 
                                                    lesser_precip_8_unbias_perc_diff_45
                                                    ),
                                     ncol = 12, nrow = 3,
                                     common.legend = TRUE, legend="bottom")

  lesser_biased_RCP45 <- ggarrange(plotlist = list(
                                                  lesser_precip_9_AV_box_45,
                                                  lesser_precip_10_AV_box_45,
                                                  lesser_precip_11_AV_box_45,
                                                  lesser_precip_12_AV_box_45,
                                                  lesser_precip_1_AV_box_45, 
                                                  lesser_precip_2_AV_box_45,
                                                  lesser_precip_3_AV_box_45,
                                                  lesser_precip_4_AV_box_45,
                                                  lesser_precip_5_AV_box_45,
                                                  lesser_precip_6_AV_box_45,
                                                  lesser_precip_7_AV_box_45,
                                                  lesser_precip_8_AV_box_45,
                                                  #
                                                  # mag of diffs
                                                  #
                                                  
                                                  lesser_precip_9_bias_mag_diff_45, 
                                                  lesser_precip_10_bias_mag_diff_45, 
                                                  lesser_precip_11_bias_mag_diff_45, 
                                                  lesser_precip_12_bias_mag_diff_45,
                                                  lesser_precip_1_bias_mag_diff_45, 
                                                  lesser_precip_2_bias_mag_diff_45, 
                                                  lesser_precip_3_bias_mag_diff_45, 
                                                  lesser_precip_4_bias_mag_diff_45, 
                                                  lesser_precip_5_bias_mag_diff_45, 
                                                  lesser_precip_6_bias_mag_diff_45, 
                                                  lesser_precip_7_bias_mag_diff_45, 
                                                  lesser_precip_8_bias_mag_diff_45, 
                                                  #
                                                  # perc of diffs
                                                  #
                                                  
                                                  lesser_precip_9_bias_perc_diff_45, 
                                                  lesser_precip_10_bias_perc_diff_45, 
                                                  lesser_precip_11_bias_perc_diff_45, 
                                                  lesser_precip_12_bias_perc_diff_45,
                                                  lesser_precip_1_bias_perc_diff_45,
                                                  lesser_precip_2_bias_perc_diff_45, 
                                                  lesser_precip_3_bias_perc_diff_45, 
                                                  lesser_precip_4_bias_perc_diff_45, 
                                                  lesser_precip_5_bias_perc_diff_45, 
                                                  lesser_precip_6_bias_perc_diff_45, 
                                                  lesser_precip_7_bias_perc_diff_45, 
                                                  lesser_precip_8_bias_perc_diff_45
                                                    ),
                                   ncol = 12, nrow = 3,
                                   common.legend = TRUE, legend="bottom")


  lesser_unbiased_RCP85 <- ggarrange(plotlist = list(
                                                    lesser_precip_9_AV_box_85,
                                                    lesser_precip_10_AV_box_85,
                                                    lesser_precip_11_AV_box_85,
                                                    lesser_precip_12_AV_box_85,
                                                    lesser_precip_1_AV_box_85, 
                                                    lesser_precip_2_AV_box_85,
                                                    lesser_precip_3_AV_box_85,
                                                    lesser_precip_4_AV_box_85,
                                                    lesser_precip_5_AV_box_85,
                                                    lesser_precip_6_AV_box_85,
                                                    lesser_precip_7_AV_box_85,
                                                    lesser_precip_8_AV_box_85,
                                                    #
                                                    # mag of diffs
                                                    #
                                                    
                                                    lesser_precip_9_unbias_mag_diff_85, 
                                                    lesser_precip_10_unbias_mag_diff_85, 
                                                    lesser_precip_11_unbias_mag_diff_85, 
                                                    lesser_precip_12_unbias_mag_diff_85,
                                                    lesser_precip_1_unbias_mag_diff_85, 
                                                    lesser_precip_2_unbias_mag_diff_85, 
                                                    lesser_precip_3_unbias_mag_diff_85, 
                                                    lesser_precip_4_unbias_mag_diff_85, 
                                                    lesser_precip_5_unbias_mag_diff_85, 
                                                    lesser_precip_6_unbias_mag_diff_85, 
                                                    lesser_precip_7_unbias_mag_diff_85, 
                                                    lesser_precip_8_unbias_mag_diff_85, 
                                                    #
                                                    # perc of diffs
                                                    #
                                                     
                                                    lesser_precip_9_unbias_perc_diff_85, 
                                                    lesser_precip_10_unbias_perc_diff_85, 
                                                    lesser_precip_11_unbias_perc_diff_85, 
                                                    lesser_precip_12_unbias_perc_diff_85,
                                                    lesser_precip_1_unbias_perc_diff_85,
                                                    lesser_precip_2_unbias_perc_diff_85, 
                                                    lesser_precip_3_unbias_perc_diff_85, 
                                                    lesser_precip_4_unbias_perc_diff_85, 
                                                    lesser_precip_5_unbias_perc_diff_85, 
                                                    lesser_precip_6_unbias_perc_diff_85, 
                                                    lesser_precip_7_unbias_perc_diff_85, 
                                                    lesser_precip_8_unbias_perc_diff_85
                                                    ),
                                     ncol = 12, nrow = 3,
                                     common.legend = TRUE, legend="bottom")

  lesser_biased_RCP85 <- ggarrange(plotlist = list(
                                                  lesser_precip_9_AV_box_85,
                                                  lesser_precip_10_AV_box_85,
                                                  lesser_precip_11_AV_box_85,
                                                  lesser_precip_12_AV_box_85,
                                                  lesser_precip_1_AV_box_85, 
                                                  lesser_precip_2_AV_box_85,
                                                  lesser_precip_3_AV_box_85,
                                                  lesser_precip_4_AV_box_85,
                                                  lesser_precip_5_AV_box_85,
                                                  lesser_precip_6_AV_box_85,
                                                  lesser_precip_7_AV_box_85,
                                                  lesser_precip_8_AV_box_85,
                                                  #
                                                  # mag of diffs
                                                  #
                                                  
                                                  lesser_precip_9_bias_mag_diff_85, 
                                                  lesser_precip_10_bias_mag_diff_85, 
                                                  lesser_precip_11_bias_mag_diff_85, 
                                                  lesser_precip_12_bias_mag_diff_85,
                                                  lesser_precip_1_bias_mag_diff_85, 
                                                  lesser_precip_2_bias_mag_diff_85, 
                                                  lesser_precip_3_bias_mag_diff_85, 
                                                  lesser_precip_4_bias_mag_diff_85, 
                                                  lesser_precip_5_bias_mag_diff_85, 
                                                  lesser_precip_6_bias_mag_diff_85, 
                                                  lesser_precip_7_bias_mag_diff_85, 
                                                  lesser_precip_8_bias_mag_diff_85, 
                                                  #
                                                  # perc of diffs
                                                  #
                                                  
                                                  lesser_precip_9_bias_perc_diff_85, 
                                                  lesser_precip_10_bias_perc_diff_85, 
                                                  lesser_precip_11_bias_perc_diff_85, 
                                                  lesser_precip_12_bias_perc_diff_85,
                                                  lesser_precip_1_bias_perc_diff_85,
                                                  lesser_precip_2_bias_perc_diff_85, 
                                                  lesser_precip_3_bias_perc_diff_85, 
                                                  lesser_precip_4_bias_perc_diff_85, 
                                                  lesser_precip_5_bias_perc_diff_85, 
                                                  lesser_precip_6_bias_perc_diff_85, 
                                                  lesser_precip_7_bias_perc_diff_85, 
                                                  lesser_precip_8_bias_perc_diff_85
                                                  ),
                                   ncol = 12, nrow = 3,
                                   common.legend = TRUE, legend="bottom")

  less_unbiased_RCP45 <- ggarrange(plotlist = list(
                                                    less_precip_9_AV_box_45,
                                                    less_precip_10_AV_box_45,
                                                    less_precip_11_AV_box_45,
                                                    less_precip_12_AV_box_45,
                                                    less_precip_1_AV_box_45, 
                                                    less_precip_2_AV_box_45,
                                                    less_precip_3_AV_box_45,
                                                    less_precip_4_AV_box_45,
                                                    less_precip_5_AV_box_45,
                                                    less_precip_6_AV_box_45,
                                                    less_precip_7_AV_box_45,
                                                    less_precip_8_AV_box_45,
                                                    #
                                                    # mag of diffs
                                                    #
                                                    less_precip_9_unbias_mag_diff_45, 
                                                    less_precip_10_unbias_mag_diff_45, 
                                                    less_precip_11_unbias_mag_diff_45, 
                                                    less_precip_12_unbias_mag_diff_45,
                                                    less_precip_1_unbias_mag_diff_45, 
                                                    less_precip_2_unbias_mag_diff_45, 
                                                    less_precip_3_unbias_mag_diff_45, 
                                                    less_precip_4_unbias_mag_diff_45, 
                                                    less_precip_5_unbias_mag_diff_45, 
                                                    less_precip_6_unbias_mag_diff_45, 
                                                    less_precip_7_unbias_mag_diff_45, 
                                                    less_precip_8_unbias_mag_diff_45, 
                                                    #
                                                    # perc of diffs
                                                    #
                                                     
                                                    less_precip_9_unbias_perc_diff_45, 
                                                    less_precip_10_unbias_perc_diff_45, 
                                                    less_precip_11_unbias_perc_diff_45, 
                                                    less_precip_12_unbias_perc_diff_45,
                                                    less_precip_1_unbias_perc_diff_45,
                                                    less_precip_2_unbias_perc_diff_45, 
                                                    less_precip_3_unbias_perc_diff_45, 
                                                    less_precip_4_unbias_perc_diff_45, 
                                                    less_precip_5_unbias_perc_diff_45, 
                                                    less_precip_6_unbias_perc_diff_45, 
                                                    less_precip_7_unbias_perc_diff_45, 
                                                    less_precip_8_unbias_perc_diff_45
                                                    ),
                                   ncol = 12, nrow = 3,
                                   common.legend = TRUE, legend="bottom")

  less_biased_RCP45 <- ggarrange(plotlist = list(
                                                  less_precip_9_AV_box_45,
                                                  less_precip_10_AV_box_45,
                                                  less_precip_11_AV_box_45,
                                                  less_precip_12_AV_box_45,
                                                  less_precip_1_AV_box_45, 
                                                  less_precip_2_AV_box_45,
                                                  less_precip_3_AV_box_45,
                                                  less_precip_4_AV_box_45,
                                                  less_precip_5_AV_box_45,
                                                  less_precip_6_AV_box_45,
                                                  less_precip_7_AV_box_45,
                                                  less_precip_8_AV_box_45,
                                                  #
                                                  # mag of diffs
                                                  #
                                                  
                                                  less_precip_9_bias_mag_diff_45, 
                                                  less_precip_10_bias_mag_diff_45, 
                                                  less_precip_11_bias_mag_diff_45, 
                                                  less_precip_12_bias_mag_diff_45,
                                                  less_precip_1_bias_mag_diff_45, 
                                                  less_precip_2_bias_mag_diff_45, 
                                                  less_precip_3_bias_mag_diff_45, 
                                                  less_precip_4_bias_mag_diff_45, 
                                                  less_precip_5_bias_mag_diff_45, 
                                                  less_precip_6_bias_mag_diff_45, 
                                                  less_precip_7_bias_mag_diff_45, 
                                                  less_precip_8_bias_mag_diff_45,
                                                  #
                                                  # perc of diffs
                                                  #
                                                   
                                                  less_precip_9_bias_perc_diff_45, 
                                                  less_precip_10_bias_perc_diff_45, 
                                                  less_precip_11_bias_perc_diff_45, 
                                                  less_precip_12_bias_perc_diff_45,
                                                  less_precip_1_bias_perc_diff_45,
                                                  less_precip_2_bias_perc_diff_45, 
                                                  less_precip_3_bias_perc_diff_45, 
                                                  less_precip_4_bias_perc_diff_45, 
                                                  less_precip_5_bias_perc_diff_45, 
                                                  less_precip_6_bias_perc_diff_45, 
                                                  less_precip_7_bias_perc_diff_45, 
                                                  less_precip_8_bias_perc_diff_45
                                                    ),
                                 ncol = 12, nrow = 3,
                                 common.legend = TRUE, legend="bottom")

  less_unbiased_RCP85 <- ggarrange(plotlist = list(
                                                    less_precip_9_AV_box_85,
                                                    less_precip_10_AV_box_85,
                                                    less_precip_11_AV_box_85,
                                                    less_precip_12_AV_box_85,
                                                    less_precip_1_AV_box_85, 
                                                    less_precip_2_AV_box_85,
                                                    less_precip_3_AV_box_85,
                                                    less_precip_4_AV_box_85,
                                                    less_precip_5_AV_box_85,
                                                    less_precip_6_AV_box_85,
                                                    less_precip_7_AV_box_85,
                                                    less_precip_8_AV_box_85,
                                                    #
                                                    # mag of diffs
                                                    #
                                                   
                                                    less_precip_9_unbias_mag_diff_85, 
                                                    less_precip_10_unbias_mag_diff_85, 
                                                    less_precip_11_unbias_mag_diff_85, 
                                                    less_precip_12_unbias_mag_diff_85,
                                                     less_precip_1_unbias_mag_diff_85, 
                                                    less_precip_2_unbias_mag_diff_85, 
                                                    less_precip_3_unbias_mag_diff_85, 
                                                    less_precip_4_unbias_mag_diff_85, 
                                                    less_precip_5_unbias_mag_diff_85, 
                                                    less_precip_6_unbias_mag_diff_85, 
                                                    less_precip_7_unbias_mag_diff_85, 
                                                    less_precip_8_unbias_mag_diff_85, 
                                                    #
                                                    # perc of diffs
                                                    #
                                                   
                                                    less_precip_9_unbias_perc_diff_85, 
                                                    less_precip_10_unbias_perc_diff_85, 
                                                    less_precip_11_unbias_perc_diff_85, 
                                                    less_precip_12_unbias_perc_diff_85,
                                                     less_precip_1_unbias_perc_diff_85,
                                                    less_precip_2_unbias_perc_diff_85, 
                                                    less_precip_3_unbias_perc_diff_85, 
                                                    less_precip_4_unbias_perc_diff_85, 
                                                    less_precip_5_unbias_perc_diff_85, 
                                                    less_precip_6_unbias_perc_diff_85, 
                                                    less_precip_7_unbias_perc_diff_85, 
                                                    less_precip_8_unbias_perc_diff_85
                                                    ),
                                   ncol = 12, nrow = 3,
                                   common.legend = TRUE, legend="bottom")

  less_biased_RCP85 <- ggarrange(plotlist = list(
                                                 less_precip_9_AV_box_85,
                                                 less_precip_10_AV_box_85,
                                                 less_precip_11_AV_box_85,
                                                 less_precip_12_AV_box_85,
                                                 less_precip_1_AV_box_85, 
                                                 less_precip_2_AV_box_85,
                                                 less_precip_3_AV_box_85,
                                                 less_precip_4_AV_box_85,
                                                 less_precip_5_AV_box_85,
                                                 less_precip_6_AV_box_85,
                                                 less_precip_7_AV_box_85,
                                                 less_precip_8_AV_box_85,
                                                 #
                                                 # mag of diffs
                                                 #
                                                 
                                                 less_precip_9_bias_mag_diff_85, 
                                                 less_precip_10_bias_mag_diff_85, 
                                                 less_precip_11_bias_mag_diff_85, 
                                                 less_precip_12_bias_mag_diff_85,
                                                 less_precip_1_bias_mag_diff_85, 
                                                 less_precip_2_bias_mag_diff_85, 
                                                 less_precip_3_bias_mag_diff_85, 
                                                 less_precip_4_bias_mag_diff_85, 
                                                 less_precip_5_bias_mag_diff_85, 
                                                 less_precip_6_bias_mag_diff_85, 
                                                 less_precip_7_bias_mag_diff_85, 
                                                 less_precip_8_bias_mag_diff_85, 
                                                 #
                                                 # perc of diffs
                                                 #
                                                 
                                                 less_precip_9_bias_perc_diff_85, 
                                                 less_precip_10_bias_perc_diff_85, 
                                                 less_precip_11_bias_perc_diff_85, 
                                                 less_precip_12_bias_perc_diff_85,
                                                 less_precip_1_bias_perc_diff_85,
                                                 less_precip_2_bias_perc_diff_85, 
                                                 less_precip_3_bias_perc_diff_85, 
                                                 less_precip_4_bias_perc_diff_85, 
                                                 less_precip_5_bias_perc_diff_85, 
                                                 less_precip_6_bias_perc_diff_85, 
                                                 less_precip_7_bias_perc_diff_85, 
                                                 less_precip_8_bias_perc_diff_85
                                                ),
                                 ncol = 12, nrow = 3,
                                 common.legend = TRUE, legend="bottom")

  most_unbiased_RCP45 <- ggarrange(plotlist = list(most_precip_9_AV_box_45,
                                                   most_precip_10_AV_box_45,
                                                   most_precip_11_AV_box_45,
                                                   most_precip_12_AV_box_45,
                                                   most_precip_1_AV_box_45, 
                                                   most_precip_2_AV_box_45,
                                                   most_precip_3_AV_box_45,
                                                   most_precip_4_AV_box_45,
                                                   most_precip_5_AV_box_45,
                                                   most_precip_6_AV_box_45,
                                                   most_precip_7_AV_box_45,
                                                   most_precip_8_AV_box_45,
                                                    #
                                                    # mag of diffs
                                                    #
                                                   
                                                    most_precip_9_unbias_mag_diff_45, 
                                                    most_precip_10_unbias_mag_diff_45, 
                                                    most_precip_11_unbias_mag_diff_45, 
                                                    most_precip_12_unbias_mag_diff_45,
                                                    most_precip_1_unbias_mag_diff_45, 
                                                    most_precip_2_unbias_mag_diff_45, 
                                                    most_precip_3_unbias_mag_diff_45, 
                                                    most_precip_4_unbias_mag_diff_45, 
                                                    most_precip_5_unbias_mag_diff_45, 
                                                    most_precip_6_unbias_mag_diff_45, 
                                                    most_precip_7_unbias_mag_diff_45, 
                                                    most_precip_8_unbias_mag_diff_45,
                                                    #
                                                    # perc of diffs
                                                    #
                                                    most_precip_9_unbias_perc_diff_45, 
                                                    most_precip_10_unbias_perc_diff_45, 
                                                    most_precip_11_unbias_perc_diff_45, 
                                                    most_precip_12_unbias_perc_diff_45, 
                                                    most_precip_1_unbias_perc_diff_45,
                                                    most_precip_2_unbias_perc_diff_45, 
                                                    most_precip_3_unbias_perc_diff_45, 
                                                    most_precip_4_unbias_perc_diff_45, 
                                                    most_precip_5_unbias_perc_diff_45, 
                                                    most_precip_6_unbias_perc_diff_45, 
                                                    most_precip_7_unbias_perc_diff_45, 
                                                    most_precip_8_unbias_perc_diff_45
                                                    ),
                                   ncol = 12, nrow = 3,
                                   common.legend = TRUE, legend="bottom")

  most_biased_RCP45 <- ggarrange(plotlist = list(
                                                most_precip_9_AV_box_45,
                                                most_precip_10_AV_box_45,
                                                most_precip_11_AV_box_45,
                                                most_precip_12_AV_box_45,
                                                most_precip_1_AV_box_45, 
                                                most_precip_2_AV_box_45,
                                                most_precip_3_AV_box_45,
                                                most_precip_4_AV_box_45,
                                                most_precip_5_AV_box_45,
                                                most_precip_6_AV_box_45,
                                                most_precip_7_AV_box_45,
                                                most_precip_8_AV_box_45,
                                                #
                                                # mag of diffs
                                                #
                                                
                                                most_precip_9_bias_mag_diff_45, 
                                                most_precip_10_bias_mag_diff_45, 
                                                most_precip_11_bias_mag_diff_45, 
                                                most_precip_12_bias_mag_diff_45,
                                                most_precip_1_bias_mag_diff_45, 
                                                most_precip_2_bias_mag_diff_45, 
                                                most_precip_3_bias_mag_diff_45, 
                                                most_precip_4_bias_mag_diff_45, 
                                                most_precip_5_bias_mag_diff_45, 
                                                most_precip_6_bias_mag_diff_45, 
                                                most_precip_7_bias_mag_diff_45, 
                                                most_precip_8_bias_mag_diff_45, 
                                                #
                                                # perc of diffs
                                                #
                                                
                                                most_precip_9_bias_perc_diff_45, 
                                                most_precip_10_bias_perc_diff_45, 
                                                most_precip_11_bias_perc_diff_45, 
                                                most_precip_12_bias_perc_diff_45, 
                                                most_precip_1_bias_perc_diff_45,
                                                most_precip_2_bias_perc_diff_45, 
                                                most_precip_3_bias_perc_diff_45, 
                                                most_precip_4_bias_perc_diff_45, 
                                                most_precip_5_bias_perc_diff_45, 
                                                most_precip_6_bias_perc_diff_45, 
                                                most_precip_7_bias_perc_diff_45, 
                                                most_precip_8_bias_perc_diff_45
                                                  ),
                                 ncol = 12, nrow = 3,
                                 common.legend = TRUE, legend="bottom")

  most_unbiased_RCP85 <- ggarrange(plotlist = list(
                                                    most_precip_9_AV_box_85,
                                                    most_precip_10_AV_box_85,
                                                    most_precip_11_AV_box_85,
                                                    most_precip_12_AV_box_85,
                                                    most_precip_1_AV_box_85, 
                                                    most_precip_2_AV_box_85,
                                                    most_precip_3_AV_box_85,
                                                    most_precip_4_AV_box_85,
                                                    most_precip_5_AV_box_85,
                                                    most_precip_6_AV_box_85,
                                                    most_precip_7_AV_box_85,
                                                    most_precip_8_AV_box_85,
                                                    #
                                                    # mag of diffs
                                                    #
                                                    
                                                    most_precip_9_unbias_mag_diff_85, 
                                                    most_precip_10_unbias_mag_diff_85, 
                                                    most_precip_11_unbias_mag_diff_85, 
                                                    most_precip_12_unbias_mag_diff_85,
                                                    most_precip_1_unbias_mag_diff_85, 
                                                    most_precip_2_unbias_mag_diff_85, 
                                                    most_precip_3_unbias_mag_diff_85, 
                                                    most_precip_4_unbias_mag_diff_85, 
                                                    most_precip_5_unbias_mag_diff_85, 
                                                    most_precip_6_unbias_mag_diff_85, 
                                                    most_precip_7_unbias_mag_diff_85, 
                                                    most_precip_8_unbias_mag_diff_85, 
                                                    #
                                                    # perc of diffs
                                                    #
                                                    
                                                    most_precip_9_unbias_perc_diff_85, 
                                                    most_precip_10_unbias_perc_diff_85, 
                                                    most_precip_11_unbias_perc_diff_85, 
                                                    most_precip_12_unbias_perc_diff_85, 
                                                    most_precip_1_unbias_perc_diff_85,
                                                    most_precip_2_unbias_perc_diff_85, 
                                                    most_precip_3_unbias_perc_diff_85, 
                                                    most_precip_4_unbias_perc_diff_85, 
                                                    most_precip_5_unbias_perc_diff_85, 
                                                    most_precip_6_unbias_perc_diff_85, 
                                                    most_precip_7_unbias_perc_diff_85, 
                                                    most_precip_8_unbias_perc_diff_85
                                                    ),
                                   ncol = 12, nrow = 3,
                                   common.legend = TRUE, legend="bottom")
  
  most_biased_RCP85 <- ggarrange(plotlist = list(
                                                  most_precip_9_AV_box_85,
                                                  most_precip_10_AV_box_85,
                                                  most_precip_11_AV_box_85,
                                                  most_precip_12_AV_box_85,
                                                  most_precip_1_AV_box_85, 
                                                  most_precip_2_AV_box_85,
                                                  most_precip_3_AV_box_85,
                                                  most_precip_4_AV_box_85,
                                                  most_precip_5_AV_box_85,
                                                  most_precip_6_AV_box_85,
                                                  most_precip_7_AV_box_85,
                                                  most_precip_8_AV_box_85,
                                                  #
                                                  # mag of diffs
                                                  #
                                                  
                                                  most_precip_9_bias_mag_diff_85, 
                                                  most_precip_10_bias_mag_diff_85, 
                                                  most_precip_11_bias_mag_diff_85, 
                                                  most_precip_12_bias_mag_diff_85,
                                                  most_precip_1_bias_mag_diff_85, 
                                                  most_precip_2_bias_mag_diff_85, 
                                                  most_precip_3_bias_mag_diff_85, 
                                                  most_precip_4_bias_mag_diff_85, 
                                                  most_precip_5_bias_mag_diff_85, 
                                                  most_precip_6_bias_mag_diff_85, 
                                                  most_precip_7_bias_mag_diff_85, 
                                                  most_precip_8_bias_mag_diff_85, 
                                                  #
                                                  # perc of diffs
                                                  #
                                                  
                                                  most_precip_9_bias_perc_diff_85, 
                                                  most_precip_10_bias_perc_diff_85, 
                                                  most_precip_11_bias_perc_diff_85, 
                                                  most_precip_12_bias_perc_diff_85,
                                                  most_precip_1_bias_perc_diff_85,
                                                  most_precip_2_bias_perc_diff_85, 
                                                  most_precip_3_bias_perc_diff_85, 
                                                  most_precip_4_bias_perc_diff_85, 
                                                  most_precip_5_bias_perc_diff_85, 
                                                  most_precip_6_bias_perc_diff_85, 
                                                  most_precip_7_bias_perc_diff_85, 
                                                  most_precip_8_bias_perc_diff_85
                                                  ),
                                 ncol = 12, nrow = 3,
                                 common.legend = TRUE, legend="bottom")

  ggsave(filename = "least_unbiased_RCP45.png",
         plot = least_unbiased_RCP45, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)

  ggsave(filename = "least_biased_RCP45.png",
         plot = least_biased_RCP45, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)
    
  ggsave(filename = "least_unbiased_RCP85.png",
         plot = least_unbiased_RCP85, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)

  ggsave(filename = "least_biased_RCP85.png",
         plot = least_biased_RCP85, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)

  ggsave(filename = "lesser_unbiased_RCP45.png",
         plot = lesser_unbiased_RCP45, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)

  ggsave(filename = "lesser_biased_RCP45.png",
         plot = lesser_biased_RCP45, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)
    
  ggsave(filename = "lesser_unbiased_RCP85.png",
         plot = lesser_unbiased_RCP85, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)

  ggsave(filename = "lesser_biased_RCP85.png",
         plot = lesser_biased_RCP85, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)

  ggsave(filename = "less_unbiased_RCP45.png",
         plot = less_unbiased_RCP45, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)

  ggsave(filename = "less_biased_RCP45.png",
         plot = less_biased_RCP45, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)
    
  ggsave(filename = "less_unbiased_RCP85.png",
         plot = less_unbiased_RCP85, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)

  ggsave(filename = "less_biased_RCP85.png",
         plot = less_biased_RCP85, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)

  ggsave(filename = "most_unbiased_RCP45.png",
         plot = most_unbiased_RCP45, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)

  ggsave(filename = "most_biased_RCP45.png",
         plot = most_biased_RCP45, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)
    
  ggsave(filename = "most_unbiased_RCP85.png",
         plot = most_unbiased_RCP85, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)

  ggsave(filename = "most_biased_RCP85.png",
         plot = most_biased_RCP85, 
         width = 28, height = 8, units = "in", 
         dpi=400, device = "png", path = plot_dir)
}

print (Sys.time() - start_time)


######
###### Takes 41.3 long minutes to run this.
######
