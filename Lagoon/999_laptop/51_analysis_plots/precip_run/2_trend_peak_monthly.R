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
in_dir_ext <- c("precip", "runbase")
unbias_dir_ext <- "/02_med_diff_med_no_bias/"

precip_AV_fileNs <- c("month_all_last_days")
runoff_AV_fileNs <- c("monthly_cum_runbase")
timeP_ty_middN <- c("month")

av_tg_col_pref <- c("monthly_cum_")
av_titles <- c("monthly ")

dt_type <- in_dir_ext[2]
timeP_ty <- 1

for (dt_type in in_dir_ext){ # precip or runoff?
  in_dir <- paste0(data_base, dt_type, "/")
  for (timeP_ty in 1:1){ # annual or chunk or wtr_yr?

    if (dt_type=="precip"){
     files <- precip_AV_fileNs
     AV_y_lab <- "cum. precip. (mm)"
     AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], dt_type)
     AV_title <- paste0(av_titles[timeP_ty], "precip.")
     } else if (dt_type=="runbase"){
      files <- runoff_AV_fileNs
      AV_y_lab <- "cum. runoff (mm)"
      AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], "runbase")
      AV_title <- paste0(av_titles[timeP_ty], "runoff.")
    }

    AVs <- readRDS(paste0(in_dir, files[timeP_ty], ".rds")) %>% data.table()
    unbias_diff <- readRDS(paste0(in_dir, unbias_dir_ext, "detail_med_diff_med_", 
                                  timeP_ty_middN[timeP_ty], "_", 
                                  dt_type, ".rds")) %>% data.table()
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
    unbias_diff <- unbias_diff %>% filter(perc_diff < 600) %>% data.table()
    unbias_diff <- unbias_diff %>% filter(perc_diff > -600) %>% data.table()
  
    plot_dir <- paste0(in_dir, "narrowed_", dt_type, "/monthly", "_", dt_type, "/peak/")
    if (dir.exists(plot_dir) == F) {
    	dir.create(path = plot_dir, recursive = T)}
    print (plot_dir)

    AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
    AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table()
    unbias_diff_45 <- unbias_diff %>% filter(emission=="RCP 4.5") %>% data.table()
    unbias_diff_85 <- unbias_diff %>% filter(emission=="RCP 8.5") %>% data.table()
    rm(AVs, unbias_diff)

    for (clust_g in cluster_types){
      curr_AVs_45 <- AVs_45 %>% filter(cluster == clust_g) %>% data.table()
      curr_AVs_85 <- AVs_85 %>% filter(cluster == clust_g) %>% data.table()
      
      curr_unbias_diff_45 <- unbias_diff_45 %>% filter(cluster == clust_g) %>% data.table()
      curr_unbias_diff_85 <- unbias_diff_85 %>% filter(cluster == clust_g) %>% data.table()
      #########
      ######### Actual value plots
      #########
      quans_85 <- find_quantiles(data_table=curr_AVs_85, tgt_col= AV_tg_col, time_type="monthly")
      quans_45 <- find_quantiles(data_table=curr_AVs_45, tgt_col= AV_tg_col, time_type="monthly")

      assign(x = "AV_box_85",
             value = {box_trend_monthly_cum(dt=curr_AVs_85, 
                                            p_type="box",
                                            y_lab = AV_y_lab, 
                                            tgt_col = AV_tg_col) + 
                      ggtitle(paste0("monthy ", dt_type, ".")) +
                      coord_cartesian(ylim = c(quans_85[1], quans_85[2]))
                      })

      assign(x = "AV_box_45",
             value = {box_trend_monthly_cum(dt = curr_AVs_45, p_type="box",
                                            y_lab = AV_y_lab, tgt_col = AV_tg_col) + 
                      ggtitle(paste0("monthy ", dt_type, ".")) + 
                      coord_cartesian(ylim = c(quans_45[1], quans_45[2]))
                      })
      #########
      ######### unbiased Percentage diffs
      #########
      box_title <- "percentage differences between future time periods and historical"
      box_subtitle <- "for each model median is\ntaken over years, separately"
      
      quans_85 <- find_quantiles(curr_unbias_diff_85, tgt_col= "perc_diff", time_type="monthly")
      quans_45 <- find_quantiles(curr_unbias_diff_45, tgt_col= "perc_diff", time_type="monthly")

      assign(x = "unbias_perc_diff_85",
             value = {box_trend_monthly_cum(dt = curr_unbias_diff_85, 
                                            p_type="box",
                                            y_lab="differences (%)", 
                                            tgt_col="perc_diff") + 
                      ggtitle(box_title) + 
                      coord_cartesian(ylim = c(quans_85[1], quans_85[2]))
                      })

      assign(x = "unbias_perc_diff_45",
             value = {box_trend_monthly_cum(dt = curr_unbias_diff_45, 
                                            p_type="box",
                                            y_lab = "differences (%)",
                                            tgt_col = "perc_diff") + 
                      ggtitle(box_title) + 
                      coord_cartesian(ylim = c(quans_45[1], quans_45[2]))
                      })
      ##################################################################################
      ##################################################################################
      ##################################################################################
      unbiased_RCP45 <- ggarrange(plotlist = list(AV_box_45, unbias_perc_diff_45),
                                  ncol = 1, nrow = 2,
                                  common.legend = TRUE, legend="bottom")

      unbiased_RCP85 <- ggarrange(plotlist = list(AV_box_85, unbias_perc_diff_85),
                                  ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")
      
      ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_", 
                               timeP_ty_middN[timeP_ty], "_unbiased_RCP45.png"),
             plot = unbiased_RCP45, width = 9, height = 5, units = "in", 
             dpi=400, device = "png", path = plot_dir)
      
      ggsave(filename = paste0(gsub("\ ", "_", clust_g), "_", 
                               timeP_ty_middN[timeP_ty], "_unbiased_RCP85.png"),
             plot = unbiased_RCP85, width = 9, height = 5, units = "in", 
             dpi=400, device = "png", path = plot_dir)
      print(paste0(gsub("\ ", "_", clust_g), "_", 
                   timeP_ty_middN[timeP_ty], "_unbiased_RCP85.png"))
    }
  }
}

print (Sys.time() - start_time)