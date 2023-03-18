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
    unbias_diff <- readRDS(paste0(in_dir, unbias_dir_ext, "detail_med_diff_med_", 
                                  timeP_ty_middN[timeP_ty], "_", 
                                  dt_type, ".rds")) %>% data.table()

    bias_diff <- bias_diff %>% filter(time_period != "2006-2025") %>% data.table()
    unbias_diff <- unbias_diff %>% filter(time_period != "2006-2025") %>% data.table()
    #
    # remove those rows whose perc diff is more than 1000%
    #
    unbias_diff <- unbias_diff %>% filter(perc_diff < 600) %>% data.table()
    unbias_diff <- unbias_diff %>% filter(perc_diff > -600) %>% data.table()
  
    plot_dir <- paste0(in_dir, "narrowed_", dt_type, "/monthly", "_", dt_type, "/perc_diffs/")
    if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
    unbias_diff_45 <- unbias_diff %>% filter(emission=="RCP 4.5") %>% data.table()
    unbias_diff_85 <- unbias_diff %>% filter(emission=="RCP 8.5") %>% data.table()
    rm(unbias_diff)

    for (clust_g in cluster_types){
      curr_unbias_diff_45 <- unbias_diff_45 %>% filter(cluster == clust_g) %>% data.table()
      curr_unbias_diff_85 <- unbias_diff_85 %>% filter(cluster == clust_g) %>% data.table()
      #########
      ######### unbiased Percentage diffs
      #########
      box_title <- "unbiased differences"
      box_subtitle <- "for each model median is\ntaken over years, separately"

      assign(x = paste0("unbias_", unlist(strsplit(clust_g, " "))[1], "_perc_diff_45"),
             value = {box_trend_monthly_cum(dt = curr_unbias_diff_45, p_type="box",
                                            y_lab = "differences (%)",
                                            tgt_col = "perc_diff") + 
                      ggtitle(box_title)
                      })

      assign(x = paste0("unbias_", unlist(strsplit(clust_g, " "))[1], "_perc_diff_85"),
             value = {box_trend_monthly_cum(dt = curr_unbias_diff_85, p_type="box",
                                            y_lab = "differences (%)", 
                                            tgt_col = "perc_diff") + 
                      ggtitle(box_title)
                      })
    }
    unbiased_RCP45 <- ggarrange(plotlist = list(unbias_least_perc_diff_45,
                                                unbias_lesser_perc_diff_45,
                                                unbias_less_perc_diff_45,
                                                unbias_most_perc_diff_45
                                                ),
                                ncol = 1, nrow = 4,
                                common.legend = TRUE, legend="bottom")

    unbiased_RCP85 <- ggarrange(plotlist = list(unbias_least_perc_diff_85,
                                                unbias_lesser_perc_diff_85,
                                                unbias_less_perc_diff_85,
                                                unbias_most_perc_diff_85
                                               ),
                                ncol = 1, nrow = 4,
                                common.legend = TRUE, legend="bottom")

    ggsave(filename = "unbiased_RCP45.png",
             plot = unbiased_RCP45, 
             width = 8, height = 7, units = "in", 
             dpi=400, device = "png",
             path = plot_dir)
    
    ggsave(filename = "unbiased_RCP85.png",
             plot = unbiased_RCP85, 
             width = 8, height = 7, units = "in", 
             dpi=400, device = "png",
             path = plot_dir)
  }
}

print (Sys.time() - start_time)