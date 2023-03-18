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

data_base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
in_dir_ext <- c("precip", "runbase")
unbias_dir_ext <- "/02_med_diff_med_no_bias/"

precip_AV_fileNs <- c("ann_all_last_days", "wtr_yr_sept_all_last_days")
runoff_AV_fileNs <- c("ann_cum_runbase", "wtr_yr_cum_runbase")
timeP_ty_middN <- c("ann", "wtr_yr")
av_tg_col_pref <- c("annual_cum_", "annual_cum_")

dt_type <-  in_dir_ext[1]
in_dir <- paste0(data_base, dt_type, "/")
timeP_ty <- 1

for (dt_type in in_dir_ext){ # precip or runoff?
  in_dir <- paste0(data_base, dt_type, "/")
  for (timeP_ty in 1:2){ # annual or wtr_yr?

    if (dt_type=="precip"){
     files <- precip_AV_fileNs
     } else if (dt_type=="runbase"){
      files <- runoff_AV_fileNs
    }
    ###############################################################

    AVs <- readRDS(paste0(in_dir, files[timeP_ty], ".rds")) %>% data.table()
    unbias_diff <- readRDS(paste0(in_dir, unbias_dir_ext, "detail_med_diff_med_", 
                                  timeP_ty_middN[timeP_ty], "_", dt_type, ".rds")) %>% 
                   data.table()
    
    # update clusters to 5 
    param_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
    new_clust <- read.csv(paste0(param_dir, "/precip_elev_5_clusters.csv"), as.is=TRUE)
    AVs <- update_clusters(data_tb = AVs, new_clusters = new_clust)
    unbias_diff <- update_clusters(data_tb = unbias_diff, new_clusters = new_clust)

    saveRDS(AVs, paste0(in_dir, files[timeP_ty], ".rds")) %>% data.table()
    saveRDS(unbias_diff, paste0(in_dir, unbias_dir_ext, "detail_med_diff_med_", 
                                timeP_ty_middN[timeP_ty], "_", dt_type, ".rds"))
  }
}


