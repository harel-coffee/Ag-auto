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
cluster_types <- c(1, 2, 3, 4, 5)
timeP_ty_middN <- c("month")

av_tg_col_pref <- c("monthly_cum_")

dt_type <- in_dir_ext[1]
timeP_ty <- 1
clust_g <- cluster_types[1]

for (dt_type in in_dir_ext){ # precip or runoff?
  in_dir <- paste0(data_base, dt_type, "/")
  for (timeP_ty in 1:1){ # annual or chunk or wtr_yr?

    if (dt_type=="precip"){
      files <- precip_AV_fileNs
      } else if (dt_type=="runbase"){
       files <- runoff_AV_fileNs
    }

    AVs <- readRDS(paste0(in_dir, files[timeP_ty], ".rds")) %>% data.table()
    unbias_diff <- readRDS(paste0(in_dir, unbias_dir_ext, "detail_med_diff_med_", 
                                  timeP_ty_middN[timeP_ty], "_", 
                                  dt_type, ".rds")) %>% data.table()

    # update clusters to 5 
    param_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
    new_clust <- read.csv(paste0(param_dir, "/precip_elev_5_clusters.csv"), as.is=TRUE)
    AVs <- update_clusters(data_tb = AVs, new_clusters = new_clust)
    unbias_diff <- update_clusters(data_tb = unbias_diff, new_clusters = new_clust)

    saveRDS(AVs, paste0(in_dir, files[timeP_ty], ".rds")) %>% data.table()
    saveRDS(unbias_diff, paste0(in_dir, unbias_dir_ext, "detail_med_diff_med_", 
                                timeP_ty_middN[timeP_ty], "_", 
                                dt_type, ".rds")) %>% data.table()
    }
}

print (Sys.time() - start_time)
