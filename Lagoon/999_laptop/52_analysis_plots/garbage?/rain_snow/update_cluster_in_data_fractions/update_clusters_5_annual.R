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

data_base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/rain_snow_fractions/"

AV_fileNs <- c("annual_fracs", "wtr_yr_fracs")
timeP_ty_middN <- c("ann", "wtr_yr")
timeP_ty <- 1

for (timeP_ty in 1:2){ # annual or wtr_yr?
  ###############################################################
  # set up title stuff
  # 
  ###############################################################
  ##################################################################################
  AVs <- readRDS(paste0(data_base, AV_fileNs[timeP_ty], ".rds")) %>% data.table()
  # update clusters to 5 
  param_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
  new_clust <- read.csv(paste0(param_dir, "/precip_elev_5_clusters.csv"), as.is=TRUE)
  AVs <- update_clusters(data_tb = AVs, new_clusters = new_clust)
  saveRDS(AVs, paste0(data_base, AV_fileNs[timeP_ty], ".rds"))
  
}


