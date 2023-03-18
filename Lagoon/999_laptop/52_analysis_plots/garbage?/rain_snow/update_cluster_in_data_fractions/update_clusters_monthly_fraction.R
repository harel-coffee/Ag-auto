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
#
# This is a lost cause. sick, set in specific ways, jumping to conclusions.
# not spending time to think.
#
source_path_1 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)
start_time <- Sys.time()
############################################################################

data_base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/rain_snow_fractions/"
in_dir <- data_base
############################################################################
AV_title <- "monthly precip."
AV_y_lab <- "cum. precip. (mm)"
AV_tg_col <- "monthly_cum_precip"

AVs <- readRDS(paste0(in_dir, "monthly_fracs.rds")) %>% data.table()
param_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
new_clust <- read.csv(paste0(param_dir, "/precip_elev_5_clusters.csv"), as.is=TRUE)
AVs <- update_clusters(data_tb = AVs, new_clusters = new_clust)
saveRDS(AVs, paste0(in_dir, "monthly_fracs.rds")) %>% data.table()
