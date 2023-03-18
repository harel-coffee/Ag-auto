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
plot_dir <- paste0(data_base, "narrowed_rain_snow_fractions/seasonal/clust_x/")
if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
print (plot_dir)
############################################################################

AV_fileNs <- "seasonal_fracs"

AV_y_lab <- "cum. precip. (mm)"
AV_title <- paste0("seasonal cum. precip.")
AV_tg_col <- "seasonal_cum_precip"

AVs <- readRDS(paste0(data_base,"seasonal_fracs.rds")) %>% data.table()
param_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
new_clust <- read.csv(paste0(param_dir, "/precip_elev_5_clusters.csv"), as.is=TRUE)
AVs <- update_clusters(data_tb = AVs, new_clusters = new_clust)
saveRDS(AVs, paste0(data_base,"seasonal_fracs.rds"))


