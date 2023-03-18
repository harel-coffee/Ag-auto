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
######################################################################
start_time <- Sys.time()
in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/precip/"
out_dir_no_bias <- file.path(in_dir, "02_med_diff_med_no_bias/")
if (dir.exists(out_dir_no_bias) == F) {dir.create(path = out_dir_no_bias, recursive = T)}

out_dir_bias <- file.path(in_dir, "02_med_diff_med_obs/")
if (dir.exists(out_dir_no_bias) == F) {dir.create(path = out_dir_no_bias, recursive = T)}

######################################################################
param_dir <- "/Users/hn/Documents/GitHub/Kirti/Lagoon/parameters/"
obs_clusters <- read.csv(paste0(param_dir, "loc_fip_clust.csv"), header=T, as.is=T)
obs_clusters <- subset(obs_clusters, select = c("location", "cluster")) %>%
                data.table()

######################################################################
##                                                                  ##
##                                                                  ##
######################################################################
raw_files <- c("seasonal_cum_precip.rds")
target_columns <- c("seasonal_cum_precip")
output_names <- c("med_diff_med_seasonal_precip.rds")
print (1:length(raw_files))
for(ii in 1:length(raw_files)){
  curr_dt <- data.table(readRDS(paste0(in_dir, raw_files[ii])))
  print (head(curr_dt, 2))
  ######################################################################
  # biased
  #
  meds_detail <- median_diff_obs_or_modeled_seasonal(dt = curr_dt, 
                                                     tgt_col=target_columns[ii], 
                                                     diff_from="1979-2016")
  meds_detail <- merge(meds_detail, obs_clusters, all.x=T, by="location")
  saveRDS(meds_detail, paste0(out_dir_bias, "detail_", output_names[ii]))
  ######################################################################
  # unbiased
  #
  meds_detail <- median_diff_obs_or_modeled_seasonal(dt = curr_dt, 
                                                     tgt_col=target_columns[ii], 
                                                     diff_from="1950-2005")
  meds_detail <- merge(meds_detail, obs_clusters, all.x=T, by="location")

  saveRDS(meds_detail, paste0(out_dir_no_bias, "detail_", output_names[ii]))
  ######################################################################

  print (out_dir_bias)
}

end_time <- Sys.time()
print( end_time - start_time)



