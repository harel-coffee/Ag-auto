###################################################################
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(lubridate)
library(dplyr)

options(digit=9)
options(digits=9)

# Time the processing of this batch of files
start_time <- Sys.time()

######################################################################
##                                                                  ##
##                      Define all paths                            ##
##                                                                  ##
######################################################################
lagoon_source_path = "/home/hnoorazar/lagoon_codes/core_lagoon.R"
source(lagoon_source_path)

data_dir <- "/data/hydro/users/Hossein/lagoon/03_rain_vs_snow/"
in_dir <- paste0(data_dir, "02_cum_rain/")

out_dir_no_bias <- file.path(data_dir, "03_med_diff_med_no_bias/")
out_dir_obs <- file.path(data_dir, "03_med_diff_med_obs/")
if (dir.exists(out_dir_no_bias) == F) {dir.create(path = out_dir_no_bias, recursive = T)}
if (dir.exists(out_dir_obs) == F) {dir.create(path = out_dir_obs, recursive = T)}
######################################################################
param_dir <- "/home/hnoorazar/lagoon_codes/parameters/"
obs_clusters <- read.csv(paste0(param_dir, "loc_fip_clust.csv"),
                         header=T, as.is=T)
obs_clusters <- subset(obs_clusters, 
                       select = c("location", "cluster")) %>%
                data.table()

######################################################################
##                                                                  ##
##                                                                  ##
######################################################################
raw_files <- c("ann_cum_rain.rds", 
               "Sept_March_cum_rain.rds", 
               "month_cum_rain.rds", 
               "wtr_yr_cum_rain.rds")

target_columns <- c("annual_cum_snow", 
                    "chunk_cum_snow", 
                    "monthly_cum_snow", 
                    "annual_cum_snow")

output_names <- c("med_diff_med_ann_snow.rds", 
                  "med_diff_med_chunk_snow.rds", 
                  "med_diff_med_month_snow.rds", 
                  "med_diff_med_wtr_yr_snow.rds")

for(ii in 1:length(raw_files)){
  if (grepl("month", raw_files[ii])){
    curr_dt <- data.table(readRDS(paste0(in_dir, raw_files[ii])))
    meds_detail_no_bias <- median_diff_obs_or_modeled_month(dt = curr_dt, 
                                                            tgt_col=target_columns[ii], 
                                                            diff_from="1950-2005")
    meds_detail_obs <- median_diff_obs_or_modeled_month(dt = curr_dt, 
                                                        tgt_col=target_columns[ii], 
                                                        diff_from="1979-2016")
      
    meds_no_bias <- median_of_diff_of_medians_month(meds_detail_no_bias)
    meds_obs <- median_of_diff_of_medians_month(meds_detail_obs)

    meds_detail_obs <- merge(meds_detail_obs, obs_clusters, all.x=T, by="location")
    meds_detail_no_bias <- merge(meds_detail_no_bias, obs_clusters, all.x=T, by="location")
    meds_no_bias <- merge(meds_no_bias, obs_clusters, all.x=T, by="location")
    meds_obs <- merge(meds_obs, obs_clusters, all.x=T, by="location")
      
    saveRDS(meds_detail_no_bias, paste0(out_dir_no_bias, "detail_", output_names[ii]))
    saveRDS(meds_no_bias, paste0(out_dir_no_bias, output_names[ii]))
    saveRDS(meds_detail_obs, paste0(out_dir_obs, "detail_", output_names[ii]))
    saveRDS(meds_obs, paste0(out_dir_obs, output_names[ii]))

    } else {
      curr_dt <- data.table(readRDS(paste0(in_dir, raw_files[ii])))
      meds_detail_no_bias <- median_diff_obs_or_modeled(dt = curr_dt, 
                                                        tgt_col=target_columns[ii], 
                                                        diff_from="1950-2005")
      meds_detail_obs <- median_diff_obs_or_modeled(dt = curr_dt, 
                                                    tgt_col=target_columns[ii], 
                                                    diff_from="1979-2016")
      
      meds_no_bias <- median_of_diff_of_medians(meds_detail_no_bias)
      meds_obs <- median_of_diff_of_medians(meds_detail_obs)

      meds_detail_obs <- merge(meds_detail_obs, obs_clusters, all.x=T, by="location")
      meds_detail_no_bias <- merge(meds_detail_no_bias, obs_clusters, all.x=T, by="location")
      meds_no_bias <- merge(meds_no_bias, obs_clusters, all.x=T, by="location")
      meds_obs <- merge(meds_obs, obs_clusters, all.x=T, by="location")
      
      saveRDS(meds_detail_no_bias, paste0(out_dir_no_bias, "detail_", output_names[ii]))
      saveRDS(meds_no_bias, paste0(out_dir_no_bias, output_names[ii]))

      saveRDS(meds_detail_obs, paste0(out_dir_obs, "detail_", output_names[ii]))
      saveRDS(meds_obs, paste0(out_dir_obs, output_names[ii]))
  }
}


end_time <- Sys.time()
print( end_time - start_time)



