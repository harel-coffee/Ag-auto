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

main_in <- "/data/hydro/users/Hossein/lagoon/01_run_off/"

in_dir <- paste0(main_in, "01_cum_runoffs/")
out_dir_no_bias <- file.path(main_in, "02_med_diff_med_no_bias/")
if (dir.exists(out_dir_no_bias) == F) {dir.create(path = out_dir_no_bias, recursive = T)}

out_dir_obs <- file.path(main_in, "02_med_diff_med_obs/")
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
raw_files <- c("all_ann_cum_runbase.rds", 
               "all_chunk_cum_runbase.rds", 
               "all_monthly_cum_runbase.rds", 
               "all_wtr_yr_cum_runbase.rds")

target_columns <- c("annual_cum_runbase", 
                    "chunk_cum_runbase", 
                    "monthly_cum_runbase", 
                    "annual_cum_runbase")

output_names <- c("med_diff_med_ann_runbase.rds", 
                  "med_diff_med_chunk_runbase.rds", 
                  "med_diff_med_month_runbase.rds", 
                  "med_diff_med_wtr_yr_runbase.rds")

for(ii in 1:length(raw_files)){
  if (grepl("month", raw_files[ii])){
     curr_dt <- data.table(readRDS(paste0(in_dir, raw_files[ii])))
     meds_detail <- median_diff_obs_or_modeled_month(dt = curr_dt, 
                                                           tgt_col=target_columns[ii], 
                                                           diff_from="1950-2005")
     meds <- median_of_diff_of_medians_month(meds_detail)

     meds_detail <- merge(meds_detail, obs_clusters, all.x=T, by="location")
     meds <- merge(meds, obs_clusters, all.x=T, by="location")
     saveRDS(meds_detail, paste0(out_dir_no_bias, "detail_", output_names[ii]))
     saveRDS(meds, paste0(out_dir_no_bias, output_names[ii]))

     meds_detail <- median_diff_obs_or_modeled_month(dt = curr_dt, 
                                                           tgt_col=target_columns[ii], 
                                                           diff_from="1979-2016")
     meds <- median_of_diff_of_medians_month(meds_detail)

     meds_detail <- merge(meds_detail, obs_clusters, all.x=T, by="location")
     meds <- merge(meds, obs_clusters, all.x=T, by="location")
      
     saveRDS(meds_detail, paste0(out_dir_obs, "detail_", output_names[ii]))
     saveRDS(meds, paste0(out_dir_obs, output_names[ii]))

    } else {
      curr_dt <- data.table(readRDS(paste0(in_dir, raw_files[ii])))
      meds_detail <- median_diff_obs_or_modeled(dt = curr_dt, 
                                                      tgt_col=target_columns[ii], 
                                                      diff_from="1950-2005")
      meds <- median_of_diff_of_medians(meds_detail)

      meds_detail <- merge(meds_detail, obs_clusters, all.x=T, by="location")
      meds <- merge(meds, obs_clusters, all.x=T, by="location")
      saveRDS(meds_detail, paste0(out_dir_no_bias, "detail_", output_names[ii]))
      saveRDS(meds, paste0(out_dir_no_bias, output_names[ii]))

      meds_detail <- median_diff_obs_or_modeled(dt = curr_dt, 
                                                      tgt_col=target_columns[ii], 
                                                      diff_from="1979-2016")
      meds <- median_of_diff_of_medians(meds_detail)

      meds_detail <- merge(meds_detail, obs_clusters, all.x=T, by="location")
      meds <- merge(meds, obs_clusters, all.x=T, by="location")
      
      saveRDS(meds_detail, paste0(out_dir_obs, "detail_", output_names[ii]))
      saveRDS(meds, paste0(out_dir_obs, output_names[ii]))

  }
}

end_time <- Sys.time()
print( end_time - start_time)



