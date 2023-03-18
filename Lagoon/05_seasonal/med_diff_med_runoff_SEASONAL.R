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
source_path_1 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/runbase/"
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
raw_files <- c("seasonal_cum_runbase.rds")
target_columns <- c("seasonal_cum_runbase")
output_names <- c("med_diff_med_seasonal_runbase.rds")

for(ii in 1:length(raw_files)){
  curr_data <- data.table(readRDS(paste0(in_dir, raw_files[ii])))
  ######################################################################v
  # biased
  #
  meds_detail <- median_diff_obs_or_modeled_seasonal(dt = curr_data,
                                                     tgt_col=target_columns[ii],
                                                     diff_from="1979-2016")
  meds_detail <- merge(meds_detail, obs_clusters, all.x=T, by="location")
  saveRDS(meds_detail, paste0(out_dir_bias, "detail_", output_names[ii]))
  ######################################################################
  # no bias
  #
  meds_detail <- median_diff_obs_or_modeled_seasonal(dt = curr_data, 
                                                     tgt_col=target_columns[ii], 
                                                     diff_from="1950-2005")

  meds_detail <- merge(meds_detail, obs_clusters, all.x=T, by="location")
  saveRDS(meds_detail, paste0(out_dir_no_bias, "detail_", output_names[ii]))
}

end_time <- Sys.time()
print( end_time - start_time)



