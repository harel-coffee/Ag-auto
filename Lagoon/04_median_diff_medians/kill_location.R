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

base_in <- "/data/hydro/users/Hossein/lagoon/"
sub_dir <- c("01_cum_precip/02_med_diff_med_no_bias/", 
             "01_cum_precip/02_med_diff_med_obs/",
             "01_run_off/02_med_diff_med_no_bias/", 
             "01_run_off/02_med_diff_med_obs/",
             "03_rain_vs_snow/03_med_diff_med_no_bias/",
             "03_rain_vs_snow/03_med_diff_med_obs/")
######################################################################
for (sub in sub_dir){
  curr_dir <- paste0(base_in, sub_dir)
  out_dir <- paste0(curr_dir, "loc_killed/")
  print(out_dir)
  if (dir.exists(out_dir) == F) {dir.create(path = out_dir, recursive = T)}

  dir_con <- dir_con[grep(pattern = "detail", x = curr_dir)]
  for (file in dir_con){
  	curr_file <- readRDS(paste0(curr_dir, file)) %>% data.table()
  	curr_file <- within(curr_file, remove(location))
  	
  	loc_killed <- median_of_diff_of_medians_kill_location(curr_file)


  	saveRDS(loc_killed, 
  		    paste0(out_dir, gsub("loc_killed", "detail",  file)))
  }
}
