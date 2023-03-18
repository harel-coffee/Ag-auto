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

for (sub in sub_dir){
  curr_dir <- paste0(base_in, sub)
  out_dir <- file.path(paste0(curr_dir, "loc_killed/"))
  if (dir.exists(out_dir) == F) {dir.create(path = out_dir, recursive = T)}
  
  files_list <- dir(file.path(curr_dir))
  dir_con <- files_list[grep(pattern = "detail", x = files_list)]
  print("_______________________________")
  print (dir_con)
  print("_______________________________")

  for (file in dir_con){
  	curr_file <- readRDS(paste0(curr_dir, file)) %>% data.table()
  	curr_file <- within(curr_file, remove(location))
  	
    if (grepl("month", file)){
        loc_killed <- median_of_diff_of_medians_kill_location_month(curr_file)
       } else {
  	    loc_killed <- median_of_diff_of_medians_kill_location(curr_file)
                  median_of_diff_of_medians_month
    }

  	saveRDS(loc_killed, 
  		    paste0(out_dir, gsub("detail", "loc_killed",  file)))
  }
}
