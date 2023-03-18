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

data_dir <- "/data/hydro/users/Hossein/lagoon/02_run_off/00_model_level_runoff_raw/"

lagoon_out = "/data/hydro/users/Hossein/lagoon/"
main_out <- file.path(lagoon_out, "/02_run_off/01_cum_runs/chunk/")
if (dir.exists(main_out) == F) {dir.create(path = main_out, recursive = T)}
######################################################################
##                                                                  ##
##                                                                  ##
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
##                                                                  ##
######################################################################
args = commandArgs(trailingOnly=TRUE)
file_inN = args[1]
raw_files <- c(file_inN)

for(file in raw_files){
  curr_dt <- data.table(readRDS(paste0(data_dir, file)))
  curr_model_N <- unique(curr_dt$model)
  
  curr_dt <- compute_chunky_cum(curr_dt, start_month=9, end_month=3)
  
  # curr_dt <- merge(curr_dt, obs_clusters, by="location", all.x=T)
  # saveRDS(curr_dt, paste0(main_out, "/chunky/", gsub("raw", "Sept_March_cum_precip", file)))
  # curr_dt <- within(curr_dt, remove(cluster))

  curr_dt <- curr_dt %>%
             group_by(location, wtr_yr, model, emission, time_period) %>%
             filter(month==3 & day==31) %>%
             data.table()

  suppressWarnings({ curr_dt <- within(curr_dt, remove(cluster, cluster.x, cluster.y))})
  curr_dt <- merge(curr_dt, obs_clusters, by="location", all.x=T)
  saveRDS(curr_dt, paste0(main_out,
                          "Sept_March_cum_", curr_model_N, ".rds" ))
}

end_time <- Sys.time()
print( end_time - start_time)



