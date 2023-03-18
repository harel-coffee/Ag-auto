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

data_dir <- "/data/hydro/users/Hossein/lagoon/00_raw_data/"

lagoon_out = "/data/hydro/users/Hossein/lagoon/"
main_out <- file.path(lagoon_out, "/01_storm_cumPrecip/cum_precip/")
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
raw_files <- c("raw_observed.rds",
               "raw_modeled_hist.rds",
               "raw_RCP45.rds", 
               "raw_RCP85.rds")

for(file in raw_files){
  curr_dt <- data.table(readRDS(paste0(data_dir, file)))
  # replace negative precips
  curr_dt <- curr_dt[precip < 0, precip := 0]
  
  curr_dt <- create_wtr_calendar(curr_dt, wtr_yr_start=10)
  curr_dt <- compute_wtr_yr_cum(curr_dt)

  # curr_dt <- merge(curr_dt, obs_clusters, by="location", all.x=T)
  # saveRDS(curr_dt, paste0(main_out, "/wtr_yr/", 
  #                         gsub("raw", "wtr_yr_sept_cum_precip", file)))
  # do the following, because we do not know
  # how that column will effect the outcome.
  # curr_dt <- within(curr_dt, remove(cluster))

  curr_dt <- curr_dt %>%
             group_by(location, wtr_yr, model, emission, time_period) %>%
             filter(month==9 & day==30) %>%
             data.table()
  
  suppressWarnings({ curr_dt <- within(curr_dt, remove(cluster, cluster.x, cluster.y))})
  curr_dt <- merge(curr_dt, obs_clusters, by="location", all.x=T)
  saveRDS(curr_dt, paste0(main_out, "/wtr_yr/", 
                          gsub("raw", "wtr_yr_cum_precip_LD", file)))
}

end_time <- Sys.time()
print( end_time - start_time)



