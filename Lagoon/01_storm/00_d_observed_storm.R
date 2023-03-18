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
main_out <- file.path(lagoon_out, "/01_storm_cumPrecip/storm/")
if (dir.exists(main_out) == F) {dir.create(path = main_out, recursive = T)}

######################################################################
param_dir <- "/home/hnoorazar/lagoon_codes/parameters/"
obs_clusters <- read.csv(paste0(param_dir, "observed_clusters.csv"),
                         header=T, as.is=T)
obs_clusters <- subset(obs_clusters, select = c("location", "cluster")) %>%
                data.table()

######################################################################
##                                                                  ##
##                                                                  ##
##                                                                  ##
######################################################################
raw_files <- c("raw_observed.rds")
obs = TRUE

for(file in raw_files){
  curr_dt <- data.table(readRDS(paste0(data_dir, file)))
  
  # fix negative precips
  curr_dt <- curr_dt[precip < 0, precip := 0]

  curr_dt <- design_storm_4_allLoc_allMod_from_raw(curr_dt, observed = obs)
  # curr_dt <- unique(curr_dt)
  curr_dt <- merge(curr_dt, obs_clusters, by="location", all.x=T)
  saveRDS(curr_dt, paste0(main_out, "/", gsub("raw", "storm", file)))
}

end_time <- Sys.time()
print( end_time - start_time)



