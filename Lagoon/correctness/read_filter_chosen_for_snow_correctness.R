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


in_dir <- "/data/hydro/users/Hossein/lagoon/03_rain_vs_snow/00_model_level/"
out_dir <- "/data/hydro/users/Hossein/lagoon/snow_correctness/"

chosen <- c("47.84375_-122.34375", "48.15625_-122.34375", 
            "47.90625_-121.84375", "48.40625_-122.53125", 
            "48.40625_-122.46875")

obs <- readRDS(paste0(in_dir, "rain_observed.rds"))
obs <- obs %>% filter(location %in% chosen)
saveRDS(obs, paste0(out_dir, "chosen_obs.rds"))

rain_RCP45 <- readRDS(paste0(in_dir, "rain_RCP45.rds"))
rain_RCP45 <- rain_RCP45 %>% filter(location %in% chosen)

rain_RCP85 <- readRDS(paste0(in_dir, "rain_RCP85.rds"))
rain_RCP85 <- rain_RCP85 %>% filter(location %in% chosen)

rain_MH <- readRDS(paste0(in_dir, "rain_modeled_hist.rds"))
rain_MH <- rain_MH %>% filter(location %in% chosen)

modeled <- rbind(rain_RCP85, rain_RCP45, rain_MH)
saveRDS(modeled, paste0(out_dir, "chosen_modeled.rds"))



