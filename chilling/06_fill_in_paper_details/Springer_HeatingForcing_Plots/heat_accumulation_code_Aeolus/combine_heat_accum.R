.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_1 = "/home/hnoorazar/bloom_codes/bloom_core.R"
source(source_1)
options(digit=9)
options(digits=9)
start_time <- Sys.time()


##########################################################################################

base_in <- "/data/hydro/users/Hossein/bloom/01_binary_to_bloom/"

modeled_in <- paste0(base_in, "modeled/")
observed_in <- paste0(base_in, "observed/")

param_dir <- "/home/hnoorazar/bloom_codes/parameters/"
##########################################################################################
locations <- read.csv(paste0(param_dir, "limited_locations.csv"))

models <- c("bcc-csm1-1", "CanESM2","CSIRO-Mk3-6-0", "HadGEM2-CC365", "IPSL-CM5A-LR", "MIROC5", "NorESM1-M", 
            "bcc-csm1-1-m",  "CCSM4", "GFDL-ESM2G", "HadGEM2-ES365", "IPSL-CM5A-MR", "MIROC-ESM-CHEM",
            "BNU-ESM", "CNRM-CM5", "GFDL-ESM2M", "inmcm4", "IPSL-CM5B-LR", "MRI-CGCM3")


modeled_bloom_dt <- data.table()


for (model in models){
  print(model)
  curr_mod_dir <- paste0(modeled_in,  model, "/rcp85/")
  for (ct in unique(locations$city)){
    Loc <- locations %>% filter(city == ct)
    curr_lat <- Loc$lat
    curr_long <- Loc$long
    file <- paste0(curr_mod_dir, "bloom_", curr_lat, "_", curr_long, ".rds")
    curr_m <- data.table(readRDS(file))
    curr_m$model <- model
    curr_m$emission <- "RCP 8.5"
    modeled_bloom_dt <- rbind(modeled_bloom_dt, curr_m)
  }
}


hist_bloom_dt <- data.table()
for (ct in unique(locations$city)){
  Loc <- locations %>% filter(city == ct)
  curr_lat <- Loc$lat
  curr_long <- Loc$long
  file <- paste0(observed_in, "bloom_", curr_lat, "_", curr_long, ".rds")
  print (file)
  curr_hist <- data.table(readRDS(file))
  curr_hist$model <- "observed"
  curr_hist$emission <- "RCP 8.5"
  hist_bloom_dt <- rbind(hist_bloom_dt, curr_hist)
}


bloom_dt <- rbind(modeled_bloom_dt, hist_bloom_dt)

out_dir <- "/data/hydro/users/Hossein/bloom/01_binary_to_bloom/"
if (dir.exists(file.path(out_dir)) == F) {
  dir.create(path = file.path(out_dir), recursive=T)
}

saveRDS(object = bloom_dt,
        file = paste0(out_dir, 
                    "/heat_accum_limit_cities.rds"))





