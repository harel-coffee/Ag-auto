.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(lubridate)
library(dplyr)

options(digit=9)
options(digits=9)

lagoon_source_path = "/home/hnoorazar/lagoon_codes/core_lagoon.R"
source(lagoon_source_path)

# Time the processing of this batch of files
start_time <- Sys.time()

################################################################
param_dir = file.path("/home/hnoorazar/lagoon_codes/parameters/")
main_in <- "/data/hydro/users/liumingdata/ForKirtiLagoon/Mingliang_Kirti_Lagoon_shortvar/"
out_dir <- "/data/hydro/users/Hossein/lagoon/02_run_off/"
if (dir.exists(out_dir) == F) {dir.create(path = out_dir, recursive = T)}
################################################################
#
# Locations of interest
#
local_files <- read.csv(file = paste0(param_dir, 
                                      "three_counties.csv"), 
                        header = T, as.is=T)

min_locations <- read.csv(file = paste0(param_dir, 
                                        "min_list_files.csv"), 
                          header = T, as.is=T)
#
# Intersection of locations
#
local_files <- local_files %>% 
                filter(location %in% min_locations$list_files)

local_files <- local_files$location
print(length(local_files))
#
# Observed Clusters
#
obs_clusters <- read.csv(paste0(param_dir, "observed_clusters.csv"),
                         header=T, as.is=T)
obs_clusters <- subset(obs_clusters, select = c("location", "cluster")) %>%
                data.table()
############
args = commandArgs(trailingOnly=TRUE)
model_names = args[1]

rcp45_dt <- data.table()
rcp85_dt <- data.table()
hist_data  <- data.table()

for (model in model_names){
  print(model)
  for (loc_f in local_files){
    current_45 <- read_min_file(paste0(main_in, model, "/rcp45/flux_", loc_f))
    current_85 <- read_min_file(paste0(main_in, model, "/rcp85/flux_", loc_f))
    current_h  <- read_min_file(paste0(main_in, model, "/historical/flux_", loc_f))
    print ("______________________")
    print (head(current_45, 2))
    print (head(current_85, 2))
    print (head(current_h, 2))
    print ("______________________")
    current_45$model <- gsub("-", "_", model)
    current_85$model <- gsub("-", "_", model)
    current_h$model  <- gsub("-", "_", model)

    current_45$location <- loc_f
    current_85$location <- loc_f
    current_h$location <- loc_f

    rcp45_dt <- rbind(rcp45_dt, current_45)
    rcp85_dt <- rbind(rcp85_dt, current_85)
    hist_data  <- rbind(hist_data, current_h)
  }
}

rcp45_dt$emission <- "RCP 4.5"
rcp85_dt$emission <- "RCP 8.5"
hist_45 <- hist_data
hist_85 <- hist_data
hist_45$emission <- "RCP 4.5"
hist_85$emission <- "RCP 8.5"

run_offs <- rbind(rcp45_dt, rcp85_dt, 
                  hist_45, hist_85)

run_offs <- merge(run_offs, obs_clusters, 
                  by="location", all.x=T)

run_offs <- put_time_period(run_offs, observed=FALSE)

run_offs$year <- as.numeric(run_offs$year)
run_offs$month <- as.numeric(run_offs$month)
run_offs$day <- as.numeric(run_offs$day)

run_offs$precip <- as.numeric(run_offs$precip)
run_offs$evap <- as.numeric(run_offs$evap)
run_offs$runoff <- as.numeric(run_offs$runoff)
run_offs$base_flow <- as.numeric(run_offs$base_flow)

print (paste0("it took ", Sys.time() - start_time , " time to reach to line 100"))
run_offs$run_p_base <- run_offs$runoff + run_offs$base_flow

saveRDS(run_offs, paste0(out_dir, "run_offs_", gsub("-", "_", model), ".rds"))


