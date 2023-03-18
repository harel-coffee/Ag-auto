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
main_in_1 <- "/data/hydro/users/liumingdata/ForKirtiLagoon/"
main_in_2 <- "Mingliang_Kirti_Lagoon_shortvar/UI_historical/"
main_in_3 <- "VIC_Binary_CONUS_to_2016/"
main_in <- paste0(main_in_1, main_in_2, main_in_3)

out_dir <- "/data/hydro/users/Hossein/lagoon/02_run_off/"
param_dir = file.path("/home/hnoorazar/lagoon_codes/parameters/")
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
print (length(local_files))
#
# Observed Clusters
#
obs_clusters <- read.csv(paste0(param_dir, "observed_clusters.csv"),
                         header=T, as.is=T)
obs_clusters <- subset(obs_clusters, select = c("location", "cluster")) %>%
                data.table()
############

hist_data  <- data.table()

for (loc_f in local_files){
  current_h  <- read_min_file(paste0(main_in, "/flux_", loc_f))
  print ("______________________")
  print (head(current_h, 2))
  print ("______________________")
  current_h$location <- loc_f
  hist_data  <- rbind(hist_data, current_h)
}

hist_45 <- hist_data
hist_85 <- hist_data
hist_45$emission <- "RCP 4.5"
hist_85$emission <- "RCP 8.5"

run_offs <- rbind(hist_45, hist_85)

run_offs <- merge(run_offs, obs_clusters, 
                  by="location", all.x=T)

run_offs <- put_time_period(run_offs, observed=TRUE)

run_offs$year <- as.numeric(run_offs$year)
run_offs$month <- as.numeric(run_offs$month)
run_offs$day <- as.numeric(run_offs$day)

run_offs$precip <- as.numeric(run_offs$precip)
run_offs$evap <- as.numeric(run_offs$evap)
run_offs$runoff <- as.numeric(run_offs$runoff)
run_offs$base_flow <- as.numeric(run_offs$base_flow)
run_offs$model  <- "observed"

run_offs$run_p_base <- run_offs$runoff + run_offs$base_flow

saveRDS(run_offs, paste0(out_dir, "run_offs_observed.rds"))

print (paste0("it took ", Sys.time() - start_time))
