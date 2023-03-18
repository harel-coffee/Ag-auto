.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)

source_path = "/home/hnoorazar/chilling_codes/current_draft/chill_core.R"
source(source_path)
options(digit=9)
options(digits=9)

start_time <- Sys.time()

######################################################################

args = commandArgs(trailingOnly=TRUE)
model_type = args[1]

observed_dt <- data.table()
######################################################################
# Define main output path

frost_out = "/data/hydro/users/Hossein/chill/frost_bloom_initial_database/"

param_dir = file.path("/home/hnoorazar/chilling_codes/parameters/")
local_files <- read.delim(file = paste0(param_dir, "file_list.txt"), header = F)
local_files <- as.vector(local_files$V1)

LocationGroups_NoMontana <- read.csv(paste0(param_dir, "LocationGroups_NoMontana.csv"), 
                                     header=T, sep=",", as.is=T)
LocationGroups_NoMontana <- within(LocationGroups_NoMontana, remove(lat, long))


######################################################################

observed_dir <- "/data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016/"

setwd(observed_dir)
print (paste0("we should be in :",  observed_dir))
print (getwd())

dir_con <- dir()

dir_con <- dir_con[grep(pattern = "data_", x = dir_con)]
dir_con <- dir_con[which(dir_con %in% local_files)]

# 3. Process the data -----------------------------------------------------

for(file in dir_con){
  
  lat <- substr(x = file, start = 6, stop = 13)
  long <- substr(x = file, start = 15, stop = 24)

  met_data <- read_binary(file_path = file, hist = T, no_vars=8)
  met_data <- data.table(met_data)
  
  # Clean it up
  met_data <- met_data %>%
              select(c(year, month, day, tmin)) %>%
              data.table()
  met_data <- form_chill_season_day_of_year_observed(met_data)

  met_data$lat <- lat
  met_data$long <- long
  met_data$model <- "Observed"
  observed_dt <- rbind(observed_dt, met_data)
}

observed_dt <- remove_montana(observed_dt, LocationGroups_NoMontana)
observed_dt <- within(observed_dt, remove(lat, long))
observed_dt <- add_time_periods_observed(observed_dt)
observed_dt <- observed_dt %>% filter(tmin <= 0)

observed_dt_till_Dec <- observed_dt %>% filter(month %in% c(9, 10, 11, 12))
observed_dt_till_Jan <- observed_dt %>% filter(month %in% c(9, 10, 11, 12, 13))
observed_dt_till_Feb <- observed_dt

rm(observed_dt)

################################################################

######## Jan  of 1950 is in the data, which belongs to chill season 1949,
######## which we do not have it in the data, so, time period for them will be 
######## NA, so we drop them.
observed_dt_till_Jan <- na.omit(observed_dt_till_Jan)
observed_dt_till_Feb <- na.omit(observed_dt_till_Feb)

if (dir.exists(frost_out) == F) { dir.create(path = frost_out, recursive = T) }

saveRDS(observed_dt_till_Dec, paste0(frost_out, "observed_dt_till_Dec.rds"))
saveRDS(observed_dt_till_Jan, paste0(frost_out, "observed_dt_till_Jan.rds"))
saveRDS(observed_dt_till_Feb, paste0(frost_out, "observed_dt_till_Feb.rds"))


end_time <- Sys.time()
print( end_time - start_time)

