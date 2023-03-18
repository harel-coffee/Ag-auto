
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(plyr)
library(lubridate)
library(purrr)
library(tidyverse)

source_path = "/home/hnoorazar/chilling_codes/current_draft/chill_core.R"
source(source_path)

options(digit=9)
options(digits=9)

# 2. Script setup ---------------------------------------------------------
# Check current folder
print("does this look right?")
getwd()
start_time <- Sys.time()

######################################################################
##                                                                  ##
##              Terminal/shell/bash arguments                       ##
##                                                                  ##
######################################################################

args = commandArgs(trailingOnly=TRUE)
season_start = args[1]

model_type = "dynamic"
######################################################################
chill_out = "/data/hydro/users/Hossein/chill/data_by_core/"
main_out <- file.path(chill_out, model_type, "/02_01_with_May_4HeatTrigger/", season_start, "/")

# Create the directory if it doesn't exist
if (dir.exists(file.path(main_out)) == F) {
  dir.create(path = main_out, recursive = T)
}

# 3. Some set up ----------------------------------------------------------

the_dir <- dir() # List of filenames


#
# we are in /data/hydro/users/Hossein/chill/data_by_core/dynamic/01/sept/observed/
# We entered here from the .sh file.
#
# Remove filenames that aren't data, if they exist
the_dir <- the_dir[grep(pattern = "chill_output_data",
                        x = the_dir)]

# Pre-allocate lists to be used
param_dir = file.path("/home/hnoorazar/chilling_codes/parameters/")
local_files <- read.delim(file = paste0(param_dir, 
                                        "file_list.txt"), 
                          header = F)
local_files <- as.vector(local_files$V1)

no_sites <- length(local_files)
data_list_historical <- vector(mode = "list", length = no_sites)

# 5. Iterate through files and process ---------------------------
#    5a. Iterate through historical files ------------------------

for(i in 1:length(the_dir)){
  
  # Read in file
  file <- read.table(file = the_dir[i],
                     header = T,
                     colClasses = c("factor", "numeric", 
                                    "numeric", "numeric",
                                    "numeric", "numeric"))
  
  names(data_list_historical)[i] <- the_dir[i]
  
  # Append it to a list following some processing
  data_list_historical[[i]] <-  heat_trigger_DoYDetector(file, 
                                                         data_type="historical", 
                                                         min_CP = 73.3, 
                                                         percent_4_heat = 0.75)
  rm(file)
}
  
# 5b. Process gathered historical data ------------------------------------

# Get medians for each location during historical period
summary_data_historical <- get_medians_heatTrigger(data_list_historical)

# Briefly want to export the raw data from the lists for use in other figs
data_historical <- ldply(data_list_historical, function(x) data.frame(x))

data_historical$year <- as.numeric(substr(x = data_historical$chill_season,
                                          start = 12, stop = 15))
data_historical$model <- "observed"
data_historical$emission <- "observed"
data_historical$lat = as.numeric(substr(x = data_historical$.id,
                                        start = 19, stop = 26))
data_historical$long = as.numeric(substr(x = data_historical$.id,
                                         start = 28, stop = 37))
data_historical <- unique(data_historical)

# Remove what's no longer needed
rm(data_list_historical)

# .id row contains originating filename of this data
data_historical <- add_time_periods_observed(data_historical)
summary_data_historical <- add_time_periods_observed(summary_data_historical)

print ("there should be time period in columns:")
print(colnames(summary_data_historical))

write.table(x = data_historical,
            file = file.path(main_out,
                             "heatTrigger_summary_obs_hist.txt"),
            row.names = F)

rm(data_historical)

# Grab lat/long
summary_data_historical <- grab_coord_heatTrigger(summary_data_historical)

summary_data_historical$model <- "observed"
summary_data_historical$emission <- "observed"
summary_data_historical <- add_time_periods_observed(summary_data_historical)
print ("there should be time period in columns:")
print(colnames(summary_data_historical))

out <- paste0(main_out, "/stats/")
if (dir.exists(file.path(out)) == F) {
  dir.create(path = out, recursive = T)
}
write.table(x = summary_data_historical,
            file = file.path(out, "heatTrigger_summary_stats_observed.txt"),
            row.names = F)
  
end_time <- Sys.time()

print( end_time - start_time)



