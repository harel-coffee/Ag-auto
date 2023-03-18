##################################################################################
#          **********                            **********
#          **********        WARNING !!!!        **********
#          **********                            **********
##
## DO NOT load any libraries here.
## And do not load any libraries on the drivers!
## Unless you are aware of conflicts between packages.
## I spent hours to figrue out what the hell is going on!
##################################################################################
.libPaths("/data/hydro/R_libs35")
.libPaths()
# library(chillR)
library(tidyverse)
library(lubridate)
library(data.table)

source_path = "/data/hydro/users/Hossein/temp_gdd/temp_gdd_core.R"
source(source_path)
options(digits=9)
##################################################################################
param_dir = "/home/hnoorazar/chilling_codes/parameters/"
local_files <- read.delim(file = paste0(param_dir, "file_list.txt"), header = F)
local_files <- as.vector(local_files$V1)

# 2b. Note if working with a directory of historical data

# Define main output path
main_out <- file.path("/data/hydro/users/Hossein/temp_gdd/observed")

if (dir.exists(main_out) == F) {
  dir.create(path = main_out, recursive = T)
}

# 2d. Prep list of files for processing

# get files in current folder
dir_con <- dir()

# remove filenames that aren't data
dir_con <- dir_con[grep(pattern = "data_", x = dir_con)]

# choose only files that we're interested in
dir_con <- dir_con[which(dir_con %in% local_files)]

# 3. Process the data -----------------------------------------------------
# Time the processing of this batch of files
start_time <- Sys.time()

for(file in dir_con){
  # 3a. read in binary meteorological data file from specified path
  met_data <- read_binary(filename = file, hist = TRUE, no_vars=8)

  # I make the assumption that lat always has same number of decimal points
  lat <- as.numeric(substr(x = file, start = 6, stop = 13))

  # data frame required
  met_data <- as.data.frame(met_data)
  
  # 3b. Clean it up
  # rename needed columns
  met_data <- met_data %>%
              select(-c(precip, windspeed, SPH, SRAD, Rmax, Rmin)) %>%
              data.frame()
  
  met_data <- add_dd_cumdd(data.table(met_data), lower=10, upper=31.11)
  met_data$tmean = (met_data$tmax + met_data$tmin)/2

  met_data$latitude = substr(x=file, start=6, stop=13)
  met_data$longitude = substr(x=file, start=15, stop=24)
  met_data$ClimateScenario = "observed"
  saveRDS(met_data, file = file.path(main_out, paste0(file, ".rds")))
  rm(met_data)
}

# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)


