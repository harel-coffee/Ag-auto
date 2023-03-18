
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(tidyverse)
library(lubridate)
library(dplyr)
library(data.table)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)

options(digit=9)
options(digits=9)

#################################################################
main_out = "/data/hydro/users/Hossein/analog/usa/percipitation/"
param_dir = file.path("/home/hnoorazar/cleaner_codes/parameters/")

locations_list = read.table(paste0(param_dir, "all_us_locations_list.txt"), header=F, sep=",")
locations_list <- as.vector(locations_list$V1)
local_files = paste0("data_", locations_list)
print ("line 23")
print (length(local_files))

# 2b. Note if working with a directory of historical data
hist <- TRUE

# 2c. If needed, make fastscratch folder matching the name of the 
#     current directory
if (dir.exists(main_out) == F) {
  dir.create(path = main_out, recursive = T)
}

# get files in current folder
dir_con <- dir()
print ("line 37")
print (length(dir_con))

# remove filenames that aren't data
dir_con <- dir_con[grep(pattern = "data_", x = dir_con)]

# choose only files that we're interested in
dir_con <- dir_con[which(dir_con %in% local_files)]

# 3. Process the data -----------------------------------------------------

start_time <- Sys.time()

for(file in dir_con){

  met_data <- read_binary(file_path = file, hist = hist, no_vars=8)
  print ("line 54")
  print (colnames(met_data))
  print (dim(met_data))
  
  # data frame required
  met_data <- data.table(met_data)
  print ("line 60")
  print (dim(met_data))
  met_data <- met_data %>%
              select(-c(tmin, tmax, SPH, SRAD, Rmax, Rmin, day, month)) %>%
              data.table()
  # compute the precipitation over a year
  met_data <- aggregate(met_data$precip, by=list(year=met_data$year), FUN=sum)

  # rename the new column generated
  colnames(met_data)[colnames(met_data)=="x"] <- "yearly_precip"

  saveRDS(met_data, paste0(main_out, file, ".rds"))
  rm(met_data, lat, long)
}

end_time <- Sys.time()
print( end_time - start_time)

