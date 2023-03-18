
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
main_out = "/data/hydro/users/Hossein/analog/local/percipitation/"
param_dir = file.path("/home/hnoorazar/cleaner_codes/parameters/")

locations_list = read.table(paste0(param_dir, "local_list.txt"), header=F, sep=",")
locations_list <- as.vector(locations_list$V1)
local_files = paste0("data_", locations_list)

# 2b. Note if working with a directory of historical data
hist <- ifelse(grepl(pattern = "historical", x = getwd()) == T, TRUE, FALSE)

# We do not want history of local sites!
# So, do nothing about them!

if (hist == FALSE){
    # Get current folder
    current_dir <- gsub(x = getwd(),
                        pattern = "/data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/",
                        replacement = "")
    
    if (dir.exists(file.path(main_out, current_dir)) == F) {
      dir.create(path = file.path(main_out, current_dir), recursive = T)
    }

    # get files in current folder
    dir_con <- dir()

    # remove filenames that aren't data
    dir_con <- dir_con[grep(pattern = "data_",
                            x = dir_con)]

    ## Choose only files that we're interested in. 
    ## For future data we want local files. 
    ## local future is supposed to be compared with USA history.

    dir_con <- dir_con[which(dir_con %in% local_files)]
    print (length(dir_con))
    start_time <- Sys.time()

    for(file in dir_con){
      
      # 3a. read in binary meteorological data file from specified path
      met_data <- read_binary(file_path = file,
                              hist = hist, 
                              no_vars=4)
      
      met_data <- as.data.frame(met_data)

      met_data <- met_data %>%
                  select(-c(month, day, tmax, tmin, windspeed)) %>%
                  data.table()

      # compute the precipitation over a year
      met_data <- aggregate(met_data$precip, by=list(year=met_data$year), FUN=sum) 
      
      # rename the new column generated
      colnames(met_data)[colnames(met_data)=="x"] <- "yearly_precip"

      saveRDS(met_data, paste0(main_out, current_dir, "/", file, ".rds"))
      rm(met_data, lat, long)
    }

    end_time <- Sys.time()
    print( end_time - start_time)
}

