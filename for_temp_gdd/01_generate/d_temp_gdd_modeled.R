###################################################################
#          **********                            **********
#          **********        WARNING !!!!        **********
#          **********                            **********
##
## DO NOT load any libraries here.
## And do not load any libraries on the drivers!
## Unless you are aware of conflicts between packages.
## I spent hours to figrue out what the hell is going on!
###################################################################
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(chillR)
library(tidyverse)
library(lubridate)
library(data.table)

source_path = "/data/hydro/users/Hossein/temp_gdd/temp_gdd_core.R"
source(source_path)
options(digits=9)

# 2. Pre-processing prep --------------------------------------------------
# 2a. Only use files in geographic locations we're interested in
param_dir = file.path("/home/hnoorazar/chilling_codes/parameters/")
local_files <- read.delim(file = paste0(param_dir, "file_list.txt"), header = F)
local_files <- as.vector(local_files$V1)

# 2b. Note if working with a directory of historical data
hist <- ifelse(grepl(pattern = "historical", x = getwd()) == T, TRUE, FALSE)

# Define main output path
main_out <- file.path("/data/hydro/users/Hossein/temp_gdd/modeled")

# Get current folder
current_dir <- gsub(x = getwd(),
                    pattern = "/data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/",
                    replacement = "")

print("does this look right?")
print(file.path(main_out, current_dir))

# current_dir was in the following two lines, after main_out.
if (dir.exists(file.path(main_out, current_dir)) == F) {
  dir.create(path = file.path(main_out, current_dir), recursive = T)
}

# 2d. Prep list of files for processing
# get files in current folder
dir_con <- dir()

# remove filenames that aren't data
dir_con <- dir_con[grep(pattern = "data_",
                        x = dir_con)]

# choose only files that we're interested in
dir_con <- dir_con[which(dir_con %in% local_files)]

# 3. Process the data -----------------------------------------------------

start_time <- Sys.time()

for(file in dir_con){
  # 3a. read in binary meteorological data file from specified path
  met_data <- read_binary(filename = file, hist=hist, no_vars=4)

  # I make the assumption that lat always has same number of decimal points
  lat <- as.numeric(substr(x = file, start = 6, stop = 13))

  # data frame required
  met_data <- as.data.frame(met_data)

  # 3b. Clean it up
  met_data <- met_data %>%
              select(-c(precip, windspeed)) %>%
              data.frame()
  
  met_data <- add_dd_cumdd(data.table(met_data), lower=10, upper=31.11)
  met_data$tmean = (met_data$tmax + met_data$tmin)/2
  
  met_data$latitude = substr(x=file, start=6, stop=13)
  met_data$longitude = substr(x=file, start=15, stop=24)
  met_data$ClimateScenario = unlist(strsplit(current_dir, split="/"))[1]


  ## Drop the years that are not in the observed data
  ## so we have a consistent time interval
  met_data = filter(met_data, year>= 1979)
  saveRDS(met_data, file = file.path(main_out, current_dir, paste0(file, ".rds")))
  rm(met_data)
}

# How long did it take?
end_time <- Sys.time()

print( end_time - start_time)

