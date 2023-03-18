###################################################################
#          **********                            **********
#          **********        WARNING !!!!        **********
#          **********                            **********
###################################################################
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_1 = "/home/hnoorazar/bloom_codes/bloom_core.R"
source_2 = "/home/hnoorazar/reading_binary/read_binary_core.R"
source(source_1)
source(source_2)
options(digit=9)
options(digits=9)
######################################################################
##                                                                  ##
##              Terminal/shell/bash arguments                       ##
##                                                                  ##
######################################################################
# Define main output path
bloom_out = "/data/hydro/users/Hossein/bloom/01_binary_to_bloom/"
main_out <- file.path(bloom_out, "/observed/")

# 2a. Only use files in geographic locations we're interested in
param_dir = file.path("/home/hnoorazar/bloom_codes/parameters/")
local_files <- read.delim(file = paste0(param_dir, 
                                        "file_list.txt"), 
                          header=FALSE, as.is=TRUE)
local_files <- as.vector(local_files$V1)

# 2b. Note if working with a directory of historical data
hist <- TRUE

# 2c. If needed, make fastscratch folder matching the name of the 
#     current directory
if (dir.exists(main_out) == F) {
  dir.create(path = main_out, recursive = T)
}

# 2d. Prep list of files for processing

# get files in current folder
dir_con <- dir()

# remove filenames that aren't data
dir_con <- dir_con[grep(pattern = "data_",
                        x = dir_con)]

# choose only files that we're interested in
dir_con <- dir_con[which(dir_con %in% local_files)]

print(head(dir_con))
# 3. Process the data -----------------------------------------------------
# Time the processing of this batch of files
start_time <- Sys.time()

for(file in dir_con){
  # 3a. read in binary meteorological data file from specified path
  met_data <- read_binary(file_path=file, hist=hist, no_vars=8) %>%
              data.table()

  # I make the assumption that lat always has same number of decimal points
  lat <- as.numeric(substr(x=file, start=6, stop=13))
  long <- as.numeric(substr(x=file, start=15, stop=25))
  met_data$lat <- lat
  met_data$long <- long
  met_data$model <- "observed"
  
  # 3b. Clean it up
  # rename needed columns
  met_data <- met_data %>%
              select(-c(precip, windspeed, SPH, SRAD, Rmax, Rmin)) %>%
              data.table()

  # 3c. get vertical DD.
  met_data <- generate_vertdd(data_tb=met_data)
  met_data$emission <- "observed"
  met_data <- put_chill_calendar(met_data, chill_start="sept")
  met_data <- add_chill_sept_DoY(met_data)

  saveRDS(object=met_data, 
          file=file.path(main_out, 
                         paste0("/bloom_", lat, "_", long, ".rds")))

  # write.table(x = met_data,
  #             file = file.path(main_out,
  #                              paste0("bloom_",
  #                                     lat, "_", long, ".txt")),
  #             row.names = F)

  rm(met_data)
}

# How long did it take?
end_time <- Sys.time()

print( end_time - start_time)

