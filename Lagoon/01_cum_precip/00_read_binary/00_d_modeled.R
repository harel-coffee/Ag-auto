###################################################################
#**********                            **********
#**********        WARNING !!!!        **********
#**********                            **********
##
## DO NOT load any libraries here.
## And do not load any libraries on the drivers!
## Unless you are aware of conflicts between packages.
## I spent hours to figrue out what the hell is going on!
###################################################################
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(lubridate)
library(dplyr)

options(digit=9)
options(digits=9)

# Time the processing of this batch of files
start_time <- Sys.time()

######################################################################
##                                                                  ##
##                      Define all paths                            ##
##                                                                  ##
######################################################################
reading_binary_source <- "/home/hnoorazar/reading_binary/read_binary_core.R"
lagoon_source_path = "/home/hnoorazar/lagoon_codes/core_lagoon.R"

source(reading_binary_source)
source(lagoon_source_path)

param_dir = file.path("/home/hnoorazar/lagoon_codes/parameters/")

lagoon_out = "/data/hydro/users/Hossein/lagoon/"
main_out <- file.path(lagoon_out, "/00_raw_data/")
if (dir.exists(main_out) == F) {dir.create(path = main_out, recursive = T)}

######################################################################
##                                                                  ##
##                                                                  ##
##                                                                  ##
######################################################################

local_files <- read.csv(file = paste0(param_dir, "three_counties.csv"), 
                        header = T, as.is=T)
local_files$location <- paste0(paste0("data_", local_files$location))

# 2b. Note if working with a directory of historical data
hist <- ifelse(grepl(pattern = "historical", x = getwd()) == T, TRUE, FALSE)

# Get current folder
current_dir <- gsub(x = getwd(),
                    pattern = "/data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/",
                    replacement = "")

current_out <- file.path(main_out, current_dir)
if (dir.exists(current_out) == F) {dir.create(path = current_out, recursive = T)}

# 2d. Prep list of files for processing
# get files in current folder
dir_con <- dir()
dir_con <- dir_con[grep(pattern = "data_", x = dir_con)]
dir_con <- dir_con[which(dir_con %in% local_files$location)] # filter LOI

# 3. read and bind the data -------
all_data <- data.table()

for(file in dir_con){
  met_data <- read_binary(file_path = file, hist = hist, no_vars=4)
  met_data <- data.table(met_data)

  location <- substr(file, start = 6, stop = 24)

  # Clean it up
  met_data <- met_data %>%
              select(c(precip, year, month, day)) %>%
              data.table()

  met_data$location <- location
  met_data <- put_time_period(met_data, observed=FALSE)

  all_data <- rbind(all_data, met_data)
}

new_col_order <- c("location", "year", "month", "day", "precip", "time_period")
setcolorder(all_data, new_col_order)
saveRDS(all_data, paste0(current_out, "/raw.rds"))

# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)

