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
##                      Define all paths                            ##
######################################################################
reading_binary_source <- "/home/hnoorazar/reading_binary/read_binary_core.R"
lagoon_source_path = "/home/hnoorazar/lagoon_codes/core_lagoon.R"

source(reading_binary_source)
source(lagoon_source_path)

param_dir = file.path("/home/hnoorazar/lagoon_codes/parameters/")

lagoon_out = "/data/hydro/users/Hossein/lagoon/"
main_out <- file.path(lagoon_out, "/00_model_level_storm/")
if (dir.exists(main_out) == F) {dir.create(path = main_out, recursive = T)}
######################################################################
param_dir <- "/home/hnoorazar/lagoon_codes/parameters/"
obs_clusters <- read.csv(paste0(param_dir, "observed_clusters.csv"),
                         header=T, as.is=T)
obs_clusters <- subset(obs_clusters, select = c("location", "cluster")) %>%
                data.table()
######################################################################
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
  
  # fix negative precips
  met_data <- met_data[precip < 0, precip := 0]

  met_data$location <- location
  all_data <- rbind(all_data, met_data)
}

all_data <- put_time_period(all_data, observed=FALSE)
all_data$model <- unlist(strsplit(getwd(), "/"))[7]
all_data$emission <- unlist(strsplit(getwd(), "/"))[8]

new_col_corder <- c("location", "year", "month", "day","precip", 
                    "time_period", "model", "emission")
setcolorder(all_data, new_col_corder)

all_data <- design_storm_4_allLoc_allMod_from_raw(data_tbl=all_data, 
                                                  observed=FALSE)

all_data <- unique(all_data)

all_data <- merge(all_data, obs_clusters, by="location", all.x=T)

saveRDS(all_data, paste0(current_out, "/storm.rds"))

# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)

