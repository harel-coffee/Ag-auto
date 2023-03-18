
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
##                                                                  ##
##              Terminal/shell/bash arguments                       ##
##                                                                  ##
######################################################################

######################################################################
# Define main output path

frost_out = "/data/hydro/users/Hossein/chill/frost_bloom_initial_database/"

param_dir = file.path("/home/hnoorazar/chilling_codes/parameters/")
local_files <- read.delim(file = paste0(param_dir, "file_list.txt"), header = F)
local_files <- as.vector(local_files$V1)

LocationGroups_NoMontana <- read.csv(paste0(param_dir, "LocationGroups_NoMontana.csv"), 
                                     header=T, sep=",", as.is=T)
LocationGroups_NoMontana <- within(LocationGroups_NoMontana, remove(lat, long))


# 2. Pre-processing prep --------------------------------------------------
# 2a. Only use files in geographic locations we're interested in

# 2b. Note if working with a directory of historical data
hist <- ifelse(grepl(pattern = "historical", x = getwd()) == T, TRUE, FALSE)

print (getwd())
print (paste0("hist: ", hist))

current_dir <- gsub(x = getwd(),
                    pattern = "/data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/",
                    replacement = "")

# 2d. Prep list of files for processing, get files in current folder
dir_con <- dir()
dir_con <- dir_con[grep(pattern = "data_", x = dir_con)]
dir_con <- dir_con[which(dir_con %in% local_files)]
print (length(dir_con))

current_out <- file.path(frost_out, "modeled", current_dir)
if (dir.exists(current_out) == F) { dir.create(path = current_out, recursive = T) }

# 3. Process the data -----------------------------------------------------

data <- data.table()
for(file in dir_con){

  met_data <- read_binary(file_path = file, hist = hist, no_vars=4)
  lat <- substr(x = file, start = 6, stop = 13)
  long <- substr(x = file, start = 15, stop = 24)

  met_data <- data.table(met_data) 

  met_data <- met_data %>%
              select(c(year, month, day, tmin)) %>%
              data.table()
  met_data <- form_chill_season_day_of_year_modeled(met_data)
  met_data$lat <- lat
  met_data$long <- long
  data <- rbind(data, met_data)
}

data <- remove_montana(data, LocationGroups_NoMontana)
data <- data %>% filter(tmin <= 0)
data <- add_time_periods_model(data)
data <- within(data, remove(lat, long))

data_till_Dec <- data %>% filter(month %in% c(9, 10, 11, 12))
data_till_Jan <- data %>% filter(month %in% c(9, 10, 11, 12, 13))
data_till_Feb <- data

################################

saveRDS(data_till_Dec, paste0(current_out, "_data_till_Dec.rds"))
saveRDS(data_till_Jan, paste0(current_out, "_data_till_Jan.rds"))
saveRDS(data_till_Feb, paste0(current_out, "_data_till_Feb.rds"))

end_time <- Sys.time()
print( end_time - start_time)


