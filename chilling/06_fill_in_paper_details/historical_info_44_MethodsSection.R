.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)


# source_path_1 <- "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_core.R"
# source_path_2 <- "/Users/hn/Documents/00_GitHub/Ag/read_binary_core/read_binary_core.R"
#
#

source_path_1 <- "/home/hnoorazar/chilling_codes/current_draft/chill_core.R"
source_path_2 <- "/home/hnoorazar/reading_binary/read_binary_core.R"

source(source_path_1)
source(source_path_2)
options(digit=9)
options(digits=9)

###
###
###
# data_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/06_fill_in_paper_details/"
# f <- "data_43.84375_-113.78125"
# param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"

data_dir <- "/data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016/"
param_dir <- "/home/hnoorazar/chilling_codes/parameters/"

###
###
###

locations <- read.csv(paste0(param_dir, "LocationGroups_NoMontana.csv"), as.is = TRUE)

local_files <- read.table(file = paste0(param_dir, "file_list.txt"), header=FALSE, as.is=TRUE)
local_files <- as.vector(local_files$V1)

hist <- TRUE
###
###
###
# avg_of_avg_of_daily_Tmin", "avg_of_avg_of_daily_Tmax",
# "avg_of_annual_precip_min", "avg_of_annual_precip_max", 
col_names <- c("location", 
               "avg_of_avg_of_daily_Tmean", 
               "avg_of_annual_precip_mean")

#
# there are 37 years in observed data which means 36 years of chill years
#
all_f_info <- setNames(data.table(matrix(nrow = length(local_files) * 36,
                                               ncol = length(col_names))),
                             col_names)

all_f_info$location <- as.character(all_f_info$location)

all_f_info$annual_temp_min <- as.double(all_f_info$annual_temp_min)
all_f_info$annual_temp_max <- as.double(all_f_info$annual_temp_max)
all_f_info$annual_temp_avg <- as.double(all_f_info$annual_temp_avg)
all_f_info$annual_temp_median <- as.double(all_f_info$annual_temp_median)
all_f_info$annual_precip_min <- as.double(all_f_info$annual_precip_min)
all_f_info$annual_precip_max <- as.double(all_f_info$annual_precip_max)
all_f_info$annual_precip_avg <- as.double(all_f_info$annual_precip_avg)
all_f_info$annual_precip_median <- as.double(all_f_info$annual_precip_median)



###
###
###
pointer = 1
for (f in local_files){
  
  met_data <- read_binary(file_path = paste0(data_dir, f), hist = hist, no_vars=8)
  curr_location = substr(f, start = 6, stop = nchar(f))

  all_f_info[pointer, "location"] = curr_location

  met_data <- put_chill_season(data_tb = met_data, chill_start = "sept")

  met_data <- met_data %>% filter(chill_season != min(met_data$chill_season))
  met_data <- met_data %>% filter(chill_season != max(met_data$chill_season))



  pointer = pointer + 



}


