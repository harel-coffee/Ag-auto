.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(lubridate)
library(tidyverse)

options(digits=9)
input_dir = "/Users/hn/Documents/GitHub/Kirti/for_temp_gdd/"
add_countyGroup <- function(data, param_dir){
	options(digits=9)
	loc_group_file_name = "LocationGroups.csv"
	locations_list = "local_list"
	loc_grp = data.table(read.csv(paste0(param_dir, loc_group_file_name)))
	loc_grp$latitude = as.numeric(loc_grp$latitude)
	loc_grp$longitude = as.numeric(loc_grp$longitude)

	data$CountyGroup = 0L
	for(i in 1:nrow(loc_grp)) {
		data[latitude == loc_grp[i, latitude] & longitude == loc_grp[i, longitude], ]$CountyGroup = loc_grp[i, locationGroup]
	}
	return (data)
}

clean_observed <- function(data, scenario="observed"){
	needed_colomns = c("year", "tmean", "Cum_dd", 
                       "ClimateGroup", "ClimateScenario",
                       "CountyGroup")
	# grab needed cols
	data = subset(data, select=needed_colomns)
    print (colnames(data))
  
    # rename col names
    colnames(data)[colnames(data) == 'ClimateScenario'] <- 'model'
    data$scenario = scenario
     
    # drop the year columnn
    data = data.table(data)
    return (data)
}

input_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/temp_vs_gdd/"
data_type = "observed"
data = data.table(readRDS(paste0(input_dir, "observed.rds")))

param_dir = "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/parameters/"
data <- add_countyGroup(data=data, param_dir)
data <- clean_observed(data=data, scenario=data_type)

data <- data[, list(mean_tmean = mean(tmean), 
                    mean_cumm_dd = mean(Cum_dd)) , 
                    by = c("ClimateGroup", "model", 
                           "scenario", "CountyGroup")]

out_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/temp_vs_gdd/"
saveRDS(data, paste0(out_dir, data_type, "_stat.rds"))

