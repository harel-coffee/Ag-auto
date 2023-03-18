.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(lubridate)
library(tidyverse)

options(digits=9)
input_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/temp_vs_gdd/"
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

clean_observed <- function(data, scenario){
    # drop 2006-2024 years
    data = filter(data, year <=2005 | year > 2025)
    data$ClimateGroup <- "Historical"

    # rename col names
    colnames(data)[colnames(data) == 'ClimateScenario'] <- 'model'
    data$scenario = scenario
    data = data.table(data)

    ######## Pick only the last day of each year for GDD
    data_gdd = data[data$month==12, ]
    data_gdd = data_gdd[data_gdd$day==31, ]
    data_gdd = within(data_gdd, remove(tmin, tmax, latitude, longitude, month, day))
    #######
    #######
    #######

    return (list(data, data_gdd))
}

input_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/temp_vs_gdd/"
data_type = "observed"
# data = data.table(readRDS(paste0(input_dir, "observed.rds")))
data = data.table(readRDS(paste0(input_dir, "observed_with_CountyG.rds")))

param_dir = "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/parameters/"
# data <- add_countyGroup(data=data, param_dir)

data_type = "observed"
data <- clean_observed(data=data, scenario=data_type)

data_tmean = data[[1]]
data_gdd = data[[2]]

data_gdd <- data_gdd[, list(mean_cumm_dd = mean(Cum_dd)) , 
                       by = c("ClimateGroup", "model", 
                              "scenario", "CountyGroup")]

data_tmean <- data_tmean[, list(mean_tmean = mean(tmean)), 
	                       by = c("ClimateGroup", "model", 
                                  "scenario", "CountyGroup")]
data <- merge(data_tmean, data_gdd)

out_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/temp_vs_gdd/"
saveRDS(data, paste0(out_dir, "observed_stat.rds"))
