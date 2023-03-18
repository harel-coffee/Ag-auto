######################################################################
##         Merging all files are too big. 
##         I will merge each model, then get their statistics, 
##         then combine those 

##         This function will merge, COMPUTE statistics, and 
##         Write just the STATISTICS to the disk.
##
######################################################################

.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(lubridate)
library(tidyverse)

args = commandArgs(trailingOnly=TRUE)
data_type = args[1]

merge <- function(data_type){
	input_dir = "/data/hydro/users/Hossein/temp_gdd/modeled/"
	output_dir= "/data/hydro/users/Hossein/temp_gdd/modeled/"

	# list of directories in input_dir
	categories = list.dirs(path = input_dir, full.names = F, recursive = F)
	
	# Remove incomplete model runs (see how you)
    # categories <- categories[-grep(x = categories$model, pattern = "incomplete"), ]
	
	versions = c("historical", "rcp45", "rcp85")
	merge_a_model(data_type=data_type, 
		          input_dir=input_dir, output_dir=output_dir, 
		          modeled_categories=categories)
}

merge_a_model <- function(data_type, input_dir, output_dir, modeled_categories){
	for (cat in modeled_categories){
		data = data.table()
		data_dir = file.path(input_dir, cat, "/", data_type, "/")
		print (paste0("The model is of type ", cat))
		print ("---------------------------------------------------")
		
		# list of .rds files in the directory
		all_files <- list.files(data_dir)
		all_files <- all_files[grep(pattern = "data_", x = all_files)]
		
		for (afile in all_files){
			curr_data <- data.table(readRDS(paste0(data_dir, afile)))
			data <- rbind(data, curr_data)
		}
	data <- add_countyGroup(data=data)
	
	# compute the average of GDD
	data <- clean(data=data, scenario=data_type)

	data_non_overlap = data[[1]]
    data_overlap = data[[2]]
    data_tmean_non_overlap = data[[3]]
    data_tmean_overlap = data[[4]]

	data_non_overlap <- data_non_overlap[, list(mean_cumm_dd = mean(Cum_dd)) , 
                                           by = c("ClimateGroup", "model", 
                                                  "scenario", "CountyGroup")]
    data_overlap <- data_overlap[, list(mean_cumm_dd = mean(Cum_dd)) , 
                                   by = c("ClimateGroup", "model", 
                                          "scenario", "CountyGroup")]

	data_tmean_non_overlap <- data_tmean_non_overlap[, list(mean_tmean = mean(tmean)), 
		                                               by = c("ClimateGroup", "model", 
                                                              "scenario", "CountyGroup")]
	data_tmean_overlap <- data_tmean_overlap[, list(mean_tmean = mean(tmean)), 
                                               by = c("ClimateGroup", "model", 
                                                      "scenario", "CountyGroup")]
    out_dir= paste0(output_dir, cat, "/")

	saveRDS(data_overlap, paste0(out_dir, data_type, "_GDD_overlap_stat.rds"))
	saveRDS(data_non_overlap, paste0(out_dir, data_type, "_GDD_non_overlap_stat.rds"))
	
	saveRDS(data_tmean_non_overlap, paste0(out_dir, data_type, "_tmean_non_overlap_stat.rds"))
	saveRDS(data_tmean_overlap, paste0(out_dir, data_type, "_tmean_overlap_stat.rds"))
	
	rm(data_overlap, data_non_overlap, data_tmean_non_overlap, data_tmean_overlap)
	}
}

clean <- function(data, scenario){

    # drop 2006-2024 years
    data = filter(data, year <=2005 | year > 2025)

    # ######## Pick only the last day of each year
    # data = data[data$month==12, ]
    # data = data[data$day==31, ]

    # # drop extra columns
    # data = within(data, remove(tmin, tmax, latitude, longitude, month, day))

    data_overlap <- data
    data_non_overlap <- data

    data_non_overlap$ClimateGroup[data_non_overlap$year >= 1979 & data_non_overlap$year <= 2005] <- "Historical"
	data_non_overlap$ClimateGroup[data_non_overlap$year > 2025  & data_non_overlap$year <= 2050] <- "2040's"
	data_non_overlap$ClimateGroup[data_non_overlap$year > 2050  & data_non_overlap$year <= 2075] <- "2060's"
	data_non_overlap$ClimateGroup[data_non_overlap$year > 2075] <- "2080's"

    data_overlap_hist = filter(data_overlap, year >= 1979 & year <= 2005)
	data_overlap_2040 = filter(data_overlap, year > 2025 & year <= 2055)
	data_overlap_2060 = filter(data_overlap, year > 2045 & year <= 2075)
	data_overlap_2080 = filter(data_overlap, year > 2065 & year <= 2095)

	if (dim(data_overlap_hist)[1]>1){data_overlap_hist$ClimateGroup = "Historical"}
	if (dim(data_overlap_2040)[1]>1){data_overlap_2040$ClimateGroup = "2040's"}
	if (dim(data_overlap_2060)[1]>1){data_overlap_2060$ClimateGroup = "2060's"}
	if (dim(data_overlap_2080)[1]>1){data_overlap_2080$ClimateGroup = "2080's"}
   
	data_overlap = rbind(data_overlap_hist, data_overlap_2040, data_overlap_2060, data_overlap_2080)
    
    # rename col names
    colnames(data_non_overlap)[colnames(data_non_overlap) == 'ClimateScenario'] <- 'model'
    colnames(data_overlap)[colnames(data_overlap) == 'ClimateScenario'] <- 'model'

    data_non_overlap$scenario = scenario
    data_overlap$scenario = scenario
     
    data_non_overlap = data.table(data_non_overlap)
    data_overlap = data.table(data_overlap)
    
    ########
    ######## Pick only the last day of each year for GDD
    ########
    gdd_non_overlap = data_non_overlap[data_non_overlap$month==12, ]
    gdd_non_overlap = gdd_non_overlap[gdd_non_overlap$day==31, ]
    gdd_non_overlap = within(gdd_non_overlap, remove(tmin, tmax, latitude, longitude, month, day))

    gdd_overlap = data_overlap[data_overlap$month==12, ]
    gdd_overlap = gdd_overlap[data_overlap$day==31, ]
    gdd_overlap = within(gdd_overlap, remove(tmin, tmax, latitude, longitude, month, day))
    
    ########
    ########       Clean data for tmean!
    ########
    data_tmean_overlap <- data_non_overlap
    data_tmean_non_overlap <- data_overlap
    rm(data_non_overlap, data_overlap)

    data_tmean_overlap_hist = filter(data_tmean_overlap, year >= 1979 & year <= 2005)
	data_tmean_overlap_2040 = filter(data_tmean_overlap, year > 2025 & year <= 2055)
	data_tmean_overlap_2060 = filter(data_tmean_overlap, year > 2045 & year <= 2075)
	data_tmean_overlap_2080 = filter(data_tmean_overlap, year > 2065 & year <= 2095)

	if (dim(data_tmean_overlap_hist)[1]>1){data_tmean_overlap_hist$ClimateGroup = "Historical"}
	if (dim(data_tmean_overlap_2040)[1]>1){data_tmean_overlap_2040$ClimateGroup = "2040's"}
	if (dim(data_tmean_overlap_2060)[1]>1){data_tmean_overlap_2060$ClimateGroup = "2060's"}
	if (dim(data_tmean_overlap_2080)[1]>1){data_tmean_overlap_2080$ClimateGroup = "2080's"}

	data_tmean_overlap = rbind(data_tmean_overlap_hist, data_tmean_overlap_2040, data_tmean_overlap_2060, data_tmean_overlap_2080)

	# rename col names
    colnames(data_tmean_non_overlap)[colnames(data_tmean_non_overlap) == 'ClimateScenario'] <- 'model'
    colnames(data_tmean_overlap)[colnames(data_tmean_overlap) == 'ClimateScenario'] <- 'model'

    data_tmean_non_overlap$scenario = scenario
    data_tmean_overlap$scenario = scenario

    data_tmean_non_overlap = data.table(data_tmean_non_overlap)
    data_tmean_overlap = data.table(data_tmean_overlap)

    return (list(gdd_non_overlap, gdd_overlap, data_tmean_non_overlap, data_tmean_overlap))
}

add_countyGroup <- function(data){
	options(digits=9)
	param_dir = "/home/hnoorazar/cleaner_codes/parameters/"
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

merge(data_type)

