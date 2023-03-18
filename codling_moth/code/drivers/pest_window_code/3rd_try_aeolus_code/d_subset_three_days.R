#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(dplyr)
# library(tidyverse)

data_dir  = "/data/hydro/users/Hossein/codling_moth_new/local/processed/"
output_dir= paste0(data_dir, "section_46_Pest/3rd_try/")
name_pref = "combined_CMPOP_rcp"
models = c("45.rds", "85.rds")

for (model in models){
	output_name = paste0("three_days_", model)
	curr_data = data.table(readRDS(paste0(data_dir, name_pref, model)))
	curr_data = subset(curr_data, select = c(ClimateScenario, 
		                                     ClimateGroup, CountyGroup, 
		                                     CumDDinF, day, month, year,
		                                     latitude, longitude))
	saveRDS(curr_data, paste0(output_dir, "columns_" ,output_name))
    # pick the months of April, June and August
	curr_data = curr_data[curr_data$month %in% c(4, 6, 8)]
    saveRDS(curr_data, paste0(output_dir, "months_" ,output_name))
	
	# pick the first days of the months
	curr_data = curr_data[curr_data$day == 1]

	curr_data$latitude = as.character(curr_data$latitude)
	curr_data$longitude = as.character(curr_data$longitude)
	curr_data$location <- paste0(curr_data$longitude, curr_data$latitude)
	curr_data <- within(curr_data, remove(longitude, latitude))
    saveRDS(curr_data, paste0(output_dir, output_name))
}