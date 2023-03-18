####### prepare data for the 3rd-try plots!

# here we read the data in the three days April 1, June 1 and Aug 1st.
# we separate each day and each model. 
# for each (location, model), we do a regression 
# between x=years and y=cumDDinF. We find the slope of the line.
# and the slope of the line is saved in a data frame.
# So, for each climate scenario, 
#
#
#
#

rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)

part_1 = "/Users/hn/Desktop/Desktop/Kirti/check_point/"
part_2 = "my_aeolus_2015/all_local/pest_control/3rd_try_data/"
data_dir = paste0(part_1, part_2)

name_prefix = "three_days_"
models = c("45", "85")

climate_scenarios = c("historical", "bcc-csm1-1-m", 
	                  "BNU-ESM", "CanESM2", "CNRM-CM5",
	                  "GFDL-ESM2G", "GFDL-ESM2M")
for (model in models){
	curr_data = data.table(readRDS(paste0(data_dir, name_prefix, model, ".rds")))
	months = c(4, 6, 8)

	april_data  = curr_data[curr_data$month == 4]
	june_data   = curr_data[curr_data$month == 6]
	august_data = curr_data[curr_data$month == 8]
	locations = unique(curr_data$location)
	rm(curr_data)

    # initialize data tables for slopes with -999.
	april_slopes <- setNames(data.table(matrix(nrow = 295, ncol = 9)), 
		                     c("location", climate_scenarios, "month"))
	april_slopes[is.na(april_slopes)] <- (-999)
	april_slopes$location <- locations

	june_slopes <- april_slopes
	august_slopes <- april_slopes

    april_slopes <- april_slopes[, month:=as.character(month)]
    june_slopes <- june_slopes[, month:=as.character(month)]
    august_slopes <- august_slopes[, month:=as.character(month)]
    
	april_slopes$month = "april"
	june_slopes$month = "june"
	august_slopes$month = "august"

	all_slopes <- setNames(data.table(matrix(nrow = 0, ncol = 9)), 
		                   c("location", climate_scenarios, "month"))

	for (curr_month in months){
		if (curr_month==4){
			curr_data  = april_data
			curr_slopes = april_slopes
			} else if(curr_month==6){
				curr_data  = june_data
				curr_slopes = june_slopes
			} else if (curr_month==8){
				curr_data  = august_data
				curr_slopes = august_slopes
			}
		for (curr_location in locations){
			data_loc = curr_data[curr_data$location == curr_location]
			for (climate_scenario in climate_scenarios){
				data = data_loc[data_loc$ClimateScenario == climate_scenario]
			    linearMod = lm(CumDDinF ~ year, data=data)
			    slope = summary(linearMod)$coefficients[2, 1]
			    curr_slopes[curr_slopes$location == curr_location, climate_scenario] = slope
			}
		}
		all_slopes = rbind(all_slopes, curr_slopes)
	}
	output_name = paste0("slopes_", model, ".rds")
	saveRDS(all_slopes, paste0(data_dir, output_name))
}
