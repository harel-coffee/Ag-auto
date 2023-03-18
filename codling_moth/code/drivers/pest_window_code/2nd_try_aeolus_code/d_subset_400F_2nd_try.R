#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(dplyr)
# library(tidyverse)

data_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/"
output_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
name_pref = "combined_CMPOP_rcp"
models = c("45.rds", "85.rds")

for (model in models){
	curr_data = data.table(readRDS(paste0(data_dir, name_pref, model)))
	curr_data = subset(curr_data, select = c(ClimateScenario, 
		                                     ClimateGroup, CountyGroup, 
		                                     CumDDinF, dayofyear, year,
		                                     PercLarvaGen1,
		                                     latitude, longitude))
	curr_data$latitude = as.character(curr_data$latitude)
	curr_data$longitude = as.character(curr_data$longitude)
	curr_data$location <- paste0(curr_data$longitude, curr_data$latitude)
	curr_data <- within(curr_data, remove(longitude, latitude))
	# add the row numbers as a new column
	# so we would find the rows that correspond to 
	# 14 days after the 400F day!
	# curr_data <- tibble::rowid_to_column(curr_data, "ID")
    # The line below does the job of the line above.
    # in case the line above does not work with the 
    # stupi libraries of Aeolus.
	curr_data$ID <- seq.int(nrow(curr_data))
	
    output_name = paste0("400F_2nd_", name_pref, model)
    saveRDS(curr_data, paste0(output_dir, output_name))
}