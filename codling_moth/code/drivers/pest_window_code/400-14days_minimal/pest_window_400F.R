#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(dplyr)

data_dir  = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
output_dir= "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
name_pref = "400F_combined_CMPOP_rcp"

models <- c("45.rds", "85.rds")
for (model in models){
	curr_data = data.table(readRDS(paste0(data_dir, name_pref, model)))
	curr_data$latitude = as.character(curr_data$latitude)
	curr_data$longitude = as.character(curr_data$longitude)
	curr_data$location <- paste0(curr_data$longitude, curr_data$latitude)
	curr_data <- within(curr_data, remove(longitude, latitude))
	
	start_400 <- curr_data %>% group_by(location, ClimateScenario, 
 		                                ClimateGroup, CountyGroup, 
 		                                year) %>% arrange(abs(CumDDinF - 400)) %>% slice(1)
	
	end_400_IDs <- start_400$ID + 13
	end_400  <- curr_data[end_400_IDs]
    
    # change column names
    colnames(start_400)[colnames(start_400)=="ID"] <- "start_ID"
    colnames(start_400)[colnames(start_400)=="CumDDinF"] <- "CumDDinF_start"
    colnames(start_400)[colnames(start_400)=="dayofyear"] <- "dayofyear_start"
    
    colnames(end_400)[colnames(end_400)=="ID"] <- "end_ID"
    colnames(end_400)[colnames(end_400)=="CumDDinF"] <- "CumDDinF_end"
    colnames(end_400)[colnames(end_400)=="dayofyear"] <- "dayofyear_end"

    start_400 <- data.table(start_400)
    curr_data <- merge(start_400, end_400, by=c("ClimateScenario", "ClimateGroup", 
    	                                "CountyGroup", "year", "location"))

    setcolorder(curr_data, c("location", "ClimateScenario", 
    	                     "ClimateGroup", "CountyGroup", 
		    	             "year", "dayofyear_start", "dayofyear_end", 
		    	             "start_ID", "end_ID", 
		    	             "CumDDinF_start", "CumDDinF_end"))

    curr_data$temp_delta <- curr_data$CumDDinF_end - curr_data$CumDDinF_start
    curr_data$CountyGroup = as.character(curr_data$CountyGroup)
    curr_data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
    curr_data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

    saveRDS(curr_data, paste0(output_dir, "window_400F_full_", model))
    
    curr_data <- subset(curr_data, select=c("ClimateGroup", "CountyGroup", 
    	                                    "CumDDinF_start", "CumDDinF_end",
    	                                    "temp_delta"))
    saveRDS(curr_data, paste0(output_dir, "window_400F_sub_", model))

    curr_data <- melt(curr_data, id=c("ClimateGroup", "CountyGroup"))
    saveRDS(curr_data, paste0(output_dir, "window_400F_sub_melt_", model))
}


