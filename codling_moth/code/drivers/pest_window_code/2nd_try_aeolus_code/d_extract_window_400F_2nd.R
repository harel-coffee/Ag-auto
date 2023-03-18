#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(dplyr)

data_dir  = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
output_dir= "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
name_pref = "400F_2nd_combined_CMPOP_rcp"

models <- c("45.rds", "85.rds")
for (model in models){
	curr_data = data.table(readRDS(paste0(data_dir, name_pref, model)))	
    #####################################################################
    ##########
    ########## Just take the 1st and the last day of window information.
    ##########
    ######################################################################
    start_400 <- curr_data %>% group_by(location, ClimateScenario, 
                                        ClimateGroup, CountyGroup, 
                                        year) %>% arrange(abs(CumDDinF - 400)) %>% slice(1)
    
    start_ID = start_400$ID 
    end_400_IDs <- start_ID + 13
	end_400  <- curr_data[end_400_IDs]
    
    # change column names
    colnames(start_400)[colnames(start_400)=="ID"] <- "start_ID"
    colnames(start_400)[colnames(start_400)=="PercLarvaGen1"] <- "PercLarvaGen1_start"
    colnames(start_400)[colnames(start_400)=="CumDDinF"] <- "CumDDinF_start"
    colnames(start_400)[colnames(start_400)=="dayofyear"] <- "dayofyear_start"
    
    colnames(end_400)[colnames(end_400)=="ID"] <- "end_ID"
    colnames(end_400)[colnames(end_400)=="PercLarvaGen1"] <- "PercLarvaGen1_end"
    colnames(end_400)[colnames(end_400)=="CumDDinF"] <- "CumDDinF_end"
    colnames(end_400)[colnames(end_400)=="dayofyear"] <- "dayofyear_end"

    start_400 <- as.data.frame(start_400)
    end_400 <- as.data.frame(end_400)

    start_end <- merge(start_400, end_400, by=c("ClimateScenario", "ClimateGroup", 
    	                                        "CountyGroup", "year", "location"))

    setcolorder(start_end, c("location", "ClimateScenario", 
    	                     "ClimateGroup", "CountyGroup", 
		    	             "year", "dayofyear_start", "dayofyear_end", 
		    	             "start_ID", "end_ID", 
		    	             "CumDDinF_start", "CumDDinF_end",
                             "PercLarvaGen1_start",
                             "PercLarvaGen1_end"))

    start_end$temp_delta <- start_end$CumDDinF_end - start_end$CumDDinF_start
    start_end$pop_delta <- start_end$PercLarvaGen1_end - start_end$PercLarvaGen1_start
    start_end$CountyGroup = as.character(start_end$CountyGroup)
    start_end[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
    start_end[CountyGroup == 2]$CountyGroup = 'Warmer Areas'

    saveRDS(start_end, paste0(output_dir, "start_end_", model))

    expanded_ID = rep(0L, 14*length(start_ID))
    for (ii in 1:length(start_ID)){
        expanded_ID[((ii-1)*14+1) : (ii*14)] = start_ID[ii]:(start_ID[ii]+13)

    }
    curr_data = curr_data[expanded_ID]

    curr_data$CountyGroup = as.character(curr_data$CountyGroup)
    curr_data[CountyGroup == 1]$CountyGroup = 'Cooler Areas'
    curr_data[CountyGroup == 2]$CountyGroup = 'Warmer Areas'
    saveRDS(curr_data, paste0(output_dir, "all_14_days_window_", model))

    # start_end <- subset(start_end, select=c("ClimateGroup", "CountyGroup", 
    #                                         "CumDDinF_start", "CumDDinF_end",
    #                                         "temp_delta"))
    # saveRDS(start_end, paste0(output_dir, "window_400F_sub_", model))

    # start_end <- melt(start_end, id=c("ClimateGroup", "CountyGroup"))
    # saveRDS(start_end, paste0(output_dir, "window_400F_sub_melt_", model))
    #####################################################################
    ##########
    ########## The whole window information
    ##########
    ######################################################################
   
    
}


