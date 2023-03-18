.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(plyr)
library(tidyverse)


data_dir = "/data/hydro/users/Hossein/temp_gdd/"
out_dir = file.path(data_dir, "cleaned_data_for_plot/")

file_names = c("modeled_rcp45.rds", "modeled_rcp85.rds", 
	           "modeled_historical.rds", "observed.rds")

################################################################################
#######
####### Clean the data:
#######                 Take columns you want
#######                 Get rid of 2006-2024 years (in rcp45, rcp85 and modeled_historical)
#######                 Rename climateScenario to model
#######                 Add column scneario (rcp45, rcp85, modeled_hist, observed)
#######                 
################################################################################
clean <- function(data, scenario){
	needed_colomns = c("year", "tmean", "Cum_dd", "ClimateGroup", "ClimateScenario")
	
	# grab needed cols
	data = subset(data, select=needed_colomns)
    
    # drop 2006-2024 years
    data = filter(data, year <=2005 | year >= 2025)

    data$ClimateGroup[data$year >= 1979 & data$year <= 2005] <- "Historical"
    data$ClimateGroup[data$year > 2025 & data$year <= 2055] <- "2040's"
    data$ClimateGroup[data$year > 2045 & data$year <= 2075] <- "2060's"
    data$ClimateGroup[data$year > 2065 & data$year <= 2095] <- "2080's"
    
    # rename col names
    colnames(data)[colnames(data) == 'ClimateScenario'] <- 'model'
    data$scenario = scenario
     
    # drop the year columnn
    data = data.table(data)
    data[, year:=NULL]
    return (data)
}

###########
########### Read, clean, bind data
###########
print
rcp45 <- data.table(readRDS(paste0(data_dir, file_names[1])))
rcp45 <- clean(data=rcp45, scenario="rcp45")
print("colnames(rcp45)")
print(colnames(rcp45))

rcp45_stat <- rcp45[, list(mean_tmean = mean(tmean), 
                     mean_cumm_dd = mean(Cum_dd)) , 
                     by = c("ClimateGroup", "model", "scenario") ]
rm(rcp45)
print ("---------------------------------------")
rcp85 <- data.table(readRDS(paste0(data_dir, file_names[2])))
rcp85 <- clean(data=rcp85, scenario="rcp85")
print("colnames(rcp85)")
print(colnames(rcp85))

rcp85_stat <- rcp85[, list(mean_tmean = mean(tmean), 
                     mean_cumm_dd = mean(Cum_dd)) , 
                     by = c("ClimateGroup", "model", "scenario") ]
rm(rcp85)
print ("---------------------------------------")
modeled_hist <- data.table(readRDS(paste0(data_dir, file_names[3])))
modeled_hist <- clean(data=modeled_hist, scenario="modeled_hist")
print("colnames(modeled_hist)")
print(colnames(modeled_hist))
modeled_hist_stat <- modeled_hist[, list(mean_tmean = mean(tmean), 
                                    mean_cumm_dd = mean(Cum_dd)) , 
                                    by = c("ClimateGroup", "model", "scenario") ]
rm(modeled_hist)
print ("---------------------------------------")

observed <- data.table(readRDS(paste0(data_dir, file_names[4])))
observed <- clean(data=observed, scenario="observed")
print("colnames(observed)")
print(colnames(observed))
observed_stat <- observed[, list(mean_tmean = mean(tmean), 
                            mean_cumm_dd = mean(Cum_dd)) , 
                            by = c("ClimateGroup", "model", "scenario") ]
rm(observed)
print ("---------------------------------------")
modeled_45_85_hist <- rbind(modeled_hist_stat, rcp45_stat, rcp85_stat)
obseved_modeled_hist <- rbind(observed_stat, modeled_hist_stat)

saveRDS(modeled_45_85_hist, paste0(out_dir, "obseved_modeled_hist.rds"))
saveRDS(obseved_modeled_hist, paste0(out_dir, "obseved_modeled_hist.rds"))

# modeled_45_85_hist = rbind(modeled_hist, rcp45, rcp85)
# print("colnames(modeled_45_85_hist)")
# print(colnames(modeled_45_85_hist))
# print ("---------------------------------------")

# obseved_modeled_hist = rbind(observed, modeled_hist)
# print("colnames(obseved_modeled_hist)")
# print(colnames(obseved_modeled_hist))
# print ("---------------------------------------")

# rm(observed, modeled_hist, rcp85, rcp45)

###########
########### Generate stats
###########

# data %>% 
# group_by("ClimateGroup", "model", "scenario") %>% 
# summarize(mean_tmean=mean(tmean), mean_Cum_dd=mean(Cum_dd))

# obseved_modeled_hist <- obseved_modeled_hist[, list(mean_tmean = mean(tmean), 
#                                              mean_cumm_dd = mean(Cum_dd)) , 
#                                              by = c("ClimateGroup", "model", "scenario") ]

# modeled_45_85_hist <- modeled_45_85_hist[, list(mean_tmean = mean(tmean), 
#                                            mean_cumm_dd = mean(Cum_dd)) , 
#                                            by = c("ClimateGroup", "model", "scenario") ]

# saveRDS(obseved_modeled_hist, paste0(out_dir, "obseved_modeled_hist.rds"))
# saveRDS(modeled_45_85_hist, paste0(out_dir, "modeled_45_85_hist.rds"))



