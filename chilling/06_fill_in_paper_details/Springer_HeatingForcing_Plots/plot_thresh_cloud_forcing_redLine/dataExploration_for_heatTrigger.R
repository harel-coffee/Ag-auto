#################################################
##
## This scrip is written to see who the data 
## look like. The data that is produced by d_observed_heattrigger(.)
## function. 
##
## I need to see what DoY the 75% of minimum CP is reached
## to plot it in the (cloudy) threshold TS plots.
## I am finding the DoY cut off point based on observed data.
##
## This is being done before Springer submission.
##
##  Feb. 26, 2021
##  Hossein Noorazar
##
##

library(data.table)
library(dplyr)
library(ggpubr)

options(digits=9)
options(digit=9)

#################################################

data_dir <- paste0("/Users/hn/Documents/01_research_data/chilling/", 
                   "01_data/02_01_with_May_4HeatTrigger/sept/")
	

####
####
####   Summary_comp files are binded version of individual "*_summary_*" files.
####
####   The "*_stats_*" files in "stats" folders have already taken median of 
####   the "threshold" columns. So, the followings are quivalent: 
####                         the part that is commented out and _stats_ file.
####   
####
####
# 
#  First part of equivalency
#
# f_name = "heatTrigger_summary_obs_hist.txt"
# heatTrig_summary_obs <- read.table(paste0(data_dir, f_name), header = TRUE) %>% 
#                         data.table()
# heatTrig_summary_obs <- within(heatTrig_summary_obs, remove(".id"))
# heatTrig_summary_obs$location <- paste0(heatTrig_summary_obs$lat, "_", heatTrig_summary_obs$long)
# heatTrig_summary_obs <- heatTrig_summary_obs %>%
#                         group_by(location, model, emission, time_period)%>% 
#                         summarise(median_heatTriggerThresh = median(heatTriggerThresh)) %>%
#                         data.table()


f_name = "stats_subfolder_heatTrigger_summary_stats_observed.txt"
heatTrig_summ_stats_obs <- read.table(paste0(data_dir, f_name), header = TRUE) %>% 
                           data.table()
heatTrig_summ_stats_obs$location <- paste0(heatTrig_summ_stats_obs$lat, "_", heatTrig_summ_stats_obs$long)
heatTrig_summ_stats_obs <- within(heatTrig_summ_stats_obs, remove(lat, long))










