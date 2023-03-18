#####
#####  HN. Never Ending project.
#####  March 11, 2021. 
#####  
#####  We want to find the DoY at which 37.5CP was accumulated (75% of 50CP)
#####  and use this day as a trigger for heat accumulation.
#####  We will use it to set vertDD in bloom data (01 step from binary to bloom data)
#####  equal to zero. And then computed cumulated vertical DD.
#####  for the chill paper!!!!!
#####
#####
rm(list=ls())
library(data.table)
library(dplyr)

options(digits=9)
options(digit=9)

#--------------------------------------------------------------------------------------------

data_dir <- "/Users/hn/Documents/01_research_data/chilling/01_data/02_02_fot_heatTrigger_lowVariety/"
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"

out_dir <- "/Users/hn/Documents/01_research_data/bloom_4_chill_paper_trigger/lowVariety/"

#--------------------------------------------------------------------------------------------

locations <- read.csv(paste0(param_dir, "4_locations.csv"), as.is=T) %>% data.table()
sept_summary_comp <- data.table(readRDS(paste0(data_dir, "summary_comp_springer_lowVariety.rds")))

#--------------------------------------------------------------------------------------------
#
#  filter the locations
#
locations$location <- paste0(locations$lat, "_", locations$long)

sept_summary_comp <- sept_summary_comp %>%
                     filter(location %in% locations$location) %>%
                     data.table()

unique(sept_summary_comp$location)

#--------------------------------------------------------------------------------------------
#
#  subset the needed columns
#  
#  We are keeping the thresh_37half since 75% of min_cp_cripps_pink = 55 is 37.5
#  73.3 came from the Australian paper.
#  
col_names <- c("location", "chill_season", "year",  "model", 
               "emission", "time_period", "thresh_37half")

sept_summary_comp <- subset(sept_summary_comp, select = col_names)

#--------------------------------------------------------------------------------------------
#
#   chill_season_1949_1950 includes the 
#   months 9-12 of 1949 and months 1-8 of 1950
#
#--------------------------------------------------------------------------------------------

sept_summary_comp_F <- sept_summary_comp %>% 
                       filter(time_period %in% c("2026-2050", "2051-2075", "2076-2099")) %>% 
                       data.table()

sept_summary_comp_obs <- sept_summary_comp %>% 
                         filter(model %in% c("Observed", "observed")) %>% 
                         data.table()

sept_summary_comp_obs$emission <- "Observed"
sept_summary_comp_obs$model <- "Observed"
sept_summary_comp_obs$time_period <- "Historical"


sept_summary_comp_F[sept_summary_comp_F$emission=="rcp45"]$emission <- "RCP 4.5"
sept_summary_comp_F[sept_summary_comp_F$emission=="rcp85"]$emission <- "RCP 8.5"


sept_summary_comp <- rbind(sept_summary_comp_obs, sept_summary_comp_F)

saveRDS(sept_summary_comp, paste0(out_dir, "heatTriggers_sept_summary_comp_lowVariety.rds"))




