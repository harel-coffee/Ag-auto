#####
#####  HN
#####  March 5, 2021
#####  
#####  We want to find the DoY at which 55CP was accumulated
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

data_dir <- "/Users/hn/Documents/01_research_data/chilling/01_data/02/"
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"

out_dir <- "/Users/hn/Documents/01_research_data/bloom_4_chill_paper_trigger/"

#--------------------------------------------------------------------------------------------

locations <- read.csv(paste0(param_dir, "4_locations.csv"), as.is=T) %>% data.table()
sept_summary_comp <- data.table(readRDS(paste0(data_dir, "sept_summary_comp.rds")))

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
#  We are keeping the thresh_55 since 75% of min_cp_cripps_pink = 73.3 is 54.975.
#  73.3 came from the Australian paper.
#  
col_names <- c("location", "chill_season", "year",  "model", 
               "emission", "time_period", "thresh_55")

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
                         filter(model %in% c("observed")) %>% 
                         data.table()

sept_summary_comp_obs$emission <- "Observed"
sept_summary_comp_obs$model <- "Observed"
sept_summary_comp_obs$time_period <- "Historical"


sept_summary_comp_F[sept_summary_comp_F$emission=="rcp45"]$emission <- "RCP 4.5"
sept_summary_comp_F[sept_summary_comp_F$emission=="rcp85"]$emission <- "RCP 8.5"


sept_summary_comp <- rbind(sept_summary_comp_obs, sept_summary_comp_F)

#
# sth in BNU model has gone wrong for Eugene at chill_season_2096_2097. 
# replace the zero with nearest neighbor. How the dell zero happened
# some how in that year/model 55CP was not reached? or sth got messed up?
#
# sept_summary_comp %>% filter(location == "44.03125_-123.09375" & emission == "RCP 8.5" & time_period == "2076-2099" & model == "BNU-ESM")
sept_summary_comp[sept_summary_comp$thresh_55==0, 'thresh_55'] <- 151

# dim(sept_summary_comp)
# sept_summary_comp <- sept_summary_comp %>% filter(thresh_55 != 0 ) %>% data.table()
# dim(sept_summary_comp)

saveRDS(sept_summary_comp, paste0(out_dir, "heatTriggers_sept_summary_comp.rds"))




