rm(list=ls())

library(data.table)
library(dplyr)
library(ggpubr)
library(ggplot2)

options(digits=9)
options(digit=9)
############################################################
###
###             local computer source
###
############################################################
bloom_source_dir <- "/Users/hn/Documents/00_GitHub/Ag/Bloom/"
chill_source_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/"

bloom_core_source      <- paste0(bloom_source_dir, "bloom_core.R")
bloom_plot_core_source <- paste0(bloom_source_dir, "bloom_plot_core.R")
chill_core_source      <- paste0(chill_source_dir, "chill_core.R")

source(bloom_core_source)
source(bloom_plot_core_source)
source(chill_core_source)

#############################################################
###
### 
###
#############################################################

in_dir <- "/Users/hn/Documents/01_research_data/bloom/"
param_dir <- paste0(bloom_source_dir, "parameters/")

plot_base_dir <- "/Users/hn/Documents/01_research_data/Ag_Papers/Chill_Paper/01_Springer_1/"
daily_vertDD_dir <- "/Users/hn/Documents/01_research_data/bloom_4_chill_paper_trigger/lowVariety/"

#############################################################
###
###               Read data off the disk
###
#############################################################

########
######## read parameters
########

limited_locations <- read.csv(file = paste0(param_dir, "limited_locations.csv"), 
                              header=TRUE, as.is=TRUE)

limited_locations$location <- paste0(limited_locations$lat, "_", limited_locations$long)
limited_locations <- within(limited_locations, remove(lat, long))

chill_doy_map <- read.csv(paste0(param_dir, "/chill_DoY_map.csv"), as.is=TRUE)

########
######## daily CP data
########
daily_vertDD <- readRDS(paste0(daily_vertDD_dir, "triggerBased_vertDD_lowVariety.rds"))

needed_cols <- c("location", "emission", "time_period", "chill_dayofyear", "chill_season")
daily_vertDD   <- subset(daily_vertDD, select = c(needed_cols, "vert_Cum_dd"))

#____________________________________________________________
#
# pick up observed and 2026-2099 time period
#
#____________________________________________________________

daily_vertDD[daily_vertDD$time_period == "1979-2016", "time_period"] <- "Observed"
daily_vertDD <- pick_obs_and_F(daily_vertDD)

#############################################################
#
#              clean up and prepare each data table
#
#############################################################

daily_vertDD <- dplyr::left_join(x = daily_vertDD, 
                                 y = limited_locations, 
                                 by = "location")

daily_vertDD <- within(daily_vertDD, remove("location"))
setnames(daily_vertDD, old=c("city"), new=c("location"))

################################################

ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")

daily_vertDD$location <- factor(daily_vertDD$location, levels = ict, order=TRUE)

daily_vertDD$emission <- factor(daily_vertDD$emission, 
                                levels = c("Observed", "RCP 4.5", "RCP 8.5"), 
                                order = TRUE)


daily_vertDD[daily_vertDD$time_period == "Observed", "time_period"] <- "Historical"

tpo <- c("Historical", "2026-2050", "2051-2075", "2076-2099")
daily_vertDD$time_period <- factor(daily_vertDD$time_period, 
                                   levels = tpo, 
                                   order = TRUE)

##################################

daily_vertDD_obs <- daily_vertDD %>% filter(time_period == "Historical") %>% data.table()

daily_vertDD_obs45 <- daily_vertDD_obs
daily_vertDD_obs85 <- daily_vertDD_obs

daily_vertDD_obs45$emission <- "RCP 4.5"
daily_vertDD_obs85$emission <- "RCP 8.5"

daily_vertDD_RCP85 <- daily_vertDD %>% filter(emission == "RCP 8.5") %>% data.table()
daily_vertDD_RCP45 <- daily_vertDD %>% filter(emission == "RCP 4.5") %>% data.table()

daily_vertDD_RCP85 <- rbind(daily_vertDD_RCP85, daily_vertDD_obs85)
daily_vertDD_RCP45 <- rbind(daily_vertDD_RCP45, daily_vertDD_obs45)

daily_vertDD <- rbind(daily_vertDD_RCP85, daily_vertDD_RCP45)
#____________________________________________________________
#
# Pick the days between Jan 15 and March 15
#

daily_vertDD <- daily_vertDD %>% 
                filter(chill_dayofyear >= 137) %>% 
                filter(chill_dayofyear <= 196) %>% 
                data.table()

daily_vertDD <- daily_vertDD %>% 
                group_by(chill_dayofyear, location, time_period, emission) %>% 
                summarise(median_over_model_yr = median(vert_Cum_dd)) %>%
                data.table()
#
# Group by (location, emission, time period) and fit linear lines
#

A <- daily_vertDD
A <- A[, list(intercept = coef(lm(median_over_model_yr ~ chill_dayofyear))[1], 
              coef      = coef(lm(median_over_model_yr ~ chill_dayofyear))[2]),
         by = c("emission", "time_period", "location")]

saveRDS(A, paste0(daily_vertDD_dir, "slopes_maxDD_CalendarBased_LV.rds"))




