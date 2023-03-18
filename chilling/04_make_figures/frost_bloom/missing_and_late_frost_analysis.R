#####################################
# frost data include 2358 locations in it.
# we need to pick up the selected limited locations
# to plot. There are 19 models in the frost data, however, there are 6
# in the bloom data

# Do we want to be consistent and have only 6 models 
# for both bloom and frost? or keep 19 models of frost?
# or we have to comoute blooms for all 19 models?

#
# We could/should create two sets of data for each (of the first two) 
# scenarios above or, we can take care of NAs
# -introduced to data by merging frost and bloom-
# in the plotting functions?
#####################################
rm(list=ls())
library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)
library(ggpubr)

options(digits=9)
options(digit=9)

source_path_1 = "/Users/hn/Documents/GitHub/Ag/chilling/chill_core.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/chilling/chill_plot_core.R"
source(source_path_1)
source(source_path_2)

param_dir <- "/Users/hn/Documents/GitHub/Ag/chilling/parameters/"
LOI <- data.table(read.csv(paste0(param_dir, "limited_locations.csv"), as.is=T))
#####################################################################################
#####################################################################################

bloom_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/bloom_new_params/for_TS_with_frost/"
bloom_LC <- readRDS(paste0(bloom_dir, "bloom_limited_50Percent.rds"))
bloom_LC$city <- as.character(bloom_LC$city)
bloom_LC <- within(bloom_LC, remove(location))
bloom_LC$time_period[bloom_LC$time_period == "1979-2015"] <- "observed"
bloom_LC$time_period[bloom_LC$time_period == "2026-2050"] <- "future"
bloom_LC$time_period[bloom_LC$time_period == "2051-2075"] <- "future"
bloom_LC$time_period[bloom_LC$time_period == "2076-2095"] <- "future"

# compute medians per location, time_periods
dues <- c("Feb") # "Dec", "Jan",
due <- "Feb"

######## Read bloom to filter the frost data by model
all_models <- c("Observed", "BNU-ESM", "CanESM2", 
                "GFDL-ESM2G", "bcc-csm1-1-m", "CNRM-CM5",
                "GFDL-ESM2M")

future_models <- c("BNU-ESM", "CanESM2", "GFDL-ESM2G", 
                   "bcc-csm1-1-m", "CNRM-CM5", "GFDL-ESM2M")
hist_model <- "Observed"
cities <- c("Hood River", "Walla Walla", "Richland", "Yakima", "Wenatchee", "Omak")
emissions <- c("RCP 4.5", "RCP 8.5")
apple_types <- c("Cripps Pink", "Gala", "Red Deli")
obs_years <- c(1979:2015)
future_years <- c(2026:2095)
missing_observed_dt <- CJ(obs_years, cities, emissions, hist_model)
missing_future_dt <- CJ(future_years, cities, emissions, future_models)
missing_observed_dt$time_period <- "observed"
missing_future_dt$time_period <- "future"

missing_test_dt <- rbind(missing_future_dt, missing_observed_dt)
setnames(missing_test_dt, old=c("V1", "V2", "V3", "V4"), 
         new=c("year", "city", "emission", "model"))

frost_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/"
frost_dir <- paste0(frost_dir, due, "/")
plot_dir <- paste0(frost_dir, "bloom_frost_in_one/")

first_frost <- data.table(readRDS(paste0(frost_dir, "first_frost_till_", due, ".rds")))
first_frost <- pick_single_cities_by_location(dt=first_frost, city_info=LOI)
first_frost <- within(first_frost, remove(location, month, day, tmin))
first_frost <- first_frost %>% filter(city %in% bloom_LC$city)

first_frost <- first_frost %>% 
             filter(time_period != "1950-2005" & 
                    model %in% all_models & 
                    time_period != "2006-2025" ) %>% 
             data.table()

first_frost <- first_frost %>% filter(year <= 2095) %>% data.table()

# change the time periods to observed and future so we can drop
# the line connecting 2015 to 2026
first_frost$time_period[first_frost$time_period == "1979-2015"] <- "observed"
first_frost$time_period[first_frost$time_period == "2076-2099"] <- "future"
first_frost$time_period[first_frost$time_period == "2026-2050"] <- "future"
first_frost$time_period[first_frost$time_period == "2051-2075"] <- "future"
first_frost <- within(first_frost, remove(chill_dayofyear))

merged_4_test <- merge(missing_test_dt, first_frost, all.x=TRUE)
missing_table <- merged_4_test %>% filter_all(any_vars(is.na(.))) %>% data.table()
missing_table <- missing_table[order(city, year),]

first_frost_observed <- first_frost %>% 
                        filter(time_period == "observed") %>% 
                        data.table()

first_frost_future <- first_frost %>% 
                      filter(time_period == "future") %>% 
                      data.table()

first_frost_future_late_frost_45 <- first_frost_future %>% 
                                    filter(extended_DoY>=365 & emission=="RCP 4.5") %>% 
                                    data.table()

first_frost_future_late_frost_85 <- first_frost_future %>% 
                                    filter(extended_DoY>=365 & emission=="RCP 8.5") %>% 
                                    data.table()
