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

start_time <- Sys.time()

param_dir <- "/Users/hn/Documents/GitHub/Ag/chilling/parameters/"
LOI <- data.table(read.csv(paste0(param_dir, "limited_locations.csv"), as.is=T))
chill_doy_map <- read.csv("/Users/hn/Documents/GitHub/Ag/chill_DoY_map.csv", as.is=TRUE)
#####################################################################################
######## Read bloom to filter the frost data by model
all_models_Observed <- c("Observed", "BNU-ESM", "CanESM2", 
                         "GFDL-ESM2G", "bcc-csm1-1-m", "CNRM-CM5",
                         "GFDL-ESM2M")

all_models_observed <- c("observed", "BNU-ESM", "CanESM2", 
                         "GFDL-ESM2G", "bcc-csm1-1-m", "CNRM-CM5",
                         "GFDL-ESM2M")
six_models <- c("BNU-ESM", "CanESM2", "GFDL-ESM2G", 
                "bcc-csm1-1-m", "CNRM-CM5", "GFDL-ESM2M")

cities <- c("Hood River", "Walla Walla", "Richland", "Yakima", "Wenatchee", "Omak")
emissions <- c("RCP 4.5", "RCP 8.5")
apple_types <- c("Cripps Pink", "Gala", "Red Deli") # 

bloom_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/bloom_new_params/for_TS_with_frost/"
bloom_LC <- readRDS(paste0(bloom_dir, "bloom_cloudy_50Percent.rds"))
bloom_LC$city <- as.character(bloom_LC$city)
bloom_LC$time_period <- as.character(bloom_LC$time_period)
########## Add/Correct chill calendar type thing
# reduce the years of days in {1, 2, ..., 243}
# add 122 to the days between 1 and 243.
bloom_LC$year[bloom_LC$dayofyear %in% c(1:243)] <- bloom_LC$year[bloom_LC$dayofyear %in% c(1:243)] - 1
bloom_LC$dayofyear[bloom_LC$dayofyear %in% c(1:243)] <- bloom_LC$dayofyear[bloom_LC$dayofyear %in% c(1:243)] + 122
bloom_LC <- bloom_LC %>% filter(year >= 2070)
bloom_LC <- within(bloom_LC, remove(time_period))

# apple, Cherry, Pear; Cherry 14 days shift, Pear 7 days shift
fruit_type <- "apple"
remove_NA <- "no"

# shift the bloom days
if (fruit_type == "Cherry"){
   bloom_LC$dayofyear <- bloom_LC$dayofyear - 14
   bloom_LC <- bloom_LC %>% filter(dayofyear >= 0)
   # apple_types <- c("Cripps Pink") # This is done just for purpose of for loop
   } else if (fruit_type == "Pear"){
    bloom_LC$dayofyear <- bloom_LC$dayofyear-7
    bloom_LC <- bloom_LC %>% filter(dayofyear>=0)
    # apple_types <- c("Cripps Pink") # This is done just for purpose of for loop
}

setcolorder(bloom_LC, c("city", "year", "emission", "apple_type", "dayofyear", "model"))
setnames(bloom_LC, old=c("dayofyear"), new=c("fifty_perc_DoY"))
bloom_LC <- data.table(bloom_LC)

# obs_years <- c(1979:2015)
# future_years <- c(2026:2095)
# complete_1st_frost <- CJ(years, cities, six_models, emission)
# future <- CJ(future_years, cities, emissions, six_models)

####################################################################################
#
#         Threshold stuff
#
chill_limited <- read.csv(paste0(param_dir, "/limited_locations.csv"), as.is=TRUE)
chill_limited$location <- paste0(chill_limited$lat, "_", chill_limited$long)
suppressWarnings({ chill_limited <- within(chill_limited, remove(lat, long))})
chill_limited <- chill_limited %>% filter(city %in% cities) %>% data.table()

thresh_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/"
thresh_dt <- readRDS(paste0(thresh_dir, "/sept_summary_comp.rds"))
thresh_dt <- thresh_dt %>% filter(year >= 2070) %>% data.table()
thresh_dt <- thresh_dt %>% 
             filter(location %in% chill_limited$location # & model %in% all_models_observed
                    ) %>% 
             data.table()

thresh_dt <- merge(thresh_dt, chill_limited, all.x=TRUE)
suppressWarnings({ thresh_dt <- within(thresh_dt, 
                   remove(location, start, sum_J1, sum_F1, sum_M1,
                          sum_A1, sum, thresh_20, thresh_25, thresh_30,
                          thresh_35, thresh_40, thresh_45, thresh_55,
                          thresh_60, thresh_65))})

thresh_dt <- remove_modeled_historical_add_time_period(thresh_dt)
thresh_dt <- remove_F0(thresh_dt)
thresh_dt <- thresh_dt %>% filter(year <= 2094)%>% data.table()
thresh_dt <- within(thresh_dt, remove(time_period))
####################################################################################
                          #                         #
                          #                         #
                          #                         #
####################################################################################
time_window <- c(paste0(min(thresh_dt$year), "_", max(thresh_dt$year)))
# output <- CJ(cities, emissions, time_window)
# setnames(output, old=c(paste0("V", c(1:3))), 
#                  new=c("city", "emission", "time_window"))

output <- setNames(data.table(matrix(nrow=0, ncol=7)), 
                  c("city", "model", "emission", "thresh", 
                    "time_window", "cross_count", "fruit_type"))

####################################################################################
# compute medians per location, time_periods
dues <- c("Feb") # "Dec", "Jan",
due <- "Feb"

for (due in dues){
  ct <- cities[1]
  em <- emissions[1]
  app_tp <- apple_types[1]
  limits <- seq(70, 75, 5)
  lim <- 75

  for (lim in limits){
    col_name <- paste0("thresh_", lim)
    for (ct in cities){
      for (em in emissions){
        ##### Threshold care
        curr_thresh <- subset(thresh_dt, select=c("city", "year", "chill_season", 
                                                  "model", "emission", col_name))
        curr_thresh <- curr_thresh %>% 
                       filter(city==ct & emission==em) %>% 
                       data.table()

        ######################################################################
        #
        #        REMOVE NAs
        #
        if (remove_NA == "yes"){
          curr_thresh <- curr_thresh %>% 
                       filter(get(col_name)<365) %>% 
                       data.table()
        }
        ######################################################################
        for (app_tp in apple_types){
          curr_bloom <- bloom_LC %>% 
                        filter(city==ct & emission==em & 
                              apple_type == gsub("\ ", "_", tolower(app_tp))) %>% 
                        data.table()
          ########################################################################
          #                                                                      #
          # layover -- layover -- layover -- layover -- layover -- layover       #
          # organize for layover plot                                            #
          # layover -- layover -- layover -- layover -- layover -- layover       #
          #                                                                      #
          ########################################################################

          suppressWarnings({ curr_bloom <- within(curr_bloom, remove(apple_type))})
          suppressWarnings({ curr_thresh <- within(curr_thresh, remove(chill_season))})
          setcolorder(curr_bloom, c("city", "year",
                                    "model", "emission", "fifty_perc_DoY"))
          if (paste0("thresh_", lim) %in% colnames(curr_thresh)){
            setnames(curr_thresh, old=c(paste0("thresh_", lim)), new=c("thresh"))
          }
          
          merged_dt <- merge(curr_thresh, curr_bloom, all.x=TRUE)
          merged_dt <- na.omit(merged_dt)
          merged_dt$thresh <- as.numeric(merged_dt$thresh)
          merged_dt$fifty_perc_DoY <- as.numeric(merged_dt$fifty_perc_DoY)
          merged_dt$bloom_thresh_diff <- merged_dt$fifty_perc_DoY - merged_dt$thresh

          merged_dt <- merged_dt %>% 
                       filter(bloom_thresh_diff<=0) %>% 
                       data.table()
          
          merged_dt <- within(merged_dt, 
                              remove(year, bloom_thresh_diff, 
                                     fifty_perc_DoY, thresh))
          merged_dt[ , `:=`( cross_count = .N ), 
                        by = c("city", "model", "emission") ]
          merged_dt <- unique(merged_dt)
          merged_dt$thresh <- paste0("threshold: ", lim)
          merged_dt$time_window <- time_window
          merged_dt$fruit_type <- paste0(fruit_type, ": ", app_tp)

          # new_row <- c(ct, em, paste0("thresh_", lim), time_window, 
          #             sum(merged_dt$bloom_thresh_diff < 0), 
          #             paste0(fruit_type, " - ", app_tp))

          # new_row <- setDT(as.list(new_row))[]
          # colnames(new_row) <- colnames(output)
          if (nrow(merged_dt)>0){
            output <- rbind(output, merged_dt)
          }
        }
      }
    }
  }
  out_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/frost_bloom/"
  out_dir <- paste0(out_dir, due, "/")
  write.table(output, 
              file = paste0(out_dir, fruit_type, "_crossovers.csv"), 
              row.names=FALSE, na="", col.names=TRUE, sep=",")
}



print(Sys.time() - start_time)