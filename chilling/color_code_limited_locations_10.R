
rm(list=ls())
.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(tidyr)
library(tidyverse)

options(digit=9)
options(digits=9)

source_path_1 = "/Users/hn/Documents/GitHub/Ag/chilling/4th_draft/chill_core.R"
source(source_path_1)

##########################################################################################
###                                                                                    ###
###                             Define Functions here                                  ###
###                                                                                    ###
##########################################################################################
define_path <- function(model_name){
  if (model_name == "dynamic"){
    in_dir <- paste0(main_in_dir, model_specific_dir_name[1])
    } else if (model == "utah"){
      in_dir <- paste0(main_in_dir, model_specific_dir_name[2])
  }
}

clean_process <- function(dt){
  dt <- subset(dt, select=c(chill_season,
                            sum_J1, sum_F1, sum_M1, sum_A1, 
                            city, scenario, model, year))
  
  dt <- dt %>% filter(year <= 2005 | year >= 2025)
  
  time_periods = c("Historical", "2025_2050", "2051_2075", "2076_2099")
  dt$time_period = 0L
  dt$time_period[dt$year <= 2005] <- time_periods[1]
  dt$time_period[dt$year >= 2025 & dt$year <= 2050] <- time_periods[2]
  dt$time_period[dt$year >  2050 & dt$year<=2075] <- time_periods[3]
  dt$time_period[dt$year >  2075] <- time_periods[4]
  dt$time_period = factor(dt$time_period, levels=time_periods, order=T)

  dt$scenario[dt$scenario == "rcp45"] <- "RCP 4.5"
  dt$scenario[dt$scenario == "rcp85"] <- "RCP 8.5"
  dt$scenario[dt$time_period == "Historical"] <- "Historical"

  jan_data <- subset(dt, select=c(sum_J1, scenario, model, time_period, chill_season, city)) %>% data.table()
  feb_data <- subset(dt, select=c(sum_F1, scenario, model, time_period, chill_season, city)) %>% data.table()
  mar_data <- subset(dt, select=c(sum_M1, scenario, model, time_period, chill_season, city)) %>% data.table()
  apr_data <- subset(dt, select=c(sum_A1, scenario, model, time_period, chill_season, city)) %>% data.table()
  return (list(jan_data, feb_data, mar_data, apr_data))
}

#############################################
###                                       ###
###                 Driver                ###
###                                       ###
#############################################

# main_in_dir = "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/non_overlapping/"
# model_names = c("dynamic") # , "utah"
# model_specific_dir_name = paste0(model_names, "_model_stats/")
# file_name = "summary_comp.rds"
# mdata <- data.table(readRDS(paste0(main_in_dir, model_specific_dir_name, file_name)))
# setnames(mdata, old=c("Chill_season"), new=c("chill_season"))
##########################################################################################

main_in_dir = "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/"
out_dir = main_in_dir
param_dir <- "/Users/hn/Documents/GitHub/Ag/chilling/parameters/"

####################################################################################
#
#      Pick single cities
#
limited_locations <- read.csv(paste0(param_dir, "limited_locations.csv"), header=T, as.is=T)
limited_locations$location <- paste0(limited_locations$lat, "_", limited_locations$long)
limited_locations <- within(limited_locations, remove(lat, long))

##########################################################################################

no_cities <- length(unique(limited_locations$city))
no_threshs <- 13

output_85 <- data.table(row_count = 1:(no_threshs*no_cities))
output_45 <- data.table(row_count = 1:(no_threshs*no_cities))

begins <- c("sept", "mid_sept", "oct", "mid_oct", "nov", "mid_nov")
for (begin in begins){

  mdata <- data.table(readRDS(paste0(main_in_dir, begin, "_summary_comp.rds")))
  mdata <- mdata %>% filter(model != "observed")
  
  mdata$location <- paste0(mdata$lat, "_", mdata$long)
  mdata <- mdata %>% 
           filter(location %in% limited_locations$location) 
           
  mdata <- dplyr::left_join(mdata, limited_locations)

  ####################################################################################  
  information <- clean_process(mdata)
 
  jan_data = information[[1]] 
  feb_data = information[[2]] 
  mar_data = information[[3]] 
  apr_data = information[[4]]
  rm(information, mdata)

  jan_result = count_years_threshs_met_limit_location(dataT = jan_data, due="Jan")
  feb_result = count_years_threshs_met_limit_location(dataT = feb_data, due="Feb")
  mar_result = count_years_threshs_met_limit_location(dataT = mar_data, due="Mar")
  apr_result = count_years_threshs_met_limit_location(dataT = apr_data, due="Apr")

  ######****************************
  ################################## JAN
  ######****************************

  quan_per_jan <- jan_result %>% 
                  group_by(time_period, city, scenario, thresh_range) %>% 
                  summarise(quan_25 = quantile(frac_passed, probs = 0.1)) %>% 
                  data.table()

  quan_per_jan_85 <- quan_per_jan  %>% 
                     filter(scenario == "RCP 8.5") %>% 
                     data.table()

  quan_per_jan_45 <- quan_per_jan  %>% 
                     filter(scenario == "RCP 4.5") %>%
                     data.table()

  quan_per_jan_85 <- quan_per_jan_85[order(time_period, thresh_range), ]
  quan_per_jan_45 <- quan_per_jan_45[order(time_period, thresh_range), ]

  quan_per_jan_85 <- quan_per_jan_85 %>% spread(time_period, quan_25)
  quan_per_jan_45 <- quan_per_jan_45 %>% spread(time_period, quan_25)

  start <- data.table(start=rep(begin, nrow(quan_per_jan_85)))
  end <- data.table(end=rep("Jan_1", nrow(quan_per_jan_85)))
  
  quan_per_jan_85 <- cbind(start, quan_per_jan_85)
  quan_per_jan_85 <- cbind(quan_per_jan_85, end)

  quan_per_jan_45 <- cbind(start, quan_per_jan_45)
  quan_per_jan_45 <- cbind(quan_per_jan_45, end)


  ######****************************
  ################################## FEB
  ######****************************
  quan_per_feb <- feb_result %>% 
                  group_by(time_period, city, scenario, thresh_range) %>% 
                  summarise(quan_25 = quantile(frac_passed, probs = 0.1)) %>% 
                  data.table()

  quan_per_feb_85 <- quan_per_feb  %>% 
                     filter(scenario == "RCP 8.5") %>% 
                     data.table()

  quan_per_feb_45 <- quan_per_feb  %>% 
                     filter(scenario == "RCP 4.5") %>%
                     data.table()

  quan_per_feb_85 <- quan_per_feb_85[order(time_period, thresh_range), ]
  quan_per_feb_45 <- quan_per_feb_45[order(time_period, thresh_range), ]

  quan_per_feb_85 <- quan_per_feb_85 %>% spread(time_period, quan_25)
  quan_per_feb_45 <- quan_per_feb_45 %>% spread(time_period, quan_25)

  start <- data.table(start=rep(begin, nrow(quan_per_jan_85)))
  end <- data.table(end=rep("Feb_1", nrow(quan_per_jan_85)))

  quan_per_feb_85 <- cbind(start, quan_per_feb_85)
  quan_per_feb_85 <- cbind(quan_per_feb_85, end)

  quan_per_feb_45 <- cbind(start, quan_per_feb_45)
  quan_per_feb_45 <- cbind(quan_per_feb_45, end)
  

  ######****************************
  ################################## MARCH
  ######****************************

  quan_per_mar <- mar_result %>% 
                  group_by(time_period, city, scenario, thresh_range) %>% 
                  summarise(quan_25 = quantile(frac_passed, probs = 0.1)) %>% 
                  data.table()

  quan_per_mar_85 <- quan_per_mar  %>% 
                     filter(scenario == "RCP 8.5") %>% 
                     data.table()

  quan_per_mar_45 <- quan_per_mar  %>% 
                     filter(scenario == "RCP 4.5") %>%
                     data.table()

  quan_per_mar_85 <- quan_per_mar_85[order(time_period, thresh_range), ]
  quan_per_mar_45 <- quan_per_mar_45[order(time_period, thresh_range), ]

  quan_per_mar_85 <- quan_per_mar_85 %>% spread(time_period, quan_25)
  quan_per_mar_45 <- quan_per_mar_45 %>% spread(time_period, quan_25)

  start <- data.table(start=rep(begin, nrow(quan_per_mar_85)))
  end <- data.table(end=rep("Mar_1", nrow(quan_per_mar_85)))

  quan_per_mar_85 <- cbind(start, quan_per_mar_85)
  quan_per_mar_85 <- cbind(quan_per_mar_85, end)

  quan_per_mar_45 <- cbind(start, quan_per_mar_45)
  quan_per_mar_45 <- cbind(quan_per_mar_45, end)


  ######****************************
  ################################## April
  ######****************************
  quan_per_apr <- apr_result %>% 
                  group_by(time_period, city, scenario, thresh_range) %>% 
                  summarise(quan_25 = quantile(frac_passed, probs = 0.1)) %>% 
                  data.table()

  quan_per_apr_85 <- quan_per_apr  %>% 
                     filter(scenario == "RCP 8.5") %>% 
                     data.table()

  quan_per_apr_45 <- quan_per_apr  %>% 
                     filter(scenario == "RCP 4.5") %>%
                     data.table()

  quan_per_apr_85 <- quan_per_apr_85[order(time_period, thresh_range), ]
  quan_per_apr_45 <- quan_per_apr_45[order(time_period, thresh_range), ]

  quan_per_apr_85 <- quan_per_apr_85 %>% spread(time_period, quan_25)
  quan_per_apr_45 <- quan_per_apr_45 %>% spread(time_period, quan_25)


  start <- data.table(start=rep(begin, nrow(quan_per_apr_85)))
  end <- data.table(end=rep("Apr_1", nrow(quan_per_apr_85)))

  quan_per_apr_85 <- cbind(start, quan_per_apr_85)
  quan_per_apr_85 <- cbind(quan_per_apr_85, end)

  quan_per_apr_45 <- cbind(start, quan_per_apr_45)
  quan_per_apr_45 <- cbind(quan_per_apr_45, end)

  output_85 <- cbind(output_85, quan_per_jan_85, quan_per_feb_85, quan_per_mar_85, quan_per_apr_85)
  output_45 <- cbind(output_45, quan_per_jan_45, quan_per_feb_45, quan_per_mar_45, quan_per_apr_45)
}

out_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/color_code_tables/limited_locs/"

set(output_85, "row_count", NULL)
set(output_45, "row_count", NULL)

write.csv(output_85, file = paste0(out_dir, "table_for_color_code_85_10th.csv"), row.names=FALSE)
write.csv(output_45, file = paste0(out_dir, "table_for_color_code_45_10th.csv"), row.names=FALSE)

