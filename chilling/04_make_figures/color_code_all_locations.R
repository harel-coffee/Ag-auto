
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
                            lat, long, warm_cold,
                            scenario, model, year))
  
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

  dt$location <- paste0(dt$lat, "_", dt$long)
  jan_data <- subset(dt, select=c(sum_J1, warm_cold, scenario, model, time_period, chill_season, location)) %>% data.table()
  feb_data <- subset(dt, select=c(sum_F1, warm_cold, scenario, model, time_period, chill_season, location)) %>% data.table()
  mar_data <- subset(dt, select=c(sum_M1, warm_cold, scenario, model, time_period, chill_season, location)) %>% data.table()
  apr_data <- subset(dt, select=c(sum_A1, warm_cold, scenario, model, time_period, chill_season, location)) %>% data.table()
  return (list(jan_data, feb_data, mar_data, apr_data))
}

#############################################
###                                       ###
###                 Driver                ###
###                                       ###
#############################################

# main_in_dir = "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/non_overlapping/"
# model_names = c("dynamic") # , "utah"
# model_specific_dir_name = paste0(model_names, "_model_stats/")
# file_name = "summary_comp.rds"
# mdata <- data.table(readRDS(paste0(main_in_dir, model_specific_dir_name, file_name)))
# setnames(mdata, old=c("Chill_season"), new=c("chill_season"))

main_in_dir = "/Users/hn/Desktop/Desktop/Ag/check_point/chilling/"
out_dir = main_in_dir

begins <- c("sept", "mid_sept", "oct", "mid_oct", "nov", "mid_nov")
begin <- "sept"
for (begin in begins){
  out_dir <- file.path(main_in_dir, begin, "/color_code_table/")
  
  if (dir.exists(file.path(out_dir)) == F) {
    dir.create(path = file.path(out_dir), recursive = T)
  }

  mdata <- data.table(readRDS(paste0(main_in_dir, begin, "_summary_comp.rds")))
  mdata <- mdata %>% filter(model != "observed")

  param_dir <- "/Users/hn/Documents/GitHub/Ag/chilling/parameters/"
  LocationGroups_NoMontana <- read.csv(paste0(param_dir, "LocationGroups_NoMontana.csv"), 
                                       header=T, sep=",", as.is=T)
  LocationGroups_NoMontana <- within(LocationGroups_NoMontana, remove(lat, long))

  mdata <- remove_montana(mdata, LocationGroups_NoMontana)
  
  information <- clean_process(mdata)
 
  jan_data = information[[1]] 
  feb_data = information[[2]] 
  mar_data = information[[3]] 
  apr_data = information[[4]]
  rm(information, mdata)

  jan_result = count_years_threshs_met_all_locations(dataT = jan_data, due="Jan")
  feb_result = count_years_threshs_met_all_locations(dataT = feb_data, due="Feb")
  mar_result = count_years_threshs_met_all_locations(dataT = mar_data, due="Mar")
  apr_result = count_years_threshs_met_all_locations(dataT = apr_data, due="Apr")

  #####################
  ##################### Add climate type back to data
  #####################
  locatin_add <- subset(LocationGroups_NoMontana, select=c(warm_cold, location))
  jan_result <- dplyr::left_join(jan_result, LocationGroups_NoMontana, by="location")
  feb_result <- dplyr::left_join(feb_result, LocationGroups_NoMontana, by="location")
  mar_result <- dplyr::left_join(mar_result, LocationGroups_NoMontana, by="location")
  apr_result <- dplyr::left_join(apr_result, LocationGroups_NoMontana, by="location")

  #####################
  #####################              RCP 8.5
  #####################
  ######****************************
  ################################## JAN
  ######****************************

  quan_per <- jan_result %>% group_by(time_period, scenario, thresh_range) %>%
                             summarise(quan_25 = quantile(frac_passed, probs = 0.25)) %>% 
                             data.table()

  ######### COOLER 8.5

  data <- quan_per %>% filter(scenario == "RCP 8.5", warm_cold=="Cooler Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)

  write.table(dattest, file = paste0(out_dir, "jan_cool_85.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  data <- quan_per %>% 
          filter(time_period == "Historical", warm_cold=="Cooler Areas", scenario=="RCP 4.5") %>% 
          data.table() %>%
          select(c(warm_cold, time_period, thresh_range, quan_25))

  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "jan_cool_hist.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  ######## WARMER

  data <- quan_per %>% filter(scenario == "RCP 8.5", warm_cold=="Warmer Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "jan_warm_85.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  data <- quan_per %>% filter(time_period == "Historical", warm_cold=="Warmer Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "jan_warm_hist.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest, quan_per)

  ######****************************
  ################################## FEB
  ######****************************
  quan_per_feb <- feb_result %>% 
                  group_by(warm_cold, time_period, scenario, thresh_range) %>% 
                  summarise(quan_25 = quantile(frac_passed, probs = 0.25)) %>% 
                  data.table()

  ######## COOLER

  data <- quan_per_feb %>% filter(scenario == "RCP 8.5", warm_cold=="Cooler Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "feb_cool_85.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  data <- quan_per_feb %>% filter(time_period == "Historical", warm_cold=="Cooler Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "feb_cool_hist.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  ######## WARMER

  data <- quan_per_feb %>% filter(scenario == "RCP 8.5", warm_cold=="Warmer Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "feb_warm_85.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  data <- quan_per_feb %>% filter(time_period == "Historical", warm_cold=="Warmer Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "feb_warm_hist.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest, quan_per_feb)

  ######****************************
  ################################## MARCH
  ######****************************

  quan_per <- mar_result %>% group_by(warm_cold, time_period, scenario, thresh_range) %>% 
                             summarise(quan_25 = quantile(frac_passed, probs = 0.25)) %>% data.table()

  ######### COOLER
  data <- quan_per %>% filter(scenario == "RCP 8.5", warm_cold=="Cooler Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period,quan_25)
  write.table(dattest, file = paste0(out_dir, "march_cool_85.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  data <- quan_per %>% filter(time_period == "Historical", warm_cold=="Cooler Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "march_cool_hist.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  ######## WARM

  data <- quan_per %>% filter(scenario == "RCP 8.5", warm_cold=="Warmer Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "march_warm_85.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  data <- quan_per %>% filter(time_period == "Historical", warm_cold=="Warmer Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "march_warm_hist.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest, quan_per)

  ######****************************
  ################################## April
  ######****************************

  quan_per <- apr_result %>% group_by(warm_cold, time_period, scenario, thresh_range) %>% 
                             summarise(quan_25 = quantile(frac_passed, probs = 0.25)) %>% data.table()

  ######### COOLER
  data <- quan_per %>% filter(scenario == "RCP 8.5", warm_cold=="Cooler Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period,quan_25)
  write.table(dattest, file = paste0(out_dir, "april_cool_85.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  data <- quan_per %>% filter(time_period == "Historical", warm_cold=="Cooler Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "april_cool_hist.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  ######## WARM

  data <- quan_per %>% filter(scenario == "RCP 8.5", warm_cold=="Warmer Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "april_warm_85.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  data <- quan_per %>% filter(time_period == "Historical", warm_cold=="Warmer Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "april_warm_hist.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest, quan_per)

  #####################
  #####################              RCP 4.5
  #####################
  ######****************************
  ################################## JAN
  ######****************************

  quan_per <- jan_result %>% group_by(warm_cold, time_period, scenario, thresh_range) %>% 
                             summarise(quan_25 = quantile(frac_passed, probs = 0.25)) %>% data.table()

  ######### COOLER

  data <- quan_per %>% filter(scenario == "RCP 4.5", warm_cold == "Cooler Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "jan_cool_45.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  ######## WARM

  data <- quan_per %>% filter(scenario == "RCP 4.5", warm_cold=="Warmer Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "jan_warm_45.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  ######****************************
  ################################## FEB
  ######****************************
  quan_per_feb <- feb_result %>% group_by(warm_cold, time_period, scenario, thresh_range) %>% 
                                 summarise(quan_25 = quantile(frac_passed, probs = 0.25)) %>% data.table()

  ######## COOLER

  data <- quan_per_feb %>% filter(scenario == "RCP 4.5", warm_cold=="Cooler Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "feb_cool_45.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  ######## WARM

  data <- quan_per_feb %>% filter(scenario == "RCP 4.5", warm_cold=="Warmer Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "feb_warm_45.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  ######****************************
  ################################## MARCH
  ######****************************

  quan_per <- mar_result %>% group_by(warm_cold, time_period, scenario, thresh_range) %>% 
                             summarise(quan_25 = quantile(frac_passed, probs = 0.25)) %>% data.table()

  ######### COOLER

  data <- quan_per %>% filter(scenario == "RCP 4.5", warm_cold=="Cooler Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "march_cool_45.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  ######## WARM

  data <- quan_per %>% filter(scenario == "RCP 4.5", warm_cold=="Warmer Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "march_warm_45.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  ######****************************
  ################################## April
  ######****************************

  quan_per <- apr_result %>% group_by(warm_cold, time_period, scenario, thresh_range) %>% 
                             summarise(quan_25 = quantile(frac_passed, probs = 0.25)) %>% data.table()

  ######### COOLER

  data <- quan_per %>% filter(scenario == "RCP 4.5", warm_cold=="Cooler Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "april_cool_45.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)

  ######## WARM

  data <- quan_per %>% filter(scenario == "RCP 4.5", warm_cold=="Warmer Areas") %>% data.table()
  data <- data[order(time_period, thresh_range), ]
  dattest <- data %>% spread(time_period, quan_25)
  write.table(dattest, file = paste0(out_dir, "april_warm_45.csv"), row.names = FALSE, col.names = TRUE, sep = ",")
  rm(data, dattest)
}





