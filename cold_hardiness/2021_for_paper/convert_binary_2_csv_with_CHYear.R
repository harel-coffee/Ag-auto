rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)
options(digits=9)
options(digit=9)



binary_core <- "/Users/hn/Documents/00_GitHub/Ag/read_binary_core/read_binary_core.R"
source(binary_core)

########################################
########
########   directories
########
########################################
binary_dir_base <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/weather/"

observed_dir <- paste0(binary_dir_base, "historical/")
modeled_dir <- paste0(binary_dir_base, "modeled/")

param_dir <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/"

########################################

locations <- data.table(read.csv(paste0(param_dir, "limited_locations_CH.csv")))
locations$location <- paste0(locations$lat, "_", locations$long)

#####
##### the observed data
#####
observed_data <- data.table()

for (a_location in locations$location){
  curr_data <- read_binary(file_path=paste0(observed_dir, "data_", a_location), hist=TRUE, no_vars=8)
  curr_data$location <- a_location
  observed_data <- rbind(observed_data, curr_data)
}
observed_data$model <- "Observed"
observed_data$emission <- "Observed"
observed_data$time_period <- "Observed"

#####
##### the modeled data
#####

models <- dir(modeled_dir)
emissions <- c("historical", "rcp45", "rcp85")

modeled_data <- data.table()

for (a_model in models){
  for (emission in emissions){
    if (emission == "historical"){
      for (a_location in locations$location){

        fileN <- paste0(modeled_dir, a_model, "/", emission, "/", "data_", a_location)
        curr_data <- read_binary(file_path=fileN, hist=TRUE, no_vars=4)
        curr_data$model <- a_model
        curr_data$emission <- "modeled_historical"
        curr_data$time_period <- "modeled_historical"
        curr_data$location <- a_location
        modeled_data <- rbind(modeled_data, curr_data)
      }
    } else {
      for (a_location in locations$location){
        fileN <- paste0(modeled_dir,  a_model, "/", emission, "/", "data_", a_location)
        curr_data <- read_binary(file_path=fileN, hist=FALSE, no_vars=4)
        curr_data$model <- a_model
        curr_data$emission <- emission
        curr_data$time_period <- "future"
        curr_data$location <- a_location
        modeled_data <- rbind(modeled_data, curr_data)
      }
    }
  }
}

common_columns <- c("location", "year",  "month", "day", "tmax", "tmin", "model", "emission", "time_period")

modeled_data <- subset(modeled_data, select=common_columns)
observed_data <- subset(observed_data, select=common_columns)

all_weather_data <- rbind(modeled_data, observed_data)

all_weather_data$hardiness_year <- ifelse(all_weather_data$month > 8, all_weather_data$year + 1, 
                                       ifelse(all_weather_data$month < 6, all_weather_data$year, NA))


write.csv(all_weather_data,    paste0(binary_dir_base, '/all_weather_data.csv'), row.names = F) 
fwrite(all_weather_data, file=paste0(binary_dir_base, '/all_weather_data.csv.gz'), row.names=F, compress="gzip")
