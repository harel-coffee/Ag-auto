#!/share/apps/R-3.2.2_gcc/bin/Rscript
library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(chron)

source_path = "/home/hnoorazar/codling_moth_2021/core.R"
source(source_path)

raw_data_dir = "/data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016/"
write_dir  =   "/data/hydro/users/Hossein/codling_moth_2021/00_LH_CMPOP/"
param_dir  =   "/home/hnoorazar/codling_moth_2021/parameters/"


file_prefix = "data_"
locations = data.table(read.csv(paste0(param_dir, "five_locations.csv")))
locations = locations$location

time_periods = list("Historical", "2040s", "2060s", "2080s")
cellByCounty = data.table(read.csv(paste0(param_dir, "CropParamCRB.csv")))

models = c("historical")

for( model in models) {
  for( location in locations) {
    
    if(model == "historical") {
      filename = paste0(file_prefix, location) # model, "/", 
      start_year = 1979 # this is observed
      end_year = 2015
    }
    else {
      filename = paste0(model, "/", file_prefix, location) 
      start_year = 2006
      end_year = 2099
    }
    temp <- prepareData_CMPOP(filename=filename, input_folder=raw_data_dir, 
                              param_dir=param_dir, 
                              cod_moth_param_name="CodlingMothparameters.txt",
                              start_year=start_year, end_year=end_year, 
                              lower=10, upper=31.11)
    temp_data <- data.table()
    if(model == "historical") {
      temp$time_period[temp$year >= 1979 & temp$year <= 2015] <- "Historical"
      temp_data <- rbind(temp_data, temp[temp$year >= 1979 & temp$year <= 2015, ])
    }
    else {
      temp$time_period[temp$year > 2025 & temp$year <= 2055] <- "2040s"
      temp_data <- rbind(temp_data, temp[temp$year > 2025 & temp$year <= 2055, ])
      temp$time_period[temp$year > 2045 & temp$year <= 2075] <- "2060s"
      temp_data <- rbind(temp_data, temp[temp$year > 2045 & temp$year <= 2075, ])
      temp$time_period[temp$year > 2065 & temp$year <= 2095] <- "2080s"
      temp_data <- rbind(temp_data, temp[temp$year > 2065 & temp$year <= 2095, ])
    }
    loc = tstrsplit(location, "_")
    options(digits=9)
    temp_data$latitude <- as.numeric(unlist(loc[1]))
    temp_data$longitude <- as.numeric(unlist(loc[2]))
    temp_data$County <- as.character(unique(cellByCounty[lat == temp_data$latitude[1] & long == temp_data$longitude[1], countyname]))
    temp_data$model <- model
    dir.create(file.path(write_dir), recursive = TRUE)
    write.table(temp_data, file = paste0(write_dir, "/CMPOP_", location), 
    	                     sep = ",", 
    	                     row.names = FALSE, 
    	                     col.names = TRUE)
  }
}


