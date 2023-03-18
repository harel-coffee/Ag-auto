#!/share/apps/R-3.2.2_gcc/bin/Rscript
library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(chron)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

raw_data_dir = "/data/hydro/users/Hossein/codling_moth_new/all_USA/raw/"
write_path = "/data/hydro/users/Hossein/codling_moth_new/all_USA/processed/future_CMPOP/"
param_dir  = "/home/hnoorazar/cleaner_codes/parameters/"

categories = c("bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CNRM-CM5", "GFDL-ESM2G", "GFDL-ESM2M")

file_prefix = "data_"
file_list = "all_us_locations_list"
version = "rcp85"

ClimateGroup = list("Historical", "2040's", "2060's", "2080's")
cellByCounty = data.table(read.csv(paste0(param_dir, "CropParamCRB.csv")))

conn = file(paste0(param_dir, file_list), open = "r")
locations = readLines(conn)
for( category in categories) {
  for( location in locations) {
    filename = paste0(category, "/", version, "/", file_prefix, location)
    if(category == "historical") {
      start_year = 1979
      end_year = 2015
    }
    else {
      start_year = 2006
      end_year = 2099
    }
    
    temp <- prepareData_CMPOP(filename=filename, input_folder=raw_data_dir, 
                              param_dir=param_dir,
                              cod_moth_param_name="CodlingMothparameters.txt",
                              start_year=start_year, end_year=end_year,
                              lower=10, upper=31.11)
    temp_data <- data.table()
    if(category == "historical") {
      temp$ClimateGroup[temp$year >= 1979 & temp$year <= 2015] <- "Historical"
      temp_data <- rbind(temp_data, temp[temp$year >= 1979 & temp$year <= 2015, ])
    }
    else {
      temp$ClimateGroup[temp$year > 2025 & temp$year <= 2055] <- "2040's"
      temp_data <- rbind(temp_data, temp[temp$year > 2025 & temp$year <= 2055, ])
      temp$ClimateGroup[temp$year > 2045 & temp$year <= 2075] <- "2060's"
      temp_data <- rbind(temp_data, temp[temp$year > 2045 & temp$year <= 2075, ])
      temp$ClimateGroup[temp$year > 2065 & temp$year <= 2095] <- "2080's"
      temp_data <- rbind(temp_data, temp[temp$year > 2065 & temp$year <= 2095, ])
    }
    loc = tstrsplit(location, "_")
    options(digits=9)
    temp_data$latitude <- as.numeric(unlist(loc[1]))
    temp_data$longitude <- as.numeric(unlist(loc[2]))
    temp_data$ClimateScenario <- category
    write_dir = paste0(write_path, category, "/", version)
    dir.create(file.path(write_dir), recursive = TRUE)
    write.table(temp_data, file = paste0(write_dir, "/CMPOP_", location), 
    	                     sep = ",", 
    	                     row.names = FALSE, 
    	                     col.names = TRUE)
  }
}
close(conn)

