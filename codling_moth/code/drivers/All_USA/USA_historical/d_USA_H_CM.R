#!/share/apps/R-3.2.2_gcc/bin/Rscript
library(chron)
library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)
options(digits=9)
raw_data_dir = "/data/hydro/users/Hossein/codling_moth_new/all_USA/raw/"
write_dir = "/data/hydro/users/Hossein/codling_moth_new/all_USA/processed/historical_CM/"
param_dir  = "/home/hnoorazar/cleaner_codes/parameters/"

file_prefix = "data_"
ClimateGroup = list("Historical", "2040s", "2060s", "2080s")
cellByCounty = data.table(read.csv(paste0(param_dir, "CropParamCRB.csv")))

args = commandArgs(trailingOnly=TRUE)
category = args[1]

file_list = "all_us_locations_list"
conn = file(paste0(param_dir, file_list), open = "r")
locations = readLines(conn)

for(location in locations) {   
  if(category == "historical") {
    start_year = 1979
    end_year = 2015
    filename = paste0(category, "/", file_prefix, location)
  }
  else {
    start_year = 2006
    end_year = 2099
    filename = paste0(category, "/", version, "/", file_prefix, location)
  }
  temp <- prepareData_CM(filename = filename, 
                         input_folder = raw_data_dir, 
                         param_dir = param_dir, 
                         cod_moth_param_name ="CodlingMothparameters.txt",
                         start_year = start_year, end_year = end_year, 
                         lower=10, upper=31.11)
  temp_data <- data.table()
  if(category == "historical") {
    temp$ClimateGroup[temp$year >= start_year & temp$year <= end_year] <- "Historical"
    temp_data <- rbind(temp_data, temp[temp$year >= start_year & temp$year <= end_year, ])
  }
  else {
    temp$ClimateGroup[temp$year > 2025 & temp$year <= 2055] <- "2040s"
    temp_data <- rbind(temp_data, temp[temp$year > 2025 & temp$year <= 2055, ])
    temp$ClimateGroup[temp$year > 2045 & temp$year <= 2075] <- "2060s"
    temp_data <- rbind(temp_data, temp[temp$year > 2045 & temp$year <= 2075, ])
    temp$ClimateGroup[temp$year > 2065 & temp$year <= 2095] <- "2080s"
    temp_data <- rbind(temp_data, temp[temp$year > 2065 & temp$year <= 2095, ])
  }
  loc = tstrsplit(location, "_")
  options(digits=9)
  temp_data$latitude <- as.numeric(unlist(loc[1]))
  temp_data$longitude <- as.numeric(unlist(loc[2]))
  # the following line does not work with all USA!
  # temp_data$County <- as.character(unique(cellByCounty[lat == temp_data$latitude[1] & 
  #                                                      long == temp_data$longitude[1], 
  #                                                      countyname]))
  temp_data$ClimateScenario <- category
  if(category != "historical") {
    dir.create(file.path(write_dir), recursive = TRUE)
    write.table(temp_data, file = paste0(write_dir, "CM_", location), sep = ",", row.names = FALSE, col.names = TRUE)
  }
  else {
    write.table(temp_data, file = paste0(write_dir, "CM_", location), sep = ",", row.names = FALSE, col.names = TRUE)
  }
}
close(conn)

