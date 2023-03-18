#!/share/apps/R-3.2.2_gcc/bin/Rscript
library(chron)
library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

raw_data_dir = "/data/hydro/users/Hossein/codling_moth/local/raw/historical/"
write_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/historical_CM/"
param_dir  = "/home/hnoorazar/cleaner_codes/parameters/"

file_prefix = "data_"
file_list = "local_list"
conn = file(paste0(param_dir, file_list), open = "r")
locations = readLines(conn)

ClimateGroup = list("Historical", "2040's", "2060's", "2080's")
cellByCounty = data.table(read.csv(paste0(param_dir, "CropParamCRB.csv")))

args = commandArgs(trailingOnly=TRUE)
category = args[1]
categories = c("historical")
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
  
  temp <- prepareData_1(filename, raw_data_dir, start_year, end_year)
  temp_data <- data.table()
  if(category == "historical") {
    temp$ClimateGroup[temp$year >= 1979 & temp$year <= 2006] <- "Historical"
    temp_data <- rbind(temp_data, temp[temp$year >= 1979 & temp$year <= 2006, ])
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
  temp_data$latitude <- as.numeric(unlist(loc[1]))
  temp_data$longitude <- as.numeric(unlist(loc[2]))
  temp_data$County <- as.character(unique(cellByCounty[lat == temp_data$latitude[1] & long == temp_data$longitude[1], countyname]))
  if(category != "historical") {
    write_dir = paste0(write_path, "data_processed/", category, "/", version)
    dir.create(file.path(write_dir), recursive = TRUE)
    write.table(temp_data, file = paste0(write_dir, "/CM_", location), sep = ",", row.names = FALSE, col.names = TRUE)
  }
  else {
    # dir.create(file.path(write_dir), recursive = TRUE)
    write.table(temp_data, file = paste0(write_dir, "/CM_", location), sep = ",", row.names = FALSE, col.names = TRUE)
  }
}
close(conn)

