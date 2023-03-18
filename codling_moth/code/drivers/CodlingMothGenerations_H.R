#!/share/apps/R-3.2.2_gcc/bin/Rscript
library(chron)
library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

data_dir = "/data/hydro/users/giridhar/giridhar/codmoth_pop/alldata_us_locations/"
write_path = "/home/hnoorazar/data/"
parameters_path = "/home/hnoorazar/cleaner_codes/parameters/"

file_prefix = "data_"
ClimateGroup = list("Historical", "2040s", "2060s", "2080s")
cellByCounty = data.table(read.csv(paste0(parameters_path, "CropParamCRB.csv")))

args = commandArgs(trailingOnly=TRUE)
category = args[1]
categories = c("historical", "BNU-ESM", "CanESM2", "GFDL-ESM2G", "bcc-csm1-1-m", "CNRM-CM5", "GFDL-ESM2M")
file_list = "all_us_locations_list"
conn = file(paste0(data_dir, file_list), open = "r")
locations = readLines(conn)

for(location in locations) {
  if(category == "historical") {
    start_year = 1979
    end_year = 2015
    filename = paste0("data/", category, "/", file_prefix, location)
  }
  else {
    start_year = 2006
    end_year = 2099
    filename = paste0("data/", category, "/", version, "/", file_prefix, location)
  }
  temp <- prepareData_1(filename, data_dir, start_year, end_year)
  temp_data <- data.table()
  if(category == "historical") {
    temp$ClimateGroup[temp$year >= 1979 & temp$year <= 2015] <- "Historical"
    temp_data <- rbind(temp_data, temp[temp$year >= 1979 & temp$year <= 2015, ])
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
  temp_data$latitude <- as.numeric(unlist(loc[1]))
  temp_data$longitude <- as.numeric(unlist(loc[2]))
  if(category != "historical") {
    write_dir = paste0(write_path, "data_processed/", category, "/", version)
    dir.create(file.path(write_dir), recursive = TRUE)
    write.table(temp_data, file = paste0(write_dir, "/CM_", location), sep = ",", row.names = FALSE, col.names = TRUE)
  }
  else {
    write_dir = paste0(write_path, "data_processed/", category, "/")
    dir.create(file.path(write_dir), recursive = TRUE)
    write.table(temp_data, file = paste0(write_dir, "/CM_", location), sep = ",", row.names = FALSE, col.names = TRUE)
  }
}
close(conn)
