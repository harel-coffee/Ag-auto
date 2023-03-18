#!/share/apps/R-3.2.2_gcc/bin/Rscript
library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(chron)

source_path = "/home/hnoorazar/analog_codes/00_biofix/core_cod_moth_bio_fix.R"
source(source_path)
options(digits=9)
#############################################################################

bad_CMPOP_dir <- "/data/hydro/users/Hossein/analog/usa/data_bases/before_biofix/"
write_path <- "/data/hydro/users/Hossein/analog/usa/data_bases/"

cod_param_dir <- "/home/hnoorazar/cleaner_codes/parameters/"
analog_param_dir <- "/home/hnoorazar/analog_codes/parameters/"

all_models = c("bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CNRM-CM5", "GFDL-ESM2G", "GFDL-ESM2M")


file_prefix = "data_"
file_list = "all_us_locations_list"

# time_period = list("Historical", "2040's", "2060's", "2080's")
ClimateGroup = list("Historical", "2040s", "2060s", "2080s")

cellByCounty = data.table(read.csv(paste0(param_dir, "CropParamCRB.csv")))

version = "rcp45"
conn = file(paste0(param_dir, file_list), open = "r")
locations = readLines(conn)
for(a_model in all_models) {
  for(location in locations) {
    filename = paste0(a_model, "/", version, "/", file_prefix, location)
    if(a_model == "historical") {
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
    if(a_model == "historical") {
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
    temp_data$ClimateScenario <- a_model
    write_dir = paste0(write_path, a_model, "/", version)
    dir.create(file.path(write_dir), recursive = TRUE)
    write.table(temp_data, file = paste0(write_dir, "/CMPOP_", location), 
    	                     sep = ",", 
    	                     row.names = FALSE, 
    	                     col.names = TRUE)
  }
}
close(conn)

