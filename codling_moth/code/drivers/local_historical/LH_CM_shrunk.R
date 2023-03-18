#!/share/apps/R-3.2.2_gcc/bin/Rscript
library(chron)
library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

raw_data_dir = "/data/hydro/users/Hossein/codling_moth/local/raw/"
write_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/"
param_dir  = "/home/hnoorazar/cleaner_codes/parameters/"

file_prefix = "data_"
file_list = "local_list"
conn = file(paste0(param_dir, file_list), open = "r")
locations = readLines(conn)
close(conn)

ClimateGroup = list("Historical", "2040s", "2060s", "2080s")
cellByCounty = data.table(read.csv(paste0(param_dir, "CropParamCRB.csv")))

start_h = 1979 # this is observed
end_h = 2015
start_f = 2006
end_f = 2099
cod_param <- "CodlingMothparameters.txt"
args = commandArgs(trailingOnly=TRUE)
category = args[1]
categories = c("historical")

for(category in categories) {
  if(category == "historical") { 
    for(location in locations) {
      filename = paste0(category, "/", file_prefix, location)  
      temp_data <- produce_CM_local(input_folder= raw_data_dir, 
                                    filename=filename,
                                    param_dir = param_dir, 
                                    cod_moth_param_name = cod_param,
                                    start_year=start_h, end_year=end_h, 
                                    lower=10, upper=31.11,
                                    location = location, category = category)
      write_dir = paste0(write_path, "historical_CM/")
      dir.create(file.path(write_dir), recursive = TRUE)
      write.table(temp_data, file = paste0(write_dir, "CM_", location), 
                             sep = ",", 
                             row.names = FALSE, 
                             col.names = TRUE)
    }
  }
  ## FUTURE CM
  else{
    for(version in versions){
      for(location in locations) {
        filename = paste0(category, "/", version, "/", file_prefix, location)
        temp_data <- produce_CM(input_folder=raw_data_dir, filename=filename,
                                param_dir=param_dir, 
                                cod_moth_param_name=cod_param,
                                start_year=start_f, end_year=end_f, 
                                lower=10, upper=31.11,
                                location=location, category=category)
        
        write_dir = paste0(write_path, "/", "future_CM/", category, "/", version)
        dir.create(file.path(write_dir), recursive = TRUE)
        write.table(temp_data, file = paste0(write_dir, "/CM_", location), 
                               sep = ",", 
                               row.names = FALSE, 
                               col.names = TRUE)
      }
    }
  }
}