#######!/share/apps/R-3.2.2_gcc/bin/Rscript

.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(lubridate)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(chron)
options(digits=9)

source_path = "/home/hnoorazar/codling_moth_2021/core.R"
source(source_path)

raw_data_dir = "/data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016/"
write_path =   "/data/hydro/users/Hossein/codling_moth_2021/00_LO_CM/"
param_dir  =   "/home/hnoorazar/codling_moth_2021/parameters/"

file_prefix = "data_"
locations = data.table(read.csv(paste0(param_dir, "five_locations.csv")))
locations = locations$location

time_periods = list("1979-2015", "2040s", "2060s", "2080s")
cellByCounty = data.table(read.csv(paste0(param_dir, "CropParamCRB.csv")))

start_h = 1979 # this is observed
end_h = 2015

cod_param <- "CodlingMothparameters.txt"
args = commandArgs(trailingOnly=TRUE)
model <- "observed"


for(location in locations) {
  filename = paste0( file_prefix, location)  # model, "/",

  temp_data <- produce_CM_local(input_folder = raw_data_dir, NoVariables = 8, # observed data have 8 columns
                                filename = filename,
                                param_dir = param_dir, 
                                cod_moth_param_name = cod_param,
                                scale_shift = 0,
                                start_year = start_h, end_year = end_h, 
                                lower = 10, upper = 31.11,
                                location = location, 
                                category = model)

  temp_data$model <- "Observed"
  write_dir = paste0(write_path) # "historical_CM/"
  dir.create(file.path(write_dir), recursive = TRUE)
  write.table(temp_data, file = paste0(write_dir, "CM_", location), 
                         sep = ",", 
                         row.names = FALSE, 
                         col.names = TRUE)
}





