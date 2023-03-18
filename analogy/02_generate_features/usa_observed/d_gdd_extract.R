.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(MESS) # has the auc function in it.
library(geepack)
library(chron)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)

write_dir = "/data/hydro/users/Hossein/analog/"
param_dir = "/home/hnoorazar/cleaner_codes/parameters/"

args = commandArgs(trailingOnly=TRUE)
time_type = args[1]

if (time_type == "future"){
    input_dir = "/data/hydro/users/Hossein/analog/local/data_bases/001_unique_CMPOP/"
    write_dir = paste0(write_dir, "local/data_bases/")
} else if (time_type == "past"){
    input_dir = "/data/hydro/users/Hossein/codling_moth_new/all_USA/processed/"
    write_dir = paste0(write_dir, "usa/data_bases/")
    version = "observed"
}


gdd_usa <- extract_gdd(in_dir=input_dir, file_name = "combined_CMPOP.rds")
saveRDS(gdd_usa, paste0(write_dir, "gdd_usa.rds"))
rm(gdd_usa)

