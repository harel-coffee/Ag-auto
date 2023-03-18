#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
#library(reshape2)
library(dplyr)
library(foreach)
library(doParallel)
#library(iterators)
source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

read_data_dir= "/data/hydro/users/Hossein/codling_moth_new/all_USA/processed/"
write_path   = "/data/hydro/users/Hossein/codling_moth_new/all_USA/processed/"
param_path   = "/home/hnoorazar/cleaner_codes/parameters/"

cats = c("historical")
locations_list = "all_us_locations_list"

args = commandArgs(trailingOnly=TRUE)
ver = args[1]
file_pref = args[2]

data <- merge_add_countyGroup(input_dir=read_data_dir, 
                              param_dir=param_path, 
                              locations_file_name = locations_list,
                              locationGroup_fileName="LocationGroups.csv",
                              categories=cats,
                              file_prefix=file_pref,
                              version=ver)

saveRDS(data, paste0(write_path, "combined_", file_pref, ".rds"))


