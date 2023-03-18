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

param_path   = "/home/hnoorazar/cleaner_codes/parameters/"
shifts = c("0", "5", "10", "15", "20")
master_path = "/data/hydro/users/Hossein/codling_moth_new/local/sensitivity"

cats = c("historical", "BNU-ESM", "CanESM2", "GFDL-ESM2G", "bcc-csm1-1-m", "CNRM-CM5", "GFDL-ESM2M")
locations_list = "local_list_1"

args = commandArgs(trailingOnly=TRUE)
ver = args[1]
file_pref = args[2]

for (shift in shifts){
	data_dir  <- paste0(master_path, "/", shift, "/")
	write_dir <- data_dir

	data <- merge_add_countyGroup(input_dir=data_dir, 
	                              param_dir=param_path, 
	                              locations_file_name = locations_list,
	                              locationGroup_fileName="LocationGroups.csv",
	                              categories=cats,
	                              file_prefix=file_pref,
	                              version=ver)

	saveRDS(data, paste0(write_dir, "combined_", file_pref, "_", ver, ".rds"))
}