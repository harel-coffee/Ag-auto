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

read_data_dir= "/data/hydro/users/Hossein/codling_moth/local/processed/"
write_path   = "/data/hydro/users/Hossein/codling_moth/local/processed/"
param_path   = "/home/hnoorazar/cleaner_codes/parameters/"


categories = c("historical", "BNU-ESM", "CanESM2", "GFDL-ESM2G", "bcc-csm1-1-m", "CNRM-CM5", "GFDL-ESM2M")
file_pref = "CMPOP"
locations_list = "local_list"

data <- merge_data(input_dir=read_data_dir, 
	               param_dir=param_path, 
	               locations_file_name=locations_list, 
	               file_prefix=file_pref, 
	               version="rcp45")

saveRDS(data, paste0(write_path, "/combined_CMPOP_rcp45.rds"))
