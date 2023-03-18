.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(MESS) # has the auc function in it.
library(geepack)
library(chron)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

print("line 13 of driver")
input_dir = "/data/hydro/users/Hossein/codling_moth_new/all_USA/processed/"
write_dir = "/data/hydro/users/Hossein/analog/usa/data_bases/"
param_dir = "/home/hnoorazar/cleaner_codes/parameters/"

result = generate_diapause_map1(input_dir, file_name, param_dir, 
	                            CodMothParams_name="CodlingMothparameters.txt", 
	                            location_group_name = "LocationGroups.csv")

saveRDS(result, paste0(write_dir, "diapause_map1.rds"))