
.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(geepack)
library(MESS)
library(chron)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

input_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/"
write_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/"
param_dir = "/home/hnoorazar/cleaner_codes/parameters/"

args = commandArgs(trailingOnly=TRUE)
version = args[1]
	
file_name = paste0("combined_CMPOP_", version)

result = generate_diapause_map1(input_dir, file_name, param_dir, 
	                            CodMothParams_name="CodlingMothparameters.txt", 
	                            location_group_name="LocationGroups.csv")

saveRDS(result, paste0(write_dir, "diapause_map1_", version,".rds"))