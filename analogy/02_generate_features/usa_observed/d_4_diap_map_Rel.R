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
    version = args[2]
    file_name = paste0("CMPOP_", version)
} else if (time_type == "past"){
    input_dir = "/data/hydro/users/Hossein/codling_moth_new/all_USA/processed/"
    write_dir = paste0(write_dir, "usa/data_bases/")
    version = "observed"
    file_name = "combined_CMPOP"
}

result = diap_map1_prep_4_analog_Rel(input_dir, file_name, param_dir,
                                     time_type,
                                     location_group_name = "LocationGroups.csv")

saveRDS(result, paste0(write_dir, "Rel_preped_4_diap_CMPOP_", version, ".rds"))

result <- gen_diap_map1_4_analog_Rel(sub1 = result, 
                                     param_dir = param_dir, time_type = time_type, 
                                     CodMothParams_name = "CodlingMothparameters.txt", 
                                     location_group_name = "LocationGroups.csv")

saveRDS(result, paste0(write_dir, "diapause_map1_rel_", version, ".rds"))

