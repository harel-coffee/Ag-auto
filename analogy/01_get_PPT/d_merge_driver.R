.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(geepack)
library(chron)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)

options(digits=9)
options(digit=9)

##################################################################
##
##            Terminal Arguments
##
##################################################################

args = commandArgs(trailingOnly=TRUE)
location_type = args[1] # either local or usa

##################################################################
main_in_dir <- "/data/hydro/users/Hossein/analog/"
main_in_dir = paste0(main_in_dir, location_type, "/percipitation/" )
print ("main_in_dir from driver:")
print (main_in_dir)
print ("____________________________")

main_out_dir <- "/data/hydro/users/Hossein/analog/"
main_out_dir <- paste0(main_out_dir, location_type, "/data_bases/")

merged_files <- merge_precip(main_in_dir, location_type)

if (location_type == "local"){
	saveRDS(merged_files[[1]], paste0(main_out_dir, "precip_local_rcp45.rds"))
	saveRDS(merged_files[[2]], paste0(main_out_dir, "precip_local_rcp85.rds"))

	} else if(location_type == "usa"){
		saveRDS(merged_files, paste0(main_out_dir, "precip_usa.rds"))
}



