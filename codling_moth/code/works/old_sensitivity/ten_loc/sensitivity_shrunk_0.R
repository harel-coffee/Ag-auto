#!/share/apps/R-3.2.2_gcc/bin/Rscript
library(chron)
library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

raw_data_dir = "/data/hydro/users/Hossein/codling_moth_new/local/raw/"
write_path = "/data/hydro/users/Hossein/codling_moth_new/local/sensitivity_wider/"
param_dir = "/home/hnoorazar/cleaner_codes/parameters/"

file_list = "local_list_10.txt"
conn = file(paste0(param_dir, file_list), open = "r")
locations = readLines(conn)
close(conn)

#locations = locations[1]
file_prefix = "data_"

ClimateGroup = list("Historical", "2040's", "2060's", "2080's")
cellByCounty = data.table(read.csv(paste0(param_dir, "CropParamCRB.csv")))

categories = c("bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CNRM-CM5", "GFDL-ESM2G", "GFDL-ESM2M", "historical")
versions = c("rcp45", "rcp85")

param_name = "CodlingMothparameters_"
params_shift = c("0", "5", "10", "15", "20")
params_shift = c("0")

start_h = 1979
end_h = 2015

start_f = 2006
end_f = 2099

for (shift in params_shift){
	cod_param = paste0(param_name, shift, ".txt")
	#############
	############# CMPOP
	#############
	for(category in categories) {
		if(category == "historical") {
			for(location in locations){
				filename = paste0(category, "/", file_prefix, location)
				temp_data <- produce_CMPOP(input_folder= raw_data_dir, filename,
                                           param_dir, cod_moth_param_name=cod_param,
                                           start_year=start_h, end_year=end_h, 
                                           lower=10, upper=31.11,
                                           location, category)
			    write_dir = paste0(write_path, shift, "/", "historical_CMPOP/")
			    dir.create(file.path(write_dir), recursive = TRUE)
			    write.table(temp_data, file = paste0(write_dir, "/CMPOP_", location), 
			    	                   sep = ",", 
			    	                   row.names = FALSE, 
			    	                   col.names = TRUE)
			}
		}
		else{
			for(version in versions) {
				for(location in locations) {
					filename <- paste0(category, "/", version, "/", file_prefix, location)
				    temp_data <- produce_CMPOP(input_folder= raw_data_dir, filename,
                                               param_dir, cod_moth_param_name=cod_param,
                                               start_year=start_f, end_year=end_f, 
                                               lower=10, upper=31.11,
                                               location, category)
				    write_dir <- paste0(write_path, shift, "/", "future_CMPOP/",category, "/", version)
				    dir.create(file.path(write_dir), recursive = TRUE)
				    write.table(temp_data, file = paste0(write_dir, "/CMPOP_", location), 
				                           sep = ",", 
				                           row.names = FALSE, 
				                           col.names = TRUE)
	            }
	        }
	    }
	}
    #############
	############# CM
	#############
	for(category in categories) {
		if(category == "historical") { 
			for(location in locations) {
				filename = paste0(category, "/", file_prefix, location)  
                temp_data <- produce_CM(input_folder= raw_data_dir, filename,
                                        param_dir, cod_moth_param_name = cod_param,
                                        start_year=start_h, end_year=end_h, 
                                        lower=10, upper=31.11,
                                        location, category)
	            write_dir = paste0(write_path, shift, "/", "historical_CM/")
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
	                temp_data <- produce_CM(input_folder=raw_data_dir, filename,
                                            param_dir, cod_moth_param_name=cod_param,
                                            start_year=start_f, end_year=end_f, 
                                            lower=10, upper=31.11,
                                            location, category)
	                write_dir = paste0(write_path, shift, "/", "future_CM/", category, "/", version)
	                dir.create(file.path(write_dir), recursive = TRUE)
	                write.table(temp_data, file = paste0(write_dir, "/CM_", location), 
                                sep = ",", 
                                row.names = FALSE, 
                                col.names = TRUE)
	            }
	        }
		}
	}
}

