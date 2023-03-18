######################################################################
#          **********                            **********
#          **********        WARNING !!!!        **********
#          **********                            **********
##
## DO NOT load any libraries here.
## And do not load any libraries on the drivers!
## Unless you are aware of conflicts between packages.
## I spent hours to figrue out what the hell is going on!
#####################################################################

.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
args = commandArgs(trailingOnly=TRUE)
data_type = args[1]

merge <- function(data_type){
	param_dir = "/home/hnoorazar/cleaner_codes/parameters/"
	loc_group_file_name = "LocationGroups.csv"
	locations_list = "local_list"

	input_dir = "/data/hydro/users/Hossein/temp_gdd/modeled/"
	output_dir= "/data/hydro/users/Hossein/temp_gdd/"

	# list of directories in input_dir
	modeled_categories = list.dirs(path = input_dir, full.names = F, recursive = F)
	versions = c("historical", "rcp45", "rcp85")
	if (data_type=="observed"){
		####################################
		##
		## merge observed data
		##
		####################################
		input_dir = "/data/hydro/users/Hossein/temp_gdd/observed/"
		observed = data.table()

		# list of .rds files in the directory
		all_files <- list.files(input_dir)
		all_files <- all_files[grep(pattern = "data_", x = all_files)]
		for (afile in all_files){
			curr_data <- data.table(readRDS(paste0(input_dir, afile)))
			observed <- rbind(observed, curr_data)
		}
		saveRDS(observed, paste0(output_dir, "observed.rds"))
		rm(observed)
	} else if (data_type=="modeled_historical"){
		####################################
		##
		## merge modeled historical data
		##
		####################################
		modeled_historical = data.table()
		for (cat in modeled_categories){
			data_dir = file.path(input_dir, cat, "/historical/")

			# list of .rds files in the directory
			all_files <- list.files(data_dir)
			all_files <- all_files[grep(pattern = "data_", x = all_files)]
			for (afile in all_files){
				curr_data <- data.table(readRDS(paste0(data_dir, afile)))
				modeled_historical<- rbind(modeled_historical, curr_data)
			}
		}
		saveRDS(modeled_historical, paste0(output_dir, "modeled_historical.rds"))
		rm(modeled_historical)
	} else if (data_type=="rcp45"){
		####################################
		##
		## merge modeled rcp45 data
		##
		####################################
		modeled_rcp45 = data.table()
		for (cat in modeled_categories){
			data_dir = file.path(input_dir, cat, "/rcp45/")

			# list of .rds files in the directory
			all_files <- list.files(data_dir)
			all_files <- all_files[grep(pattern = "data_", x = all_files)]
			for (afile in all_files){
				curr_data <- data.table(readRDS(paste0(data_dir, afile)))
				modeled_rcp45<- rbind(modeled_rcp45, curr_data)
			}
		}
		saveRDS(modeled_rcp45, paste0(output_dir, "modeled_rcp45.rds"))
		rm(modeled_rcp45)
	} else if (data_type=="rcp85"){
	    ####################################
		##
		## merge modeled rcp85 data
		##
		####################################
		modeled_rcp85 = data.table()
		for (cat in modeled_categories){
			data_dir = file.path(input_dir, cat, "/rcp85/")
			# list of .rds files in the directory
			all_files <- list.files(data_dir)
			all_files <- all_files[grep(pattern = "data_", x = all_files)]
			for (afile in all_files){
				curr_data <- data.table(readRDS(paste0(data_dir, afile)))
				modeled_rcp85 <- rbind(modeled_rcp85, curr_data)
			}
		}
		saveRDS(modeled_rcp85, paste0(output_dir, "modeled_rcp85.rds"))
		rm(modeled_rcp85)
	}
}

merge(data_type)

