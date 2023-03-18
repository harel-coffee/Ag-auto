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

param_dir = "/home/hnoorazar/cleaner_codes/parameters/"
loc_group_file_name = "LocationGroups.csv"
locations_list = "local_list"

output_dir= "/data/hydro/users/Hossein/temp_gdd/"

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


