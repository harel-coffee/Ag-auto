#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)
write_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"
input_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"
param_dir = "/home/hnoorazar/cleaner_codes/parameters/"

versions = c("rcp45", "rcp85")

for (version in versions){
	file_name = paste0("combined_CMPOP_", version)
	output = diapause_map1_prep(input_dir, 
	                            file_name,
                                param_dir, 
                                location_group_name="LocationGroups.csv")
	saveRDS(output, paste0(write_dir, "prepared_4_map_1_", version, ".rds"))

}