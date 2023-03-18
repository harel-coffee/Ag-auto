#!/share/apps/R-3.2.2_gcc/bin/Rscript
library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(chron)

input_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/"
write_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/small_samples/"
files = list.files(input_dir, "*.rds")
files = c("vertdd_combined_CMPOP_rcp45.rds", "vertdd_combined_CMPOP_rcp85.rds")
files = c("combined_CMPOP_rcp45.rds", "combined_CMPOP_rcp85.rds")
for (file in files){
	file_name = paste0(input_dir, file)
	data <- data.table(readRDS(file_name))
	data <- data[61098635:61098735, ]
	saveRDS(data, paste0(write_dir, file))
}

# input_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/discovery/Girids/"
# write_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/discovery/samples/"

# files = list.files(input_dir, "*.rds")
# files = c("allData_vertdd_new_rcp45.rds", "allData_vertdd_new.rds", "allData_vertdd.rds")
# for (file in files){
#	file_name = paste0(input_dir, file)
#	data <- data.table(readRDS(file_name))
#	data <- data[61098635:61198635, ]
#	saveRDS(data, paste0(write_dir, file))
#}