######################################################################
##         
##         Merge statistics of all models into one.
##         
######################################################################

.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(lubridate)
library(tidyverse)

input_dir = "/data/hydro/users/Hossein/temp_gdd/modeled/"

categories = list.dirs(path = input_dir, full.names = F, recursive = F)
versions = c("historical", "rcp45", "rcp85")
print (categories)
overlap = data.table()
non_overlap = data.table()

for (cat in categories){
	data_dir = file.path(input_dir, cat, "/")
	print (data_dir)
	
	hist_non_gdd <- data.table(readRDS(paste0(data_dir, "historical_GDD_non_overlap_stat.rds")))
	hist_non_tmean <- data.table(readRDS(paste0(data_dir, "historical_tmean_non_overlap_stat.rds")))
	hist_non <- merge(hist_non_gdd, hist_non_tmean)

	rcp45_non_gdd <- data.table(readRDS(paste0(data_dir, "rcp45_GDD_non_overlap_stat.rds")))
	rcp45_non_tmean <- data.table(readRDS(paste0(data_dir, "rcp45_tmean_non_overlap_stat.rds")))
	rcp45_non <- merge(rcp45_non_gdd, rcp45_non_tmean)

	rcp85_non_gdd <- data.table(readRDS(paste0(data_dir, "rcp85_GDD_non_overlap_stat.rds")))
	rcp85_non_tmean <- data.table(readRDS(paste0(data_dir, "rcp85_tmean_non_overlap_stat.rds")))
	rcp85_non <- merge(rcp85_non_gdd, rcp85_non_tmean)
	
	hist_gdd <- data.table(readRDS(paste0(data_dir, "historical_GDD_overlap_stat.rds")))
	hist_tmean <- data.table(readRDS(paste0(data_dir, "historical_tmean_overlap_stat.rds")))
	hist <- merge(hist_gdd, hist_tmean)

	rcp45_gdd <- data.table(readRDS(paste0(data_dir, "rcp45_GDD_overlap_stat.rds")))
	rcp45_tmean <- data.table(readRDS(paste0(data_dir, "rcp45_tmean_overlap_stat.rds")))
	rcp45 <- merge(rcp45_gdd, rcp45_tmean)

	rcp85_gdd <- data.table(readRDS(paste0(data_dir, "rcp85_GDD_overlap_stat.rds")))
	rcp85_tmean <- data.table(readRDS(paste0(data_dir, "rcp85_tmean_overlap_stat.rds")))
	rcp85 <- merge(rcp85_gdd, rcp85_tmean)

	
	overlap = rbind(overlap, hist, rcp45, rcp85)
	non_overlap = rbind(non_overlap, hist_non, rcp45_non, rcp85_non)
	rm(hist, rcp45, rcp85, hist_non, rcp45_non, rcp85_non)
}
output_dir= "/data/hydro/users/Hossein/temp_gdd/cleaned_data_for_plot/"
saveRDS(overlap, paste0(output_dir, "overlap_stat.rds"))
saveRDS(non_overlap, paste0(output_dir, "non_overlap_stat.rds"))
rm(data)

