#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
#library(ggplot2)
#library(reshape2)
library(dplyr)
library(foreach)
library(doParallel)
#library(iterators)

#print(getwd())
#data_dir = getwd()
data_dir = "/data/hydro/users/giridhar/giridhar/codmoth_pop"
categories = c("historical", "BNU-ESM", "CanESM2", "GFDL-ESM2G", "bcc-csm1-1-m", "CNRM-CM5", "GFDL-ESM2M")
#file_prefix = "CMPOP_"
file_prefix = "CM_"
locations_list = "list"
data = data.table()
conn = file(paste0(data_dir, "/", locations_list), open = "r")
locations = readLines(conn)
close(conn)

#args = commandArgs(trailingOnly=TRUE)
#category = args[1]
for( category in categories) {
for( location in locations) {
	if(category != "historical") {
		filename <- paste0(data_dir, "/", category, "/rcp45/", file_prefix, location)
	}
	else {
		filename <- paste0(data_dir, "/", category, "/", file_prefix, location)
	}
	#print(filename)
	data <- rbind(data, read.table(filename, header = TRUE, sep = ","))
}
}
##print(data)

#cl <- makeCluster(6)
#registerDoParallel(cl)
#data <- foreach(category = categories, .combine = rbind) %:% 
#  foreach(location = locations, .combine = rbind) %dopar% {
#    filename <- paste0(data_dir, "/", category, "/", file_prefix, location)
#    print(filename)
#    read.table(filename, header = TRUE, sep = ",")
#  }
#stopCluster(cl)
#saveRDS(data, paste0(data_dir, "/", category, "Data_rcp45.rds"))
saveRDS(data, paste0(data_dir, "/combinedData_rcp45.rds"))
