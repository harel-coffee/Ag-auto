#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)

#data_dir = getwd()
data_dir = "/data/hydro/users/giridhar/giridhar/codmoth_pop"
categories = c("historical", "BNU-ESM", "CanESM2", "GFDL-ESM2G", "bcc-csm1-1-m", "CNRM-CM5", "GFDL-ESM2M")
data = data.table()

for(category in categories) {
	#filename <- paste0(data_dir, "/", category, "Data.rds")
	if(category != "historical") {
		filename <- paste0(data_dir, "/", category, "Data_rcp45.rds")
	}
	else {
		filename <- paste0(data_dir, "/", category, "Data.rds")
	}
	#print(filename)
	temp <- data.table(readRDS(filename))
	temp$ClimateScenario <- category
	data <- rbind(data, temp)
}

saveRDS(data, paste0(data_dir, "/", "allData_rcp45.rds"))

#filename = paste0(data_dir, "/", "allData_revised.rds")
