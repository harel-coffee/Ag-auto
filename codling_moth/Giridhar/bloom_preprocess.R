#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)

#data_dir = getwd()
data_dir = "/data/hydro/users/giridhar/giridhar/codmoth_pop/"
#filename <- paste0(data_dir, "/allData_vertdd_new.rds")
filename <- paste0(data_dir, "/allData_vertdd_new_rcp45.rds")
data <- data.table(readRDS(filename))

data = subset(data, select = c("ClimateGroup","latitude", "longitude","County","ClimateScenario","year","month","day","dayofyear","cripps_pink","gala","red_deli"))
data = melt(data, id.vars = c("ClimateGroup","latitude", "longitude","County","ClimateScenario","year","month","day","dayofyear"), variable.name = "apple_type")
data = data[value >= 1.000000e+00,]
data = data[, head(.SD, 1), by = c("ClimateGroup","latitude", "longitude","County","ClimateScenario","year", "apple_type")]
data = data[, .(medDoY = as.integer(median(dayofyear))), by = c("ClimateGroup","latitude", "longitude","County", "apple_type")]

saveRDS(data, paste0(data_dir, "/", "allData_bloom_rcp45.rds"))
