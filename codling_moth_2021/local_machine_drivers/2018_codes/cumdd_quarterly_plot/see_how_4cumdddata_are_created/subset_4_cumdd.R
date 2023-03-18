#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)

write_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/cumdd_data/"
filename = "/data/hydro/users/Hossein/codling_moth_new/local/processed/combined_CMPOP_rcp85.rds"

data <- data.table(readRDS(filename))
data = subset(data, select = c("ClimateGroup", "CountyGroup", "dayofyear", "CumDDinF", "year"))

#data = data[1:as.integer(dim(data)[1]/4), ]
saveRDS(data, paste0(write_dir, "cumdd_CMPOP_rcp85.rds"))

filename = "/data/hydro/users/Hossein/codling_moth_new/local/processed/combined_CMPOP_rcp45.rds"
data <- data.table(readRDS(filename))
data = subset(data, select = c("ClimateGroup", "CountyGroup", "dayofyear", "CumDDinF", "year"))
saveRDS(data, paste0(write_dir, "cumdd_CMPOP_rcp45.rds"))