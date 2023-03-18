#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)

#data_dir = getwd()
data_dir = "/data/hydro/users/giridhar/giridhar/codmoth_pop/"
filename <- paste0(data_dir, "/allData_vertdd_new.rds")
data <- data.table(readRDS(filename))

data = data[, .(cripp

saveRDS(data, paste0(data_dir, "/", "allData_vertdd_map.rds"))
#saveRDS(data, paste0(data_dir, "/", "allData_vertdd.rds"))
