########################################################################
#
# We are trying here to extract the 10 locations chosen for
# chill project stuff, to plot the bloom time for them.
# 
########################################################################
.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)

source_path1 = "/home/hnoorazar/cleaner_codes/core.R"
source_path2 = "/home/hnoorazar/cleaner_codes/core_plot.R"
source(source_path1)
source(source_path2)

#########
######### Running this is gonna take at least 40 minutes, 
######### IF it is successful! R memory allocation problem!
#########
data_dir <- "/data/hydro/users/Hossein/codling_moth_new/local/processed/vertdd_with_new_normal_params/"
out_dir <- "/data/hydro/users/Hossein/chill/frost_bloom_initial_database/bloom_limited_cities/"
param_dir <- "/home/hnoorazar/chilling_codes/parameters/"

if (dir.exists(file.path(out_dir)) == F) {
  dir.create(path = out_dir, recursive = T)
}

data_dt <- readRDS(paste0(data_dir, "vertdd_combined_CMPOP_", "rcp45", ".rds"))
if (!("location" %in% colnames(data_dt))){
data_dt$location <- paste0(data_dt$latitude, "_",data_dt$longitude)
}
print (head(data_dt, 2))
data_dt <- subset(data_dt, select = c("location"))
data_dt <- unique(data_dt)
saveRDS(data_dt, paste0(out_dir, "bloom_locations.rds"))









