.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_1 = "/home/hnoorazar/bloom_codes/bloom_core.R"
source(source_1)
options(digit=9)
options(digits=9)
start_time <- Sys.time()
###############################################################

# in directory
data_in <- "/data/hydro/users/Hossein/bloom/sensitivity_4_chillPaper/02_merged_01_Step/"
 
# out directory
out_dir <- "/data/hydro/users/Hossein/bloom/sensitivity_4_chillPaper/02_merged_01_Step/"
if (dir.exists(out_dir) == F) {
  dir.create(path = out_dir, recursive = T)}


rcp45 <- readRDS(paste0(data_in, "future_blooms_rcp45.rds"))
rcp85 <- readRDS(paste0(data_in, "future_blooms_rcp85.rds"))


all_bloom <- rbind(rcp45, rcp85)
saveRDS(all_bloom, paste0(out_dir, "/future_blooms.rds"))

end_time <- Sys.time()
print( end_time - start_time)

