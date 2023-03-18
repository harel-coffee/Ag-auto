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
bloom_base <- "/data/hydro/users/Hossein/bloom/sensitivity_4_chillPaper/"
model_in <- paste0(bloom_base, "/01_modeled_bloom/")

# out directory
out_dir <- "/data/hydro/users/Hossein/bloom/sensitivity_4_chillPaper/02_merged_01_Step/"
if (dir.exists(out_dir) == F) {
  dir.create(path = out_dir, recursive = T)}

models <- c("bcc-csm1-1", "bcc-csm1-1-m", "BNU-ESM", "CanESM2", 
            "CCSM4", "CNRM-CM5", "CSIRO-Mk3-6-0", "GFDL-ESM2G", 
            "GFDL-ESM2M", "HadGEM2-CC365", "HadGEM2-ES365", "inmcm4", 
            "IPSL-CM5A-LR", "IPSL-CM5A-MR", "IPSL-CM5B-LR", "MIROC5", 
            "MIROC-ESM-CHEM", "MRI-CGCM3", "NorESM1-M")
model_counter = 0

rcp45 <- data.table()

for (model in models){
  print (paste0("line 39: ", model))
  file_name <- paste0("all_dt.rds")

  curr_45 <- readRDS(paste0(model_in, model, "/rcp45/", file_name))
  rcp45 <- rbind(rcp45, curr_45)
  
  model_counter = model_counter + 1
  print (paste0("model_counter = ", model_counter))
}

rcp45$time_period <- "future"
saveRDS(rcp45, paste0(out_dir, "/future_blooms_rcp45.rds"))

end_time <- Sys.time()
print( end_time - start_time)

