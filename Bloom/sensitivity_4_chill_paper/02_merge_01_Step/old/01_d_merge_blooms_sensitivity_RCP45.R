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

st_dates <- c(1, 7, 14, 21)
dist_means <- c(420, 425, 430, 435, 440, 445, 450, 455)
st_dates <- c(1, 7, 14, 21, 28, 35, 42)
dist_means <- c(327, 348.8, 370.6, 392.4, 414.2, 436, 457.8, 479.6, 501.4, 523.2, 545)

local_files <- c("48.40625_-119.53125", "46.59375_-120.53125",
                 "46.03125_-118.34375", "44.03125_-123.09375")

for (st_date in st_dates){
  for (dt_mu in dist_means){
    for (loc in local_files){
      for (model in models){
        print (paste0("line 39: ", model))
        file_name <- paste0("bloom_", loc, "_start_Jan_", st_date, "_NormalMean_", dt_mu, ".rds")

        curr_45 <- readRDS(paste0(model_in, model, "/rcp45/", file_name))
        rcp45 <- rbind(rcp45, curr_45)
        
        model_counter = model_counter + 1
        print (paste0("model_counter = ", model_counter))
      }
    }
  }
}

rcp45$time_period <- "future"
saveRDS(rcp45, paste0(out_dir, "/future_blooms_rcp45.rds"))

end_time <- Sys.time()
print( end_time - start_time)

