#!/share/apps/R-3.2.2_gcc/bin/Rscript

library(data.table)
library(dplyr)
library(foreach)
library(doParallel)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)
             
args = commandArgs(trailingOnly=TRUE)
version = args[1]
bloom_cut_off = as.numeric(args[2])
print(bloom_cut_off)
print(class(bloom_cut_off))

data_dir  = "/data/hydro/users/Hossein/codling_moth_new/local/processed/vertdd_with_new_normal_params/"
write_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/"

filename <- paste0(data_dir, "vertdd_combined_CMPOP_", version, ".rds")

data <- data.table(readRDS(filename))

st_time <- Sys.time()
data <- bloom_per_year(data, bloom_cut_off)
data$location <- paste0(data$latitude, "_", data$longitude)
data <- within(data, remove(latitude, longitude))

saveRDS(data, paste0(write_dir, "bloom_", version, "_", bloom_cut_off, "_new_TS_4_frostPlot.rds"))
print(  Sys.time() - st_time )


