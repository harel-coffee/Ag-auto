#!/share/apps/R-3.2.2_gcc/bin/Rscript

library(data.table)
library(dplyr)
library(foreach)
library(doParallel)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

data_dir  = "/data/hydro/users/Hossein/codling_moth/local/processed/"
write_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"

args = commandArgs(trailingOnly=TRUE)
version = args[1]

filename <- paste0(data_dir, "vertdd_combined_CMPOP_", version, ".rds")

data <- data.table(readRDS(filename))

data = bloom(data)

saveRDS(data, paste0(write_dir, "bloom_", version, ".rds"))
