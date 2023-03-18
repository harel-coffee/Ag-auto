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
write_dir = "/data/hydro/users/Hossein/codling_moth_new/local/processed/new_4_frost/"
if (dir.exists(write_dir) == F) {dir.create(path = write_dir, recursive = T)}
print(write_dir)

filename <- paste0(data_dir, "vertdd_combined_CMPOP_", version, ".rds")

data <- data.table(readRDS(filename))

st_time <- Sys.time()

data <- bloom(data, bloom_cut_off)
saveRDS(data, paste0(write_dir, "bloom_no_median_full_info_", version, "_", bloom_cut_off, "_new.rds"))


bloom_medians_across_models <- bloom_per_year_median_accross_models(data)
bloom_medians_across_models_time_periods <- bloom_medians_across_models_time_periods(data)

saveRDS(bloom_medians_across_models, paste0(write_dir, "bloom_medians_across_models_", version, "_", bloom_cut_off, "_new.rds"))
saveRDS(bloom_medians_across_models, paste0(write_dir, "bloom_medians_across_models_time_periods_", version, "_", bloom_cut_off, "_new.rds"))

p_dir <- "/home/hnoorazar/chilling_codes/parameters/"
limited_cities <- read.csv(paste0(p_dir, "bloom_limited_cities.csv"), as.is=TRUE)
data$location <- paste0(data$latitude, "_", data$longitude)
data <- data %>% filter(location %in% limited_cities$location) %>% data.table()
limited_cities <- within(limited_cities, remove(lat, long))
data <- merge(data, limited_cities, all.x=TRUE)
saveRDS(data, paste0(write_dir, "bloom_no_median_full_info_limited_CTs_", version, "_", bloom_cut_off, "_new.rds"))

print(  Sys.time() - st_time )


