#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(dplyr)

data_dir  = "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
output_dir= "/data/hydro/users/Hossein/codling_moth_new/local/processed/section_46_Pest/"
name_pref = "400F_combined_CMPOP_rcp45.rds"
curr_data = data.table(readRDS(paste0(data_dir, name_pref)))
size <- as.integer(dim(curr_data)[1]/8)

curr_data <- curr_data[1:size, ]

saveRDS(curr_data, paste0(output_dir, "eigth_of_rcp45.rds"))