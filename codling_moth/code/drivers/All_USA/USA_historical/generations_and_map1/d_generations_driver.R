.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(MESS) # has the auc function in it.
library(geepack)
library(chron)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

input_dir = "/data/hydro/users/Hossein/codling_moth_new/all_USA/processed/"
write_dir = "/data/hydro/users/Hossein/analog/usa/data_bases/"
file_name = paste0("combined_CMPOP.rds")

output = generations_func(input_dir, file_name)
generations_Aug = data.table(output[[1]])
generations_Nov = data.table(output[[2]])

saveRDS(generations_Aug, paste0(write_dir, "generations_Aug_combined_CMPOP.rds"))
saveRDS(generations_Nov, paste0(write_dir, "generations_Nov_combined_CMPOP.rds"))