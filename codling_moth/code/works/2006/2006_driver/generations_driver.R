#!/share/apps/R-3.2.2_gcc/bin/Rscript
#library(chron)
library(data.table)
library(ggplot2)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

args = commandArgs(trailingOnly=TRUE)
version = args[1]

write_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"
input_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"
file_name = paste0("combined_CMPOP_", version, ".rds")


output = generations_func(input_dir, file_name)
generations = data.table(output[[1]])
generations1 = data.table(output[[2]])

saveRDS(generations, paste0(write_dir, "generations_combined_CMPOP_", version, ".rds"))
saveRDS(generations1, paste0(write_dir, "generations1_combined_CMPOP_", version, ".rds"))