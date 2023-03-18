#!/share/apps/R-3.2.2_gcc/bin/Rscript
library(chron)
library(data.table)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)

source_path = "/home/hnoorazar/cleaner_codes/core.R"
source(source_path)

input_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"
output_dir = "/data/hydro/users/Hossein/codling_moth/local/processed/"
low_temp = 4.5
up_temp = 24.28
#lower = 10 # 50 F
#upper = 31.11 # 88 F


args = commandArgs(trailingOnly=TRUE)
file_name = args[1]


vertdd = generate_vertdd(input_dir, file_name, lower_temp = low_temp, upper_temp=up_temp)

output_name = paste0("vertdd_", file_name)
saveRDS(vertdd, paste0(output_dir, output_name, ".rds"))