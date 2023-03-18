#######!/share/apps/R-3.2.2_gcc/bin/Rscript

.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(lubridate)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(chron)
options(digits=9)

source_path = "/home/hnoorazar/codling_moth_2021/core.R"
source(source_path)

#####################################################################################
param_dir = "/home/hnoorazar/codling_moth_2021/parameters/"

data_dir <- "/data/hydro/users/Hossein/codling_moth_2021/03_01_WSS_CM_CMPOP_Merged/"
output_dir <- "/data/hydro/users/Hossein/codling_moth_2021/03_02_WSS_diapause_generations/"
#******************************************************************************************************


shifts = as.character(seq(0, 20, 1)/100)

########################################################################
######
######  observed 
######

output = generations_func(input_dir = data_dir, file_name = "combined_LO_CMPOP_WSS.rds")
generations_Aug = data.table(output[[1]])
generations_Nov = data.table(output[[2]])

saveRDS(generations_Aug, paste0(output_dir, "LO_gens_Aug_combined_CMPOP.rds"))
saveRDS(generations_Nov, paste0(output_dir, "LO_gens_Nov_combined_CMPOP.rds"))

########################################################################
######
######  modeled
######

output = generations_func(input_dir = data_dir, file_name = "combined_LM_CMPOP_WSS.rds")
generations_Aug = data.table(output[[1]])
generations_Nov = data.table(output[[2]])

saveRDS(generations_Aug, paste0(output_dir, "LM_gens_Aug_combined_CMPOP.rds"))
saveRDS(generations_Nov, paste0(output_dir, "LM_gens_Nov_combined_CMPOP.rds"))





