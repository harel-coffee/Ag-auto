.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(raster)
library(FNN)
library(RColorBrewer)
library(colorRamps)
library(adehabitatLT)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)

################################################################################
# 
#                   Directories
# 
################################################################################
main_dir <- "/data/hydro/users/Hossein/analog/"

main_us_dir <- file.path(main_dir, "usa/ready_features/")
main_local_dir <- file.path(main_dir, "local/ready_features/one_file_4_all_locations/")
main_out <- file.path(main_dir, "03_analogs/sigma/")
################################################################################
################################################################################
# 
#                   Terminal arguments
# 
################################################################################
args = commandArgs(trailingOnly=TRUE)
model_type = args[1]
emission_type = args[2]

################################################################################
# 
#                   Read Data
# 
################################################################################

all_dt_usa <- data.table(readRDS(paste0(main_us_dir, "all_data_usa.rds")))
local_dt <- data.table(readRDS(paste0(main_local_dir, "feat_", model_type, "_", emission_type, ".rds")))

###########################################################################
# create subdirectory for specific emission types

out_dir = file.path(main_out, "no_ICV", emission_type)
if (dir.exists(out_dir) == F) {
  dir.create(path = out_dir, recursive = T)
}
################################################################################
# 
#                   Set parameters and run the code
# 
################################################################################

n_neighbors = 500

information = find_NN_info_W4G(all_dt_usa, local_dt, n_neighbors=n_neighbors)

NN_dist_tb = information[[1]]
NN_loc_year_tb = information[[2]]
NN_sigma_tb = information[[3]]

saveRDS(NN_dist_tb, paste0(out_dir, "/NN_dist_tb_", model_type, ".rds"))
saveRDS(NN_loc_year_tb, paste0(out_dir, "/NN_loc_year_tb_", model_type, ".rds"))
saveRDS(NN_sigma_tb, paste0(out_dir, "/NN_sigma_tb_", model_type, ".rds"))


