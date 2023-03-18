.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(raster)
library(FNN)
library(RColorBrewer)
library(colorRamps)
library(EnvStats, lib.loc = "~/.local/lib/R3.5.1")

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)

################################################################################
################################################################################
# 
#                   Terminal arguments and parameters
# 
################################################################################
args = commandArgs(trailingOnly=TRUE)

precip = args[1]   # w_precip # no_recip
gen_3 = args[2]    # w_gen3   # no_gen3
model_ty = args[3] # bcc, BNU stuff
emission = args[4] # rcp45 or rcp85 
int_file = args[5] # file of interest that looks like: feat_45_96875_119_34375_2085.rds

n_nghs = 35000

################################################################################
# 
#                     set up proper directories
# 
################################################################################

main_dir <- "/data/hydro/users/Hossein/analog/"

main_us_dir <- file.path(main_dir, "usa/ready_features/")
main_local_dir <- file.path(main_dir, "local/ready_features/broken_down_location_year_level/")
int_local_dir <- file.path(main_local_dir, emission, model_ty, "/")

main_out <- file.path(main_dir, "03_analogs/fine_analogs/")
out_dir = file.path(main_out, gen_3, precip, n_nghs, emission, model_ty)

if (dir.exists(out_dir) == F) { dir.create(path = out_dir, recursive = T) }

################################################################################
# 
#                                 Read Data
# 
################################################################################

all_dt_usa <- data.table(readRDS(paste0(main_us_dir, "all_data_usa.rds")))
local_dt <- data.table(readRDS(paste0(int_local_dir, int_file)))

################################################################################
start_time <- Sys.time()

if (precip== "w_precip") {precip= TRUE} else {precip= FALSE}
if (gen_3 == "w_gen3")   {gen_3 = TRUE} else {gen_3 = FALSE}


information = find_NN_info_W4G_ICV(ICV=all_dt_usa, 
                                   historical_dt=all_dt_usa, 
                                   future_dt=local_dt, 
                                   n_neighbors=n_nghs, 
                                   precipitation=precip,
                                   gen3 = gen_3)

NN_dist_tb = information[[1]]
NN_loc_year_tb = information[[2]]
NN_sigma_tb = information[[3]]

################################################################################
# 
#                                 extract location and year
# 
################################################################################
lat <- substr(int_file, 6, 13)
long <- substr(int_file, 16, 23)
yr <- substr(int_file, 25, 28)

lat_long_yr_extension <- substr(int_file, 6, 32)

################################################################################

saveRDS(NN_dist_tb, paste0(out_dir, "/NN_dist_tb_", lat_long_yr_extension))
saveRDS(NN_loc_year_tb, paste0(out_dir, "/NN_loc_year_tb_", lat_long_yr_extension))
saveRDS(NN_sigma_tb, paste0(out_dir, "/NN_sigma_tb_", lat_long_yr_extension))

end_time <- Sys.time()
print( end_time - start_time)

