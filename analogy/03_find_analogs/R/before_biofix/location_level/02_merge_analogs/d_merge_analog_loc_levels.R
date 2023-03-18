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

####################################################################################################

#                Terminal arguments and parameters

####################################################################################################
args = commandArgs(trailingOnly=TRUE)
precip = args[1] # w_precip # no_recip
gen_3 = args[2]  # w_gen3   # no_gen3
model_ty = args[3] # Do a full name of models
emission = args[4] # rcp45 rcp85

n_nghs = 47841

main_in <- file.path("/data/hydro/users/Hossein/analog/03_analogs/location_level/")
in_dir <- paste0(main_in, precip, "_", gen_3, "_", emission, "/")
out_dir <- paste0(in_dir, "merged/")
if (dir.exists(out_dir) == F) { dir.create(path = out_dir, recursive = T) }

setwd(in_dir)
getwd()

the_dir <- dir(in_dir, pattern = model_ty)

# remove filenames that aren't of model of interest

NN_dist_tb_list <- the_dir[grep(pattern = "NN_dist_tb_", x = the_dir)]
NN_loc_year_tb_list <- the_dir[grep(pattern = "NN_loc_year_tb_", x = the_dir)]
NN_sigma_tb_list <- the_dir[grep(pattern = "NN_sigma_tb_", x = the_dir)]

time_periods <- c("2026_2050", "2051_2075", "2076_2095")

start_time <- Sys.time()

for (time_p in time_periods){
  NN_dist_tb_list_int <- NN_dist_tb_list[grep(pattern = time_p, x = NN_dist_tb_list)]
  NN_loc_year_tb_list_int <- NN_loc_year_tb_list[grep(pattern = time_p, x = NN_loc_year_tb_list)]
  NN_sigma_tb_list_int <- NN_sigma_tb_list[grep(pattern = time_p, x = NN_sigma_tb_list)]
  
  print (NN_sigma_tb_list_int)
  print (time_p)
  print("")
  print ("__________________________")
  print("")
  NN_dist_tb <- data.table()
  NN_loc_year_tb <- data.table()
  NN_sigma_tb <- data.table()
  
  print(length(NN_sigma_tb_list_int))

  for (counter in 1:(length(NN_sigma_tb_list_int))){
    print(counter)
    NN_dist_tb <- rbind(NN_dist_tb, data.table(readRDS(NN_dist_tb_list_int[counter])))
    NN_loc_year_tb <- rbind(NN_loc_year_tb,data.table(readRDS(NN_loc_year_tb_list_int[counter])))
    NN_sigma_tb <- rbind(NN_sigma_tb, data.table(readRDS(NN_sigma_tb_list_int[counter])))
  }
  saveRDS(NN_dist_tb, paste0(out_dir, "/NN_dist_tb_",  model_ty, "_", time_p,  ".rds"))
  saveRDS(NN_loc_year_tb, paste0(out_dir, "/NN_loc_year_tb_", model_ty, "_", time_p,  ".rds"))
  saveRDS(NN_sigma_tb, paste0(out_dir, "/NN_sigma_tb_", model_ty, "_", time_p,  ".rds"))
}


end_time <- Sys.time()
print( end_time - start_time)



