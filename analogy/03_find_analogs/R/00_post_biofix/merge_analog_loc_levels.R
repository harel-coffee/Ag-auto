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

precip = args[1] # w_precip # no_recip
gen_3 = args[2]  # w_gen3   # no_gen3
model_ty = args[3] # Do a full name of models
emission = args[4] # rcp45 rcp85

n_nghs = 47841

main_in <- file.path("/data/hydro/users/Hossein/analog/03_analogs/location_level/")
in_dir <- file.path(main_in, gen_3, precip, n_nghs, emission, "/")
out_dir <- file.path(in_dir, "merged/")

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
  NN_dist_tb_list <- NN_dist_tb_list[grep(pattern = time_p, x = NN_dist_tb_list)]
  NN_loc_year_tb_list <- NN_loc_year_tb_list[grep(pattern = time_p, x = NN_loc_year_tb_list)]
  NN_sigma_tb_list <- NN_sigma_tb_list[grep(pattern = time_p, x = NN_sigma_tb_list)]

  NN_dist_tb <- data.table()
  NN_loc_year_tb <- data.table()
  NN_sigma_tb <- data.table()

  for (counter in 1:(length(NN_sigma_tb_list))){
    NN_dist_tb_2026_2050 <- rbind(NN_dist_tb_2026_2050, data.table(readRDS(NN_dist_tb_list_2026_2050[counter])))
    NN_loc_year_tb_2026_2050 <- rbind(NN_loc_year_tb_2026_2050,data.table(readRDS(NN_loc_year_tb_2026_2050[counter])))
    NN_sigma_tb_2026_2050 <- rbind(NN_sigma_tb_2026_2050, data.table(readRDS(NN_loc_year_tb_2026_2050[counter])))
  }
  saveRDS(NN_dist_tb, paste0(out_dir, "/NN_dist_tb_",  model_ty, "_", time_p,  ".rds"))
  saveRDS(NN_loc_year_tb, paste0(out_dir, "/NN_loc_year_tb_", model_ty, "_", time_p,  ".rds"))
  saveRDS(NN_sigma_tb, paste0(out_dir, "/NN_sigma_tb_", model_ty, "_", time_p,  ".rds"))
}

# break it into three time periods

# NN_dist_tb_list_2026_2050 <- NN_dist_tb_list[grep(pattern = "2026_2050", x = NN_dist_tb_list)]
# NN_loc_year_tb_list_2026_2050 <- NN_loc_year_tb_list[grep(pattern = "2026_2050", x = NN_loc_year_tb_list)]
# NN_sigma_tb_list_2026_2050 <- NN_sigma_tb_list[grep(pattern = "2026_2050", x = NN_sigma_tb_list)]


# NN_dist_tb_list_2051_2075 <- NN_dist_tb_list[grep(pattern = "2051_2075", x = NN_dist_tb_list)]
# NN_loc_year_tb_list_2051_2075 <- NN_loc_year_tb_list[grep(pattern = "2051_2075", x = NN_loc_year_tb_list)]
# NN_sigma_tb_list_2051_2075 <- NN_sigma_tb_list[grep(pattern = "2051_2075", x = NN_sigma_tb_list)]


# NN_dist_tb_list_2076_2095 <- NN_dist_tb_list[grep(pattern = "2076_2095", x = NN_dist_tb_list)]
# NN_loc_year_tb_list_2076_2095 <- NN_loc_year_tb_list[grep(pattern = "2076_2095", x = NN_loc_year_tb_list)]
# NN_sigma_tb_list_2076_2095 <- NN_sigma_tb_list[grep(pattern = "2076_2095", x = NN_sigma_tb_list)]


# NN_dist_tb_2026_2050 <- data.table()
# NN_loc_year_tb_2026_2050 <- data.table()
# NN_sigma_tb_2026_2050 <- data.table()

# NN_dist_tb_2051_2075 <- data.table()
# NN_loc_year_tb_2051_2075 <- data.table()
# NN_sigma_tb_2051_2075 <- data.table()

# NN_dist_tb_2076_2095 <- data.table()
# NN_loc_year_tb_2076_2095 <- data.table()
# NN_sigma_tb_2076_2095 <- data.table()

end_time <- Sys.time()
print( end_time - start_time)



