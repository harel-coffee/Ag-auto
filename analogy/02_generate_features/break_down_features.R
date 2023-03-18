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


################################################################################
# 
#                     set up proper directories
# 
################################################################################

main_dir <- "/data/hydro/users/Hossein/analog/"
main_local_dir <- file.path(main_dir, "local/ready_features/one_file_4_all_locations/")

main_out <- file.path(main_dir, "03_analogs")

out_dir = file.path(main_out, gen_3, precip, n_nghs, emission_type)
if (dir.exists(out_dir) == F) {
  dir.create(path = out_dir, recursive = T)
}

################################################################################
# 
#                   Read Data
# 
################################################################################

all_dt_usa <- data.table(readRDS(paste0(main_us_dir, "all_data_usa.rds")))
local_dt <- data.table(readRDS(paste0(main_local_dir, "feat_", model_type, "_", emission_type, ".rds")))

################################################################################
start_time <- Sys.time()

# run the model for separate time frames
time_frames = c("2026_2050") #

if (precip== "w_precip") {precip= TRUE} else {precip= FALSE}
if (gen_3 == "w_gen3")   {gen_3 = TRUE} else {gen_3 = FALSE}

for (time in time_frames){
  if (time == time_frames[1]){
    local_dt_time <- local_dt %>% filter(year >= 2026 & year <= 2050)
    } else if (time == time_frames[2]) {
      local_dt_time <- local_dt %>% filter(year >= 2051 & year <= 2075)
    } else if (time == time_frames[3]) {
     local_dt_time <- local_dt %>% filter(year >= 2076 & year <= 2095)
  }

  information = find_NN_info_W4G_ICV(ICV=all_dt_usa, 
                                     historical_dt=all_dt_usa, 
                                     future_dt=local_dt_time, 
                                     n_neighbors=n_nghs, 
                                     precipitation=precip,
                                     gen3 = gen_3)

  NN_dist_tb = information[[1]]
  NN_loc_year_tb = information[[2]]
  NN_sigma_tb = information[[3]]

  saveRDS(NN_dist_tb, paste0(out_dir, "/NN_dist_tb_", model_type, "_", time,  ".rds"))
  saveRDS(NN_loc_year_tb, paste0(out_dir, "/NN_loc_year_tb_", model_type, "_", time,  ".rds"))
  saveRDS(NN_sigma_tb, paste0(out_dir, "/NN_sigma_tb_", model_type, "_", time,  ".rds"))
}

end_time <- Sys.time()
print( end_time - start_time)

