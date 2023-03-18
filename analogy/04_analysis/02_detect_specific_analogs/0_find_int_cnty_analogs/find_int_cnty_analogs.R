.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(raster)
library(FNN)
library(RColorBrewer)
library(colorRamps)
library(EnvStats, lib.loc = "~/.local/lib/R3.5.1")
# library(swfscMisc) # has na.count(.) in it not available in aeolus at this time.

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)

################################################################################
start_time <- Sys.time()
################################################################################
# 
#                   Terminal arguments and parameters
# 
################################################################################
args = commandArgs(trailingOnly=TRUE)

precip = args[1]   # \in {w_precip, no_recip}
emission = args[2] # \in {rcp45, rcp85}
sigma_bd = args[3] # \in {1, 2}, (for 1_sigma or 2_sigma)

print (paste0("precip ", precip))
print (paste0("emission ", emission))
################################################################################
# 
#                   Set up directories
# 
################################################################################
main_in <- "/data/hydro/users/Hossein/analog/02_features_post_biofix/"

out_dir <- "/data/hydro/users/Hossein/analog/03_analogs/biofixed/detected_4_plots/01_intr_cnty_analogs/"
out_dir <- paste0(out_dir, precip, "_", emission, "/", sigma_bd, "_sigma/")

print (paste0("out_dir is : ",  out_dir))
if (dir.exists(out_dir) == F) { dir.create(path = out_dir, recursive = T) }

################################################################################
# 
#                   Read parameters
# 
################################################################################
param_dir <- "/home/hnoorazar/analog_codes/parameters/"

hist_1300_fip_dt <- data.table(read.csv(paste0(param_dir, "all_us_1300_county_fips_locations.csv"),
                                        as.is=T, header=T))

top_3_dir <- "/home/hnoorazar/analog_codes/parameters/top_3_files/"
top_3_name <- paste0(sigma_bd, "_sigma_", precip, "_", emission, "_top_3.csv")
top_3_dt <- data.table(read.csv(paste0(top_3_dir, top_3_name), as.is=T, header=T))
top_3_dt <- within(top_3_dt, remove(top_2_fip, top_3_fip, emission))

################################################################################
# 
#                   Read files
# 
################################################################################
#
future_dt <- data.table(readRDS(paste0(main_in, "CDD_precip_", emission, ".rds")))
hist_dt <- data.table(readRDS(paste0(main_in, "hist_CDD_precip.rds")))
#
# filter the  9 locations that are not in historical data
#
future_dt <- future_dt %>% filter(location %in% hist_dt$location)
#
# add time period to future data so we can use it for filtering
#
future_dt$time_period = 0L
future_dt$time_period[future_dt$year <= 2050] <- "F1"
future_dt$time_period[future_dt$year >= 2051 & future_dt$year <= 2075] <- "F2"
future_dt$time_period[future_dt$year >= 2076] <- "F3"

################################################################################
# 
#                   Run!
# 
################################################################################
#
# add  fips to the data to use it for filtering
#
future_dt <- merge(future_dt, hist_1300_fip_dt, all.x=T, by="location")
hist_dt <- merge(hist_dt, hist_1300_fip_dt, all.x=T, by="location")

n_rows_top_3_dt <- nrow(top_3_dt)

for (top_3_row in seq(n_rows_top_3_dt)){
  
  curr_top_row <- top_3_dt[top_3_row, ]
  
  curr_model <- curr_top_row$model
  curr_time <- curr_top_row$time_period
  curr_fip <- curr_top_row$future_fip
  curr_hist_fip <- curr_top_row$top_1_fip

  curr_future_data <- future_dt %>% 
                      filter(fips == curr_fip & model == curr_model & time_period==curr_time) %>% 
                      data.table()

  curr_hist_data <- hist_dt %>% 
                    filter(fips == curr_hist_fip) %>%
                    data.table()
  curr_ICV <- hist_dt %>% 
              filter(fips == curr_fip) %>%
              data.table()

  n_nghs = nrow(curr_hist_data)
  
  non_numeric_cols <- c("location", "year", "model")

  if (precip == "w_precip") {
    numeric_col = c("yearly_precip", "CumDDinF_Aug23")
    } else {
    numeric_col = c("CumDDinF_Aug23")
  }

  information = find_NN_info_biofix(ICV = curr_ICV, 
                                    historical_dt = curr_hist_data, 
                                    future_dt = curr_future_data, 
                                    n_neighbors = n_nghs, 
                                    numeric_cols = numeric_col, 
                                    non_numeric_cols = non_numeric_cols)
  NN_dist_tb = information[[1]]
  NN_loc_year_tb = information[[2]]
  NN_sigma_tb = information[[3]]

  if (curr_time== "F1"){
    time <- "2026_2050"
    } else if (curr_time== "F2"){
      time <- "2051_2075"
    } else {
      time <- "2076_2095"
  }
  saveRDS(NN_loc_year_tb, paste0(out_dir, "/NN_loc_year_tb_", curr_fip, "_", curr_model, "_", time, ".rds"))
  saveRDS(NN_sigma_tb, paste0(out_dir, "/NN_sigma_tb_", curr_fip, "_", curr_model, "_", time, ".rds"))

}

end_time <- Sys.time()
print(end_time - start_time)





