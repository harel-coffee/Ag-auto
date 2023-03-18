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
# for laptop tests:
sigma_bd <- 1
in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/analogs/detect_4_plots/"
int_name <- "tb_53047_BNU-ESM_2026_2050.rds"
# for laptop tests ^^^^^^^^^^^^^

args = commandArgs(trailingOnly=TRUE)
precip = args[1]   # \in {w_precip, no_recip}
emission = args[2] # \in {rcp45, rcp85}
sigma_bd = args[3] # \in {1, 2}, (for 1_sigma or 2_sigma)
int_name = args[4] # input name, in form of tb_16027_bcc-csm1-1-m_2026_2050.rds


################################################################################
# 
#                   Checking stuff, print to check 
# 
################################################################################
#
# Assuming all fips (of local counties) are 5 digits
#
L <- nchar(int_name)
target_fip <- substr(int_name, 4, 8)
target_model <- substr(int_name, 10, (L-14))
target_time_period <- substr(int_name, (L-12), (L-4))

print (paste0("precip ", precip))
print (paste0("emission ", emission))

print (paste0("int_name is ", int_name))
print (paste0("(target_fip, target_time_period, target_model) = (", 
               target_fip, ", ", target_time_period, ", ", target_model, ")"))

################################################################################
# 
#                   Set up directories
# 
################################################################################

main_in <- "/data/hydro/users/Hossein/analog/03_analogs/biofixed/detected_4_plots/01_intr_cnty_analogs/"
param_dir <- "/home/hnoorazar/analog_codes/parameters/to_detect/"

main_out <- "/data/hydro/users/Hossein/analog/04_analysis/biofixed/analog_points_4_contour_plot/"

in_dir <- paste0(main_in, precip, "_", emission, "/", sigma_bd, "_sigma/")
################################################################################
# 
#                   Read data
# 
################################################################################


NN_loc_year <- data.table(readRDS(paste0(in_dir, "NN_loc_year_", int_name)))
NN_sigma_tb <- data.table(readRDS(paste0(in_dir, "NN_sigma_", int_name)))

year_loc_target <- NN_sigma_tb[, 1:2]

NN_loc_year <- within(NN_loc_year, remove(year, location))
NN_sigma_tb <- within(NN_sigma_tb, remove(year, location))

col_dim <- ncol(NN_sigma_tb)
NN_loc_locations <- NN_loc_year[, c(rep(c(FALSE, TRUE), col_dim)), with = FALSE]
NN_loc_yearsssss <- NN_loc_year[, c(rep(c(TRUE, FALSE), col_dim)), with = FALSE]

NN_loc_yearsssss[NN_sigma_tb >= sigma_bd] <- "NO"
NN_loc_locations[NN_sigma_tb >= sigma_bd] <- "NO"
########################################################################
#
# form the outout here, so, save time, to avoid binding
#
start_time <- Sys.time()
n_ouput_rows <- (nrow(NN_loc_yearsssss) * ncol(NN_loc_yearsssss))
output_dt <- setNames(data.table(matrix(nrow = n_ouput_rows, ncol = 2)), c("year", "location"))
output_dt[, 1] <- rep("NA", n_ouput_rows)
output_dt[, 2] <- rep("NA", n_ouput_rows)
########################################################################

curr_row_output <- 0
for (row in (1 : nrow(NN_loc_yearsssss))){
  for (col in (1 : ncol(NN_loc_yearsssss))){
    if (NN_loc_yearsssss[row, col, with = FALSE] != "NO"){
      curr_row_output = curr_row_output + 1
      # 
      # NN_loc_yearsssss[row, col] would not work, either have to add "with=FALSE"
      # or use NN_loc_yearsssss[row, ..col]
      #  R thing
      #
      output_dt[curr_row_output, 1] <- NN_loc_yearsssss[row, col, with=FALSE]
      output_dt[curr_row_output, 2] <- NN_loc_locations[row, col, with=FALSE]
    }
  }
}

output_dt <- output_dt %>% filter(year != "NA") %>% data.table()

out_dir <- paste0(main_out, sigma_bd, "_sigma/", precip, "_", emission, "/")
if (dir.exists(out_dir) == F) {dir.create(path = out_dir, recursive = T)}
saveRDS(output_dt, paste0(out_dir, "/analogs_year_loc_", 
	                        target_fip, "_", target_model, "_", target_time_period, ".rds"))

print (paste0("it took ",  (Sys.time() - start_time)))









