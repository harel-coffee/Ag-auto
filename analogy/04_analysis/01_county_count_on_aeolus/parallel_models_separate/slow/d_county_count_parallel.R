
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(lubridate)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)

options(digit=9)
options(digits=9)

# Check current folder
print("does this look right?")
getwd()
start_time <- Sys.time()

######################################################################
##                                                                  ##
##                     Terminal arguments                           ##
##                                                                  ##
######################################################################

args = commandArgs(trailingOnly=TRUE)
carbon_type= args[1] # rcp45 or rcp85
precip_type= args[2] # include precip or no_precip
sigma_bd= args[3]    # sigma cut off for sigma dissimilarity 1 or 2 or 3 or what?
all_model_names = args[4]

n_nghs = 47841
######################################################################
##                                                                  ##
##                     set up directories                           ##
##                                                                  ##
######################################################################

# main_in <- file.path("/data/hydro/users/Hossein/analog/03_analogs/location_level/")
dt_dir <- file.path(main_in, precip_type, carbon_type, "merged/")

main_out <- file.path("/data/hydro/users/Hossein/analog/04_analysis/")
out_dir <- file.path(main_out, precip_type, "/", sigma_bd, "/")

if (dir.exists(out_dir) == F) { dir.create(path = out_dir, recursive = T) }
print (out_dir)

param_dir <- "/home/hnoorazar/analog_codes/parameters/"

######################################################################
##                                                                  ##
##                          read files                              ##
##                                                                  ##
######################################################################
# all_model_names <- c("bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CNRM-CM5", "GFDL-ESM2G", "GFDL-ESM2M")

all_close_analogs <- data.table()
all_close_analogs_unique <- data.table()

time_periods <- c("_2026_2050", "_2051_2075", "_2076_2095")

for (model_type in all_model_names){
  for (time in time_periods){
    print (paste0(time, " - ", model_type))
    NNs_name <- paste0(dt_dir, "/NN_loc_year_tb_", model_type, time, ".rds")
    dist_name <- paste0(dt_dir, "/NN_dist_tb_", model_type, time,  ".rds")
    sigma_name <- paste0(dt_dir, "/NN_sigma_tb_", model_type, time, ".rds")

    NNs <- data.table(readRDS(NNs_name))
    dists <- data.table(readRDS(dist_name))
    sigmas <- data.table(readRDS(sigma_name))

    print (dim(NNs))
    print (dim(dists))
    print (dim(sigmas))
    
    county_list <- data.table(read.csv(paste0(param_dir, "/Min_fips_st_county_location.csv"), 
                                       header=T, sep=",", as.is=T)
                              )
    print ("line 76")
    a_model_output <- count_NNs_per_counties_all_locs(NNs=NNs, dists=dists, sigmas=sigmas, 
                                                      county_list=county_list, 
                                                      sigma_bd=sigma_bd, novel_thresh=4)

    close_analogs <- a_model_output[[1]]
    close_analogs_unique <- a_model_output[[2]]

    close_analogs$ClimateScenario <- model_type
    close_analogs_unique$ClimateScenario <- model_type

    all_close_analogs <- rbind(all_close_analogs, close_analogs)
    all_close_analogs_unique <- rbind(all_close_analogs_unique, close_analogs_unique)

    saveRDS(all_close_analogs, paste0(out_dir, "all_close_analogs_", model_type, "_", carbon_type, time, ".rds"))
    saveRDS(all_close_analogs_unique, paste0(out_dir, "all_close_analogs_unique_", model_type, "_", carbon_type, time, ".rds"))

  }  
}







