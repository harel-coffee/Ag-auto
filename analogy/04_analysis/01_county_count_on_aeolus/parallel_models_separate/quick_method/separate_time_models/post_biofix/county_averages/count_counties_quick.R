
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
sigma_bd = args[3]   # sigma cut off for sigma dissimilarity 1 or 2 or 3 or what?
all_model_names <- args[4]
time_periods <- args[5]

n_nghs = 47841
######################################################################
##                                                                  ##
##                     set up directories                           ##
##                                                                  ##
######################################################################

main_in <- file.path("/data/hydro/users/Hossein/analog/03_analogs/biofixed/county_averages/", "/")
dt_dir <- paste0(main_in, precip_type, "_", carbon_type, "/merged/")

main_out <- file.path("/data/hydro/users/Hossein/analog/04_analysis/biofixed/county_averages/")
out_dir <- paste0(main_out, sigma_bd, "_sigma/", precip_type, "_", carbon_type, "/")

if (dir.exists(out_dir) == F) { dir.create(path = out_dir, recursive = T) }
print ("__________________________")
print ("out_dir:")
print (out_dir)
print ("__________________________")
param_dir <- "/home/hnoorazar/analog_codes/parameters/"

######################################################################
##                                                                  ##
##                          read files                              ##
##                                                                  ##
######################################################################

for (model_type in all_model_names){
  for (time in time_periods){
    print (paste0(time, " - ", model_type))
    st_time <- Sys.time()
    NNs_name <- paste0(dt_dir, "/NN_loc_year_tb_", model_type, time, ".rds")
    # dist_name <- paste0(dt_dir, "/NN_dist_tb_", model_type, time,  ".rds")
    sigma_name <- paste0(dt_dir, "/NN_sigma_tb_", model_type, time, ".rds")

    NNs <- data.table(readRDS(NNs_name))
    sigmas <- data.table(readRDS(sigma_name))

    print (dim(NNs))
    print (dim(sigmas))
    
    county_list <- data.table(read.csv(paste0(param_dir, "/Min_fips_st_county_location.csv"), 
                                       header=T, sep=",", as.is=T))
    
    a_model_analog_output <- count_analogs_counties_quick_cnty_avgs(NNs, sigmas, 
                                                                    county_list, 
                                                                    sigma_bd=sigma_bd)
    print ("line 80")
    a_model_novel_output <- count_novel_quick_cnty_avgs(NNs, sigmas, county_list, novel_bd=4)
    print ("line 82")

    a_model_analog_output$model <- model_type
    a_model_novel_output$model <- model_type

    saveRDS(a_model_analog_output,paste0(out_dir, "analog_",model_type, "_", carbon_type, time, ".rds"))
    saveRDS(a_model_novel_output, paste0(out_dir, "novel_", model_type, "_", carbon_type, time, ".rds"))
    print (Sys.time() - st_time)
  }  
}











