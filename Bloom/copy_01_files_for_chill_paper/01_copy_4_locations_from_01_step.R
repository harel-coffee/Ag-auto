###################################################################
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

options(digits=9)
options(digit=9)

######################################################################
##                                                                  ##
##                                                                  ##
######################################################################
# to use put_time_period() function
source_path = "/home/hnoorazar/chilling_codes/current_draft/chill_core.R"
source(source_path)
options(digit=9)
options(digits=9)


#
# Define main output path
#
data_dir_base <- "/data/hydro/users/Hossein/bloom/01_binary_to_bloom/"
main_out <- "/data/hydro/users/Hossein/bloom/01_binary_to_bloom_copies_of_4_locations_for_chill_paper/"
param_dir = file.path("/home/hnoorazar/bloom_codes/parameters/")

#___________________________________________________________________________________________
locations <- c("48.40625_-119.53125", "44.03125_-123.09375", 
               "46.59375_-120.53125", "46.03125_-118.34375")

modeled_dir_base <- paste0(data_dir_base, "modeled/")
models <- c("bcc-csm1-1", "CanESM2", "CSIRO-Mk3-6-0", "HadGEM2-CC365", 
            "IPSL-CM5A-LR", "MIROC5", "NorESM1-M", "bcc-csm1-1-m", "CCSM4", 
            "GFDL-ESM2G", "HadGEM2-ES365", "IPSL-CM5A-MR", "MIROC-ESM-CHEM",
            "BNU-ESM", "CNRM-CM5", "GFDL-ESM2M", "inmcm4", "IPSL-CM5B-LR", 
            "MRI-CGCM3")

emissions <- c("historical", "rcp45", "rcp85") # 

obs_dir_base <- paste0(data_dir_base, "observed/")
#___________________________________________________________________________________________
modeled_01_data <- data.table()
observed_01_data <- data.table()

start_time <- Sys.time()

#
#  observed loop
#
for (a_location in locations){
  a_file <- data.table(readRDS(paste0(obs_dir_base, "bloom_", a_location, ".rds")))
  a_file$model <- "Observed"
  a_file$emission <- "Observed"
  a_file <- put_time_period(data_tb = a_file, observed=TRUE)
  observed_01_data <- rbind(observed_01_data, a_file)

}

#
#  modeled loop
#
for (a_model in models){
  for (em in emissions){
    curr_dir <- paste0(modeled_dir_base, a_model, "/", em, "/")
    for (a_location in locations){
      a_file <- data.table(readRDS(paste0(curr_dir, "bloom_", a_location, ".rds")))
      a_file$model <- a_model
      if (em == "historical"){
        emm = "Historical"
        } else if (em == "rcp45"){
        emm = "RCP 4.5"
        } else if (em == "rcp85"){
          emm = "RCP 8.5"
      }

      a_file$emission <- emm

      a_file <- put_time_period(data_tb = a_file, observed=FALSE)
      modeled_01_data <- rbind(modeled_01_data, a_file)

    }
  }
}

bloom_01_4locations_for_chill <- rbind(modeled_01_data, observed_01_data)
bloom_01_4locations_for_chill <- data.table(bloom_01_4locations_for_chill)

if (dir.exists(file.path(main_out)) == F) {
  dir.create(path = file.path(main_out), recursive=T)
}

saveRDS(object = bloom_01_4locations_for_chill,
        file   = paste0(main_out, "bloom_01Step_4locations_for_chill.rds")
        )

end_time <- Sys.time()

print( end_time - start_time)

