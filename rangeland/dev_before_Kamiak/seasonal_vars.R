
library(data.table)
library(dplyr)

binary_source_path = "/home/h.noorazar/Sid/sidFabio/read_binary_core.R" # Kamiak
source(binary_source_path)
options(digit=9)
options(digits=9)

source_path = "/home/h.noorazar/rangeland/rangeland_core.R"
source(source_path)

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 9)

start_time <- Sys.time()
#
#
#   Years:      1992-2017 (Gridmet data are from 1979)
#   Grid count: 146229 in 25 states of interest.
#   Seasons:    Season 1: Jan - Mar
#               Season 2: Apr - Jul
#               Season 3: Aug - Sep
#               Season 4: Oct - Dec
#
#
########################################################################
######
######      Terminal/shell/bash arguments                      
######
########################################################################

args = commandArgs(trailingOnly=TRUE)
a_state = args[1] 
# model_type = args[2] # observed, modeled_hist, or GFDL-ESM2M HadGEM2-ES365 MIROC-ESM-CHEM IPSL-CM5A-LR NorESM1-M
# model_type = gsub("-", "", model_type)

########################################################################
######
######      Directories
######
########################################################################
data_base <- "/data/project/agaid/h.noorazar/rangeland/"
param_dir <- paste0(data_base, "parameters/")

binary_path_1 <- paste0("/data/project/agaid/rajagopalan_agroecosystems/", 
                        "commondata/meteorologicaldata/gridded/gridMET/gridmet/historical/")

binary_path_2 <- paste0("/data/adam/data/metdata/VIC_ext_maca_v2_binary_westUSA/", model_type, "/rcp85/")

out_dir = "/data/project/agaid/h.noorazar/rangeland/seasonal_variables/00_gridwise/"
if (dir.exists(out_dir) == F) {
  dir.create(path = out_dir, recursive = T)
}
########################################################################
######
######      Body
######
########################################################################


#
#   Define parameters
#
model_type = "observed"


grids_25states <- data.table(read.csv(paste0(param_dir, "grids_25states.csv")))
curr_state_grids = grids_25states[grids_25states$state==a_state]

missing_count <- 0
missing_grids <- setNames(data.table(matrix(nrow = 146229, ncol = 1)), 
                                     c("missing_grids_from_binary_path_1"))
missing_grids$missing_grids_from_binary_path_1 = as.character(missing_grids$missing_grids_from_binary_path_1)


for (a_file in curr_state_grids$grid){
  if (file.exists(paste0(binary_path_1, a_file))){
  	curr_loc_dt <- seasonal_weather_aggregate_oneLoc(path_=binary_path_1, file_name=a_file, data_type_="observed")
  	write.csv(curr_loc_dt, 
              file = paste0(current_out, a_file, ".csv"), 
              row.names=FALSE)

    } else {
      missing_count = missing_count + 1
      missing_grids[missing_count, "missing_grids_from_binary_path_1"] = a_file
  }
}

write.csv(missing_grids, 
          file = paste0(current_out, "missing_grids_from_binary_path_1.csv"), 
          row.names=FALSE)


# How long did it take?
end_time <- Sys.time()
print( end_time - start_time)








