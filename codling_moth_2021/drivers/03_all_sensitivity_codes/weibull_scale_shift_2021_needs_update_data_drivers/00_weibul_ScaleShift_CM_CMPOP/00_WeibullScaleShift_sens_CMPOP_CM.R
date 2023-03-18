#######!/share/apps/R-3.2.2_gcc/bin/Rscript

.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(lubridate)
library(reshape2)
library(dplyr)
library(foreach)
library(iterators)
library(chron)
options(digits=9)

source_path = "/home/hnoorazar/codling_moth_2021/core.R"
source(source_path)

write_path = "/data/hydro/users/Hossein/codling_moth_2021/03_00_WSS_sensitivity_data/"
param_dir  =   "/home/hnoorazar/codling_moth_2021/parameters/"

sens_param_dir <- paste0(param_dir, "sensitivity_params/")

locations = data.table(read.csv(paste0(param_dir, "five_locations.csv")))
locations = locations$location

time_periods = list("1979-2015", "2040s", "2060s", "2080s")

#
#   This is weird. cellByCounty is accessible inside 
#   functions produce_CMPOP_local() and produce_CM_local().
#
cellByCounty = data.table(read.csv(paste0(param_dir, "CropParamCRB.csv")))

#####*********************
# 
# historical here means observed
# 
#####*********************

## move models to .sh files. 
## models = c("bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CNRM-CM5", "GFDL-ESM2G", "GFDL-ESM2M", "historical")

# scale_shift has to be in percent format.
# scale_shift = seq(0, 20, 1)/100

args = commandArgs(trailingOnly=TRUE)
models = args[1]
scale_shift = as.numeric(args[2])

emissions = c("rcp45", "rcp85")

start_h = 1979
end_h = 2015

start_f = 2006
end_f = 2099

for (shift in scale_shift){
    cod_param = paste0("CodlingMothparameters_", shift, ".txt")
    #############
    ############# CMPOP
    #############
    for(model in models) {
        if(model == "observed") {
            raw_data_dir = "/data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016/"
            cod_param = "CodlingMothparameters_0.txt"
            for(location in locations){
                filename = paste0("data_", location)

                temp_data <- produce_CMPOP_local(input_folder = raw_data_dir, 
                                                 noVariables = 8, # observed have 8 columns
                                                 filename = filename,
                                                 param_dir = sens_param_dir, 
                                                 cod_moth_param_name = cod_param,
                                                 scale_shift = shift,
                                                 start_year = start_h, 
                                                 end_year = end_h, 
                                                 lower = 10, 
                                                 upper = 31.11,
                                                 location = location, 
                                                 category = model)
                
                write_dir = paste0(write_path, shift, "/", "00_LO_CMPOP/")
                dir.create(file.path(write_dir), recursive = TRUE)
                write.table(temp_data, file = paste0(write_dir, "/CMPOP_", location), 
                                       sep = ",", 
                                       row.names = FALSE, 
                                       col.names = TRUE)
            }
        }
        else{
            raw_data_dir = "/data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/"
            for(version in emissions) {
                for(location in locations) {
                    filename <- paste0(model, "/", version, "/", "data_", location)
                    temp_data <- produce_CMPOP_local(input_folder = raw_data_dir, 
                                                     noVariables = 4, # modeled have 4 columns
                                                     filename = filename,
                                                     param_dir = sens_param_dir, 
                                                     cod_moth_param_name = cod_param,
                                                     scale_shift = shift,
                                                     start_year = start_f, 
                                                     end_year = end_f, 
                                                     lower = 10, 
                                                     upper = 31.11,
                                                     location = location, 
                                                     category = model)

                    write_dir <- paste0(write_path, shift, "/", "00_LM_CMPOP/", model, "/", version)
                    dir.create(file.path(write_dir), recursive = TRUE)
                    write.table(temp_data, file = paste0(write_dir, "/CMPOP_", location), 
                                           sep = ",", 
                                           row.names = FALSE, 
                                           col.names = TRUE)
                }
            }
        }
    }
    #############
    ############# CM
    #############
    for(model in models) {
        if(model == "observed") {
            raw_data_dir = "/data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016/"
            cod_param = "CodlingMothparameters_0.txt"
            for(location in locations) {
                filename = paste0("data_", location)
                
                temp_data <- produce_CM_local(input_folder = raw_data_dir, 
                                              NoVariables = 8,  # historical have 8 columns
                                              filename = filename,
                                              param_dir = sens_param_dir, 
                                              cod_moth_param_name = cod_param,
                                              scale_shift = shift,
                                              start_year = start_h, 
                                              end_year = end_h, 
                                              lower = 10, 
                                              upper = 31.11,
                                              location = location, 
                                              category = model)

                write_dir = paste0(write_path, shift, "/", "00_LO_CM/")
                dir.create(file.path(write_dir), recursive = TRUE)
                write.table(temp_data, file = paste0(write_dir, "CM_", location), 
                                       sep = ",", 
                                       row.names = FALSE, 
                                       col.names = TRUE)
            }
        }
        ##
        ## FUTURE CM
        ##
        else{
            raw_data_dir = "/data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/"
            for(version in emissions){
                for(location in locations) {
                    filename = paste0(model, "/", version, "/", "data_", location)
                    temp_data <- produce_CM_local(input_folder = raw_data_dir, 
                                                  NoVariables = 4,  # modeled have 4 columns
                                                  filename = filename,
                                                  param_dir = sens_param_dir, 
                                                  cod_moth_param_name = cod_param,
                                                  scale_shift = shift,
                                                  start_year = start_f, 
                                                  end_year = end_f, 
                                                  lower = 10, 
                                                  upper = 31.11,
                                                  location = location, 
                                                  category = model)

                    write_dir = paste0(write_path, shift, "/", "00_LM_CM/", model, "/", version)
                    dir.create(file.path(write_dir), recursive = TRUE)
                    write.table(temp_data, file = paste0(write_dir, "/CM_", location), 
                                sep = ",", 
                                row.names = FALSE, 
                                col.names = TRUE)
                }
            }
        }
    }
}


