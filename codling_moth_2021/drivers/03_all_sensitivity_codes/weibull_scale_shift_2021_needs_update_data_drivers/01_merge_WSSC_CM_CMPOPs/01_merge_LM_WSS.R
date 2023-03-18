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

#####################################################################################
param_dir = "/home/hnoorazar/codling_moth_2021/parameters/"

data_base = "/data/hydro/users/Hossein/codling_moth_2021/03_00_WSS_sensitivity_data/"

locations = data.table(read.csv(paste0(param_dir, "five_locations.csv")))
locations_wCities <- locations
locations = locations$location


shifts = as.character(seq(0, 20, 1)/100)
models = c("BNU-ESM", "CanESM2", "GFDL-ESM2G", "bcc-csm1-1-m", "CNRM-CM5", "GFDL-ESM2M")
emissions = c("rcp45", "rcp85")
#####################################################################################
# /data/hydro/users/Hossein/codling_moth_2021/03_00_WSS_sensitivity_data/0/00_LM_CM/bcc-csm1-1-m
#####################################################################################
all_LM_CM    <- data.table()
all_LM_CMPOP <- data.table()

for (shift in shifts){
  data_dir_S  <- paste0(data_base, shift, "/")
  
  data_dir_S_CM    <- paste0(data_dir_S, "00_LM_CM", "/")
  data_dir_S_CMPOP <- paste0(data_dir_S, "00_LM_CMPOP", "/")

  for (a_model in models){
    data_dir_S_CM_Model <- paste0(data_dir_S_CM, a_model, "/")
    data_dir_S_CMPOP_Model <- paste0(data_dir_S_CMPOP, a_model, "/")

    for (em in emissions){
      data_dir_S_CM_Model_em    <- paste0(data_dir_S_CM_Model, em, "/")
      data_dir_S_CMPOP_Model_em <- paste0(data_dir_S_CMPOP_Model, em, "/")

      for (a_location in locations){
        filename_CM    = paste0(data_dir_S_CM_Model_em, "CM_", a_location)
        filename_CMPOP = paste0(data_dir_S_CMPOP_Model_em, "CMPOP_", a_location)
        data_to_add_CM    = read.table(filename_CM, header = TRUE, sep = ",")
        data_to_add_CMPOP = read.table(filename_CMPOP, header = TRUE, sep = ",")

        data_to_add_CM$model <- a_model
        data_to_add_CMPOP$model <- a_model

        if (em == "rcp45"){
          emiss = "RCP 4.5"
         } else if (em == "rcp85"){
          emiss = "RCP 8.5"
        }

        data_to_add_CM$emission <- emiss
        data_to_add_CMPOP$emission <- emiss

        data_to_add_CM$shift <- shift
        data_to_add_CMPOP$shift <- shift

        all_LM_CM    <- rbind(all_LM_CM   , data_to_add_CM)
        all_LM_CMPOP <- rbind(all_LM_CMPOP, data_to_add_CMPOP)
      }
    }
  }
}

write_dir <- "/data/hydro/users/Hossein/codling_moth_2021/03_01_WSS_CM_CMPOP_Merged/"


locations_wCities <- within(locations_wCities, remove(lat, long))
all_LM_CM$location <- paste0(all_LM_CM$latitude, "_", all_LM_CM$longitude)
all_LM_CMPOP$location <- paste0(all_LM_CMPOP$latitude, "_", all_LM_CMPOP$longitude)

all_LM_CM <- merge(all_LM_CM, locations_wCities, by="location", all.x=T)
all_LM_CMPOP <- merge(all_LM_CMPOP, locations_wCities, by="location", all.x=T)


saveRDS(all_LM_CM, paste0(write_dir, "combined_LM_CM_WSS.rds"))
saveRDS(all_LM_CMPOP, paste0(write_dir, "combined_LM_CMPOP_WSS.rds"))






