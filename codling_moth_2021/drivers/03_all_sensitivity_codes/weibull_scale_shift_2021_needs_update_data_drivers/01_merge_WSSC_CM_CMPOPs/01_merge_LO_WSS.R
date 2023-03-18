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

data_base = "/data/hydro/users/Hossein/codling_moth_2021/03_00_WSS_sensitivity_data/"
param_dir = "/home/hnoorazar/codling_moth_2021/parameters/"

locations = data.table(read.csv(paste0(param_dir, "five_locations.csv")))
locations_wCities <- locations
locations = locations$location


shifts = as.character(seq(0, 20, 1)/100)
model = "observed"
emission = "observed"
#####################################################################################

#####################################################################################
all_LO_CM    <- data.table()
all_LO_CMPOP <- data.table()

for (shift in shifts){
  data_dir_S  <- paste0(data_base, shift, "/")
  
  data_dir_S_CM    <- paste0(data_dir_S, "00_LO_CM", "/")
  data_dir_S_CMPOP <- paste0(data_dir_S, "00_LO_CMPOP", "/")

  for (a_location in locations){
    filename_CM    = paste0(data_dir_S_CM, "CM_", a_location)
    filename_CMPOP = paste0(data_dir_S_CMPOP, "CMPOP_", a_location)
    data_to_add_CM    = read.table(filename_CM, header = TRUE, sep = ",")
    data_to_add_CMPOP = read.table(filename_CMPOP, header = TRUE, sep = ",")

    data_to_add_CM$model <- "observed"
    data_to_add_CMPOP$model <- "observed"

    data_to_add_CM$emission <- "observed"
    data_to_add_CMPOP$emission <- "observed"

    data_to_add_CM$shift <- shift
    data_to_add_CMPOP$shift <- shift

    all_LO_CM    <- rbind(all_LO_CM   , data_to_add_CM)
    all_LO_CMPOP <- rbind(all_LO_CMPOP, data_to_add_CMPOP)
  }
}

write_dir <- "/data/hydro/users/Hossein/codling_moth_2021/03_01_WSS_CM_CMPOP_Merged/"


locations_wCities <- within(locations_wCities, remove(lat, long))
all_LO_CM$location <- paste0(all_LO_CM$latitude, "_", all_LO_CM$longitude)
all_LO_CMPOP$location <- paste0(all_LO_CMPOP$latitude, "_", all_LO_CMPOP$longitude)

all_LO_CM <- merge(all_LO_CM, locations_wCities, by="location", all.x=T)
all_LO_CMPOP <- merge(all_LO_CMPOP, locations_wCities, by="location", all.x=T)

saveRDS(all_LO_CM, paste0(write_dir, "combined_LO_CM_WSS.rds"))
saveRDS(all_LO_CMPOP, paste0(write_dir, "combined_LO_CMPOP_WSS.rds"))






