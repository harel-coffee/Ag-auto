.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(chillR)
library(tidyverse)
library(lubridate)
             
source_path = "/home/hnoorazar/chilling_codes/current_draft/chill_core.R"
source(source_path)
options(digit=9)
options(digits=9)


###########################################################################
param_dir <- "/home/hnoorazar/chilling_codes/parameters/"
data_dir_base <- "/data/hydro/users/Hossein/chill_proj/data_by_core/dynamic/01/sept/"


observed_base <- paste0(data_dir_base, "observed/")
modeled_base <- paste0(data_dir_base, "modeled/")

model_names <- c("MRI-CGCM3", "bcc-csm1-1", "BNU-ESM", "CCSM4", "CSIRO-Mk3-6-0",
                 "GFDL-ESM2M", "HadGEM2-ES365", "IPSL-CM5A-LR", "IPSL-CM5B-LR",
                 "MIROC-ESM-CHEM", "NorESM1-M", "bcc-csm1-1-m", "CanESM2",   
                 "CNRM-CM5", "GFDL-ESM2G", "HadGEM2-CC365", 
                 "inmcm4", "IPSL-CM5A-MR", "MIROC5")

modeled_emissions <- c("historical", "rcp45", "rcp85")

four_locations <- read.csv(paste0(param_dir, "4_locations.csv"), as.is=TRUE)

four_locations$location <- paste0(four_locations$lat, "_", four_locations$long)
four_locations_vector <- four_locations$location
four_locations <- within(four_locations, remove(lat, long))

pre_name <- "chill_output_data_"

## 
## observed
## 
observed_data <- data.table()

for (a_location in four_locations_vector){
  curr_location <- read.table(file = paste0(observed_base, pre_name, a_location, ".txt"),
                              header = T,
                              colClasses = c("factor", "numeric", "numeric", "numeric",
                                             "numeric", "numeric"))
  curr_location$location <- a_location
  observed_data <- rbind(observed_data, curr_location)

}

observed_data$emission <- "observed"
observed_data$model <- "observed"

###
### trim the incomplete chill seasons 
###
observed_data <- observed_data %>% filter(chill_season != "chill_1978-1979")
observed_data <- observed_data %>% filter(chill_season != "chill_2015-2016")

out_dir <- "/data/hydro/users/Hossein/chill_proj/data_by_core/dynamic/combined_01Step_for_fourLocations_paper_revision/"

if (dir.exists(file.path(out_dir)) == F) {
  dir.create(path = out_dir, recursive = T)
}

observed_data <- dplyr::left_join(x = observed_data, y = four_locations, by = "location")
write.table(observed_data, file = paste0(out_dir, "observed_daily_CP_for_CDF.csv"),
            row.names=FALSE, col.names=TRUE, na="", sep=",")

print (head(observed_data, 2))


## 
## modeled
## 
modeled_data <- data.table()

for (a_location in four_locations_vector){
  for (a_model in model_names){

  	curr_location_45 <- read.table(file = paste0(modeled_base, a_model, "/rcp45/", pre_name, a_location, ".txt"),
                                   header = T,
                                   colClasses = c("factor", "numeric", "numeric", "numeric",
                                                  "numeric", "numeric"))

    curr_location_85 <- read.table(file = paste0(modeled_base, a_model, "/rcp85/", pre_name, a_location, ".txt"),
                                   header = T,
                                   colClasses = c("factor", "numeric", "numeric", "numeric",
                                                  "numeric", "numeric"))

    curr_location_hist <- read.table(file = paste0(modeled_base, a_model, "/historical/", pre_name, a_location, ".txt"),
                                     header = T,
                                     colClasses = c("factor", "numeric", "numeric", "numeric",
                                                    "numeric", "numeric"))

    
    curr_location_45$emission <- "RCP 4.5"
    curr_location_85$emission <- "RCP 8.5"
    curr_location_hist$emission <- "modeled historical"

    curr_location_45$location <- a_location
    curr_location_85$location <- a_location
    curr_location_hist$location <- a_location

    curr_location_45$model <- a_model
    curr_location_85$model <- a_model
    curr_location_hist$model <- a_model

    ###
    ### trim the incomplete chill seasons 
    ###

    curr_location_45 <- curr_location_45 %>% filter(chill_season != "chill_2005-2006")
    curr_location_45 <- curr_location_45 %>% filter(chill_season != "chill_2099-2100")

    curr_location_85 <- curr_location_85 %>% filter(chill_season != "chill_2005-2006")
    curr_location_85 <- curr_location_85 %>% filter(chill_season != "chill_2099-2100")

    curr_location_hist <- curr_location_hist %>% filter(chill_season != "chill_1949-1950")
    curr_location_hist <- curr_location_hist %>% filter(chill_season != "chill_2005-2006")

    
    modeled_data <- rbind(modeled_data, curr_location_45, curr_location_85, curr_location_hist)

  }
  
}

modeled_data <- dplyr::left_join(x = modeled_data, y = four_locations, by = "location")

print (dim(modeled_data))
print (dim(observed_data))
modeled_data <- rbind(modeled_data, observed_data)
print (dim(modeled_data))



out_dir <- "/data/hydro/users/Hossein/chill_proj/data_by_core/dynamic/combined_01Step_for_fourLocations_paper_revision/"

if (dir.exists(file.path(out_dir)) == F) {
  dir.create(path = out_dir, recursive = T)
}

write.table(modeled_data, file = paste0(out_dir, "daily_CP_for_CDF.csv"),
            row.names=FALSE, col.names=TRUE, na="", sep=",")








