.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)

start_time <- Sys.time()
###
### This takes 2 hours or so.
###

source_path = "/home/hnoorazar/chilling_codes/current_draft/chill_core.R"
source(source_path)
options(digit=9)
options(digits=9)

historical <- data.table()
rcp45 <- data.table()
rcp85 <- data.table()

main_in <- "/data/hydro/users/Hossein/chill/frost_bloom_initial_database/modeled/"
out_dir <- "/data/hydro/users/Hossein/chill/frost_bloom_initial_database/frost_RDS/"
if (dir.exists(out_dir) == F) { dir.create(path = out_dir, recursive = T) }

models <- c("bcc-csm1-1", "bcc-csm1-1-m", "BNU-ESM", "CanESM2", 
            "CCSM4", "CNRM-CM5", "CSIRO-Mk3-6-0", "GFDL-ESM2G", 
            "GFDL-ESM2M", "HadGEM2-CC365", "HadGEM2-ES365", "inmcm4", 
            "IPSL-CM5A-LR", "IPSL-CM5A-MR", "IPSL-CM5B-LR", "MIROC5", 
            "MIROC-ESM-CHEM", "MRI-CGCM3", "NorESM1-M")

emission_types <- c("historical", "rcp45", "rcp85")
model_counter = 0
for (model in models){
  print (paste0("line 33: ", model))
  curr_hist <- data.table(readRDS(paste0(main_in, model, "/historical_data_till_Jan.rds")))
  curr_45 <- data.table(readRDS(paste0(main_in, model, "/rcp45_data_till_Jan.rds")))
  curr_85 <- data.table(readRDS(paste0(main_in, model, "/rcp85_data_till_Jan.rds")))
    
  curr_hist$model <- model
  curr_45$model <- model
  curr_85$model <- model

  curr_45$emission <- "RCP 4.5"
  curr_85$emission <- "RCP 8.5"
  print (paste("dimensions curr_hist are "))
  print (dim(curr_hist))
  print ("__________________")
  print (paste("dimensions curr_45 are "))
  print (dim(curr_45))
  print ("__________________")
  print (paste("dimensions curr_85 are "))
  print (dim(curr_85))
  print ("__________________")
  
  historical <- rbind(historical, curr_hist)
  rcp45 <- rbind(rcp45, curr_45)
  rcp85 <- rbind(rcp85, curr_85)
  
  model_counter = model_counter + 1
  print (paste0("model_counter = ", model_counter))
  print ("__________________")
  print (paste("dimensions historical are "))
  print (dim(historical))
  print ("__________________")
  print (paste("dimensions rcp45 are "))
  print (dim(rcp45))
  print ("__________________")
  print (paste("dimensions rcp85 are "))
  print (dim(rcp85))

}

historical_45 <- historical
historical_85 <- historical

historical_45$emission <- "RCP 4.5"
historical_85$emission <- "RCP 8.5"
rm(historical)

main_in_obs <- "/data/hydro/users/Hossein/chill/frost_bloom_initial_database/"
observed <- data.table(readRDS(paste0(main_in_obs, "observed_dt_till_Jan.rds")))
print ("dim observed is:")
print (dim(observed))
print ("line 71")
observed_45 <- observed
observed_85 <- observed

rm(observed)
observed_45$emission <- "RCP 4.5"
observed_85$emission <- "RCP 8.5"

rcp45 <- rbind(rcp45, historical_45, observed_45)
rcp85 <- rbind(rcp85, historical_85, observed_85)

rm(historical_45, historical_85, observed_45, observed_85)
saveRDS(rcp45, paste0(out_dir, "/rcp45_till_Jan.rds"))
saveRDS(rcp85, paste0(out_dir, "/rcp85_till_Jan.rds"))
print ("line 85")
# ###################################################################
# #
# #     Compute first and fifth frost day of year
# #
# ###################################################################
in_n_out <- "/data/hydro/users/Hossein/chill/frost_bloom_initial_database/frost_RDS/"
# rcp45 <- data.table(readRDS(paste0(in_n_out, "rcp45_till_Jan.rds")))
# rcp85 <- data.table(readRDS(paste0(in_n_out, "rcp85_till_Jan.rds")))

first_frost_45 <- kth_smallest_in_group(rcp45, target_column = "chill_dayofyear", k=1)
first_frost_85 <- kth_smallest_in_group(rcp85, target_column = "chill_dayofyear", k=1)
first_frost <- rbind(first_frost_45, first_frost_85)
saveRDS(first_frost, paste0(out_dir, "first_frost_till_Jan.rds"))
rm(first_frost_45, first_frost_85, first_frost)

print ("line 101")
fifth_frost_45 <- kth_smallest_in_group(rcp45, target_column = "chill_dayofyear", k=5)
fifth_frost_85 <- kth_smallest_in_group(rcp85, target_column = "chill_dayofyear", k=5)
fifth_frost <- rbind(fifth_frost_45, fifth_frost_85)
saveRDS(fifth_frost, paste0(out_dir, "fifth_frost_till_Jan.rds"))

rm(fifth_frost_45, fifth_frost_85, fifth_frost)
####################################################################
#
#      Compute medians
#
####################################################################
in_n_out <- "/data/hydro/users/Hossein/chill/frost_bloom_initial_database/frost_RDS/"

#######################################
####################################### FIRST frost median (1 st)
#######################################

first_frost <- data.table(readRDS(paste0(in_n_out, "first_frost_till_Jan.rds")))

print ("line 122")
first_frost_medians <- first_frost %>%
                       group_by(time_period, model, location, emission) %>%
                       summarise(median = median(chill_dayofyear)) %>%
                       data.table()

saveRDS(first_frost_medians, paste0(out_dir, "first_frost_medians_till_Jan.rds"))
rm(first_frost_medians)

#######################################
####################################### FIFTH frost median (5 th)
#######################################
fifth_frost <- data.table(readRDS(paste0(in_n_out, "fifth_frost_till_Jan.rds")))

fifth_frost_medians <- fifth_frost %>%
                       group_by(time_period, model, location, emission) %>%
                       summarise(median = median(chill_dayofyear)) %>%
                       data.table()

saveRDS(fifth_frost_medians, paste0(out_dir, "fifth_frost_medians_till_Jan.rds"))


end_time <- Sys.time()
print( end_time - start_time)




