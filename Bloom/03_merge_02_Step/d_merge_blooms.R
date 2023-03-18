.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

source_1 = "/home/hnoorazar/bloom_codes/bloom_core.R"
source(source_1)
options(digit=9)
options(digits=9)
start_time <- Sys.time()
###############################################################

# in directory
bloom_base <- "/data/hydro/users/Hossein/bloom/"
main_in <- paste0(bloom_base, "02_bloomCut_first_frost/bloom/")
model_in <- paste0(main_in, "modeled/")

# out directory
out_dir <- "/data/hydro/users/Hossein/bloom/03_merge_02_Step/"
if (dir.exists(out_dir) == F) {
  dir.create(path = out_dir, recursive = T)}

models <- c("bcc-csm1-1", "bcc-csm1-1-m", "BNU-ESM", "CanESM2", 
            "CCSM4", "CNRM-CM5", "CSIRO-Mk3-6-0", "GFDL-ESM2G", 
            "GFDL-ESM2M", "HadGEM2-CC365", "HadGEM2-ES365", "inmcm4", 
            "IPSL-CM5A-LR", "IPSL-CM5A-MR", "IPSL-CM5B-LR", "MIROC5", 
            "MIROC-ESM-CHEM", "MRI-CGCM3", "NorESM1-M")
model_counter = 0

historical <- data.table()
rcp45 <- data.table()
rcp85 <- data.table()

for (model in models){
  print (paste0("line 35: ", model))
  curr_hist <- readRDS(paste0(model_in, model, 
                              "/historical/", 
                              "fullbloom_50percent_day_", 
                              gsub("-", "_", model),
                              "_historical", ".rds"))
  
  curr_45 <- readRDS(paste0(model_in, model, 
                            "/rcp45/", 
                            "fullbloom_50percent_day_", 
                            gsub("-", "_", model),
                            "_rcp45", ".rds"))
  
  curr_85 <- readRDS(paste0(model_in, model, 
                            "/rcp85/", 
                            "fullbloom_50percent_day_", 
                            gsub("-", "_", model),
                            "_rcp85", ".rds"))
  
  historical <- rbind(historical, curr_hist)
  rcp45 <- rbind(rcp45, curr_45)
  rcp85 <- rbind(rcp85, curr_85)
  
  model_counter = model_counter + 1
  print (paste0("model_counter = ", model_counter))
}
historical$time_period <- "modeled_hist"
historical_45 <- historical
historical_85 <- historical

historical_45$emission <- "RCP 4.5"
historical_85$emission <- "RCP 8.5"
rm(historical)

###################################################
#
#                 Read observed
#
###################################################

observed <- readRDS(paste0(main_in, 
                          "fullbloom_50percent_day_observed.rds")) %>%
            data.table()
observed$time_period <- "observed"
observed_45 <- observed
observed_85 <- observed

observed_45$emission <- "RCP 4.5"
observed_85$emission <- "RCP 8.5"

rcp45$time_period <- "future"
rcp85$time_period <- "future"

rcp45 <- rbind(rcp45, historical_45, observed_45)
rcp85 <- rbind(rcp85, historical_85, observed_85)

# saveRDS(rcp45, paste0(out_dir, "/fullbloom_50percent_day_rcp45.rds"))
# saveRDS(rcp85, paste0(out_dir, "/fullbloom_50percent_day_rcp85.rds"))

all_bloom <- rbind(rcp45, rcp85)
saveRDS(all_bloom, paste0(out_dir, "/fullbloom_50percent_day.rds"))

end_time <- Sys.time()
print( end_time - start_time)

