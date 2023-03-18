###################################################################
#**********                            **********
#**********        WARNING !!!!        **********
#**********                            **********
##
## DO NOT load any libraries here.
## And do not load any libraries on the drivers!
## Unless you are aware of conflicts between packages.
## I spent hours to figrue out what the hell is going on!
###################################################################
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(lubridate)
library(dplyr)

options(digit=9)
options(digits=9)

# Time the processing of this batch of files
start_time <- Sys.time()

################################################################
main_in <- "/data/hydro/users/Hossein/lagoon/03_rain_vs_snow/00_model_level/"
out_dir <- "/data/hydro/users/Hossein/lagoon/03_rain_vs_snow/01_combined/"

################################################################
model_names <- c("bcc-csm1-1", "CanESM2", "CSIRO-Mk3-6-0",
                 "HadGEM2-CC365", "IPSL-CM5A-LR", "MIROC5",
                 "NorESM1-M", "bcc-csm1-1-m", "CCSM4", 
                 "GFDL-ESM2G", "HadGEM2-ES365", "IPSL-CM5A-MR",
                 "MIROC-ESM-CHEM", "BNU-ESM", "CNRM-CM5",
                 "GFDL-ESM2M", "inmcm4", "IPSL-CM5B-LR",
                 "MRI-CGCM3")

emission <- c("historical", "rcp45", "rcp85")

rcp45_data <- data.table()
rcp85_data <- data.table()
hist_data <- data.table()

for (model in model_names){
  current_45 <- readRDS(paste0(main_in, model, "/rcp45/", "rain_snow.rds"))
  current_85 <- readRDS(paste0(main_in, model, "/rcp85/", "rain_snow.rds"))
  current_hist<-readRDS(paste0(main_in, model, "/historical/", "rain_snow.rds"))

  current_45$model <- gsub("-", "_", model)
  current_85$model <- gsub("-", "_", model)
  current_hist$model <- gsub("-", "_", model)

  rcp45_data <- rbind(rcp45_data, current_45)
  rcp85_data <- rbind(rcp85_data, current_85)
  hist_data <- rbind(hist_data, current_hist)
}

rcp45_data$emission <- "RCP 4.5"
rcp85_data$emission <- "RCP 8.5"
hist_data$emission <- "hist_mod"

saveRDS(rcp45_data, paste0(out_dir, "rain_RCP45.rds"))
saveRDS(rcp85_data, paste0(out_dir, "rain_RCP85.rds"))
saveRDS(hist_data, paste0(out_dir, "rain_modeled_hist.rds"))

hist_data_45 <- hist_data
hist_data_85 <- hist_data

hist_data_45$emission <- "RCP 4.5"
hist_data_85$emission <- "RCP 8.5"
rm(hist_data)

all_modeled <- rbind(rcp45_data, rcp85_data, hist_data_45, hist_data_85)
saveRDS(all_modeled, paste0(out_dir, "rain_all_modeled.rds"))

############################################################################
############################################################################

in_dir <- "/data/hydro/users/Hossein/lagoon/03_rain_vs_snow/00_model_level/"
A <- readRDS(paste0(in_dir, "rain_modeled_hist.rds"))
B <- readRDS(paste0(in_dir, "rain_RCP45.rds"))
C <- readRDS(paste0(in_dir, "rain_RCP85.rds"))
all_modeled <- rbind(A, B, C)


observed <- readRDS(paste0(main_in, "/observed/", "rain_observed.rds"))
observed_45 <- observed
observed_85 <- observed

observed_45$emission <- "RCP 4.5"
observed_85$emission <- "RCP 8.5"
rm(observed)

all_rains <- rbind(all_modeled, observed_45, observed_85)
saveRDS(all_rains, paste0(out_dir, "all_rains.rds"))



