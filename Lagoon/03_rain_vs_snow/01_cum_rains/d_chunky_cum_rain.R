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

######################################################################
##                                                                  ##
##                      Define all paths                            ##
##                                                                  ##
######################################################################
lagoon_source_path = "/home/hnoorazar/lagoon_codes/core_lagoon.R"
source(lagoon_source_path)

base_dir <- "/data/hydro/users/Hossein/lagoon/03_rain_vs_snow/"
data_dir <- paste0(base_dir, "01_combined/")
out_dir <- paste0(base_dir, "02_cum_rain/chunky/")
if (dir.exists(out_dir) == F) {dir.create(path = out_dir, recursive = T)}
######################################################################
##                                                                  ##
######################################################################
raw_files <- c("rain_observed.rds", "rain_modeled_hist.rds",
               "rain_RCP45.rds", "rain_RCP85.rds")

for(file in raw_files){
  curr_dt <- data.table(readRDS(paste0(data_dir, file)))
  curr_dt <- curr_dt %>% filter(time_period != "2006-2025")%>%data.table()
  curr_dt <- curr_dt[precip < 0, precip := 0] # replace negative precips
  
  curr_dt <- chunky_cum_rain(curr_dt, start_month=9, end_month=3)
  curr_dt <- curr_dt %>%
             group_by(location, wtr_yr, model, emission, time_period) %>%
             filter(month==3 & day==31) %>%
             data.table()
  
  saveRDS(curr_dt, paste0(out_dir, 
                          gsub("rain", "Sept_March_cum_rain", file)))
}

end_time <- Sys.time()
print( end_time - start_time)



