rm(list=ls())

###################################################################
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)

options(digits=9)
options(digit=9)


binary_core_path = "/home/hnoorazar/reading_binary/read_binary_core.R"
chill_core_path = "/home/hnoorazar/chilling_codes/current_draft/chill_core.R"

source(chill_core_path)
source(binary_core_path)


param_dir <- "/home/hnoorazar/chilling_codes/parameters/"
observed_dir <- "/data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS_to_2016/"

###############################################################
all_locations <- read.csv(paste0(param_dir, "LocationGroups_NoMontana.csv"), as.is=TRUE)

col_names <- c("location", 
               "annual_tmean", "annual_precip_mean", 
               "annual_winterYR_tmean", "annual_winterYR_precip_mean")

output_dt <- setNames(data.table(matrix(nrow = nrow(all_locations), ncol = length(col_names))), 
                     col_names)

output_dt$location <- as.character(output_dt$annual_tmean)
output_dt$annual_tmean <- as.numeric(output_dt$annual_tmean)
output_dt$annual_precip_mean <- as.numeric(output_dt$annual_precip_mean)
output_dt$annual_winterYR_tmean <- as.numeric(output_dt$annual_winterYR_tmean)
output_dt$annual_winterYR_precip_mean <- as.numeric(output_dt$annual_winterYR_precip_mean)


row_count = 1
for (loc in all_locations$location){
  
  output_dt[row_count, "location"] = loc
  
  data_dt <- read_binary(paste0(observed_dir, "data_", loc), hist=TRUE, no_vars=8)
  data_dt$t_mean <- (data_dt$tmax + data_dt$tmin)/2
  
  data_dt_temp <- data_dt[, .(Tmean_per_Yr = mean(t_mean)), by = c("year")]  
  output_dt[row_count, "annual_tmean"] <- mean(data_dt_temp$Tmean_per_Yr)
  
  #
  # precip
  #
  data_dt_precip <- data_dt
  data_dt_precip[, annual_cum_precip := cumsum(precip), by=list(year)]
  data_dt_precip <- data_dt_precip %>% filter(month==12 & day == 31) %>% data.table()
  output_dt[row_count, "annual_precip_mean"] <- mean(data_dt_precip$annual_cum_precip)
  
  # data_dt <- read_binary(paste0(observed_dir, "data_", loc), hist=TRUE, no_vars=8)
  # data_dt$t_mean <- (data_dt$tmax + data_dt$tmin)/2

  data_dt <- put_chill_season(data_dt, chill_start = "sept")
  data_dt <- data_dt %>% filter(!(month %in% c(4, 5, 6, 7, 8))) %>% data.table()

  # remove the incomplete chill seasons
  data_dt <- data_dt %>% filter(chill_season != min(data_dt$chill_season))
  data_dt <- data_dt %>% filter(chill_season != max(data_dt$chill_season)) %>% data.table()
  
  data_dt_temp <- data_dt[, .(Tmean_per_ChillYr = mean(t_mean)), by = c("chill_season")]
  output_dt[row_count, "annual_winterYR_tmean"] <- mean(data_dt_temp$Tmean_per_ChillYr)

  data_dt_precip <- data_dt
  #
  # add DoY to the data
  #
  data_dt_precip$DoWintSeason <- 1
  data_dt_precip[, DoWintSeason := cumsum(DoWintSeason), by=list(chill_season)]

  #
  # compute accumulated precip
  #
  data_dt_precip[, annual_cum_precip := cumsum(precip), by=list(chill_season)]

  data_dt_precip <- data_dt_precip %>% filter(month==3 & day == 31) %>% data.table()

  output_dt[row_count, "annual_winterYR_precip_mean"] <- mean(data_dt_precip$annual_cum_precip)
  row_count = row_count + 1
}

out_dir <- "/data/hydro/users/Hossein/chill_proj/topography/"
write.csv(output_dt, 
	      paste0(out_dir, "annual_avgs.csv"), row.names = FALSE)

