.libPaths("/data/hydro/R_libs35")
.libPaths()

library(plyr)
library(tidyverse)
library(data.table)

main_in <- "/data/hydro/users/Hossein/chill/data_by_core/dynamic/02/"

start_season <- c("mid_sept", 
                  "oct", "mid_oct",
                  "nov", "mid_nov") # "sept", 
for (st in start_season){
  in_dir <- file.path(main_in, st, "non_overlap/")
  setwd(in_dir)

  the_dir <- dir()
  the_dir <- the_dir[grep(pattern = ".txt", x = the_dir)]  
  the_dir <- the_dir[grep(pattern = "summary", x = the_dir)]
  the_dir <- the_dir[-grep(pattern = "summary_stats", x = the_dir)]
  the_dir_summary <- the_dir
  
  summary_comp <- lapply(the_dir_summary, read.table, header = T)
  summary_comp <- do.call(bind_rows, summary_comp)

  # Combine the data with cold/warm geographic designations
  param_dir = "/home/hnoorazar/cleaner_codes/parameters/"
  cold_warm <- read.csv(paste0(param_dir, "LocationGroups.csv"))

  summary_comp <- inner_join(x = summary_comp, y = cold_warm,
                             by = c("long" = "longitude",
                                    "lat" = "latitude")) %>%
                  mutate(climate_type = case_when( # create var for cool/warm designation
                                                  locationGroup == 1 ~ "Cooler Area",
                                                  locationGroup == 2 ~ "Warmer Area")) %>%
                  select(-locationGroup, -.id)

  summary_comp$scenario[summary_comp$scenario=="historical"] = "Historical"
  summary_comp$scenario[summary_comp$scenario=="rcp45"] = "RCP 4.5"
  summary_comp$scenario[summary_comp$scenario=="rcp85"] = "RCP 8.5"

  saveRDS(summary_comp, paste0(in_dir, "summary_comp.rds"))

  summary_comp <- summary_comp %>% filter(model != "observed")
  dt = data.table(summary_comp)
  
  dt$time_period[dt$year<=2006] = "Historical"
  dt$time_period[dt$year>=2025 & dt$year<=2055] = "2025_2055"
  dt$time_period[dt$year>=2056 & dt$year<=2075] = "2056_2075"
  dt$time_period[dt$year>=2076] = "2076_2099"

  okanagan <- dt %>% filter(lat== "48.40625" & long=="-119.53125")
  okanagan$location = "okanagan"
                                   
  richland <- dt %>% filter(lat== "46.28125" & long=="-119.34375")
  richland$location = "richland"

  okan_rich = rbind(okanagan, richland)

  rm(dt, richland, okanagan)

  medians <- okan_rich %>%
             group_by(location, scenario, time_period) %>%
             summarise_at(.funs = funs(med = median), vars(sum_J1:sum_A1)) %>% 
             data.table()

  means <- okan_rich %>%
           group_by(location, scenario, time_period) %>%
           summarise_at(.funs = funs(means = mean), vars(sum_J1:sum_A1)) %>% 
           data.table()
  means <- as.data.frame(means)
  medians <- as.data.frame(medians)
  mean_medians <- merge(means, medians)
  rm(means, medians)
  write.table(x = mean_medians, row.names=F, col.names = T, sep=",",
              file = paste0(in_dir, "/mean_medians_", st, ".csv"))
}

