rm(list=ls())
library(plyr)
library(tidyverse)
library(data.table)

source_path = "/Users/hn/Documents/GitHub/Kirti/Chilling/4th_draft/chill_plot_core.R"
source(source_path)

main_in <- "/Users/hn/Desktop/Desktop/Kirti/check_point/chilling/"
setwd(main_in)

start_season <- c("sept", "mid_sept", 
                  "oct", "mid_oct",
                  "nov", "mid_nov") # "sept", 
for (st in start_season){
  dt <- data.table(readRDS(paste0(st, "_summary_comp.rds")))
  
  dt <- dt %>% filter(model != "observed")
  dt <- pick_single_cities(dt)

  dt$time_period[dt$year<=2006] = "Historical"
  dt$time_period[dt$year>=2025 & dt$year<=2055] = "2025_2055"
  dt$time_period[dt$year>=2056 & dt$year<=2075] = "2056_2075"
  dt$time_period[dt$year>=2076] = "2076_2099"
  dt <- na.omit(dt)

  medians <- dt %>%
             group_by(city, scenario, time_period) %>%
             summarise_at(.funs = funs(med = median), vars(sum_J1:sum_A1)) %>% 
             data.table()

  means <- dt %>%
           group_by(city, scenario, time_period) %>%
           summarise_at(.funs = funs(means = mean), vars(sum_J1:sum_A1)) %>% 
           data.table()
  
  means <- as.data.frame(means)
  medians <- as.data.frame(medians)

  mean_medians <- merge(means, medians)
  rm(means, medians)
  write.table(x = mean_medians, row.names=F, col.names = T, sep=",",
              file = paste0(main_in, "/mean_medians_", st, ".csv"))

}

