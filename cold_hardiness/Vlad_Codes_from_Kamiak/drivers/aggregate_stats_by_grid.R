setwd('~/CSANR/Cold/')
library(ggplot2)
library(data.table)
library(lubridate)
library(zoo)

by_year <- T
data_dir <- 'data/stats by grid/'
data_files <- setdiff(list.files(data_dir), 'Old')
if (by_year) {
  cols <- c('variety', 'CH_year', "DD_h", 'is_eco', "switch_day", "avg_slope_endo", "avg_slope_eco",
            "biweekly_slope_endo", "biweekly_slope_eco", 'is_CD', "lat", "long", "model", "scenario")
} else
  cols <- c('variety', "lat", "long", "model", "scenario", "range", "DD_h", "no_eco_rate", "switch_day",
            "CDI", "avg_slope_endo", "avg_slope_eco", "biweekly_slope_endo", "biweekly_slope_eco")
all <- read.csv(paste0(data_dir, data_files[1]))[cols]
for (data_file in data_files[2:length(data_files)]) {
  print(data_file)
  all <- rbind(all, read.csv(paste0(data_dir, data_file))[cols])
}
write.csv(all, sprintf('data/CH_stats_by_loc%s.csv', ifelse(by_year, '_by_year', '')), row.names = F)
