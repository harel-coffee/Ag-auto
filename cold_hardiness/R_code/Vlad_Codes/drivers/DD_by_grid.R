setwd('~/CSANR/Cold/')
library(ggplot2)
library(data.table)
library(lubridate)
source("read_binary_core.R")

min_CH_30y_start <- 1979
max_CH_30y_start <- 1987
n_missing <- 0
n_loc <- 0
DD_by_loc <- NA
data_dir <- 'data/WA_gridmet/'
for (loc_file_name in list.files(data_dir)) {
  if (!startsWith(loc_file_name, 'CH_')) next
  n_loc <- n_loc + 1
  loc_file_name_parts <- strsplit(tools::file_path_sans_ext(loc_file_name), '_')[[1]]
  loc_lat <- loc_file_name_parts[5]
  loc_long <- loc_file_name_parts[6]
  loc <- read.csv(paste0(data_dir, loc_file_name))[
    c('Date', 'base10_chilling_sum', 'DD_heating_sum', 'dormancy_period')]
  names(loc)[2:4] <- c('DD_c', 'DD_h', 'period')
  loc$Date <- as.Date(loc$Date)
  loc$month <- month(loc$Date)
  loc$year <- year(loc$Date)
  loc$CH_year <- ifelse(loc$month > 8, loc$year + 1, ifelse(loc$month < 6, loc$year, NA))
  loc <- na.omit(loc)
  DD_h <- aggregate(list(DD_h = loc$DD_h), by = list(CH_year = loc$CH_year), max)
  loc_period_1 <- loc[loc$period == 1, ]
  DD_c <- aggregate(list(DD_c = -loc_period_1$DD_c,# day_switch=loc_period_1$Date),
                         day_switch = as.integer(loc_period_1$Date - as.Date(sprintf('%s-09-01', loc_period_1$CH_year-1)))),
                    by = list(CH_year = loc_period_1$CH_year), max)
  #DD <- aggregate(list(DD_c=-loc$DD_c, DD_h=loc$DD_h), by = list(CH_year=loc$CH_year), max)
  DD <- merge(DD_c, DD_h)
  loc_DD <- data.frame('lat' = rep(loc_lat, 2), 'long' = rep(loc_long, 2), 'DD_type' = c('c', 'h'))
  for (CH_30y_start in min_CH_30y_start:max_CH_30y_start) {
    CH_30y_end <- CH_30y_start + 29
    CH_year_mask <- DD$CH_year >= CH_30y_start & DD$CH_year <= CH_30y_end
    loc_DD[1, sprintf('DD_%d_%d', CH_30y_start, CH_30y_start+29)] <- -mean(DD$DD_c[CH_year_mask])
    loc_DD[1, sprintf('day_%d_%d', CH_30y_start, CH_30y_start+29)] <- mean(DD$day_switch[CH_year_mask])
    loc_DD[2, sprintf('DD_%d_%d', CH_30y_start, CH_30y_start+29)] <- mean(DD$DD_h[CH_year_mask])
    loc_DD[2, sprintf('day_%d_%d', CH_30y_start, CH_30y_start+29)] <- NA
  }
  
  if (!is.data.frame(DD_by_loc)) {
    DD_by_loc <- loc_DD
  } else {
    DD_by_loc <- rbind(DD_by_loc, loc_DD)
  }
  
}
print(sprintf('total: %i, missing: %i', n_loc, n_missing))
write.csv(DD_by_loc, 'data/DD_by_loc.csv', row.names = F)
