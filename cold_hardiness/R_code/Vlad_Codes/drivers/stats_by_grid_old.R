setwd('~/CSANR/Cold/')
library(ggplot2)
library(data.table)
library(lubridate)
library(zoo)

stats_by_loc <- NA
data_dir <- 'data/Prosser vs Cascades/'
for (subdir in list.dirs(data_dir, full.names = F, recursive = F)) {
  data_subdir <- paste0(data_dir, subdir, '/')
  if (subdir != 'obs_hist')
    data_subdir <- paste0(data_subdir, 'rcp85/')
  
  for (loc_file_name in list.files(data_subdir)) {
    if (!startsWith(loc_file_name, 'CH_')) next
    loc_file_name_parts <- strsplit(tools::file_path_sans_ext(loc_file_name), '_')[[1]]
    loc_lat <- loc_file_name_parts[3]
    loc_long <- loc_file_name_parts[4]
    loc <- read.csv(paste0(data_subdir, loc_file_name))[
      c('Date', 't_min', 'predicted_Hc', 'base10_chilling_sum', 'DD_heating_sum', 'dormancy_period')]
    names(loc)[3:6] <- c('t_CH', 'DD_c', 'DD_h', 'period')
    loc$Date <- as.Date(loc$Date)
    loc$month <- month(loc$Date)
    loc$year <- year(loc$Date)
    loc$CH_year <- ifelse(loc$month > 8, loc$year + 1, ifelse(loc$month < 6, loc$year, NA))
    loc <- na.omit(loc)
    loc$slope <- append(NA, diff(loc$t_CH))
    loc$slope[loc$month == 9 & day(loc$Date) == 1] <- NA
    #loc$week_slope <- append(rep(NA, 13), rollapply(loc$slope, 14, FUN=function(x) mean(x, na.rm=TRUE)))
    loc$biweekly_slope <- append(rep(NA, 13), rollmean(loc$slope, 14))
    loc$monthly_slope <- append(rep(NA, 29), rollmean(loc$slope, 30))
    
    # Aggregate by CH year.
    loc_by_year <- aggregate(list(DD_h = loc$DD_h, period = loc$period),
                             by = list(CH_year = loc$CH_year), max)
    loc_by_year$is_eco <- loc_by_year$period == 2
    loc_by_year$DD_h[!loc_by_year$is_eco] <- NA
    loc_period_eco <- loc[loc$period == 2, ]
    loc_by_year <- merge(loc_by_year, aggregate(list(Date = loc_period_eco$Date),
                                                by = list(CH_year = loc_period_eco$CH_year), min), all.x = T)
    loc_by_year$switch_day <- as.integer(
      loc_by_year$Date - as.Date(sprintf('%s-09-01', loc_by_year$CH_year-1)))
    loc_period_endo <- loc[loc$period == 1, ]
    loc_by_year <- merge(loc_by_year, aggregate(
      list(avg_slope_endo = loc_period_endo$slope), by = list(CH_year = loc_period_endo$CH_year),
      na.rm = T, na.action = NULL, mean), all.x = T)
    loc_by_year <- merge(loc_by_year, aggregate(
      list(avg_slope_eco = loc_period_eco$slope), by = list(CH_year = loc_period_eco$CH_year), mean),
      all.x = T)
    loc_by_year <- merge(loc_by_year, aggregate(
      list(max_slope_endo = loc_period_endo$slope, biweekly_slope_endo = loc_period_endo$biweekly_slope,
           monthly_slope_endo = loc_period_endo$monthly_slope),
      by = list(CH_year = loc_period_endo$CH_year), na.rm = T, na.action = NULL, min), all.x = T)
    loc_by_year <- merge(loc_by_year, aggregate(
      list(max_slope_eco = loc_period_eco$slope, biweekly_slope_eco = loc_period_eco$biweekly_slope,
           monthly_slope_eco = loc_period_eco$monthly_slope),
      by = list(CH_year = loc_period_eco$CH_year), max), all.x = T)
    loc_by_year <- merge(loc_by_year, aggregate(
      list(is_CD = loc$t_min < loc$t_CH), by = list(CH_year = loc$CH_year), any), all.x = T)
    
    # Compile statistics for each time frame from loc_by_year.
    loc_stats <- data.frame(
      lat = character(), long = character(), model = character(), range = character(), DD_h = numeric(),
      no_eco_rate = integer(), switch_day = numeric(), CDI = numeric(), avg_slope_endo = numeric(),
      avg_slope_eco = numeric(), max_slope_endo = numeric(), max_slope_eco = numeric(),
      biweekly_slope_endo = numeric(), biweekly_slope_eco = numeric(),
      monthly_slope_endo = numeric(), monthly_slope_eco = numeric(), stringsAsFactors = F)
    if (max(loc$CH_year) == 2100) {
      for (CH_year_from in c(2026, 2051, 2076)) {
        CH_year_to <- CH_year_from + 24
        loc_frame <- loc_by_year[loc_by_year$CH_year >= CH_year_from & loc_by_year$CH_year <= CH_year_to, ]
        loc_stats[nrow(loc_stats) + 1, ] <- list(
          loc_lat, loc_long, subdir, paste(CH_year_from, CH_year_to, sep = '-'),
          mean(loc_frame$DD_h, na.rm = T), 1 - mean(loc_frame$is_eco), mean(loc_frame$switch_day, na.rm = T),
          mean(loc_frame$is_CD), mean(loc_frame$avg_slope_endo, na.rm = T),
          mean(loc_frame$avg_slope_eco, na.rm = T), mean(loc_frame$max_slope_endo, na.rm = T),
          mean(loc_frame$max_slope_eco, na.rm = T), mean(loc_frame$biweekly_slope_endo, na.rm = T),
          mean(loc_frame$biweekly_slope_eco, na.rm = T), mean(loc_frame$monthly_slope_endo, na.rm = T),
          mean(loc_frame$monthly_slope_eco, na.rm = T))
      }
    } else {
      loc_stats[1, ] <- list(
        loc_lat, loc_long, subdir, paste(min(loc_by_year$CH_year), max(loc_by_year$CH_year), sep = '-'),
        mean(loc_by_year$DD_h, na.rm = T), 1 - mean(loc_by_year$is_eco),
        mean(loc_by_year$switch_day, na.rm = T), mean(loc_by_year$is_CD),
        mean(loc_by_year$avg_slope_endo, na.rm = T), mean(loc_by_year$avg_slope_eco, na.rm = T),
        mean(loc_by_year$max_slope_endo, na.rm = T), mean(loc_by_year$max_slope_eco, na.rm = T),
        mean(loc_by_year$biweekly_slope_endo, na.rm = T), mean(loc_by_year$biweekly_slope_eco, na.rm = T),
        mean(loc_by_year$monthly_slope_endo, na.rm = T), mean(loc_by_year$monthly_slope_eco, na.rm = T))
    }
    
    if (!is.data.frame(stats_by_loc)) {
      stats_by_loc <- loc_stats
    } else {
      stats_by_loc <- rbind(stats_by_loc, loc_stats)
    }
    
  }
  
}
write.csv(stats_by_loc, 'data/stats_by_loc.csv', row.names = F)
