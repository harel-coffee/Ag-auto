#setwd('~/CSANR/Cold/')
library(ggplot2)
library(data.table)
library(lubridate)
library(zoo)

args = commandArgs(trailingOnly = TRUE)
by_year <- T
stats_by_loc <- NA
data_dir <- '/data/rajagopalan/CH_output_data/modeled/'
lats <- c(46.46875, 46.53125)
longs <- c(-120.28125, -119.28125)
locs <- paste(lats, longs, sep='_')
#data_dir <- 'data/Prosser vs Cascades/'
for (model in list.dirs(data_dir, full.names = F, recursive = F)[as.integer(args[1]):as.integer(args[2])]) {
  model_subdir <- paste0(data_dir, model, '/')
  for (scenario in c('rcp85', 'rcp45')) {
    if (model != 'obs_hist') {
      data_subdir <- paste0(model_subdir, scenario, '/')
    } else
      data_subdir <- model_subdir
    if (model == 'obs_hist' & scenario == 'rcp45')
      next

    print(data_subdir)
    ii=0#!!remove
    for (loc_file_name in list.files(data_subdir)) {
      if (!startsWith(loc_file_name, 'CH_')) next
      ii=ii+1#!! remove
      #print(sprintf('%d %s %s', ii, model, scenario))#!! remove
      loc_file_name_parts <- strsplit(tools::file_path_sans_ext(loc_file_name), '_')[[1]]
      loc_lat <- loc_file_name_parts[length(loc_file_name_parts) - 1]
      loc_long <- loc_file_name_parts[length(loc_file_name_parts)]
      if (length(locs) > 0 & !(paste(loc_lat, loc_long, sep='_') %in% locs)) next
      loc <- read.csv(paste0(data_subdir, loc_file_name))[
        c('variety', 'Date', 't_min', 'predicted_Hc', 'base10_chilling_sum', 'DD_heating_sum', 'dormancy_period')]
      names(loc)[4:7] <- c('t_CH', 'DD_c', 'DD_h', 'period')
      loc$Date <- as.Date(loc$Date)
      loc$month <- month(loc$Date)
      loc$year <- year(loc$Date)
      loc$CH_year <- ifelse(loc$month > 8, loc$year + 1, ifelse(loc$month < 6, loc$year, NA))
      loc <- na.omit(loc)
      loc$slope <- append(NA, diff(loc$t_CH))
      loc$slope[loc$month == 9 & day(loc$Date) == 1] <- NA
      loc$biweekly_slope <- append(rep(NA, 13), rollmean(loc$slope, 14))
      # Aggregate by CH year.
      loc_by_year <- aggregate(list(DD_h = loc$DD_h, period = loc$period),
                               by = list(variety=loc$variety, CH_year=loc$CH_year), max)
      loc_by_year$is_eco <- loc_by_year$period == 2
      loc_by_year$DD_h[!loc_by_year$is_eco] <- NA
      loc_period_eco <- loc[loc$period == 2, ]
      if (nrow(loc_period_eco) > 0) {
        loc_by_year <- merge(loc_by_year, aggregate(
          list(Date = loc_period_eco$Date), by=list(variety=loc_period_eco$variety,
                                                    CH_year=loc_period_eco$CH_year), min), all.x = T)
        loc_by_year$switch_day <- as.integer(
          loc_by_year$Date - as.Date(sprintf('%s-09-01', loc_by_year$CH_year-1)))
        loc_by_year <- merge(loc_by_year, aggregate(
          list(avg_slope_eco = loc_period_eco$slope),
          by = list(variety=loc_period_eco$variety, CH_year=loc_period_eco$CH_year), mean), all.x = T)
        loc_by_year <- merge(loc_by_year, aggregate(
          list(biweekly_slope_eco = loc_period_eco$biweekly_slope),
          by = list(variety=loc_period_eco$variety, CH_year=loc_period_eco$CH_year), max), all.x = T)
      } else {
        loc_by_year$Date <- NA
        loc_by_year$switch_day <- NA
        loc_by_year$avg_slope_eco <- NA
        loc_by_year$biweekly_slope_eco <- NA
      }
      loc_period_endo <- loc[loc$period == 1, ]
      loc_by_year <- merge(loc_by_year, aggregate(
        list(avg_slope_endo = loc_period_endo$slope),
        by = list(variety=loc_period_endo$variety, CH_year=loc_period_endo$CH_year),
        na.rm = T, na.action = NULL, mean), all.x = T)
      loc_by_year <- merge(loc_by_year, aggregate(
        list(biweekly_slope_endo = loc_period_endo$biweekly_slope), 
        by=list(variety=loc_period_endo$variety, CH_year=loc_period_endo$CH_year),
        na.rm = T, na.action = NULL, min), all.x = T)
      loc_by_year <- merge(loc_by_year, aggregate(
        list(is_CD = loc$t_min < loc$t_CH),
        by=list(variety=loc$variety, CH_year=loc$CH_year), any), all.x = T)

      loc_stats <- NULL
      for (var in unique(loc_by_year$variety)) {
        if (by_year) {
          loc_stats_var <- loc_by_year[loc_by_year$variety == var,
                                   setdiff(colnames(loc_by_year), c('Date', 'period'))]
          loc_stats_var$lat <- loc_lat
          loc_stats_var$long <- loc_long
          loc_stats_var$model <- model
          loc_stats_var$scenario <- ifelse(model != 'obs_hist', scenario, NA)
        } else {
          # Compile statistics for each time frame from loc_by_year.
          loc_stats_var <- data.frame(
            variety=character(), lat=character(), long=character(), model=character(),
            scenario=character(), range=character(), DD_h=numeric(), no_eco_rate=integer(),
            switch_day=numeric(), CDI=numeric(), avg_slope_endo=numeric(), avg_slope_eco=numeric(),
            biweekly_slope_endo = numeric(), biweekly_slope_eco = numeric(), stringsAsFactors = F)
          if (model != 'obs_hist') {
            for (CH_year_from in c(2025, 2050, 2075)) {
              CH_year_to <- CH_year_from + 24
              loc_frame <- loc_by_year[loc_by_year$CH_year >= CH_year_from & loc_by_year$CH_year <= CH_year_to, ]
              loc_stats_var[nrow(loc_stats_var) + 1, ] <- list(
                var, loc_lat, loc_long, model, scenario, paste(CH_year_from, CH_year_to, sep = '-'),
                mean(loc_frame$DD_h, na.rm = T), 1 - mean(loc_frame$is_eco),
                mean(loc_frame$switch_day, na.rm = T), mean(loc_frame$is_CD),
                mean(loc_frame$avg_slope_endo, na.rm = T), mean(loc_frame$avg_slope_eco, na.rm = T),
                mean(loc_frame$biweekly_slope_endo, na.rm = T), mean(loc_frame$biweekly_slope_eco, na.rm = T))
            }
          } else {
            loc_stats_var[1, ] <- list(
              var, loc_lat, loc_long, model, NA, paste(min(loc_by_year$CH_year), max(loc_by_year$CH_year), sep = '-'),
              mean(loc_by_year$DD_h, na.rm = T), 1 - mean(loc_by_year$is_eco),
              mean(loc_by_year$switch_day, na.rm = T), mean(loc_by_year$is_CD),
              mean(loc_by_year$avg_slope_endo, na.rm = T), mean(loc_by_year$avg_slope_eco, na.rm = T),
              mean(loc_by_year$biweekly_slope_endo, na.rm = T), mean(loc_by_year$biweekly_slope_eco, na.rm = T))
          }
        }
        loc_stats <- rbind(loc_stats, loc_stats_var)
      }
      
      if (!is.data.frame(stats_by_loc)) {
        stats_by_loc <- loc_stats
      } else {
        stats_by_loc <- rbind(stats_by_loc, loc_stats)
      }
    }
    
  }
  
}
write.csv(stats_by_loc, sprintf('/data/rajagopalan/CH_output_data/stats_by_loc/CH_stats_%sby_loc_%s.csv',
                                ifelse(by_year, 'by_year_', ''), args[1]), row.names = F)
#write.csv(stats_by_loc, 'data/CH_stats_by_loc.csv', row.names = F)
print('Done.')
