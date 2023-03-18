setwd('/data/rajagopalan/CH_output_data/')
library(lubridate)
library(zoo)

input_file <- 'CH_stats_by_loc.csv'
# Find locations with higher CDI in any future range.
all <- read.csv(input_file, stringsAsFactors=F)[c('lat', 'long', 'model', 'scenario', 'range', 'CDI')]
hist_mask <- all$model == 'obs_hist'
future <- merge(all[!hist_mask, ], all[hist_mask, c('lat', 'long', 'CDI')], by=c('lat', 'long'))
future <- future[future$CDI.x > future$CDI.y, c('lat', 'long', 'model', 'scenario', 'range')]
future_files <- paste('modeled', future$model, future$scenario,
                      sprintf('CH_data_%s_%s.csv', future$lat, future$long), sep='/')
hist_files <- paste('modeled', 'obs_hist', sprintf('CH_observed_historical_data_%s_%s.csv', future$lat, future$long), sep='/')

prev_future_file <- ''
prev_hist_file <- ''
CD_locs <- NULL
for (i in 1:length(future_files)) {
  print(sprintf('%d/%d', i, length(future_files)))
  # if (i < 349)#!!
  #   next#!!
  # if (i > 349)#!!
  #   break#!!
  hist_file <- hist_files[i]
  if (hist_file != prev_hist_file) {
    hist_loc <- read.csv(hist_file)[c('Date', 't_min', 'predicted_Hc')]
    # print('hist_loc')#!!
    # print(head(hist_loc))#!!
    hist_loc_CD <- hist_loc[hist_loc$t_min < hist_loc$predicted_Hc, ]
    # print('hist_loc_CD')#!!
    # print(head(hist_loc_CD))#!!
    if (nrow(hist_loc_CD) > 0) {
      hist_loc_CD <- cbind(hist_loc_CD, future[i, c('lat', 'long')])
      names(hist_loc_CD)[-3:-1] <- c('lat', 'long')
      hist_loc_CD$model <- 'obs_hist'
      hist_loc_CD$scenario <- NA
      hist_loc_CD$range <- '1979-2016'
      CD_locs <- rbind(CD_locs, hist_loc_CD)
    }

  }
  prev_hist_file <- hist_file
  
    future_file <- future_files[i]
    if (future_file != prev_future_file)
      future_loc <- read.csv(future_file)[c('year', 'Date', 't_min', 'predicted_Hc')]
    prev_future_file <- future_file
    # print(head(future_loc))#!!
    # print(future_file)#!!
    if (nrow(future_loc) > 1) {
      time_frame <- as.integer(strsplit(future$range[i], '-')[[1]])
      future_loc$Date <- as.Date(future_loc$Date)
      future_loc$month <- month(future_loc$Date)
      future_loc$CH_year <- ifelse(future_loc$month > 8, future_loc$year + 1,
                                   ifelse(future_loc$month < 6, future_loc$year, NA))
      future_loc_CD <- future_loc[
        with(future_loc, CH_year >= time_frame[1] & CH_year <= time_frame[2] & t_min < predicted_Hc),
        c('Date', 't_min', 'predicted_Hc')]
      # print(head(future_loc_CD))#!!
      if (nrow(future_loc_CD) == 0) {
        print(future_file)
        print('No cold damage in future data, terminating.')
        break
      }
      future_loc_CD <- cbind(future_loc_CD, future[i, ])
      names(future_loc_CD)[-3:-1] <- names(future)
      CD_locs <- rbind(CD_locs, future_loc_CD)
  }
}

write.csv(CD_locs, 'cold_damage.csv', row.names=F)
