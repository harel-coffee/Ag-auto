library(lubridate)
library(plyr)
library(tidyr)
library(ggplot2)
library(data.table)

read_loc <- function(path, i) {
  lat_long <- strsplit(tools::file_path_sans_ext(basename(path)), '_')[[1]]
  lat <- lat_long[length(lat_long) - 1]
  long <- lat_long[length(lat_long)]
  loc <- read.csv(path)[c('variety', 'Date', 't_min', 't_max', 'predicted_Hc')]
  names(loc)[5] <- 't_CH'
  loc$Date <- as.Date(loc$Date)
  loc$jday <- yday(loc$Date)
  loc$month <- month(loc$Date)
  loc$year <- year(loc$Date)
  loc$CH_year <- ifelse(loc$month > 8, loc$year + 1, ifelse(loc$month < 6, loc$year, NA))
  loc <- na.omit(loc)
  loc <- loc[loc$CH_year >= range_starts[i] & loc$CH_year <= range_ends[i], ]
  if (nrow(loc) == 0)
    return(NULL)

  loc <- loc[!leap_year(loc$year) | loc$jday != 60, ] # remove Feb 29
  is_leap_jday <- leap_year(loc$year) & loc$month > 2 
  loc[is_leap_jday, 'jday'] <- loc[is_leap_jday, 'jday'] - 1 # decrease jday in leap years
  loc$jday <- (loc$jday - 244) %% 365
  loc <- loc
  loc$sm <- loc$t_min - loc$t_CH # safety margin
  loc$lat <- lat
  loc$long <- long

  return(list(lat=lat, long=long, loc=loc[loc$jday <= 257, ]))
}

data_dir <- '/data/rajagopalan/CH_output_data/modeled'
output_dir <- '/data/rajagopalan/CH_output_data/for_whiteboard/'
obs_hist <- 'obs_hist'
scenarios <- c('rcp45', 'rcp85')
range_starts <- c(1980, 2025, 2050, 2075)
range_ends <- c(2016, 2049, 2074, 2099)
data_descs <- paste(range_starts, range_ends, sep='-')

start <- Sys.time()

temp <- NULL
hist_temp <- NULL
args = commandArgs(trailingOnly = TRUE)
mod_ix <- (as.integer(args[1]) %/% 2) + 1
scenario_ix <- (as.integer(args[1]) %% 2) + 1
model <- setdiff(basename(list.dirs(path=data_dir, recursive=F)), obs_hist)[mod_ix]
scenario <- scenarios[scenario_ix]
print(paste(model, scenario))
hist_CDI <- NULL
for (path in list.files(path=paste(data_dir, obs_hist, sep='/'), pattern="*.csv", full.names=T)) {
  if (startsWith(basename(path), 'CH_')) {
    res <- read_loc(path, 1)
    if (!is.null(res)) {
      loc_temp <- cbind(lat=res$lat, long=res$long, model=model, scenario=scenario, aggregate(
        cbind(hist_t_mean = (t_min + t_max)/2) ~ month + variety, data=res$loc, mean))
      hist_temp <- rbind(hist_temp, loc_temp)
      
      loc_CDI <- cbind(lat=res$lat, long=res$long, model=model, scenario=scenario, aggregate(
        CDI ~ variety, data=aggregate(cbind(CDI = sm < 0) ~ CH_year + variety, data=res$loc, any), mean))
      hist_CDI <- rbind(hist_CDI, loc_CDI)
    }
  }
}

for (i in 2:4) {
  future_temp <- NULL
  future_CDI <- NULL
  for (path in list.files(path=paste(data_dir, model, scenario, sep='/'), pattern="*.csv", full.names=T)) {
    if (startsWith(basename(path), 'CH_')) {
      res <- read_loc(path, i)
      if (!is.null(res)) {
        loc_temp <- cbind(lat=res$lat, long=res$long, model=model, scenario=scenario, aggregate(
          cbind(t_mean = (t_min + t_max)/2) ~ month + variety, data=res$loc, mean))
        future_temp <- rbind(future_temp, loc_temp)
        
        loc_CDI <- cbind(lat=res$lat, long=res$long, model=model, scenario=scenario, aggregate(
          CDI ~ variety, data=aggregate(cbind(CDI = sm < 0) ~ CH_year + variety, data=res$loc, any), mean))
        # loc_CDI <- data.frame(res$lat, res$long, model=model, scenario=scenario, CDI=mean(aggregate(
        #   list(is_CD=res$loc$sm < 0), by=list(CH_year=res$loc$CH_year, variety=res$loc$variety), any)$is_CD))
        future_CDI <- rbind(future_CDI, loc_CDI)
      }
    }
  }
  print(sprintf('%s-%s average temperatures by month loaded', range_starts[i], range_ends[i]))
  
  CDI_increase <- merge(future_CDI, hist_CDI, by=c('lat', 'long', 'variety'), all.x=T)
  names(CDI_increase)[1:2] <- c('lat', 'long')
  CDI_increase$CDI_increase <- CDI_increase$CDI.x - CDI_increase$CDI.y
  CDI_increase <- CDI_increase[c('lat', 'long', 'variety', 'CDI_increase')]
  print('CDI increase loaded.')
  
  time_frame_temp <- merge(merge(future_temp, hist_temp), CDI_increase)
  time_frame_temp$time_frame <- i
  temp <- rbind(temp, time_frame_temp)
}

write.csv(temp, file=sprintf('%s%s_%s_temp.csv', output_dir, model, scenario), row.names=F)
print(sprintf('Done in %s.', Sys.time() - start))
#=====================================
