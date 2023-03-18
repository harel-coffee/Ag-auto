library(lubridate)
library(plyr)
library(tidyr)
library(ggplot2)
library(data.table)

read_loc <- function(path) {
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
  loc <- loc[loc$CH_year >= range_starts[i] & loc$CH_year <= range_ends[i] & loc$variety == 'Cabernet Sauvignon', ]
  if (nrow(loc) == 0)
    return(NULL)

  loc <- loc[!leap_year(loc$year) | loc$jday != 60, ] # remove Feb 29
  is_leap_jday <- leap_year(loc$year) & loc$month > 2 
  loc[is_leap_jday, 'jday'] <- loc[is_leap_jday, 'jday'] - 1 # decrease jday in leap years
  loc$jday <- (loc$jday - 244) %% 365
  loc$sm <- loc$t_min - loc$t_CH # safety margin
  loc$lat <- lat
  loc$long <- long
  
  return(list(lat=lat, long=long, loc=loc[loc$jday <= 257, ]))
}

data_dir <- '/data/rajagopalan/CH_output_data/modeled/'
output_dir <- '/data/rajagopalan/CH_output_data/for_comparison/'
dirs <- c('obs_hist', 'bcc-csm1-1-m/rcp45')
variety <- 'Cabernet Sauvignon'
range_starts <- c(1980, 2025)
range_ends <- c(2016, 2049)
data_descs <- paste(range_starts, range_ends, sep='-')

CDI_increase_path <- sprintf('%sCDI_increase_%s_%s.csv', output_dir, data_descs[1], data_descs[2])
if (!file.exists(CDI_increase_path)) {
  CDI <- list('1'=NULL, '2'=NULL)
  for (i in 1:2) {
    for (path in list.files(path=paste0(data_dir, dirs[i]), pattern="*.csv", full.names=T)) {
      if (startsWith(basename(path), 'CH_')) {
        res <- read_loc(path)
	if (!is.null(res)) {
        loc_CDI <- data.frame(res$lat, res$long, CDI=mean(aggregate(
          list(is_CD = res$loc$sm < 0), by = list(CH_year = res$loc$CH_year), any)$is_CD))
        CDI[[i]] <- rbind(CDI[[i]], loc_CDI)
	}
      }
    }
  }
  CDI_increase <- merge(CDI[[2]], CDI[[1]], by=c('res.lat', 'res.long'))
  names(CDI_increase)[1:2] <- c('lat', 'long')
  CDI_increase$CDI_increase <- CDI_increase$CDI.x - CDI_increase$CDI.y
  CDI_increase <- CDI_increase[c('lat', 'long', 'CDI_increase')]
  write.csv(CDI_increase, file=CDI_increase_path, row.names=F)
} else {
  CDI_increase <- read.csv(CDI_increase_path)
}
print('CDI increase loaded.')

start <- Sys.time()
df_plot_paths <- sprintf('%scomparison_%s.csv', output_dir, data_descs)
df_plot <- list('1'=NULL, '2'=NULL)
df_ensav_paths <- sprintf('%sensav_for_comparison_%s.csv', output_dir, data_descs)
df_ensav <- list('1'=NULL, '2'=NULL)
if (!all(file.exists(df_plot_paths))) {
  probs <- (0:10)/10 
  for (i in 1:2) {
    df <- NULL
    n_path <- 0
    files_list <- list.files(path=paste0(data_dir, dirs[i]), pattern="*.csv", full.names=T)
    for (path in files_list) {
      if (startsWith(basename(path), 'CH_')) {
      n_path <- n_path + 1
      res <- read_loc(path)
      if (!is.null(res))
        df <- rbind(df, res$loc[c('jday', 'sm', 't_CH', 't_min', 't_max', 'CH_year', 'lat', 'long')])
      if (n_path %% 100 == 0)
      print(sprintf('%s %s %d/%d loaded.', Sys.time() - start, data_descs[i], n_path, length(files_list)))
      }
    }
    df <- merge(df, CDI_increase)
    df$CDI_increase_cat <- ifelse(
      df$CDI_increase < -.05, 1, ifelse(df$CDI_increase < .05, 2, ifelse(df$CDI_increase < .2, 3, 4)))
    print(sprintf('%s %s CDI increase cat computed.', Sys.time() - start, data_descs[i]))
    df <- data.frame(melt(data.table(df[setdiff(names(df), 'CDI_increase')]),
               id.vars=c('CDI_increase_cat', 'jday', 'lat', 'long', 'CH_year'), variable.name='var'))
    print(sprintf('%s %s reshaped.', Sys.time() - start, data_descs[i]))
    df_ensav[[i]] <- aggregate(value ~ CDI_increase_cat + jday + var, df, mean, na.rm=T)
    names(df_ensav[[i]])[4] <- 'ens_av'
    print(sprintf('%s %s ensav computed', Sys.time() - start, data_descs[i]))
    write.csv(df_ensav[[i]], file=df_ensav_paths[i], row.names=F)
    
    for (j in (1:4)) {
      df_cat_m <- data.frame(dcast(data.table(df[df$CDI_increase_cat == j, ]),
                                   CDI_increase_cat + jday + var ~ lat + long + CH_year))
      print(sprintf('%s %s df_cat_m computed', Sys.time() - start, data_descs[i]))
      # calculate percentiles across grids and years
      arr <- t(apply(df_cat_m[, -(1:3)], 1, quantile, prob=probs, na.rm=T))
      df_cat_q <- data.frame(df_cat_m[, 1:3], arr)
      names(df_cat_q) <- c('CDI_increase_cat', 'jday', 'var', colnames(arr))
      df_cat_q_m <- melt(data.table(df_cat_q), id.vars=1:3)
      df_cat_q_m$delta <- df_cat_q_m$variable # add inter-quantile (per) range as delta 
      levels(df_cat_q_m$delta) <- abs(probs - rev(probs))*100
      
      df_plot_cat <- ddply(df_cat_q_m, .(CDI_increase_cat, jday, var, delta), summarize,
                           quantmin=min(value), quantmax=max(value))
      print(sprintf('%s %s df_plot_cat computed', Sys.time() - start, data_descs[i]))
      
      df_plot[[i]] <- rbind(df_plot[[i]], df_plot_cat)
    }
    write.csv(df_plot[[i]], file=df_plot_paths[i], row.names=F)
  }
} else {
  for (i in 1:2) {
    df_plot[[i]] <- read.csv(df_plot_paths[i])
    df_ensav[[i]] <- read.csv(df_ensav_paths[i])
  }
}
print(sprintf('%s all loaded.', Sys.time() - start))
#=====================================
