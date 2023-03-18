setwd('~/CSANR/Cold/')
library(ggplot2)
library(data.table)
library(lubridate)
library(cowplot)

make_season <- function(date) {
  month <- month(date)
  ifelse(month >= 3 & month <= 5, 'spring', ifelse(month <= 2 | month == 12, 'winter',
                                                   ifelse(month >= 9 & month <= 11, 'fall', NA)))
}

spring_only <- F
from_1980 <- T
hist_range <- ifelse(from_1980, '1980-2016', '1979-2016')
model <- 'bcc-csm1-1-m'
time_frame = '2025-2049'
scenario = 'rcp45'
#==============================
df <- read.csv('data/cold_damage_future.csv', stringsAsFactors=F)[-1]
df$Date <- as.Date(df$Date)
df$season <- make_season(df$Date)
df <- df[!is.na(df$season), ]
if (spring_only)
  df <- df[!is.na(df$season) & df$season == 'spring', ]

future <- df[df$model == model & df$range == time_frame & df$scenario == scenario, ]
unique_locs <- unique(df[c('lat', 'long')])

hist <- NULL
for (i in 1:nrow(unique_locs)) {
  hist_loc <- read.csv(sprintf('data/WA_gridmet/CH_observed_historical_data_%s_%s.csv',
                               unique_locs$lat[i], unique_locs$long[i]))[c('Date', 't_min', 'predicted_Hc')]
  hist_loc$lat <- unique_locs$lat[i]
  hist_loc$long <- unique_locs$long[i]
  hist_loc <- hist_loc[hist_loc$t_min < hist_loc$predicted_Hc, ]
  hist <- rbind(hist, hist_loc)
}
hist$model <- 'obs_hist'
hist$scenario <- NA
hist$range <- '1979-2016'
hist$Date <- as.Date(hist$Date)
hist$season <- make_season(hist$Date)
hist <- hist[!is.na(hist$season), ]
if (from_1980)
  hist <- hist[hist$Date >= as.Date('1980-09-01'), ]
if (spring_only)
  hist <- hist[!is.na(hist$season) & hist$season == 'spring', ]

df <- rbind(future, hist)
df$type <- factor(ifelse(df$model == 'obs_hist', hist_range, time_frame), levels=c(hist_range, time_frame))
df$doy <- yday(df$Date)
df$month <- month(df$Date)
df$year <- year(df$Date)
df$CH_year <- ifelse(df$month > 8, df$year + 1, ifelse(df$month < 6, df$year, NA))
df$day <- as.integer(df$Date - as.Date(sprintf('%s-09-01', df$CH_year-1)))

#==============================
# df <- read.csv('data/cold_damage.csv', stringsAsFactors=F)
# df$Date <- as.Date(df$Date)
# df$season <- make_season(df$Date)
# df <- df[!is.na(df$season), ]
# df$type <- factor(ifelse(df$model == 'obs_hist', '1979-2016', time_frame), levels=c('1979-2016', time_frame))
#==============================

t_min = min(future$t_min, hist$t_min)
t_max = max(future$predicted_Hc, hist$predicted_Hc)
color_vals <- c('fall' = '#F8766D', 'winter' = "#619CFF", 'spring' = "#00BA38")
title <- sprintf('%s RCP45 cold damage comparison', model, scenario)

# ggplot(df, aes(x = t_min, y = predicted_Hc)) + geom_point(aes(color = season), alpha=.5, size=.5) +
#   xlim(t_min, t_max) + ylim(t_min, t_max) + facet_wrap(~ type) + 
#   geom_abline(intercept = 0, slope = 1, linetype = "dashed", size=.125) + 
#   labs(x=expression('°'*t[min]), y=expression('°'*t[CH]), title=title) + scale_color_manual(values=color_vals)
# ggsave('plots/spring_CD1.png', width = 7, height = 4)
# 
# ggplot(df, aes(x = type, y = predicted_Hc)) + geom_boxplot(outlier.shape=NA) + 
#   geom_jitter(height=0, aes(color=type))

#ggplot(df, aes(x = type, y = doy)) + geom_boxplot(outlier.shape=NA, aes(fill=type)) #+ geom_jitter(height=0, aes(color=type))

ggplot(df, aes(x = day, fill=type, color=type)) + geom_density(alpha=.1) +# + geom_histogram(aes(y=..density..))
  labs(x='days since Sep 1', y='cold damage frequency')
ggsave(sprintf('plots/CD_frequency_%s_bcc-csm1-1-m_rcp45_2025-2049.png', hist_range), width = 7, height = 4)
# ggplot(df, aes(x = doy, fill=type, color=type)) + geom_histogram(binwidth=20, alpha=.1) +
#   labs(x='day of year', y='cold damage count')
# ggsave('plots/CD_count_bcc-csm1-1-m_rcp45_2025-2049.png', width = 7, height = 4)

# ggplot(df, aes(x = Date, color=season)) +
#   geom_point(aes(y=t_min), shape=4, alpha=.5, size=.75) +
#   geom_segment(aes(x=Date-70, y=predicted_Hc, xend=Date+70, yend=predicted_Hc), alpha=.75) +
#   geom_segment(aes(x=Date, y=predicted_Hc, xend=Date, yend=t_min), linetype='dotted', alpha=.75, size=.25) + 
#   scale_color_manual(values=color_vals) + 
#   facet_wrap(~ type, scales="free_x") + labs(y = '°t', x = 'Date', title=title)
# ggsave('plots/CD3.png', width = 7, height = 4)
# 
# is_hist <- df$model == 'obs_hist'
# hist <- df[is_hist, ]
# future <- df[!is_hist, ]
# 
# ggplot(future) + geom_segment(aes(y = t_min, x = Date, yend = predicted_Hc, xend=Date, color=season)) + 
#   scale_color_manual(values=c('fall' = 'red', 'winter' = "blue", 'spring' = "green"))
# 
# 
# output_file <- 'cold_damage_increase.png'
