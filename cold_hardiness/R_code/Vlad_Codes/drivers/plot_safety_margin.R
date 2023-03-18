setwd('~/CSANR/Cold/')
library(lubridate)
library(plyr)
library(tidyr)
library(ggplot2)
library(data.table)
library(cowplot)
library(ggnewscale)

range_starts <- c(1980, 2025)
range_ends <- c(2016, 2049)
data_types <- c(paste(range_starts[1], range_ends[1], sep='–'),
                paste(range_starts[2], range_ends[2], sep='–'))
sm_path <- sprintf('data/safety_margins_%s-%s_%s-%s.csv',
                   range_starts[1], range_ends[1], range_starts[2], range_ends[2])
CDI_path <- sprintf('data/CDI_%s-%s_%s-%s.csv',
                    range_starts[1], range_ends[1], range_starts[2], range_ends[2])
if (!file.exists(sm_path)) {
  dirs <- c('WA_gridmet', 'bcc-csm1-1-m rcp45')
  sm <- NULL
  CDI <- NULL
  for (i in 1:2) {
    for (path in list.files(path=paste0('data/', dirs[i]), pattern="*.csv", full.names=T)) {
      lat_long <- strsplit(tools::file_path_sans_ext(basename(path)), '_')[[1]]
      lat <- lat_long[length(lat_long) - 1]
      long <- lat_long[length(lat_long)]
      loc <- read.csv(path)[c('Date', 't_min', 'predicted_Hc')]
      loc$Date <- as.Date(loc$Date)
      loc$jday <- yday(loc$Date)
      loc$month <- month(loc$Date)
      loc$year <- year(loc$Date)
      loc$CH_year <- ifelse(loc$month > 8, loc$year + 1, ifelse(loc$month < 6, loc$year, NA))
      loc <- na.omit(loc)
      loc <- loc[!leap_year(loc$year) | loc$jday != 60, ] # remove Feb 29
      is_leap_jday <- leap_year(loc$year) & loc$month > 2 
      loc[is_leap_jday, 'jday'] <- loc[is_leap_jday, 'jday'] - 1 # decrease jday in leap years
      loc <- loc[loc$CH_year >= range_starts[i] & loc$CH_year <= range_ends[i], ]
      loc$jday <- (loc$jday - 244) %% 365
      loc$sm <- loc$t_min - loc$predicted_Hc # safety margin
      
      loc_CDI <- data.frame(lat, long, type=data_types[i], CDI=mean(aggregate(
        list(is_CD = loc$sm < 0), by = list(CH_year = loc$CH_year), any)$is_CD))
      CDI <- rbind(CDI, loc_CDI)
      
      loc_sm <- data.table(loc)[, .(avg=mean(sm), min=min(sm), max=max(sm)), by = .(jday)]
      loc_sm$lat <- lat
      loc_sm$long <- long
      loc_sm$type <- data_types[i]
      sm <- rbind(sm, loc_sm)
      
    }
  }
  write.csv(sm, file=sm_path, row.names=F)
  write.csv(CDI, file=CDI_path, row.names=F)
} else {
  sm <- read.csv(sm_path)
  CDI <- read.csv(CDI_path)
}

# !! only select grids with CDI increase > .2
is_CDI_hist <- CDI$type == data_types[1]
CDI_matched <- merge(CDI[!is_CDI_hist, -3], CDI[is_CDI_hist, -3], by=c('lat', 'long'))
CDI_matched$CDI_increase <- CDI_matched$CDI.x - CDI_matched$CDI.y
sm <- merge(sm, CDI_matched[c('lat', 'long', 'CDI_increase')])
sm <- sm[sm$CDI_increase >= .2, setdiff(names(sm), 'CDI_increase')]

sm <- data.frame(dcast(melt(data.table(sm), id.vars=c('jday', 'lat', 'long', 'type'),
                            variable.name='sm_type'), jday + type + sm_type ~ lat + long))
probs <- c(0:10)/10 
arr <- t(apply(sm[, -(1:3)], 1, quantile, prob=probs, na.rm=T))
sm_q <- data.frame(sm[, 1:3], arr)
names(sm_q) <- c('jday', 'data_type', 'sm_type', colnames(arr))
sm_q_melted <- melt(data.table(sm_q), id.vars=1:3)
sm_q_melted$delta <- sm_q_melted$variable # add inter-quantile (per) range as delta 
levels(sm_q_melted$delta) <- abs(probs- rev(probs))*100

sm_to_plot <- ddply(
  sm_q_melted, .(jday, data_type, sm_type, delta), summarize, quantmin=min(value), quantmax=max(value))

#!! remove days after May 16 (jday 257) because they are absent from the future data (due to hardiness_driver.R)
sm_to_plot <- sm_to_plot[sm_to_plot$jday <= 257, ]
sm <- sm[sm$jday <= 257, ]

# sm$jday <- as.factor(sm$jday)
# sm_to_plot$jday <- as.factor(sm_to_plot$jday)


is_hist <- sm_to_plot$data_type=='1980–2016'
sm_to_plot_hist <- sm_to_plot[is_hist, ]
sm_to_plot_future <- sm_to_plot[!is_hist, ]
is_hist <- sm$type=='1980–2016'
sm_hist <- sm[is_hist, ]
sm_future <- sm[!is_hist, ]
sm_hist_ensemble_av <- data.frame(
  jday=sm_hist[, 1], sm_type=sm_hist[, 3], ens_av=apply(sm_hist[, -c(1, 3)][, -(1:3)], 1, mean, na.rm=T))
sm_future_ensemble_av <- data.frame(
  jday=sm_future[, 1], sm_type=sm_future[, 3], ens_av=apply(sm_future[, -c(1, 3)][, -(1:3)], 1, mean, na.rm=T))
sm_future_ensemble_av$variable <- as.factor("Mean")



colors <- c('1980–2016'='brown', '2025–2049'='blue3')
ggplot() + geom_hline(yintercept=0, linetype='dashed', size=.5) +
  geom_ribbon(data=sm_to_plot_hist, alpha=.5, show.legend=F,
              aes(x=jday, ymin=quantmin, ymax=quantmax,group=rev(delta), fill=as.numeric(delta))) +
  scale_fill_gradient(low = "pink", high = "darkred") +
  geom_line(data=sm_hist_ensemble_av, aes(jday, ens_av, color='1980–2016'), size=.2) + 
  new_scale_fill() +
  geom_ribbon(data=sm_to_plot_future, alpha=.5, show.legend=F,
              aes(x=jday, ymin=quantmin, ymax=quantmax,group=rev(delta), fill=as.numeric(delta))) +
  scale_fill_gradient(low = "cadetblue1", high = "dodgerblue3") +
  geom_line(data=sm_future_ensemble_av, aes(jday, ens_av, color='2025–2049'), size=.2) +
  labs(x='', y='safety margin, °C', title='grids with CDI increase ≥ 20%', color='') + theme(legend.position="top") + 
  facet_wrap(~ sm_type, nrow=3) +  scale_color_manual(values=colors) + 
  guides(color=guide_legend(override.aes=list(size=1, alpha=.5))) +
  scale_x_continuous(breaks=c(0, 30, 61, 91, 122, 153, 181, 212, 242), labels=sprintf('%s-01', c(
    "Sep", 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May')))

ggsave('plots/safety_margin_comparison_1980_2016_bcc-csm1-1-m_rcp45_2025-2049.png', width = 15, height = 10)
