setwd('~/CSANR/Cold/')
library(lubridate)
library(plyr)
library(tidyr)
library(ggplot2)
library(data.table)
library(cowplot)
library(ggnewscale)

read_loc <- function(path) {
  lat_long <- strsplit(tools::file_path_sans_ext(basename(path)), '_')[[1]]
  lat <- lat_long[length(lat_long) - 1]
  long <- lat_long[length(lat_long)]
  loc <- read.csv(path)[c('Date', 't_min', 't_max', 'predicted_Hc')]
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
  loc$lat <- lat
  loc$long <- long
  
  return(list(lat=lat, long=long, loc=loc[loc$jday <= 257, ]))
}

dirs <- c('WA_gridmet', 'bcc-csm1-1-m rcp45')
range_starts <- c(1980, 2025)
range_ends <- c(2016, 2049)
data_descs <- paste(range_starts, range_ends, sep='-')

CDI_increase_path <- sprintf('data/CDI_increase_%s_%s.csv', data_descs[1], data_descs[2])
if (!file.exists(CDI_increase_path)) {
  CDI <- list('1'=NULL, '2'=NULL)
  for (i in 1:2) {
    for (path in list.files(path=paste0('data/', dirs[i]), pattern="*.csv", full.names=T)) {
      res <- read_loc(path)
      loc_CDI <- data.frame(res$lat, res$long, CDI=mean(aggregate(
        list(is_CD = res$loc$sm < 0), by = list(CH_year = res$loc$CH_year), any)$is_CD))
      CDI[[i]] <- rbind(CDI[[i]], loc_CDI)
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

df_plot_paths <- sprintf('data/for_comparison_%s.csv', data_descs)
df_plot <- list('1'=NULL, '2'=NULL)
df_ensav_paths <- sprintf('data/ensav_for_comparison_%s.csv', data_descs)
df_ensav <- list('1'=NULL, '2'=NULL)
if (!all(file.exists(df_plot_paths))) {
  probs <- (0:10)/10 
  for (i in 1:2) {
    df <- NULL
    n_path <- 0
    files_list <- list.files(path=paste0('data/', dirs[i]), pattern="*.csv", full.names=T)
    for (path in files_list) {
      n_path <- n_path + 1
      df <- rbind(df, read_loc(path)$loc[c('jday', 'sm', 't_min', 't_max', 'predicted_Hc', 'CH_year', 'lat', 'long')])
      if (n_path %% 100 == 0)
      print(sprintf('%s %s %d/%d loaded.', var, data_descs[i], n_path, length(files_list)))
      
    }
    df <- merge(df, CDI_increase)
    df$CDI_increase_cat <- ifelse(
      df$CDI_increase < -.05, 1, ifelse(df$CDI_increase < .05, 2, ifelse(df$CDI_increase < .2, 3, 4)))
    df <- data.frame(melt(data.table(df[setdiff(names(df), 'CDI_increase')]),
               id.vars=c('CDI_increase_cat', 'jday', 'lat', 'long', 'CH_year'), variable.name='var'))
    df_ensav[[i]] <- aggregate(value ~ CDI_increase_cat + jday + var, df, mean, na.rm=T)
    names(df_ensav[[i]])[4] <- 'ens_av'
    write.csv(df_ensav[[i]], file=df_ensav_paths[i], row.names=F)
    
    for (j in (1:4)) {
      df_cat_m <- data.frame(dcast(data.table(df[df$CDI_increase_cat == j, ]),
                                   CDI_increase_cat + jday + var ~ lat + long + CH_year))
      # calculate percentiles across grids and years
      arr <- t(apply(df_cat_m[, -(1:3)], 1, quantile, prob=probs, na.rm=T))
      df_cat_q <- data.frame(df_cat_m[, 1:3], arr)
      names(df_cat_q) <- c('CDI_increase_cat', 'jday', 'var', colnames(arr))
      df_cat_q_m <- melt(data.table(df_cat_q), id.vars=1:3)
      df_cat_q_m$delta <- df_cat_q_m$variable # add inter-quantile (per) range as delta 
      levels(df_cat_q_m$delta) <- abs(probs - rev(probs))*100
      
      df_plot_cat <- ddply(df_cat_q_m, .(CDI_increase_cat, jday, var, delta), summarize,
                           quantmin=min(value), quantmax=max(value))
      
      df_plot[[i]] <- rbind(df_plot[[i]], df_plot_cat)
    }
    write.csv(df_plot[[i]], file=df_plot_paths[i], row.names=F)
  }
} else {
  for (i in 1:2)
    df_plot[[i]] <- read.csv(df_plot_paths[i])
}
print(sprintf('%s loaded.', var))
#=====================================

colors <- c('1980–2016'='brown', '2025–2049'='blue3')
for (i in 1:2) {
  df_plot[[i]]$var[df_plot[[i]]$var == 'predicted_Hc'] <- 't_CH'
  df_plot[[i]]$CDI_increase_cat <- as.factor(df_plot[[i]]$CDI_increase_cat)
  levels(df_plot[[i]]$CDI_increase_cat) <- c('< -5%', '[-5%, +5%)', '[+5%, +20%)', '> +20%')
  df_ensav[[i]]$var[df_ensav[[i]]$var == 'predicted_Hc'] <- 't_CH'
  df_ensav[[i]]$CDI_increase_cat <- as.factor(df_ensav[[i]]$CDI_increase_cat)
  levels(df_ensav[[i]]$CDI_increase_cat) <- c('< -5%', '[-5%, +5%)', '[+5%, +20%)', '> +20%')
}

for (var in c('sm', 't_min', 't_max', 't_CH')) {
  hist_plot_var <- df_plot[[1]][df_plot[[1]]$var == var, ]
  future_plot_var <- df_plot[[2]][df_plot[[2]]$var == var, ]
  hist_ensav_var <- df_ensav[[1]][df_ensav[[1]]$var == var, ]
  future_ensav_var <- df_ensav[[2]][df_ensav[[2]]$var == var, ]
  ggplot() + geom_hline(yintercept=0, linetype='dashed', size=.5) +
    geom_ribbon(data=hist_plot_var, alpha=.5, show.legend=F,
                aes(x=jday, ymin=quantmin, ymax=quantmax, group=rev(delta), fill=as.numeric(delta))) +
    scale_fill_gradient(low = "pink", high = "darkred") +
    geom_line(data=hist_ensav_var, aes(jday, ens_av, color='1980–2016'), size=.2) +
    new_scale_fill() +
    geom_ribbon(data=future_plot_var, alpha=.5, show.legend=F,
                aes(x=jday, ymin=quantmin, ymax=quantmax,group=rev(delta), fill=as.numeric(delta))) +
    scale_fill_gradient(low = "cadetblue1", high = "dodgerblue3") +
    geom_line(data=future_ensav_var, aes(jday, ens_av, color='2025–2049'), size=.2) +
    labs(x='', y='safety margin, °C', title='grids with CDI increase ≥ 20%', color='') + theme(legend.position="top") +
    facet_wrap(~ CDI_increase_cat + var, ncol=1) +  scale_color_manual(values=colors) +
    guides(color=guide_legend(override.aes=list(size=1, alpha=.5))) +
    scale_x_continuous(breaks=c(0, 30, 61, 91, 122, 153, 181, 212, 242), labels=sprintf('%s-01', c(
      "Sep", 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May')))
  
  ggsave(sprintf('plots/%s_comparison_1980_2016_bcc-csm1-1-m_rcp45_2025-2049.png', var),
         width = 15, height = 10)
}