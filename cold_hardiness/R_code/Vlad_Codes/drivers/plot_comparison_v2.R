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
  loc <- read.csv(path)[c('variety', 'Date', 't_min', 't_max', 'predicted_Hc')]
  names(loc)[5] <- 't_CH'
  loc$Date <- as.Date(loc$Date)
  loc$jday <- yday(loc$Date)
  loc$month <- month(loc$Date)
  loc$year <- year(loc$Date)
  loc$CH_year <- ifelse(loc$month > 8, loc$year + 1, ifelse(loc$month < 6, loc$year, NA))
  loc <- na.omit(loc)
  loc <- loc[!leap_year(loc$year) | loc$jday != 60, ] # remove Feb 29
  is_leap_jday <- leap_year(loc$year) & loc$month > 2 
  loc[is_leap_jday, 'jday'] <- loc[is_leap_jday, 'jday'] - 1 # decrease jday in leap years
  loc <- loc[loc$CH_year >= range_starts[i] & loc$CH_year <= range_ends[i] & loc$variety == 'Cabernet Sauvignon', ]
  loc$jday <- (loc$jday - 244) %% 365
  loc$sm <- loc$t_min - loc$t_CH # safety margin
  loc$lat <- lat
  loc$long <- long
  
  return(list(lat=lat, long=long, loc=loc[loc$jday <= 257, ]))
}

dirs <- c('WA_gridmet', 'bcc-csm1-1-m rcp45')
variety <- 'Cabernet Sauvignon'
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

#=====================================
# start <- Sys.time()
# for (var in c('sm', 't_CH', 't_max', 't_min')) {
#   
#   df_plot_paths <- sprintf('data/%s_comparison_%s.csv', var, data_descs)
#   df_plot <- list('1'=NULL, '2'=NULL)
#   df_ensav_paths <- sprintf('data/%s_ensav_for_comparison_%s.csv', var, data_descs)
#   df_ensav <- list('1'=NULL, '2'=NULL)
#   if (!all(file.exists(df_plot_paths))) {
#     probs <- (0:10)/10 
#     for (i in 1:2) {
#       df <- NULL
#       n_path <- 0
#       files_list <- list.files(path=paste0('data/', dirs[i]), pattern="*.csv", full.names=T)
#       for (path in files_list) {
#         n_path <- n_path + 1
#         df <- rbind(df, read_loc(path)$loc[c('jday', var, 'CH_year', 'lat', 'long')])
#         if (n_path %% 100 == 0)
#         print(sprintf('%s %s %s %d/%d loaded.', Sys.time() - start, var, data_descs[i], n_path, length(files_list)))
#         
#       }
#       df <- merge(df, CDI_increase)
#       df$CDI_increase_cat <- ifelse(
#         df$CDI_increase < -.05, 1, ifelse(df$CDI_increase < .05, 2, ifelse(df$CDI_increase < .2, 3, 4)))
#       print(sprintf('%s %s %s CDI increase cat computed.', Sys.time() - start, var, data_descs[i]))
#       df <- data.frame(melt(data.table(df[setdiff(names(df), 'CDI_increase')]),
#                  id.vars=c('CDI_increase_cat', 'jday', 'lat', 'long', 'CH_year'), variable.name='var'))
#       print(sprintf('%s %s %s reshaped.', Sys.time() - start, var, data_descs[i]))
#       df_ensav[[i]] <- aggregate(value ~ CDI_increase_cat + jday + var, df, mean, na.rm=T)
#       names(df_ensav[[i]])[4] <- 'ens_av'
#       print(sprintf('%s %s %s ensav computed', Sys.time() - start, var, data_descs[i]))
#       write.csv(df_ensav[[i]], file=df_ensav_paths[i], row.names=F)
#       
#       for (j in (1:4)) {
#         df_cat_m <- data.frame(dcast(data.table(df[df$CDI_increase_cat == j, ]),
#                                      CDI_increase_cat + jday + var ~ lat + long + CH_year))
#         print(sprintf('%s %s %s df_cat_m computed', Sys.time() - start, var, data_descs[i]))
#         # calculate percentiles across grids and years
#         arr <- t(apply(df_cat_m[, -(1:3)], 1, quantile, prob=probs, na.rm=T))
#         df_cat_q <- data.frame(df_cat_m[, 1:3], arr)
#         names(df_cat_q) <- c('CDI_increase_cat', 'jday', 'var', colnames(arr))
#         df_cat_q_m <- melt(data.table(df_cat_q), id.vars=1:3)
#         df_cat_q_m$delta <- df_cat_q_m$variable # add inter-quantile (per) range as delta 
#         levels(df_cat_q_m$delta) <- abs(probs - rev(probs))*100
#         
#         df_plot_cat <- ddply(df_cat_q_m, .(CDI_increase_cat, jday, var, delta), summarize,
#                              quantmin=min(value), quantmax=max(value))
#         print(sprintf('%s %s %s df_plot_cat computed', Sys.time() - start, var, data_descs[i]))
#         
#         df_plot[[i]] <- rbind(df_plot[[i]], df_plot_cat)
#       }
#       write.csv(df_plot[[i]], file=df_plot_paths[i], row.names=F)
#     }
#   } else {
#     for (i in 1:2) {
#       df_plot[[i]] <- read.csv(df_plot_paths[i])
#       df_ensav[[i]] <- read.csv(df_ensav_paths[i])
#     }
#   }
#   print(sprintf('%s %s loaded.', Sys.time() - start, var))
# }
#=====================================
df_plot_paths <- sprintf('data/comparison_%s.csv', data_descs)
df_plot <- list('1'=NULL, '2'=NULL)
df_ensav_paths <- sprintf('data/ensav_for_comparison_%s.csv', data_descs)
df_ensav <- list('1'=NULL, '2'=NULL)

colors <- c('1980–2016'='brown', '2025–2049'='blue3')
for (i in 1:2) {
  df_plot[[i]] <- read.csv(df_plot_paths[i])
  df_ensav[[i]] <- read.csv(df_ensav_paths[i])
  df_plot[[i]]$CDI_increase_cat <- as.factor(df_plot[[i]]$CDI_increase_cat)
  levels(df_plot[[i]]$CDI_increase_cat) <- sprintf('CDI change %s', c('< -5%', 'in [-5%, +5%)', 'in [+5%, +20%)', '> +20%'))
  for (cat in unique(df_plot[[i]]$CDI_increase_cat))  
    for (jday in unique(df_plot[[i]]$jday))
      for (delta in unique(df_plot[[i]]$delta)) {
        df <- df_plot[[i]][df_plot[[i]]$CDI_increase_cat == cat & df_plot[[i]]$jday == jday & df_plot[[i]]$delta == delta, ]
        t_mean_quantmin <- (df[df$var == 't_min', 'quantmin'] + df[df$var == 't_max', 'quantmin'])/2
        t_mean_quantmax <- (df[df$var == 't_min', 'quantmax'] + df[df$var == 't_max', 'quantmax'])/2
        df_plot[[i]] <- rbind(df_plot[[i]], data.frame(
          CDI_increase_cat=cat, jday=jday, var='t_mean', delta=delta, quantmin=t_mean_quantmin, quantmax=t_mean_quantmax))
      }
  df_ensav[[i]]$CDI_increase_cat <- as.factor(df_ensav[[i]]$CDI_increase_cat)
  levels(df_ensav[[i]]$CDI_increase_cat) <- sprintf('CDI change %s', c('< -5%', 'in [-5%, +5%)', 'in [+5%, +20%)', '> +20%'))
  for (cat in unique(df_ensav[[i]]$CDI_increase_cat))  
    for (jday in unique(df_ensav[[i]]$jday)) {
      df <- df_ensav[[i]][df_ensav[[i]]$CDI_increase_cat == cat & df_ensav[[i]]$jday == jday, ]
      t_mean_ens_av <- (df[df$var == 't_min', 'ens_av'] + df[df$var == 't_max', 'ens_av'])/2
      df_ensav[[i]] <- rbind(df_ensav[[i]], data.frame(
        CDI_increase_cat=cat, jday=jday, var='t_mean', ens_av=t_mean_ens_av))
    }
}

alpha <- .2
jdays_show <- c(122, 153, 181, 212, 242)
for (var in c('sm', 't_CH', 't_max', 't_min', 't_mean')) {
  var_name <- ifelse(var == 'sm', 'safety margin, °C', ifelse(
    var == 't_CH', 'tCH, °C', ifelse(var == 't_max', 'tmax, °C', ifelse(var == 't_mean', 'tmean, °C', 'tmin, °C'))))
  hist_plot_var <- df_plot[[1]][df_plot[[1]]$var == var, ]
  future_plot_var <- df_plot[[2]][df_plot[[2]]$var == var, ]
  hist_ensav_var <- df_ensav[[1]][df_ensav[[1]]$var == var, ]
  future_ensav_var <- df_ensav[[2]][df_ensav[[2]]$var == var, ]
  p <- ggplot() +
    geom_ribbon(data=hist_plot_var, alpha=alpha, show.legend=F,
                aes(x=jday, ymin=quantmin, ymax=quantmax, group=rev(delta), fill=as.numeric(delta))) +
    scale_fill_gradient(low = "pink", high = "darkred") +
    new_scale_fill() +
    geom_ribbon(data=future_plot_var, alpha=alpha, show.legend=F,
                aes(x=jday, ymin=quantmin, ymax=quantmax,group=rev(delta), fill=as.numeric(delta))) +
    scale_fill_gradient(low = "cadetblue1", high = "dodgerblue3") +
    labs(x='', y=var_name, color='') + theme(legend.position="top") +
    facet_wrap(~ CDI_increase_cat, ncol=1) +  scale_color_manual(values=colors) +
    guides(color=guide_legend(override.aes=list(size=1, alpha=alpha))) +
    scale_x_continuous(breaks=c(0, 30, 61, 91, 122, 153, 181, 212, 242), labels=sprintf('%s-01', c(
      "Sep", 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'))) +
    geom_line(data=hist_ensav_var, aes(jday, ens_av, color='1980–2016'), size=.5) +
    geom_line(data=future_ensav_var, aes(jday, ens_av, color='2025–2049'), size=.5)
  if (var == 'sm') {
    p <- p + geom_hline(yintercept=0, linetype='dashed', size=.5)
  } else if (var == 't_min' | var == 't_max' | var == 't_mean') {
    for (cat in unique(hist_ensav_var$CDI_increase_cat)) {
      for (j in 1:(length(jdays_show)-1)) {
        #      for (jday in c(122, 137, 153, 167, 181, 196, 212, 227, 242)) {
        jday_from <- jdays_show[j]
        jday_to <- jdays_show[j + 1]
        hist_month_av <- data.frame(
          CDI_increase_cat=cat, jday=(jday_from + jday_to - 1)/2, var=var, ens_av=mean(hist_ensav_var[
            hist_ensav_var$jday >= jday_from & hist_ensav_var$jday < jday_to & 
              hist_ensav_var$CDI_increase_cat == cat & hist_ensav_var$var == var, 'ens_av']))
        future_month_av <- data.frame(
          CDI_increase_cat=cat, jday=(jday_from + jday_to - 1)/2, var=var, ens_av=mean(future_ensav_var[
            future_ensav_var$jday >= jday_from & future_ensav_var$jday < jday_to & 
              future_ensav_var$CDI_increase_cat == cat & future_ensav_var$var == var, 'ens_av']))
        if (hist_month_av$ens_av > future_month_av$ens_av) {
          max_month_av <- hist_month_av
          max_color <- colors[1]
          min_month_av <- future_month_av
          min_color <- colors[2]
        } else {
          max_month_av <- future_month_av
          max_color <- colors[2]
          min_month_av <- hist_month_av
          min_color <- colors[1]
        }
        p <- p + 
          geom_text(data=max_month_av, aes(x=jday, y=ens_av+4, label = paste0(round(ens_av, 1), '°C'),
                                           group = CDI_increase_cat), size = 2, color=max_color, inherit.aes = FALSE) +
          geom_text(data=min_month_av, aes(x=jday, y=ens_av-4, label = paste0(round(ens_av, 1), '°C'),
                                           group = CDI_increase_cat), size = 2, color=min_color, inherit.aes = FALSE) +
          geom_segment(data=max_month_av, x=jday_from, y=max_month_av$ens_av, xend=jday_to,
                       yend=max_month_av$ens_av, linetype='dashed', size=.25, color=max_color) +
          geom_segment(data=min_month_av, x=jday_from, y=min_month_av$ens_av, xend=jday_to,
                       yend=min_month_av$ens_av, linetype='dashed', size=.25, color=min_color)
      }
    }
  }
  
  ggsave(sprintf('plots/%s_comparison_1980_2016_bcc-csm1-1-m_rcp45_2025-2049.png', var),
         width = 15, height = 10)
}
