setwd('~/CSANR/Cold/')
library(ggplot2)
library(data.table)
library(lubridate)
library(cowplot)
library(scales)

raw_grid <- read.table('data/all_calibrated_plus_uncalibrated_soil.txt')[c(3:4, 78)]
names(raw_grid) <- c('lat', 'long', 'elev')

prosser_lat <- 46.28125
prosser_long <- -119.71875
prosser_elev <- raw_grid[raw_grid$lat == prosser_lat & raw_grid$long == prosser_long, 'elev']
cascades_lat <- 48.96875
cascades_long <- -120.21875
cascades_elev <- raw_grid[raw_grid$lat == cascades_lat & raw_grid$long == cascades_long, 'elev']
chelan_lat <- 47.84375
chelan_long <- -120.09375
chelan_elev <- raw_grid[raw_grid$lat == chelan_lat & raw_grid$long == chelan_long, 'elev']

loc_texts <- c(
  sprintf('Prosser grid %s, %s (elev. %dm)', prosser_lat, prosser_long, round(prosser_elev)),
  sprintf('Cascades grid %s, %s (elev. %dm)', cascades_lat, cascades_long, round(cascades_elev)),
  sprintf('Chelan grid %s, %s (elev. %dm)', chelan_lat, chelan_long, round(chelan_elev)))
lats <- c(prosser_lat, cascades_lat, chelan_lat)
longs <- c(prosser_long, cascades_long, chelan_long)
elevs <- c(prosser_elev, cascades_elev, chelan_elev)
file_names <- c('Prosser_stats.png', 'Cascades_stats.png', 'Chelan_stats.png')
stats_by_loc_all_scenarios <- read.csv('data/stats_by_loc.csv')
stats_by_loc_all_scenarios <- stats_by_loc_all_scenarios[stats_by_loc_all_scenarios$CH_year != 2016, ]
stats_by_loc_all_scenarios$switch_day <- stats_by_loc_all_scenarios$switch_day - 61 # since Nov 1
stats_by_loc_all_scenarios$model <- factor(stats_by_loc_all_scenarios$model, levels = c(
  'obs_hist', "MIROC-ESM-CHEM", "HadGEM2-ES365", "HadGEM2-CC365", "BNU-ESM", "bcc-csm1-1", "CanESM2",
  "CNRM-CM5", "IPSL-CM5A-MR", "IPSL-CM5A-LR", "bcc-csm1-1-m", "GFDL-ESM2G", "IPSL-CM5B-LR",
  "CCSM4", "CSIRO-Mk3-6-0", "MIROC5", "NorESM1-M", "MRI-CGCM3", "GFDL-ESM2M", "inmcm4"))
n_models <- length(levels(stats_by_loc_all_scenarios$model))
colors <- c('black', hue_pal()(n_models + 6)[1:n_models - 1])

CDI <- list()
no_switch <- list()
switch_day <- list()
DD_h <- list()
avg_endo_slope <- list()
biweekly_endo_slope <- list()
avg_eco_slope <- list()
biweekly_eco_slope <- list()
for (scenario in c('rcp45', 'rcp85')) {
  stats_by_loc <- stats_by_loc_all_scenarios[
    stats_by_loc_all_scenarios$scenario == scenario | stats_by_loc_all_scenarios$model == 'obs_hist', ]
  for (i in 1:length(elevs)) {
    stats <- stats_by_loc[stats_by_loc$lat == lats[i] & stats_by_loc$long == longs[i], ]

    is_timeframe_1 <- stats$model == 'obs_hist'
    is_timeframe_2 <- stats$CH_year >= 2025 & stats$CH_year < 2050
    is_timeframe_3 <- stats$CH_year >= 2050 & stats$CH_year < 2075
    is_timeframe_4 <- stats$CH_year >= 2075 & stats$CH_year < 2100
    # To place individual years chronologically on the x-axis.
    stats$CH_year[is_timeframe_1] <- (stats$CH_year[is_timeframe_1] - 1979)/45 + .6
    stats$CH_year[is_timeframe_2] <- (stats$CH_year[is_timeframe_2] - 2025)/30 + 1.6
    stats$CH_year[is_timeframe_3] <- (stats$CH_year[is_timeframe_3] - 2050)/30 + 2.6
    stats$CH_year[is_timeframe_4] <- (stats$CH_year[is_timeframe_4] - 2075)/30 + 3.6
    
    CDI_1 <- mean(stats$is_CD[is_timeframe_1])
    CDI_2 <- mean(stats$is_CD[is_timeframe_2])
    CDI_3 <- mean(stats$is_CD[is_timeframe_3])
    CDI_4 <- mean(stats$is_CD[is_timeframe_4])
    no_switch_1 <- mean(!stats$is_eco[is_timeframe_1])
    no_switch_2 <- mean(!stats$is_eco[is_timeframe_2])
    no_switch_3 <- mean(!stats$is_eco[is_timeframe_3])
    no_switch_4 <- mean(!stats$is_eco[is_timeframe_4])
    label_0 <- annotate('text', x=.5, y=Inf, label = 'CDI:\nno eco:', size=4)
    label_1 <- annotate('text', x=1, y=Inf, label = sprintf('%1.2f%%\n%1.2f%%', 100*CDI_1, 100*no_switch_1), size=4)
    label_2 <- annotate('text', x=2, y=Inf, label = sprintf('%1.2f%%\n%1.2f%%', 100*CDI_2, 100*no_switch_2), size=4)
    label_3 <- annotate('text', x=3, y=Inf, label = sprintf('%1.2f%%\n%1.2f%%', 100*CDI_3, 100*no_switch_3), size=4)
    label_4 <- annotate('text', x=4, y=Inf, label = sprintf('%1.2f%%\n%1.2f%%', 100*CDI_4, 100*no_switch_4), size=4)
    stats$range <- ifelse(
      is_timeframe_1, 1, ifelse(is_timeframe_2, 2, ifelse(is_timeframe_3, 3, ifelse(is_timeframe_4, 4, NA))))
    stats <- stats[!is.na(stats$range), ]
    stats$range <- as.factor(stats$range)
    levels(stats$range) <- c('1979-2015', '2025-2049', '2050-2074', '2075-2099')
    
    stats_CDI <- aggregate(list(CDI = stats$is_CD), by = list(range = stats$range, model = stats$model), mean)
    CDI[[scenario]][[i]] <- ggplot(stats_CDI, aes(x = range, y = CDI)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') +
      geom_jitter(aes(color = model), size=3, position = position_jitter(w = 0.4, h = 0)) + 
      scale_color_manual(values = colors) + label_0 + label_1 + label_2 + label_3 + label_4 +
      scale_y_continuous(breaks = (1:4)/4, limits = c(0, 1.1), labels = percent)
    stats_no_switch <- aggregate(list(no_switch = !stats$is_eco), by = list(range = stats$range, model = stats$model), mean)
    no_switch[[scenario]][[i]]<- ggplot(stats_no_switch, aes(x = range, y = no_switch)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') +
      geom_jitter(aes(color = model), size=3, position = position_jitter(w = 0.4, h = 0)) + 
      scale_color_manual(values = colors) + label_0 + label_1 + label_2 + label_3 + label_4 + 
      scale_y_continuous(breaks = (1:4)/4, limits = c(0, 1.1), labels = percent)
    switch_day[[scenario]][[i]] <- ggplot(stats, aes(x = range, y = switch_day)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') + 
      geom_point(aes(x = CH_year, color = model), size=1) + scale_color_manual(values = colors) +
      label_0 + label_1 + label_2 + label_3 + label_4
    DD_h[[scenario]][[i]] <- ggplot(stats, aes(x = range, y = DD_h)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') + 
      geom_point(aes(x = CH_year, color = model), size=1) + scale_color_manual(values = colors) +
      label_0 + label_1 + label_2 + label_3 + label_4
    avg_endo_slope[[scenario]][[i]] <- ggplot(stats, aes(x = range, y = avg_slope_endo)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') + 
      geom_point(aes(x = CH_year, color = model), size=1) + scale_color_manual(values = colors) +
      label_0 + label_1 + label_2 + label_3 + label_4
    biweekly_endo_slope[[scenario]][[i]] <- ggplot(stats, aes(x = range, y = biweekly_slope_endo)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') + 
      geom_point(aes(x = CH_year, color = model), size=1) + scale_color_manual(values = colors) +
      label_0 + label_1 + label_2 + label_3 + label_4
    avg_eco_slope[[scenario]][[i]] <- ggplot(stats, aes(x = range, y = avg_slope_eco)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') + 
      geom_point(aes(x = CH_year, color = model), size=1) + scale_color_manual(values = colors) +
      label_0 + label_1 + label_2 + label_3 + label_4
    biweekly_eco_slope[[scenario]][[i]] <- ggplot(stats, aes(x = range, y = biweekly_slope_eco)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') + 
      geom_point(aes(x = CH_year, color = model), size=1) + scale_color_manual(values = colors) +
      label_0 + label_1 + label_2 + label_3 + label_4
    
  }
}

grid_labels <- c('Cascades RCP4.5','Cascades RCP8.5', 'Prosser RCP4.5','Prosser RCP8.5',
                 'Chelan RCP4.5', 'Chelan RCP8.5')
model_legend <- get_legend(CDI[['rcp45']][[1]] + guides(fill = guide_legend(nrow=1, title = 'model')) + 
                             theme(legend.position = 'bottom'))
style <- theme(legend.position="none", axis.text.y=element_text(size=11), axis.text.x=element_text(size=11))

CDIs <- plot_grid(
   CDI[['rcp45']][[2]] + style, CDI[['rcp85']][[2]] + style, CDI[['rcp45']][[1]] + style,
   CDI[['rcp85']][[1]] + style, CDI[['rcp45']][[3]] + style, CDI[['rcp85']][[3]] + style,
   ncol = 2, labels = grid_labels, label_y = 1.03, label_x = 0.1, hjust = 0, scale = 0.93)
top <- plot_grid(model_legend, labels = 'CDI', label_x = 0.5)
plot_grid(top, CDIs, nrow = 2, rel_heights = c(.23, 2.5), axis = 'lrtb')
ggsave(paste0('plots/CDI_stats.png'), width = 15, height = 25, limitsize = F)

no_switches <- plot_grid(
  no_switch[['rcp45']][[2]] + style, no_switch[['rcp85']][[2]] + style, no_switch[['rcp45']][[1]] + style,
  no_switch[['rcp85']][[1]] + style, no_switch[['rcp45']][[3]] + style, no_switch[['rcp85']][[3]] + style,
  ncol = 2, labels = grid_labels, label_y = 1.03, label_x = 0.1, hjust = 0, scale = 0.93)
top <- plot_grid(model_legend, labels = 'no ecodormancy rate', label_x = 0.4)
plot_grid(top, no_switches, nrow = 2, rel_heights = c(.23, 2.5), axis = 'lrtb')
ggsave(paste0('plots/no_eco_stats.png'), width = 15, height = 25, limitsize = F)

switch_days <- plot_grid(
  switch_day[['rcp45']][[2]] + style, switch_day[['rcp85']][[2]] + style, switch_day[['rcp45']][[1]] + style,
  switch_day[['rcp85']][[1]] + style, switch_day[['rcp45']][[3]] + style, switch_day[['rcp85']][[3]] + style,
  ncol = 2, labels = grid_labels, label_y = 1.03, label_x = 0.1, hjust = 0, scale = 0.93)
top <- plot_grid(model_legend, labels = 'day of ecodormancy switch (from Nov 01)', label_x = 0.25)
plot_grid(top, switch_days, nrow = 2, rel_heights = c(.23, 2.5), axis = 'lrtb')
ggsave(paste0('plots/switch_day_stats.png'), width = 15, height = 25, limitsize = F)

avg_endo_slopes <- plot_grid(
  avg_endo_slope[['rcp45']][[2]] + style, avg_endo_slope[['rcp85']][[2]] + style, avg_endo_slope[['rcp45']][[1]] + style,
  avg_endo_slope[['rcp85']][[1]] + style, avg_endo_slope[['rcp45']][[3]] + style, avg_endo_slope[['rcp85']][[3]] + style,
  ncol = 2, labels = grid_labels, label_y = 1.03, label_x = 0.1, hjust = 0, scale = 0.93)
top <- plot_grid(model_legend, labels = 'mean acclimation slope', label_x = 0.38)
plot_grid(top, avg_endo_slopes, nrow = 2, rel_heights = c(.23, 2.5), axis = 'lrtb')
ggsave(paste0('plots/avg_endo_slope_stats.png'), width = 15, height = 25, limitsize = F)

biweekly_endo_slopes <- plot_grid(
  biweekly_endo_slope[['rcp45']][[2]] + style, biweekly_endo_slope[['rcp85']][[2]] + style, biweekly_endo_slope[['rcp45']][[1]] + style,
  biweekly_endo_slope[['rcp85']][[1]] + style, biweekly_endo_slope[['rcp45']][[3]] + style, biweekly_endo_slope[['rcp85']][[3]] + style,
  ncol = 2, labels = grid_labels, label_y = 1.03, label_x = 0.1, hjust = 0, scale = 0.93)
top <- plot_grid(model_legend, labels = 'max biweekly mean acclimation slope', label_x = 0.3)
plot_grid(top, biweekly_endo_slopes, nrow = 2, rel_heights = c(.23, 2.5), axis = 'lrtb')
ggsave(paste0('plots/biweekly_endo_slope_stats.png'), width = 15, height = 25, limitsize = F)

avg_eco_slopes <- plot_grid(
  avg_eco_slope[['rcp45']][[2]] + style, avg_eco_slope[['rcp85']][[2]] + style, avg_eco_slope[['rcp45']][[1]] + style,
  avg_eco_slope[['rcp85']][[1]] + style, avg_eco_slope[['rcp45']][[3]] + style, avg_eco_slope[['rcp85']][[3]] + style,
  ncol = 2, labels = grid_labels, label_y = 1.03, label_x = 0.1, hjust = 0, scale = 0.93)
top <- plot_grid(model_legend, labels = 'mean deacclimation slope', label_x = 0.37)
plot_grid(top, avg_eco_slopes, nrow = 2, rel_heights = c(.23, 2.5), axis = 'lrtb')
ggsave(paste0('plots/avg_eco_slope_stats.png'), width = 15, height = 25, limitsize = F)

biweekly_eco_slopes <- plot_grid(
  biweekly_eco_slope[['rcp45']][[2]] + style, biweekly_eco_slope[['rcp85']][[2]] + style, biweekly_eco_slope[['rcp45']][[1]] + style,
  biweekly_eco_slope[['rcp85']][[1]] + style, biweekly_eco_slope[['rcp45']][[3]] + style, biweekly_eco_slope[['rcp85']][[3]] + style,
  ncol = 2, labels = grid_labels, label_y = 1.03, label_x = 0.1, hjust = 0, scale = 0.93)
top <- plot_grid(model_legend, labels = 'max biweekly mean deacclimation slope', label_x = 0.29)
plot_grid(top, biweekly_eco_slopes, nrow = 2, rel_heights = c(.23, 2.5), axis = 'lrtb')
ggsave(paste0('plots/biweekly_eco_slope_stats.png'), width = 15, height = 25, limitsize = F)

