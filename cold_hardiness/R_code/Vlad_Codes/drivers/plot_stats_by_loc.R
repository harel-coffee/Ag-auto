setwd('~/CSANR/Cold/')
library(ggplot2)
library(data.table)
library(lubridate)
library(cowplot)
library(scales)

raw_grid <- read.table('data/all_calibrated_plus_uncalibrated_soil.txt')[c(3:4, 78)]
names(raw_grid) <- c('lat', 'long', 'elev')

# prosser_lat <- 46.28125
# prosser_long <- -119.71875
# prosser_elev <- raw_grid[raw_grid$lat == prosser_lat & raw_grid$long == prosser_long, 'elev']
# cascades_lat <- 48.96875
# cascades_long <- -120.21875
# cascades_elev <- raw_grid[raw_grid$lat == cascades_lat & raw_grid$long == cascades_long, 'elev']
# chelan_lat <- 47.84375
# chelan_long <- -120.09375
# chelan_elev <- raw_grid[raw_grid$lat == chelan_lat & raw_grid$long == chelan_long, 'elev']
min_CDI_change_lat <- 46.53125
min_CDI_change_long <- -119.28125
min_CDI_change_elev <- raw_grid[raw_grid$lat == min_CDI_change_lat & raw_grid$long == min_CDI_change_long, 'elev']
max_CDI_change_lat <- 46.46875
max_CDI_change_long <- -120.28125
max_CDI_change_elev <- raw_grid[raw_grid$lat == max_CDI_change_lat & raw_grid$long == max_CDI_change_long, 'elev']

lats <- c(min_CDI_change_lat, max_CDI_change_lat)
longs <- c(min_CDI_change_long, max_CDI_change_long)
elevs <- round(c(min_CDI_change_elev, max_CDI_change_elev))
CDI_changes <- c('decreased', 'increased')
loc_texts <- sprintf('%s CDI grid %s, %s (elev. %dm)', CDI_changes, lats, longs, elevs)
file_names <- sprintf('%s_stats.png', CDI_changes)
stats_by_loc_all_scenarios <- read.csv('data/CH_stats_by_loc_by_year.csv')
stats_by_loc_all_scenarios <- stats_by_loc_all_scenarios[
  stats_by_loc_all_scenarios$CH_year != 2016 & stats_by_loc_all_scenarios$variety == 'Cabernet Sauvignon', ]
stats_by_loc_all_scenarios$switch_day <- stats_by_loc_all_scenarios$switch_day - 61 # since Nov 1
stats_by_loc_all_scenarios$model <- factor(stats_by_loc_all_scenarios$model, levels = c(
  'obs_hist', "MIROC-ESM-CHEM", "HadGEM2-ES365", "HadGEM2-CC365", "BNU-ESM", "bcc-csm1-1", "CanESM2",
  "CNRM-CM5", "IPSL-CM5A-MR", "IPSL-CM5A-LR", "bcc-csm1-1-m", "GFDL-ESM2G", "IPSL-CM5B-LR",
  "CCSM4", "CSIRO-Mk3-6-0", "MIROC5", "NorESM1-M", "MRI-CGCM3", "GFDL-ESM2M", "inmcm4"))
n_models <- length(levels(stats_by_loc_all_scenarios$model))
colors <- c('black', hue_pal()(n_models + 6)[1:n_models - 1])
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
    # label_0 <- geom_text(x=0.25, y=1.1, label = 'mean CDI:\nmean no switch:')
    # label_1 <- geom_text(x=1, y=1.1, label = sprintf('%1.2f%%\n%1.2f%%', 100*CDI_1, 100*no_switch_1))
    # label_2 <- geom_text(x=2, y=1.1, label = sprintf('%1.2f%%\n%1.2f%%', 100*CDI_2, 100*no_switch_2))
    # label_3 <- geom_text(x=3, y=1.1, label = sprintf('%1.2f%%\n%1.2f%%', 100*CDI_3, 100*no_switch_3))
    # label_4 <- geom_text(x=4, y=1.1, label = sprintf('%1.2f%%\n%1.2f%%', 100*CDI_4, 100*no_switch_4))
    stats$range <- ifelse(
      is_timeframe_1, 1, ifelse(is_timeframe_2, 2, ifelse(is_timeframe_3, 3, ifelse(is_timeframe_4, 4, NA))))
    stats <- stats[!is.na(stats$range), ]
    stats$range <- as.factor(stats$range)
    levels(stats$range) <- c('1979-2015', '2025-2049', '2050-2074', '2075-2099')
    
    stats_CDI <- aggregate(list(CDI = stats$is_CD), by = list(range = stats$range, model = stats$model), mean)
    CDI <- ggplot(stats_CDI, aes(x = range, y = CDI)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') +
      geom_jitter(aes(color = model), size=3, position = position_jitter(w = 0.4, h = 0)) + 
      scale_color_manual(values = colors) + geom_text(x=0.25, y=1.1, label = 'mean:') +
      annotate('text', x=1, y=1.1, label = sprintf('%1.2f%%', 100*CDI_1)) + 
      annotate('text', x=2, y=1.1, label = sprintf('%1.2f%%', 100*CDI_2)) + 
      annotate('text', x=3, y=1.1, label = sprintf('%1.2f%%', 100*CDI_3)) + 
      annotate('text', x=4, y=1.1, label = sprintf('%1.2f%%', 100*CDI_4)) + 
      scale_y_continuous(breaks = (1:4)/4, limits = c(0, 1.1), labels = percent)
    stats_no_switch <- aggregate(list(no_switch = !stats$is_eco), by = list(range = stats$range, model = stats$model), mean)
    no_switch <- ggplot(stats_no_switch, aes(x = range, y = no_switch)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') +
      geom_jitter(aes(color = model), size=3, position = position_jitter(w = 0.4, h = 0)) + 
      scale_color_manual(values = colors) + geom_text(x=0.25, y=1.1, label = 'mean:') +
      annotate('text', x=1, y=1.1, label = sprintf('%1.2f%%', 100*no_switch_1)) + 
      annotate('text', x=2, y=1.1, label = sprintf('%1.2f%%', 100*no_switch_2)) + 
      annotate('text', x=3, y=1.1, label = sprintf('%1.2f%%', 100*no_switch_3)) + 
      annotate('text', x=4, y=1.1, label = sprintf('%1.2f%%', 100*no_switch_4)) + 
      scale_y_continuous(breaks = (1:4)/4, limits = c(0, 1.1), labels = percent)
    switch_day <- ggplot(stats, aes(x = range, y = switch_day)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + geom_jitter(aes(color = model), size=1) + labs(x = '', y = '') +
      coord_cartesian(clip = 'off') + scale_color_manual(values = colors) #label_0 + label_1 + label_2 + label_3 + label_4 +
    DD_h <- ggplot(stats, aes(x = range, y = DD_h)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') + 
      geom_point(aes(x = CH_year, color = model), size=1) + scale_color_manual(values = colors)
      #label_0 + label_1 + label_2 + label_3 + label_4
    avg_endo_slope <- ggplot(stats, aes(x = range, y = avg_slope_endo)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') + 
      geom_point(aes(x = CH_year, color = model), size=1) + scale_color_manual(values = colors)
      #label_0 + label_1 + label_2 + label_3 + label_4 + 
    biweekly_endo_slope <- ggplot(stats, aes(x = range, y = biweekly_slope_endo)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') + 
      geom_point(aes(x = CH_year, color = model), size=1) + scale_color_manual(values = colors)
      #label_0 + label_1 + label_2 + label_3 + label_4 + 
    avg_eco_slope <- ggplot(stats, aes(x = range, y = avg_slope_eco)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') + 
      geom_point(aes(x = CH_year, color = model), size=1) + scale_color_manual(values = colors)
      #label_0 + label_1 + label_2 + label_3 + label_4 + 
    biweekly_eco_slope <- ggplot(stats, aes(x = range, y = biweekly_slope_eco)) + geom_boxplot(outlier.shape = NA) + 
      stat_boxplot(geom ='errorbar') + labs(x = '', y = '') + coord_cartesian(clip = 'off') + 
      geom_point(aes(x = CH_year, color = model), size=1) + scale_color_manual(values = colors)
      #label_0 + label_1 + label_2 + label_3 + label_4
    model_legend <- get_legend(CDI + guides(fill = guide_legend(nrow=1, title = 'model')) + 
                                 theme(legend.position = 'bottom'))
    
    title <- ggdraw() + draw_label(sprintf('%s, scenario "%s"', loc_texts[i], scenario), fontface='bold')
    row0 <- plot_grid(
      CDI + theme(legend.position="none"), no_switch + theme(legend.position="none"), ncol = 2,
      labels = c('CDI', 'no ecodormancy rate'), label_y = 1.01, label_x = 0.1, hjust = 0, scale = 0.95)
    row1 <- plot_grid(
      switch_day + theme(legend.position="none"), DD_h + theme(legend.position="none"), ncol = 2,
      labels = c('day of ecodormancy switch (from Nov 01)', 'DD_h'), label_y = 1.01, label_x = 0.1, hjust = 0, scale = 0.95)
    row2 <- plot_grid(
      avg_endo_slope + theme(legend.position="none"), biweekly_endo_slope + theme(legend.position="none"),
      ncol = 2, labels = c('mean acclimation slope', 'max biweekly mean acclimation slope'),
      label_y = 1.01, scale = 0.95, label_x = 0.1, hjust = 0)
    row3 <- plot_grid(
      avg_eco_slope + theme(legend.position="none"), biweekly_eco_slope + theme(legend.position="none"),
      ncol = 2, labels = c('mean deacclimation slope', 'max biweekly mean deacclimation slope'),
      label_y = 1.01, scale = 0.95, label_x = 0.1, hjust = 0)
    top <- plot_grid(title, model_legend, nrow = 2, rel_heights = c(.75, 4), scale = .95)
    plot_grid(top, row0, row1, row2, row3, nrow = 5, rel_heights = c(.7, 2.5, 2.5, 2.5, 2.5), axis = 'lrtb')
  
    ggsave(paste0('plots/', scenario, '_', file_names[i]), width = 16, height = 25, limitsize = F)
  }
}



