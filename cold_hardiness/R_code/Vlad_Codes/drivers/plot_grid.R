setwd('~/CSANR/Cold/')
library(lubridate)
library(tidyr)
library(ggplot2)
library(data.table)
library(ggmap)
library(cowplot)
library(geosphere)

in_detail <- TRUE
CH_data_prefix <- ifelse(in_detail, 'CH', 'output')
col_names <- c('Date', 't_max', 't_min', 'predicted_Hc')
if (in_detail) col_names <- append(
  col_names, c('dormancy_period', 'base10_chilling_sum', 'DD_heating_sum'))
raw_grid <- read.table('data/all_calibrated_plus_uncalibrated_soil.txt')[c(3:4, 78)]
names(raw_grid) <- c('lat', 'long', 'elev')
# grid step is .06250, ref. point is (45.59375, -122.15625)
step <- 0.0625
# cascades_lats <- c(47.96875, 48.90625, 45.65625, 47.21875)
# cascades_longs <- c(-121.84375, -122.21875, -122.28125, -120.78125)
# olympic_lats <- c(47.34375, 47.90625, 47.96875)
# olympic_longs <- c(-123.53125, -123.09375, -124.28125)
cascades_lats <- c(48.96875, 47.46875, 48.78125, 46.84375, 46.21875)
cascades_longs <- c(-120.21875, -120.78125, -121.84375, -121.78125, -121.46875)
olympic_lats <- c(47.84375, 47.71875)
olympic_longs <- c(-123.34375, -123.34375)
lats <- c(cascades_lats, olympic_lats)
longs <- c(cascades_longs, olympic_longs)
longs <- c(cascades_longs, olympic_longs)
#lats <- c(46.28125)
#longs <- c(-119.71875)

data_dir <- 'data/WA_gridmet/'
for (i in 1:length(lats)) {
  loc_lat <- lats[i]
  loc_long <- longs[i]
  loc_type <- ifelse(i > length(cascades_lats), 'Olympic', 'Cascades')
  #loc_type <- 'Prosser'

  loc_elev <- raw_grid[raw_grid$lat == loc_lat & raw_grid$long == loc_long, 'elev']
  near_lats <- c()
  near_longs <- c()
  for (n in -50:50)
    for (m in -50:50)
      if (distm(c(loc_long, loc_lat), c(loc_long + n*step, loc_lat + m*step),
               fun = distHaversine) <= 20000) {
        near_lats <- append(near_lats, loc_lat + m*step)
        near_longs <- append(near_longs, loc_long + n*step)
      }
  near_grid <- data.frame(lat = near_lats, long = near_longs)
  avg_elev <- mean(merge(near_grid, raw_grid)$elev)
  
  loc_file_name <- sprintf('%s_observed_historical_data_%s_%s.csv',
                           CH_data_prefix, loc_lat, loc_long)
  loc <- read.csv(paste0(data_dir, loc_file_name))[col_names]
  names(loc)[2:4] <- c('tmax', 'tmin', 'tCH')
  #loc[2:4] <- (loc[2:4] - 32) * 5/9
  loc$Date <- as.Date(loc$Date)
  loc$month <- month(loc$Date)
  loc$year <- year(loc$Date)
  loc$CH_year <- as.factor(ifelse(loc$month > 9, loc$year + 1, ifelse(loc$month < 5, loc$year, NA)))
  CH_years <- as.integer(levels(loc$CH_year))
  levels(loc$CH_year) <- sprintf('%d-%d', CH_years-1, CH_years)
  loc <- na.omit(loc)
  
  loc_temperatures <- ggplot(loc, aes(x = Date))
  if (in_detail) {
    rle_res <- rle(loc$dormancy_period)
    ends <- cumsum(rle_res$lengths)
    starts <- append(1, ends[1:(length(ends) - 1)])
    rects <- data.frame(CH_year = loc$CH_year[ends], start = loc$Date[starts],
                        end = loc$Date[ends], col = as.factor(rle_res$values))
    chils <- data.frame(CH_year = loc$CH_year[ends], chil = loc$DD_chilling_sum[starts],
                        Date = loc$Date[starts], col = as.factor(rle_res$values))
    chils$lab <- sprintf("DD['c']:%s", round(chils$chil))
    heats <- data.frame(CH_year = loc$CH_year[ends], heat = loc$DD_heating_sum[ends],
                        Date = loc$Date[ends], col = as.factor(rle_res$values))
    heats$lab <- sprintf("DD['h']:%s", round(heats$heat))
    is_end_CH_year <- month(rects$start) == 4 & day(rects$start) == 30
    rects$start[is_end_CH_year] <- dmy(sprintf('01/10/%s', year(rects$start[is_end_CH_year])))
    rects$Date <- rects$start
    loc_temperatures <- loc_temperatures +
      geom_rect(data = rects[rects$col == 1, ],
                aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf, linetype = "1"),
                alpha = .4, fill = 'blue') +
      geom_rect(data = rects[rects$col == 2, ],
                aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf, linetype = "2"),
                alpha = .4, fill = 'green') + 
      scale_linetype_manual(values = c("1" = 0, "2" = 0), name = "dormancy period",
                            guide = guide_legend(override.aes = list(fill = c("blue", "green"),
                                                                     alpha = .27)))
  }
  loc_temperatures <- loc_temperatures + 
    geom_ribbon(aes(ymin = tmin, ymax = tmax, fill = 'grid t°')) +
    geom_line(aes(y = tCH, color = 'grid t°CH'), linetype = 3) +
     scale_x_date(date_labels = "%b 1", date_breaks = "1 months") +
    facet_wrap(~ CH_year, scales = 'free') +
    scale_fill_manual(name = '', values = c('grid t°' = 'orangered3')) +
    scale_colour_manual(breaks=c('grid t°CH'), name = '', values = c('orangered3')) +
    guides(fill = guide_legend(order = 1), color = guide_legend(order = 2, override.aes = list(
      linetype = c(3)))) +
    theme(legend.position="top", legend.key.width = unit(2.6, "line"),
          axis.text.x = element_text(size = 7)) +
    labs(x = 'date', y = 'temperatures, °C') + coord_cartesian(clip = "off")
  if (in_detail)
    loc_temperatures <- loc_temperatures + 
      geom_text(data = chils[rects$col == 2, ], size = 2.5, hjust = 1, vjust = 1,
                aes(x = Date, y = Inf, label = lab), parse = T) + 
      geom_text(data = heats[rects$col == 2, ], size = 2.5, hjust = 1, vjust = 1,
                aes(x = Date, y = Inf, label = lab), parse = T)

  # Map grid location.
  google_key <- as.character(read.table('GOOGLE_API_KEY')[1, 1])
  register_google(key=google_key) # your Google API key here
  wa_map <- get_map(c(-124.5, 45.25, -116.8, 49.25)) # only AG stations in WA
  loc_map <- ggmap(wa_map) + 
    geom_point(x = loc_long, y = loc_lat, color = 'orangered3', size = 3, pch = 4, stroke = 1)
    
  title <- ggdraw() + draw_label(sprintf(
    'elevation %dm (average elevation in 20km radius %dm), grid %s, %s (%s)',
    round(loc_elev), round(avg_elev), loc_lat, loc_long, loc_type), fontface='bold')
  plot_grid(title, loc_map, loc_temperatures, nrow = 3, labels = c('', '', ''),
            rel_heights = c(.1, 1.25, 3), axis = 'lrtb')
  ggsave(sprintf('plots/%s_grid_%s_%s.png', loc_type, loc_lat, loc_long), width = 16, height = 9, limitsize = F)
  #ggsave(sprintf('plots/%s.tiff', station_name), width = 16, height = 9, limitsize = F)
  
}
ggmap(wa_map) +
  geom_point(x = -123.34375, y = 47.84375, color = 'orangered3', size = 3, pch = 4, stroke = 1)
