setwd('~/CSANR/Cold/')
library(lubridate)
library(tidyr)
library(ggplot2)
library(data.table)
library(ggmap)
library(cowplot)
library(geosphere)

raw_grid <- read.table('data/all_calibrated_plus_uncalibrated_soil.txt')[c(3:4, 78)]
names(raw_grid) <- c('lat', 'long', 'elev')

col_names <- c('Date', 't_max', 't_min', 'predicted_Hc', 'dormancy_period',
               'base10_chilling_sum', 'DD_heating_sum')
prosser_lat <- 46.28125
prosser_long <- -119.71875
prosser_elev <- raw_grid[raw_grid$lat == prosser_lat & raw_grid$long == prosser_long, 'elev']
prosser_file_name <- sprintf('CH_data_%s_%s.csv', prosser_lat, prosser_long)
cascades_lat <- 48.96875
cascades_long <- -120.21875
cascades_elev <- raw_grid[raw_grid$lat == cascades_lat & raw_grid$long == cascades_long, 'elev']
cascades_file_name <- sprintf('CH_data_%s_%s.csv', cascades_lat, cascades_long)

data_dir <- 'data/Prosser vs Cascades/'
file_names <- c(prosser_file_name, cascades_file_name)
loc_names <- c('Prosser', 'Cascades')
loc_texts <- c(sprintf('Prosser grid %s, %s (elev. %dm)', prosser_lat, prosser_long, round(prosser_elev)),
               sprintf('Cascades grid %s, %s (elev. %dm)', cascades_lat, cascades_long, round(cascades_elev)))
# Determine sorting order (warmest to coldest) of the model/scenario combinations.
avg_temps <- list()
for (dir in list.dirs(data_dir, recursive = F, full.names = F)) {
  if (dir == 'obs_hist') 
    next
  loc <- read.csv(paste0(data_dir, dir, '/rcp85/', file_names[1]))[col_names]
  loc$Date <- as.Date(loc$Date)
  loc$year <- year(loc$Date)
  loc$month <- month(loc$Date)
  loc <- loc[loc$year >= 2050 & loc$year <= 2080 & (loc$month == 12 | loc$month == 1 | loc$month == 2), ]
  avg_temps[dir] <- mean(loc$t_max + loc$t_min)/2
}
sorted_dirs <- names(avg_temps)[rev(order(unlist(avg_temps)))]

for (loc_i in 1:2) {
  loc_temps <- list(NULL)
  for (dir in c('obs_hist', list.dirs(data_dir, recursive = F, full.names = F))) {
    i = ifelse(dir == 'obs_hist', 1, 2)
    if (i == 1 & !is.null(loc_temps[[1]]))
      next
    for (scenario in c('rcp85', 'rcp45')) {
      loc <- read.csv(paste0(data_dir, dir, ifelse(dir == 'obs_hist', '/', sprintf('/%s/', scenario)),
                             file_names[loc_i]))[col_names]
      names(loc)[2:4] <- c('tmax', 'tmin', 'tCH')
      loc$Date <- as.Date(loc$Date)
      loc$month <- month(loc$Date)
      loc$year <- year(loc$Date)
      loc$CH_year <- as.factor(ifelse(loc$month > 9, loc$year + 1, ifelse(loc$month < 5, loc$year, NA)))
      CH_years <- as.integer(levels(loc$CH_year))
      levels(loc$CH_year) <- sprintf('%d-%d', CH_years-1, CH_years)
      loc <- na.omit(loc)
      
      loc$dormancy_period[loc$month == 4 & day(loc$Date) == 30] <- 2
      rle_res <- rle(loc$dormancy_period)
      ends <- cumsum(rle_res$lengths)
      starts <- append(1, ends[1:(length(ends) - 1)] + 1)
      bad_interval <- ends == starts
      chils <- data.frame(CH_year = loc$CH_year[ends],
                          chil = loc$base10_chilling_sum[starts],
                          Date = loc$Date[starts],
                          col = as.factor(rle_res$values))
      chils$lab <- sprintf("DD['c']:%s", round(chils$chil))
      chils$lab[!bad_interval] <- ''
      starts <- starts[!bad_interval]
      ends <- ends[!bad_interval]
      rle_res$values <- rle_res$values[!bad_interval]
      rects <- data.frame(CH_year = loc$CH_year[ends], start = loc$Date[starts],
                          end = loc$Date[ends], col = as.factor(rle_res$values))
      heats <- data.frame(CH_year = loc$CH_year[ends], heat = loc$DD_heating_sum[ends],
                          Date = loc$Date[ends], col = as.factor(rle_res$values))
      heats$lab <- sprintf("DD['h']:%s", round(heats$heat))
      is_end_CH_year <- month(rects$start) == 4 & day(rects$start) == 30
      rects$start[is_end_CH_year] <- dmy(sprintf('01/10/%s', year(rects$start[is_end_CH_year])))
      rects$Date <- rects$start
      
      loc_temps[[i]] <- ggplot(loc, aes(x = Date)) +
        geom_rect(data = rects[rects$col == 1, ],
                  aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf, linetype = "1"),
                  alpha = .4, fill = 'blue') +
        geom_rect(data = rects[rects$col == 2, ],
                  aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf, linetype = "2"),
                  alpha = .4, fill = 'green') + 
        scale_linetype_manual(values = c("1" = 0, "2" = 0), name = "dormancy period",
                              guide = guide_legend(override.aes = list(fill = c("blue", "green"),
                                                                       alpha = .27)))
      loc_temps[[i]] <- loc_temps[[i]] + 
        geom_ribbon(aes(ymin = tmin, ymax = tmax, fill = 'grid t°')) +
        geom_line(aes(y = tCH, color = 'grid t°CH'), linetype = 3) +
        scale_x_date(date_labels = "%b 1", date_breaks = "1 months") +
        facet_wrap(~ CH_year, scales = 'free', ncol = 6) +
        scale_fill_manual(name = '', values = c('grid t°' = 'orangered3')) +
        scale_colour_manual(breaks=c('grid t°CH'), name = '', values = c('orangered3')) +
        guides(fill = guide_legend(order = 1), color = guide_legend(order = 2, override.aes = list(
          linetype = c(3)))) +
        theme(legend.position="top", legend.key.width = unit(2.6, "line"),
              axis.text.x = element_text(size = 7)) +
        labs(x = ifelse(i == 2, 'date', ''), y = 'temperatures, °C') + coord_cartesian(clip = "off")
      
      loc_temps[[i]] <- loc_temps[[i]] + 
        geom_text(data = chils[chils$col == 2, ], size = 2.5, hjust = 1, vjust = 1,
                  aes(x = Date, y = Inf, label = lab), parse = T) + 
        geom_text(data = heats[rects$col == 2, ], size = 2.5, hjust = 1, vjust = 1,
                  aes(x = Date, y = Inf, label = lab), parse = T)
      
      if (i == 2) {
        loc_temps[[i]] <- loc_temps[[i]] + guides(fill = F, color = F, linetype = F) + labs(title = ' ')
        title_text <- paste0(loc_texts[loc_i], sprintf(', model: %s, scenario: %s', dir, scenario))
        title <- ggdraw() + draw_label(title_text, fontface='bold')
        plot_grid(title, loc_temps[[1]], loc_temps[[2]], nrow = 3, hjust = 0, labels = c('', '', ''),
                  rel_heights = c(.1, 1.2, 2.6), axis = 'lrtb')
        dir_i <- which(sorted_dirs == dir)
        ggsave(sprintf('plots/prosser vs cascades/%2d_%s_%s_%s.png', dir_i, dir, scenario,
                       loc_names[loc_i]), width = 16, height = 40, limitsize = F)
      }
    }
    
  }
}
