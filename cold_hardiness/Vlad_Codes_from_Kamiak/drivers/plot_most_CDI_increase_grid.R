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
               'base10_chilling_sum', 'DD_heating_sum', 'variety')
lat <- 47.09375
long <- -118.59375
elev <- raw_grid[raw_grid$lat == prosser_lat & raw_grid$long == prosser_long, 'elev']
hist_file <- sprintf('data/WA_gridmet/CH_observed_historical_data_%s_%s.csv', lat, long)
model = 'bcc-csm1-1-m'
scenario = 'RCP45'
future_file <- sprintf('data/CH_data_%s_%s.csv', lat, long) # bcc-csm1-1-m, RCP45
datafiles = list('1'=hist_file, '2'=future_file)
loc_temps <- list(NULL)
for (i in 1:2) {
  loc <- read.csv(datafiles[[i]], stringsAsFactors = F)[col_names]
  loc <- loc[loc$variety == 'Cabernet Sauvignon', col_names[1:ncol(loc)-1]]
  names(loc)[2:4] <- c('tmax', 'tmin', 'tCH')
  loc$Date <- as.Date(loc$Date)
  loc$month <- month(loc$Date)
  loc$year <- year(loc$Date)
  loc$CH_year <- ifelse(loc$month > 9, loc$year + 1, ifelse(loc$month < 5, loc$year, NA))
  if (i == 2)
    loc <- loc[!is.na(loc$CH_year) & loc$CH_year >= 2025 & loc$CH_year <= 2049, ]
  loc$CH_year <- as.factor(loc$CH_year)
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
    facet_wrap(~ CH_year, scales = 'x_free', ncol = 10) +
    scale_fill_manual(name = '', values = c('grid t°' = 'orangered3')) +
    scale_colour_manual(breaks=c('grid t°CH'), name = '', values = c('orangered3')) +
    guides(fill=guide_legend(order=1), color=guide_legend(order=2, override.aes=list(linetype=c(3)))) +
    theme(legend.position="top", legend.key.width=unit(2.6, "line"), axis.text.x=element_text(size=7)) +
    labs(x=ifelse(i==2, 'date', ''), y='temperatures, °C') + coord_cartesian(clip="off", ylim=c(-27, 30))
  
  loc_temps[[i]] <- loc_temps[[i]] + 
    geom_text(data = chils[chils$col == 2, ], size = 2.5, hjust = 1, vjust = 1,
              aes(x = Date, y = Inf, label = lab), parse = T) + 
    geom_text(data = heats[rects$col == 2, ], size = 2.5, hjust = 1, vjust = 1,
              aes(x = Date, y = Inf, label = lab), parse = T)
  
  if (i == 2) {
    loc_temps[[i]] <- loc_temps[[i]] + guides(fill = F, color = F, linetype = F) + labs(title = ' ')
    title_text <- sprintf('grid with most CDI increase (%s, %s) under %s, %s', lat, long, model, scenario)
    title <- ggdraw() + draw_label(title_text, fontface='bold')
    plot_grid(title, loc_temps[[1]], loc_temps[[2]], nrow = 3, hjust = 0, labels = c('', '', ''),
              rel_heights = c(.1, 1.075, .8), axis = 'lrtb')
    ggsave(sprintf('plots/CD_increase_%s_%s_%s_%s.png', lat, long, model, scenario),
           width = 26, height = 15, limitsize = F)
  }
}

library(data.table)
hist <- read.csv(hist_file, stringsAsFactors = F)[col_names]
hist$type = '1980-2016'
future <- read.csv(future_file, stringsAsFactors = F)[col_names]
future$type = '2025-2049'
df <- rbind(hist, future)
names(df)[4] <- 't_CH'
df$Date <- as.Date(df$Date)
df$month <- month(df$Date)
df$year <- year(df$Date)
df$CH_year <- ifelse(df$month > 8, df$year + 1, ifelse(df$month < 6, df$year, NA))
df <- na.omit(df)
df <- df[df$Date >= as.Date('1980-09-01') & 
           (df$type == '1980-2016' | (df$CH_year >= 2025 & df$CH_year < 2050)), ]
df$day <- as.integer(df$Date - as.Date(sprintf('%s-09-01', df$CH_year-1)))
#df$month <- as.ordered(month(df$Date))
dt <- data.table(df)[ , .(avg=mean(t_CH), min=min(t_CH), max=max(t_CH)), by = .(type, day)]
m <- data.frame(melt(dt, measure.vars=c('avg', 'min', 'max'),
                      variable.name = "aggregation", value.name = "t_CH"))

ggplot(m, aes(x=day, y=t_CH, color=type)) + geom_path() + facet_wrap(~ aggregation, ncol=1) +
  labs(x='days since Sep 1')
ggsave('plots/tCH_comparison_1980_2016_bcc-csm1-1-m_rcp45_2025-2049.png', width = 9, height = 7)
