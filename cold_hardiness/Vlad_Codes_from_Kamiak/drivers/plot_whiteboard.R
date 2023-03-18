setwd('~/CSANR/Cold/')
library(lubridate)
library(plyr)
library(tidyr)
library(ggplot2)
library(data.table)

scenarios <- c('rcp45', 'rcp85')

temp <- na.omit(read.csv('data/future_temp.csv'))
temp$month <- as.factor(temp$month)
levels(temp$month) <- c('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Sep', 'Oct', 'Nov', 'Dec')
temp$month <- factor(temp$month, levels=c('Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'))
temp$time_frame <- as.factor(temp$time_frame)
levels(temp$time_frame) <- c('2025–2049', '2050–2074', '2075–2099')

for (scenario in scenarios)
  for (variety in unique(temp$variety)) {
    if (variety == 'Chardonnay') {thresholds <- c(14, 3)} else if (variety == 'Cabernet Sauvignon') thresholds <- c(13, 5)
    
    df <- temp[temp$scenario == scenario & temp$variety == variety, ]
    df <- data.frame(melt(setDT(df), measure.vars=c('t_mean', 'hist_t_mean'), variable.name='data_type', value.name='t_mean'))
    levels(df$data_type) <- c('modeled', 'obs_hist')
    df$data_type <- factor(df$data_type, levels=c('obs_hist', 'modeled'))
    ggplot(df, aes(x=data_type, y=t_mean, color=CDI_increase)) + 
      geom_point(size = 0.15, stroke = 0, position = position_jitter(w = 0.3, h = 0)) + labs(x='', y='t_mean, °C') +
      scale_color_gradient2(midpoint=0, low="blue", mid="white", high="red") + 
      facet_wrap(~ time_frame + month, nrow=3) + geom_hline(yintercept=thresholds, linetype='dashed',size=.1)
    ggsave(sprintf('plots/future_temp_%s_%s.png', variety, scenario), width = 15, height = 10)
  }

#df <- read.csv('data/CDI_increase_1980-2016_2025-2049.csv')
#nrow(temp[temp$scenario == 'rcp45' & temp$variety == unique(temp$variety)[2] & temp$CDI_increase > 0, ])
