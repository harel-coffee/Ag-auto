library(lubridate)
library(plyr)
library(tidyr)
library(ggplot2)
library(data.table)

output_dir <- '/data/rajagopalan/CH_output_data/for_whiteboard/'
scenarios <- c('rcp45', 'rcp85')
future_temp <- NULL

for (path in list.files(path=output_dir, pattern="*.csv", full.names=T)) {
  print(path)
  if (!startsWith(basename(path), 'CDI_increase') & !startsWith(basename(path), 'future_temp'))
    future_temp <- rbind(future_temp, read.csv(path))
}

write.csv(future_temp, file=paste0(output_dir, 'future_temp.csv'), row.names=F)
