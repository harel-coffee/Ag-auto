library(lubridate)
library(tidyr)
library(ggplot2)
library(data.table)

output_dir <- '/data/rajagopalan/CH_output_data/bcc-csm1-1-m_rcp45_Cabernet_2025+/'
print(length(list.files(path=output_dir, pattern="*.csv", full.names=T)))
n <- 0
for (path in list.files(path="/data/rajagopalan/CH_output_data/modeled/bcc-csm1-1-m/rcp45/",
                        pattern="*.csv", full.names=T)) {
  df <- read.csv(path)
  n <- n + 1
  write.csv(df[df$variety == 'Cabernet Sauvignon' & df$hardiness_year >= 2025, ],
            file=paste0(output_dir, basename(path)))
}
print(n)
print(length(list.files(path=output_dir, pattern="*.csv", full.names=T)))
