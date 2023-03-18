rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)

source_path_1 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

options(digit=9)
options(digits=9)

in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/cum_precip/"


ann_all_doomsday <- readRDS(paste0(in_dir, "/ann_all_last_days.rds")) %>%
                    data.table()

observed <- ann_all_doomsday %>% 
            filter(time_period == "1979-2016" &
                   emission == "RCP 4.5") %>%
            data.table()

observed <- subset(observed, select=c(location, year,
                                      annual_cum_precip,
                                      cluster))

observed <- observed %>% 
            group_by(location, cluster) %>%
            summarise(avg_ann_prec = mean(annual_cum_precip))%>%
            data.table()

wettest_locs <- observed %>%
                group_by(cluster) %>%
                slice(which.max(avg_ann_prec))%>%
                data.table()

wettest_locs$condition <- "wettest"

driest_locs <- observed %>%
                group_by(cluster) %>%
                slice(which.min(avg_ann_prec))%>%
                data.table()
driest_locs$condition <- "driest"

extremes_per_group <- rbind(wettest_locs, driest_locs)

write.table(extremes_per_group, 
            file = paste0(in_dir, "extremes_per_group.csv"), 
            row.names=FALSE, col.names=TRUE, 
            sep=",", na="")


