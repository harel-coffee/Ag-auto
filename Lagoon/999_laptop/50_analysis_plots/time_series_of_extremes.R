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
plot_dir <- paste0(in_dir, "plots/")

###############
ann_all_doomsday <- readRDS(paste0(in_dir, "/ann_all_last_days.rds")) %>%
                    data.table()

extremes <- read.csv(file = paste0(in_dir, "extremes_per_group.csv"), 
                     header = T, as.is=T)

##############

ann_all_doomsday <- orig_dt
selected_models <- c("BNU_ESM")

ann_all_doomsday <- ann_all_doomsday %>%
                    filter(location %in% extremes$location &
                           time_period != 1950-2005 &
                           model %in% selected_models) %>%
                    data.table()

ann_all_doomsday <- within(ann_all_doomsday, 
                           remove(month, day, precip, time_period))

extremes <- within(extremes, remove(cluster, avg_ann_prec))
ann_all_doomsday <- merge(ann_all_doomsday, extremes, all.x=TRUE)

dt_db <- ann_all_doomsday
dt_db$cluster <- factor(dt_db$cluster, levels=c(4, 3, 2, 1))

cluster_label <- c(4, 3, 2, 1)
categ_label <- c("most precip", "less precip", 
                 "lesser precip", "least precip")

TS <- ggplot(dt_db) + 
      geom_line(aes(x=year, y=annual_cum_precip)) +
      facet_grid(. ~ emission ~ cluster ~ condition)

ggsave(filename = paste0("TS.png"), 
       plot = TS, 
       width = 8, height = 4, units = "in", 
       dpi=600, device = "png",
       path=plot_dir)





