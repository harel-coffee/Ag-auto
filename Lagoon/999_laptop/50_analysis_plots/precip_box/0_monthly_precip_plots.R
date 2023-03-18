rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
options(digit=9)
options(digits=9)

source_path_1 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/cum_precip/"
plot_dir <- paste0(in_dir, "plots/monthly/")

##############################

file <- "month_all_last_days"
plotting_col <- "monthly_cum_precip"
dt_tb <- data.table(readRDS(paste0(in_dir, file, ".rds")))
head(dt_tb, 2)

y_lab <- "monthly cum. precip. (mm)"

###############################
#
# Box plot
#
###############################
tg_col <- "monthly_cum_precip"
box_plt <- box_trend_monthly_cum(dt=dt_tb, p_type="box", 
                                 y_lab = y_lab, tgt_col= tg_col)

ggsave(filename = "AV_monthly_box.png", 
       plot = box_plt, 
       width = 14, height = 6, units = "in", 
       dpi=300, device = "png",
       path = plot_dir)

###############################
#
# Separate Nov and Dec.
#
dt_tb_NOV_Dec <- dt_tb %>% filter(month %in% c(11, 12)) %>% data.table()
nov_Dec <- Nov_Dec_cum_box(dt=dt_tb_NOV_Dec, y_lab = y_lab, tgt_col= tg_col)
ggsave(filename = "AV_Nov_Dec_box.png", 
       plot = nov_Dec, 
       width = 10, height = 6, units = "in", 
       dpi=300, device = "png",
       path = plot_dir)

###############################
#
# Trend line
#
###############################

# trend_med <- box_trend_monthly_cum(dt=dt_tb, p_type="trend", 
#                                    trend_type="median", y_lab=y_lab)
# ggsave(filename = paste0(file, "_trend_med.png"), 
#        plot = trend_med, 
#        width = 40, height = 20, units = "in", 
#        dpi=600, device = "png",
#        path = plot_dir)


