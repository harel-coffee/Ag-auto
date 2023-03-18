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

in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/rain/"
plot_dir <- paste0(in_dir, "plots/monthly/")
if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
##############################
fileN <- "month_cum_rain"
dt_tb <- data.table(readRDS(paste0(in_dir, fileN, ".rds")))
head(dt_tb, 2)
##################################################################
#
# Rain
#
tg_col <- "monthly_cum_rain"
y_labb <- "rain (mm)"
ttl <- "monthly cum. rain"
subttl <- " "

box_plt <- box_trend_monthly_cum(dt=dt_tb, p_type="box", 
                                 y_lab=y_labb, tgt_col = tg_col# ,ttl, subttl
                                 )

box_plt <- box_plt + ggtitle(ttl) # , subtitle=subttl
ggsave(filename = "AV_rain_monthly_box.png", 
       plot = box_plt, 
       width = 14, height = 6, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)

##################################################################
# Snow
tg_col <- "monthly_cum_snow"
y_labb <- "snow (mm)"
ttl <- "monthly cum. snow"
subttl <- " "

box_plt <- box_trend_monthly_cum(dt=dt_tb, p_type="box", 
                                 y_lab=y_labb, tgt_col = tg_col# ,ttl, subttl
                                 )

box_plt <- box_plt + ggtitle(ttl) # , subtitle=subttl
ggsave(filename = "AV_snow_monthly_box.png", 
       plot = box_plt, 
       width = 14, height = 6, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)
##################################################################
#
# Rain
#
tg_col <- "monthly_cum_rain"
y_labb <- "rain (mm)"
ttl <- "monthly cum. rain"
subttl <- " "

dt_tb <- dt_tb %>% filter(month %in% c(11, 12)) %>% data.table()
nov_Dec <- Nov_Dec_cum_box(dt=dt_tb, y_lab = y_labb, tgt_col= tg_col)
nov_Dec <- nov_Dec +  ggtitle(ttl) # , , subtitle=subttl

ggsave(filename = "AV_rain_nov_Dec_box.png", 
       plot = nov_Dec, 
       width = 11, height = 6, units = "in", 
       dpi=300, device = "png",
       path = plot_dir)
##########################################################
#
# Snow
#
##########################################################
tg_col <- "monthly_cum_snow"
y_labb <- "snow (mm)"
ttl <- "monthly cum. snow"
subttl <- " "

nov_Dec <- Nov_Dec_cum_box(dt=dt_tb, y_lab = y_labb, tgt_col= tg_col)
nov_Dec <- nov_Dec +  ggtitle(ttl) # , , subtitle=subttl

ggsave(filename = "AV_snow_nov_Dec_box.png", 
       plot = nov_Dec, 
       width = 11, height = 6, units = "in", 
       dpi=300, device = "png",
       path = plot_dir)



