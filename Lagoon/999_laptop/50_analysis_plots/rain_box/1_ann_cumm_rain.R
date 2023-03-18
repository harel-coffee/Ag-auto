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
plot_dir <- paste0(in_dir, "plots/annual/")
if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
##############################
fileN <- "ann_cum_rain"
dt_tb <- data.table(readRDS(paste0(in_dir, fileN, ".rds")))
head(dt_tb, 2)

###
#
# separate files for separate plots
#
###
dt_tb_45 <- dt_tb %>% filter(emission=="RCP 4.5") %>% data.table()
dt_tb_85 <- dt_tb %>% filter(emission=="RCP 8.5") %>% data.table()

# dt_tb_noMH <- dt_tb %>% filter(time_period != "1950-2005") %>% data.table()
# dt_tb_45_noMH <- dt_tb_noMH %>% filter(emission=="RCP 4.5") %>% data.table()
# dt_tb_85_noMH <- dt_tb_noMH %>% filter(emission=="RCP 8.5") %>% data.table()

##############################################

plot_col <- "annual_cum_rain"
y_lab <- "rain (mm)"

ttl <- "annual cum. rain"
subttl <- " "

ann_box_p <- ann_wtrYr_chunk_cum_box_cluster_x(dt=dt_tb, 
                                               y_lab=y_lab, 
                                               tgt_col = plot_col, 
                                               ttl, subttl)

ann_box_p <- ann_box_p +  ggtitle(ttl) # , , subtitle=subttl
ggsave(filename = paste0("AV_", fileN, ".png"), 
       plot = ann_box_p, 
       width = 10, height = 3, units = "in", 
       dpi=300, device = "png",
       path = plot_dir)

#############################################
#
# separate plots
#
ann_box_p <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb_45, y_lab, 
                                               tgt_col = plot_col, 
                                               ttl, subttl)
ann_box_p <- ann_box_p +  ggtitle(ttl) # , , subtitle=subttl

ggsave(filename = "AV_ann_cum_rain_45.png",
       plot = ann_box_p, 
       width = 5.5, height = 3.5, units = "in", 
       dpi=300, device = "png",
       path = plot_dir)

ann_box_p <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb_85, y_lab, tgt_col = plot_col,
                                               ttl, subttl)

ann_box_p <- ann_box_p +  ggtitle(ttl) # , , subtitle=subttl
ggsave(filename = "AV_ann_cum_rain_85.png",
       plot = ann_box_p, 
       width = 5.5, height = 3.5, units = "in", 
       dpi=300, device = "png",
       path = plot_dir)
################################################
################################################
################################################
########
########     Snow
########

plot_col <- "annual_cum_snow"
y_lab <- "snow (mm)"

ttl <- "annual cum. snow"
subttl <- " "

ann_box_p <- ann_wtrYr_chunk_cum_box_cluster_x(dt=dt_tb, 
                                               y_lab=y_lab, 
                                               tgt_col = plot_col, 
                                               ttl, subttl)

ann_box_p <- ann_box_p +  ggtitle(ttl) # , , subtitle=subttl
ggsave(filename = "AV_ann_cum_snow.png", 
       plot = ann_box_p, 
       width = 10, height = 3, units = "in", 
       dpi=300, device = "png",
       path = plot_dir)

#############################################
#
# separate plots
#
ann_box_p <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb_45, y_lab, 
                                               tgt_col = plot_col, 
                                               ttl, subttl)
ann_box_p <- ann_box_p +  ggtitle(ttl) # , , subtitle=subttl

ggsave(filename = "AV_ann_cum_snow_45.png",
       plot = ann_box_p, 
       width = 5.5, height = 3.5, units = "in", 
       dpi=300, device = "png",
       path = plot_dir)

ann_box_p <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb_85, y_lab, tgt_col = plot_col,
                                               ttl, subttl)

ann_box_p <- ann_box_p +  ggtitle(ttl) # , , subtitle=subttl
ggsave(filename = "AV_ann_cum_snow_85.png",
       plot = ann_box_p, 
       width = 5.5, height = 3.5, units = "in", 
       dpi=300, device = "png",
       path = plot_dir)