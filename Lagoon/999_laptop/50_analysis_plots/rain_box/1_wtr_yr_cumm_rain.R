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
plot_dir <- paste0(in_dir, "plots/wtr_yr/")
if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
##############################
fileN <- "wtr_yr_cum_rain"
dt_tb <- data.table(readRDS(paste0(in_dir, fileN, ".rds")))
head(dt_tb, 2)

#######################################
#
# separate data 4 separate plots

dt_tb_45 <- dt_tb %>% filter(emission=="RCP 4.5") %>% data.table()
dt_tb_85 <- dt_tb %>% filter(emission=="RCP 8.5") %>% data.table()
#######################################
plot_col <- "annual_cum_rain"
y_lab <- "rain (mm)"
ttl <- "annual (water year) cum. rain"
subttl <- " "
wtr_yr_cum_run <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb, 
                                                    y_lab=y_lab, 
                                                    tgt_col=plot_col,
                                                    ttl, 
                                                    subttl)
wtr_yr_cum_run <- wtr_yr_cum_run + ggtitle(ttl) # , , subtitle=subttl
ggsave(filename = paste0("AV_", fileN, ".png"), 
       plot = wtr_yr_cum_run, 
       width = 10, height = 3.5, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)

#
# separate plots
#
wtr_yr_cum_run <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb_45, 
                                                    y_lab=y_lab, 
                                                    tgt_col=plot_col)

wtr_yr_cum_run <- wtr_yr_cum_run + ggtitle(ttl)
ggsave(filename = paste0("AV_", fileN, "_45.png"), 
       plot = wtr_yr_cum_run, 
       width = 5.5, height = 3.5, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)


wtr_yr_cum_run <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb_85, 
                                                    y_lab=y_lab, 
                                                    tgt_col=plot_col)

ggsave(filename = paste0("AV_", fileN, "_85.png"), 
       plot = wtr_yr_cum_run, 
       width = 5.5, height = 3.5, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)


#######
#######3 Snow
#######
plot_col <- "annual_cum_snow"
y_lab <- "snow (mm)"
ttl <- "annual (water year) cum. snow"
subttl <- " "
wtr_yr_cum_run <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb, 
                                                    y_lab=y_lab, 
                                                    tgt_col=plot_col,
                                                    ttl, 
                                                    subttl)
wtr_yr_cum_run <- wtr_yr_cum_run + ggtitle(ttl) # , , subtitle=subttl
ggsave(filename = "AV_wtr_yr_cum_snow.png",
       plot = wtr_yr_cum_run, 
       width = 10, height = 3.5, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)

#
# separate plots
#
wtr_yr_cum_run <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb_45, 
                                                    y_lab=y_lab, 
                                                    tgt_col=plot_col)

wtr_yr_cum_run <- wtr_yr_cum_run + ggtitle(ttl)
ggsave(filename = "AV_wtr_yr_cum_snow_45.png", 
       plot = wtr_yr_cum_run, 
       width = 5.5, height = 3.5, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)


wtr_yr_cum_run <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb_85, 
                                                    y_lab=y_lab, 
                                                    tgt_col=plot_col)

ggsave(filename = "AV_wtr_yr_cum_snow_85.png", 
       plot = wtr_yr_cum_run, 
       width = 5.5, height = 3.5, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)
