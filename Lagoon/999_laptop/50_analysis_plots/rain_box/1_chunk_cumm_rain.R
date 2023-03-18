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
plot_dir <- paste0(in_dir, "plots/chunky/")
if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
##############################
fileN <- "Sept_March_cum_rain"
dt_tb <- data.table(readRDS(paste0(in_dir, fileN, ".rds")))
head(dt_tb, 2)

############################################################
#
# separate files 4 separate plots
dt_tb_45 <- dt_tb %>% filter(emission=="RCP 4.5") %>% data.table()
dt_tb_85 <- dt_tb %>% filter(emission=="RCP 8.5") %>% data.table()

############################################################

plot_col <- "chunk_cum_rain"
y_lab <- "rain (mm)"

ttl <- "Sept. - Mar. cum. rain"
subttl <- " "
chunk_box <- ann_wtrYr_chunk_cum_box_cluster_x(dt=dt_tb,
                                               y_lab = y_lab,
                                               tgt_col=plot_col,
                                               ttl, subttl)

chunk_box <- chunk_box + ggtitle(ttl) # , , subtitle=subttl
ggsave(filename = paste0("AV_rain_", fileN, ".png"), 
       plot = chunk_box, 
       width = 10, height = 3, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)

chunk_box <- ann_wtrYr_chunk_cum_box_cluster_x(dt=dt_tb_45,
                                                y_lab = y_lab,
                                                tgt_col=plot_col,
                                                ttl, subttl)

chunk_box <- chunk_box + ggtitle(ttl) # , , subtitle=subttl
ggsave(filename = paste0("AV_rain_", fileN, "_45.png"), 
       plot = chunk_box, 
       width = 5.5, height = 3.5, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)


chunk_box <- ann_wtrYr_chunk_cum_box_cluster_x(dt=dt_tb_85,
                                                y_lab = y_lab,
                                                tgt_col=plot_col,
                                                ttl, subttl)
chunk_box <- chunk_box + ggtitle(ttl) # , , subtitle=subttl
ggsave(filename = paste0("AV_rain_", fileN, "_85.png"), 
       plot = chunk_box, 
       width = 5.5, height = 3.5, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)
##################################################################
#
#      Snow
#
##################################################################

plot_col <- "chunk_cum_snow"
y_lab <- "snow (mm)"

ttl <- "Sept. - Mar. cum. snow"
subttl <- " "
chunk_box <- ann_wtrYr_chunk_cum_box_cluster_x(dt=dt_tb,
                                               y_lab = y_lab,
                                               tgt_col=plot_col,
                                               ttl, subttl)

chunk_box <- chunk_box + ggtitle(ttl) # , , subtitle=subttl
ggsave(filename = "AV_Sept_March_cum_snow.png",
       plot = chunk_box, 
       width = 10, height = 3, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)

chunk_box <- ann_wtrYr_chunk_cum_box_cluster_x(dt=dt_tb_45,
                                                y_lab = y_lab,
                                                tgt_col=plot_col,
                                                ttl, subttl)

chunk_box <- chunk_box + ggtitle(ttl) # , , subtitle=subttl
ggsave(filename = paste0("AV_Snow_", fileN, "_45.png"), 
       plot = chunk_box, 
       width = 5.5, height = 3.5, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)

chunk_box <- ann_wtrYr_chunk_cum_box_cluster_x(dt=dt_tb_85,
                                                y_lab = y_lab,
                                                tgt_col=plot_col,
                                                ttl, subttl)
chunk_box <- chunk_box + ggtitle(ttl) # , , subtitle=subttl
ggsave(filename = paste0("AV_Snow_", fileN, "_85.png"), 
       plot = chunk_box, 
       width = 5.5, height = 3.5, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)

