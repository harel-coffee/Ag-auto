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
plot_dir <- paste0(in_dir, "plots/wtr_yr/")
if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}

##############################

fileN <- "wtr_yr_sept_all_last_days"
dt_tb <- data.table(readRDS(paste0(in_dir, fileN, ".rds")))
dt_tb <- cluster_numeric_2_str(dt_tb)
head(dt_tb, 2)

plot_col <- "annual_cum_precip"
y_lab <- "annual (water year) cum. precip. (mm)"
###########################################################
#
# separate data 4 separate plots
dt_tb_45 <- dt_tb %>% filter(emission=="RCP 4.5") %>% data.table()
dt_tb_85 <- dt_tb %>% filter(emission=="RCP 8.5") %>% data.table()

# dt_tb_noMH <- dt_tb %>% filter(time_period != "1950-2005") %>% data.table()
# dt_tb_45_noMH <- dt_tb_noMH %>% filter(emission=="RCP 4.5") %>% data.table()
# dt_tb_85_noMH <- dt_tb_noMH %>% filter(emission=="RCP 8.5") %>% data.table()

###########################################################
wtr_yr_cum_prec <- ann_wtrYr_chunk_cum_box_cluster_x(dt = dt_tb, 
                                                     y_lab = y_lab, 
                                                     tgt_col = plot_col)

ggsave(filename = paste0("AV_", fileN, ".png"), 
       plot = wtr_yr_cum_prec, 
       width = 10, height = 3, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)


wtr_yr_cum_prec <- ann_wtrYr_chunk_cum_box_cluster_x(dt = dt_tb_45, 
                                                     y_lab = y_lab, 
                                                     tgt_col = plot_col)
ggsave(filename = paste0("AV_", fileN, "_45.png"), 
       plot = wtr_yr_cum_prec, 
       width = 5, height = 3, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)

wtr_yr_cum_prec <- ann_wtrYr_chunk_cum_box_cluster_x(dt = dt_tb_85, 
                                                     y_lab = y_lab, 
                                                     tgt_col = plot_col)
ggsave(filename = paste0("AV_", fileN, "_85.png"), 
       plot = wtr_yr_cum_prec, 
       width = 5, height = 3, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)


# wtr_yr_cum_prec <- ann_wtrYr_chunk_cum_box_cluster_x(dt = dt_tb_noMH, 
#                                                      y_lab = y_lab, 
#                                                      tgt_col = plot_col)
# ggsave(filename = paste0("AV_", fileN, "_noMH.png"), 
#        plot = wtr_yr_cum_prec, 
#        width = 10, height = 3, units = "in", 
#        dpi=400, device = "png",
#        path = plot_dir)

# wtr_yr_cum_prec <- ann_wtrYr_chunk_cum_box_cluster_x(dt = dt_tb_45_noMH, 
#                                                      y_lab = y_lab, 
#                                                      tgt_col = plot_col)
# ggsave(filename = paste0("AV_", fileN, "_45_noMH.png"), 
#        plot = wtr_yr_cum_prec, 
#        width = 5, height = 3, units = "in", 
#        dpi=400, device = "png",
#        path = plot_dir)

# wtr_yr_cum_prec <- ann_wtrYr_chunk_cum_box_cluster_x(dt = dt_tb_85_noMH, 
#                                                      y_lab = y_lab, 
#                                                      tgt_col = plot_col)
# ggsave(filename = paste0("AV_", fileN, "_85_noMH.png"), 
#        plot = wtr_yr_cum_prec, 
#        width = 5, height = 3, units = "in", 
#        dpi=400, device = "png",
#        path = plot_dir)

###########################################################



