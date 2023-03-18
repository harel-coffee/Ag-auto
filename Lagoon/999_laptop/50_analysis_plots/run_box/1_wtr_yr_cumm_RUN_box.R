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
in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/runoff/"
plot_dir <- paste0(in_dir, "plots/wtr_yr/")

##############################
fileN <- "all_wtr_yr_cum_runoff_LD"
dt_tb <- data.table(readRDS(paste0(in_dir, fileN, ".rds")))
head(dt_tb, 2)

#######################################
#
# separate data 4 separate plots
# dt_tb_noMH <- dt_tb %>% filter(time_period != "1950-2005") %>% data.table()

dt_tb_45 <- dt_tb %>% filter(emission=="RCP 4.5") %>% data.table()
dt_tb_85 <- dt_tb %>% filter(emission=="RCP 8.5") %>% data.table()

# dt_tb_45_noMH <- dt_tb_noMH %>% filter(emission=="RCP 4.5") %>% data.table()
# dt_tb_85_noMH <- dt_tb_noMH %>% filter(emission=="RCP 8.5") %>% data.table()

#######################################
plot_col <- "annual_cum_runbase"
y_lab <- "runoff (mm)"
ttl <- "annual (water year) cum. runoff"
subttl <- " "
wtr_yr_cum_run <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb, 
                                                    y_lab=y_lab, 
                                                    tgt_col=plot_col,
                                                    ttl, 
                                                    subttl)
wtr_yr_cum_run <- wtr_yr_cum_run + ggtitle(ttl) # , , subtitle=subttl
ggsave(filename = paste0(fileN, ".png"), 
       plot = wtr_yr_cum_run, 
       width = 10, height = 3.5, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)

#########################################
#

# wtr_yr_cum_run <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb_noMH, 
#                                                     y_lab=y_lab, 
#                                                     tgt_col=plot_col)

# ggsave(filename = paste0(fileN, "_noMH.png"), 
#        plot = wtr_yr_cum_run, 
#        width = 9, height = 3.5, units = "in", 
#        dpi=600, device = "png",
#        path = plot_dir)
#
# separate plots
#
wtr_yr_cum_run <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb_45, 
                                                    y_lab=y_lab, 
                                                    tgt_col=plot_col)

wtr_yr_cum_run <- wtr_yr_cum_run + ggtitle(ttl)
ggsave(filename = paste0(fileN, "_45.png"), 
       plot = wtr_yr_cum_run, 
       width = 5.5, height = 3.5, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)


wtr_yr_cum_run <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb_85, 
                                                    y_lab=y_lab, 
                                                    tgt_col=plot_col)

ggsave(filename = paste0(fileN, "_85.png"), 
       plot = wtr_yr_cum_run, 
       width = 5.5, height = 3.5, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)

# wtr_yr_cum_run <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb_45_noMH, 
#                                                     y_lab=y_lab, 
#                                                     tgt_col=plot_col)

# ggsave(filename = paste0(fileN, "_45_noMH.png"), 
#        plot = wtr_yr_cum_run, 
#        width = 5, height = 3.5, units = "in", 
#        dpi=600, device = "png",
#        path = plot_dir)


# wtr_yr_cum_run <- ann_wtrYr_chunk_cum_box_cluster_x(dt_tb_85_noMH, 
#                                                     y_lab=y_lab, 
#                                                     tgt_col=plot_col)

# ggsave(filename = paste0(fileN, "_85_noMH.png"), 
#        plot = wtr_yr_cum_run, 
#        width = 5, height = 3.5, units = "in", 
#        dpi=600, device = "png",
#        path = plot_dir)

