rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)

source_path_1 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

options(digit=9)
options(digits=9)
########################################################################
########################################################################

in_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/storm/"
plot_dir <- paste0(in_dir, "plots/")
           
all_storms <- readRDS(paste0(in_dir, "all_storms.rds"))
all_storms <- convert_5_numeric_clusts_to_alphabet(data_tb = all_storms)
head(all_storms, 2)

# all_storms <- cluster_numeric_2_str(all_storms)
# all_storms$return_period <- factor(all_storms$return_period,
#                                    levels = c("1950-2005", "1979-2016", 
#                                                "2006-2025", "2026-2050",
#                                                "2051-2075", "2076-2099"))

all_storms <- all_storms %>%
              filter(return_period != "1979-2016")%>%
              data.table()

storm_diffs <- storm_diff_obs_or_modeled(dt_dt =all_storms, diff_from="2006-2025")

diff_quan <- storm_25_quantiles(storm_diffs, tgt_col= "storm_diff") 
perc_diff_quan <- storm_25_quantiles(storm_diffs, tgt_col= "perc_diff")

storm_diffs_box <- storm_diff_box_25yr(data_tb=storm_diffs, tgt_col="storm_diff") + 
                   coord_cartesian(ylim = c(diff_quan[1], diff_quan[2]))

storm_diffs_percentage_box <- storm_diff_box_25yr(storm_diffs, tgt_col="perc_diff") + 
                              coord_cartesian(ylim = c(perc_diff_quan[1], perc_diff_quan[2]))

ggsave(filename = "storm_diffs_box.png",
       plot = storm_diffs_box, 
       width = 9, height = 3.5, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)

ggsave(filename = "storm_diffs_percentage_box.png",
       plot = storm_diffs_percentage_box, 
       width = 9, height = 3.5, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)

