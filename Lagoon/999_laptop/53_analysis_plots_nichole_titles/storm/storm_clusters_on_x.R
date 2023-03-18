rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)

b <- "/Users/hn/Documents/GitHub/Ag/Lagoon/"
source_path_1 = paste0(b, "core_lagoon.R")
source_path_2 = paste0(b, "core_plot_lagoon.R")
source(source_path_1)
source(source_path_2)

options(digit=9)
options(digits=9)
###########################################################
###########################################################
base_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
in_dir <- paste0(base_dir, "storm/")
plot_dir <- paste0(base_dir, "plots/storm/new_2_storm/")
if (dir.exists(plot_dir) == F){
  dir.create(path = plot_dir, recursive = T)}

all_storms <- readRDS(paste0(in_dir, "all_storms.rds"))
# all_storms <- get_ridof_canada(all_storms)

head(all_storms, 2)
all_storms <- all_storms %>% 
              filter(return_period != "1979-2016") %>% 
              data.table()
              
all_storms <- all_storms %>% 
              filter(return_period != "2006-2025") %>% 
              data.table()
all_storms <- convert_5_numeric_clusts_to_alphabet(all_storms)

all_storms <- within(all_storms, 
                    remove(five_years, ten_years, 
                           fifteen_years, twenty_years))

###############################################################

for (em in unique(all_storms$emission)){
  AV_title <- paste0("25-year/24-hr design storm intensity")
  
  curr_dt <- all_storms %>% 
             filter(emission == em) %>% 
             data.table()

  quans <- storm_25_quantiles(curr_dt, 
                              tgt_col= "twenty_five_years")

  
  AV_plt <- box_dt_25_clust_x(within(curr_dt, remove(location, model))) +
            ggtitle(label = AV_title) +
            coord_cartesian(ylim = c(quans[1], quans[2]))
  
  ##################
  ################## Unbias diffs
  ##################
  box_title <- "Difference (%) in design storm intensity"
  unbias_diffs <- storm_diff_obs_or_modeled(dt_dt = curr_dt, 
                                            diff_from = "1950-2005")
  quans_diff <- storm_25_quantiles(unbias_diffs,
                                   tgt_col = "perc_diff")

  unbias_diffs_perc_box <- storm_diff_box_25yr_clust_x(unbias_diffs, 
                                                       tgt_col="perc_diff") + 
                           ggtitle(label=box_title) +
                           coord_cartesian(ylim = c(quans_diff[1], 
                                                    quans_diff[2]))
  storm_plt <- ggarrange(plotlist = list(AV_plt, 
                                         unbias_diffs_perc_box),
                       ncol = 1, nrow = 2, 
                       common.legend = TRUE, legend="bottom")

  storm_plt <- annotate_figure(storm_plt,
                              top = text_grob(em, 
                                              face = "bold", 
                                              size = 10,
                                              color="red"))
  
  ggsave(filename = paste0("storm_", 
                           substr(em, 5, 5), 
                           substr(em, 7, 7), ".png"),
         plot = storm_plt, 
         width=5.5, height=4, units = "in", 
         dpi=600, device = "png", 
         path = plot_dir)

}

