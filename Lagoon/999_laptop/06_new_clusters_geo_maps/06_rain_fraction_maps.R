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

in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/"
rain <- readRDS(paste0(in_dir, "/rain_snow_fractions/annual_fracs.rds"))
rain <- rain %>% filter(emission == "RCP 8.5") %>% data.table()
rain <- subset(rain, select=c(location, time_period, rain_fraction))

rain <- rain[, .(avg_rain_frac = mean(rain_fraction)), 
              by = c("location", "time_period")]
rain[,(c("avg_rain_frac")) := round(.SD, 2), .SDcols=c("avg_rain_frac")]

rain_obs_map <- geo_map_of_rain_frac(rain[time_period == "1979-2016"])
rain_hist_map <- geo_map_of_rain_frac(rain[time_period == "1950-2005"])
rain_F1_map <- geo_map_of_rain_frac(rain[time_period == "2026-2050"])
rain_F2_map <- geo_map_of_rain_frac(rain[time_period == "2051-2075"])
rain_F3_map <- geo_map_of_rain_frac(rain[time_period == "2076-2099"])

obs_future <- ggarrange(plotlist = list(rain_obs_map, 
                                        rain_F1_map, rain_F2_map, 
                                        rain_F3_map),
                        ncol = 4, nrow = 1, common.legend = FALSE)

mod_hist_future <- ggarrange(plotlist = list(rain_hist_map, 
                                            rain_F1_map, rain_F2_map, 
                                            rain_F3_map),
                             ncol = 4, nrow = 1, common.legend = FALSE)

obs_mod_hist_future <- ggarrange(plotlist = list(rain_obs_map, rain_hist_map,
                                                 rain_F1_map, rain_F2_map, 
                                                 rain_F3_map),
                                 ncol = 5, nrow = 1, common.legend = FALSE)

plot_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/rain_frac_maps/"
if (dir.exists(plot_dir) == F){dir.create(path = plot_dir, recursive = T)}

ggsave(filename = paste0("rain_obs_map.png"), 
       plot = rain_obs_map, 
       width = 4, height = 4, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)


ggsave(filename = paste0("rain_hist_map.png"), 
       plot = rain_hist_map, 
       width = 4, height = 4, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)


ggsave(filename = paste0("rain_F1_map", ".png"), 
       plot = rain_F1_map, 
       width = 4, height = 4, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)


ggsave(filename = paste0("rain_F2_map", ".png"), 
       plot = rain_F2_map, 
       width = 4, height = 4, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)

ggsave(filename = paste0("rain_F3_map", ".png"), 
       plot = rain_F3_map, 
       width = 4, height = 4, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)


ggsave(filename = paste0("obs_future", ".png"), 
       plot = obs_future, 
       width = 16, height = 4, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)

ggsave(filename = paste0("mod_hist_future", ".png"), 
       plot = mod_hist_future, 
       width = 16, height = 4, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)

ggsave(filename = paste0("obs_mod_hist_future", ".png"), 
       plot = obs_mod_hist_future, 
       width = 20, height = 4, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)

