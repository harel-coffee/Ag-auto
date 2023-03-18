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

in_dir <- "/Users/hn/Documents/GitHub/Kirti/Lagoon/parameters/"
cluster_info <- read.csv(paste0(in_dir, "loc_fip_clust.csv"), header=T, as.is=T)
cluster_info <- within(cluster_info, remove(centroid, cluster))

precip_map <- geo_map_of_precip(cluster_info)
plot_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/"
ggsave(filename = paste0("precip_map.png"), 
       plot = precip_map, 
       width = 4, height = 4, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)

