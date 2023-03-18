
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
############################################################
#
# plotting the 5 clusters on the map was originally 
# done in the following code:
# /Users/hn/Documents/GitHub/Ag/Lagoon/000_laptop_codes/01_clustering_laptop/1_precip_elevation_clustering/cluster_precip_elev_5_clusters.R 
# where clustering was also done.
# The current code, the one you are reading/running now,
# is written to change the cluster labels in increasing order
# from West to East.
#
############################################################

in_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
cluster_info <- read.csv(paste0(in_dir, "precip_elev_5_clusters.csv"), 
                          header=T, as.is=T)
cluster_info <- within(cluster_info, remove(elevation, ann_prec_mean, 
                                            elev_centriod, prec_centroid))
cluster_info$cluster <- factor(cluster_info$cluster)

cluster_plt <- geo_map_of_clusters(cluster_info) + 
               ggtitle("clustering by both precip and elevation")


# cluster_info <- cluster_info %>%
#                 mutate(season = case_when(cluster == 1 ~ 4,
#                                           cluster == 2 ~ 3,
#                                           cluster == 3 ~ 1,
#                                           cluster == 4 ~ 2,
#                                           cluster == 5 ~ 5)
#                        ) %>% data.table()

# cluster_info <- within(cluster_info, remove(cluster))
# setnames(cluster_info, old=c("season"), new=c("cluster"))

cluster_info$cluster <- factor(cluster_info$cluster)

cluster_info <- convert_5_numeric_clusts_to_alphabet(cluster_info)

cluster_plt <- geo_map_of_clusters(cluster_info) + 
               ggtitle("clustering by both precip and elevation")

write.table(cluster_info, 
            file = paste0(in_dir, "precip_elev_5_clusters.csv"),
            row.names = FALSE, na="", 
            col.names=TRUE, sep=",")

plt_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
ggsave(filename = "precip_elev_5_clusters_8_by_6.png", 
       plot = cluster_plt, device = "png",
       width = 8, height = 6, units = "in", dpi=600,
       path=plt_dir)

cluster_info <- get_ridof_canada(cluster_info)
cluster_plt <- geo_map_of_clusters(cluster_info)
ggsave(filename = "precip_elev_5_clusters_8_by_6_no_canada.png", 
       plot = cluster_plt, device = "png",
       width = 8, height = 6, units = "in", dpi=600,
       path=plt_dir)






