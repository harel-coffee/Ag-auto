rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
library(maps)

source_path_1 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

options(digit=9)
options(digits=9)

##### read file
in_dir <- "/Users/hn/Documents/GitHub/Kirti/Lagoon/parameters/"
plot_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/"
observed <- read.csv(paste0(in_dir, "loc_fip_clust_elev.csv"), as.is=T)
head(observed, 2)
observed <- within(observed, remove(centroid, cluster, fips))
head(observed, 2)

outputs <- cluster_by_precip_elev(observed, scale=FALSE, no_clusters=5)
clusters <- outputs[[1]]
cluster_obj <- outputs[[2]]
head(clusters, 2)
head(observed, 2)

centes_4_2d_plot <- data.table(cluster_obj$centers)
centes_4_2d_plot$cluster <- 5:1
write.table(centes_4_2d_plot, file = paste0(plot_dir, "five_centroids.csv"),
            row.names=FALSE, na="", col.names=TRUE, sep=",")
################################################
########################
# Do the following so in the map they are discrete 
# for sake of coloring.
clusters$cluster <- factor(clusters$cluster)
cluster_plt <- geo_map_of_clusters(clusters) + ggtitle("clustering by both precip and elevation")

plot_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/"
ggsave(filename = "precip_elev_5_clusters.png", 
       plot = cluster_plt, device = "png",
       width = 6, height = 6, units = "in", dpi=500,
       path=plot_dir)

# round columns to 2 decimal
clusters <- clusters %>% 
            mutate_at(vars(elevation, ann_prec_mean, 
                           elev_centriod, prec_centroid), funs(round(., 2)))
# save in parameter
out_dir <- "/Users/hn/Documents/GitHub/Kirti/Lagoon/parameters/"
write.table(clusters, 
            file = paste0(out_dir, "precip_elev_5_clusters.csv"),
            row.names = FALSE, na="", 
            col.names=TRUE, sep=",")




