rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
library(maps)

source_path_1 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

options(digit=9)
options(digits=9)
# --------------------------------------------------------------------
# location and elevations
param_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
info <- read.csv(paste0(param_dir, "useless_clusters/loc_fip_clust_elev.csv"), as.is=T)
elevation_info <- subset(info, select=c(location, elevation))
precip_info <- subset(info, select=c(location, ann_prec_mean))
# --------------------------------------------------------------------
#######################
#######################      PRECIP. PLOT
#######################
##### read file
in_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
file_name <- "precip_avgs.rds"

observed_dt <- readRDS(paste0(in_dir, file_name)) %>% data.table()
setnames(observed_dt, old=c("precip_avg"), new=c("annual_cum_precip"))
outputs <- cluster_yr_avging(observed_dt, scale=FALSE, no_clusters=4)

clusters <- outputs[[1]]
cluster_obj <- outputs[[2]]
clusters$cluster <- factor(clusters$cluster)
########################################### for table
###########################################
clusters <- data.table(clusters)
clusters[, .(prec_mean_mean = mean(ann_prec_mean)), by = c("cluster")]
clusters[, .(prec_mean_range = range(ann_prec_mean)), by = c("cluster")]

############################ 
clusters <- merge(clusters, elevation_info)
clusters[, .(elev_mean = mean(elevation)), by = c("cluster")]
clusters[, .(elev_range = range(elevation)), by = c("cluster")]
######################### end of table

precip_plt <- geo_map_of_clusters(clusters) +
               ggtitle("clustering by avg. annual. precip. (observed)")
# --------------------------------------------------------------------
#######################
#######################      ELEVATION PLOT
#######################
##### read file
in_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
observed <- read.csv(paste0(in_dir, "loc_fip_clust_elev.csv"), as.is=T)
observed <- within(observed, remove(centroid, ann_prec_mean, cluster, fips))

outputs <- cluster_by_elevation(observed_dt=observed, scale=FALSE, no_clusters=4)
clusters <- outputs[[1]]
cluster_obj <- outputs[[2]]

clusters$cluster <- factor(clusters$cluster)
########################################### for table
###########################################
clusters <- data.table(clusters)
clusters <- merge(clusters, precip_info)

clusters[, .(prec_mean_mean = mean(ann_prec_mean)), by = c("cluster")]
clusters[, .(prec_mean_range = range(ann_prec_mean)), by = c("cluster")]

clusters[, .(elev_mean = mean(elevation)), by = c("cluster")]
clusters[, .(elev_range = range(elevation)), by = c("cluster")]
######################### end of table

elevation_plt <- geo_map_of_clusters(clusters) + 
                 ggtitle("clustering by elevation")
# --------------------------------------------------------------------
#######################
#######################      Precip-Elevation 4 clusters
#######################
in_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
plot_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
observed <- read.csv(paste0(in_dir, "loc_fip_clust_elev.csv"), as.is=T)
observed <- within(observed, remove(centroid, cluster, fips))

outputs <- cluster_by_precip_elev(observed_dt = observed, scale=FALSE, no_clusters=4)
clusters <- outputs[[1]]
cluster_obj <- outputs[[2]]
clusters$cluster <- factor(clusters$cluster)

########################################### for table
###########################################
clusters <- data.table(clusters)

clusters[, .(prec_mean_mean = mean(ann_prec_mean)), by = c("cluster")]
clusters[, .(prec_mean_range = range(ann_prec_mean)), by = c("cluster")]

clusters[, .(elev_mean = mean(elevation)), by = c("cluster")]
clusters[, .(elev_range = range(elevation)), by = c("cluster")]
######################### end of table

PE_4_plt <- geo_map_of_clusters(clusters) + 
            ggtitle("clustering by both precip and elevation")
PE_4_plt
# --------------------------------------------------------------------
#######################
#######################      Precip-Elevation 5 clusters
#######################
in_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
plot_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
observed <- read.csv(paste0(in_dir, 
	                        "useless_clusters/", 
	                        "loc_fip_clust_elev.csv"), as.is=T)

observed <- within(observed, remove(centroid, cluster, fips))

outputs <- cluster_by_precip_elev(observed, scale=FALSE, no_clusters=5)
clusters <- outputs[[1]]
cluster_obj <- outputs[[2]]
clusters$cluster <- factor(clusters$cluster)
########################################### for table
###########################################
clusters <- data.table(clusters)

clusters[, .(prec_mean_mean = mean(ann_prec_mean)), by = c("cluster")]
clusters[, .(prec_mean_range = range(ann_prec_mean)), by = c("cluster")]

clusters[, .(elev_mean = mean(elevation)), by = c("cluster")]
clusters[, .(elev_range = range(elevation)), by = c("cluster")]
######################### end of table

PE_5_plt <- geo_map_of_clusters(clusters) + 
            ggtitle("clustering by both precip and elevation")
PE_5_plt
# --------------------------------------------------------------------
#######################
#######################      arrange
#######################

all_p <- ggarrange(plotlist = list(precip_plt, elevation_plt, 
                                   PE_4_plt, PE_5_plt),
                   ncol = 4, nrow = 1, common.legend = FALSE)

# --------------------------------------------------------------------
#######################
#######################      save
#######################
ggsave(filename = "all_clusters.png", 
       plot = all_p, device = "png",
       width = 24, height = 6, units = "in", dpi=400,
       path=plot_dir)
