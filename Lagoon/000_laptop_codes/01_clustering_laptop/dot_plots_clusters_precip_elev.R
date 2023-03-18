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

######################################################################
#
#      4 clusters based on (precip, elevation)
#

in_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
plot_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
observed <- read.csv(paste0(in_dir, 
                            "useless_clusters/", 
                            "loc_fip_clust_elev.csv"), 
                    as.is=T)
observed <- within(observed, remove(centroid, cluster, fips))

outputs <- cluster_by_precip_elev(observed_dt = observed, 
                                  scale=FALSE, 
                                  no_clusters=4)
clusters <- outputs[[1]]
cluster_obj <- outputs[[2]]
clusters$cluster <- factor(clusters$cluster)

four_clusters <- ggplot(clusters, aes(x=elevation, y=ann_prec_mean, color=cluster)) +
                 geom_point() + 
                 theme(legend.position = "bottom",
                       legend.title = element_blank(),
                       legend.spacing.x = unit(.1, 'line'),
                       legend.text = element_text(size = 9, face="bold"))+
                 guides(colour = guide_legend(override.aes = list(size=3)))
######################################################################
#
#      5 clusters based on (precip, elevation)
#

in_dir <- "/Users/hn/Documents/GitHub/Ag/Lagoon/parameters/"
plot_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
observed <- read.csv(paste0(in_dir, "loc_fip_clust_elev.csv"), as.is=T)
observed <- within(observed, remove(centroid, cluster, fips))

outputs <- cluster_by_precip_elev(observed, 
                                  scale=FALSE, 
                                  no_clusters=5)
clusters <- outputs[[1]]
cluster_obj <- outputs[[2]]
clusters$cluster <- factor(clusters$cluster)
five_clusters <- ggplot(clusters, aes(x=elevation, y=ann_prec_mean, color=cluster)) +
                 geom_point() + 
                 theme(legend.position = "bottom",
                       legend.title = element_blank(),
                       legend.spacing.x = unit(.1, 'line'),
                       legend.text = element_text(size = 9, face="bold"))+
                 guides(colour = guide_legend(override.aes = list(size=3)))

# ---------------------------------------------------------------------------------
#######################
#######################      arrange
#######################
all_p <- ggarrange(plotlist = list(four_clusters, five_clusters),
                   ncol = 2, nrow = 1, common.legend = FALSE)

# ---------------------------------------------------------------------------------
#######################
#######################      save
#######################
plot_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"

ggsave(filename = "cluster_visualization.png", 
       plot = all_p, device = "png",
       width = 7, height = 7, units = "in", dpi=400,
       path=plot_dir)
